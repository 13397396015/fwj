"""
layers.py
CurvEdgeNet的神经网络层组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np


class CurvatureCalculator:
    """计算网络的Ollivier-Ricci曲率"""

    @staticmethod
    def compute_edge_curvature(adj_matrix, similarity_matrix):
        """
        计算边的离散曲率
        Args:
            adj_matrix: 邻接矩阵 [N, N]
            similarity_matrix: 相似性矩阵 [N, N]
        Returns:
            curvature_matrix: 曲率矩阵 [N, N]
        """
        n = adj_matrix.shape[0]
        curvature_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:  # 存在边
                    # 计算最短路径距离
                    if similarity_matrix[i, j] > 0:
                        d_ij = 1.0 / similarity_matrix[i, j]
                    else:
                        d_ij = 1.0

                    # 边权重
                    w_ij = similarity_matrix[i, j]

                    # 曲率计算: K(i,j) = 1 - d(i,j)/W(i,j)
                    if w_ij > 0:
                        curvature_matrix[i, j] = 1 - d_ij / w_ij
                    else:
                        curvature_matrix[i, j] = 0

        return curvature_matrix

    @staticmethod
    def normalize_curvature(curvature_matrix):
        """
        归一化曲率值到[-1, 1]
        Args:
            curvature_matrix: 原始曲率矩阵
        Returns:
            normalized: 归一化后的曲率矩阵
        """
        k_min = curvature_matrix.min()
        k_max = curvature_matrix.max()

        if k_max - k_min > 0:
            normalized = (curvature_matrix - k_min) / (k_max - k_min) * 2 - 1
        else:
            normalized = curvature_matrix

        return normalized


class CurvatureEnhancedGAT(nn.Module):
    """曲率增强的图注意力网络"""

    def __init__(self, in_features, hidden_features, out_features,
                 num_heads=1, num_layers=3, gamma=0.3, dropout=0.1):
        super(CurvatureEnhancedGAT, self).__init__()
        self.gamma = gamma
        self.num_layers = num_layers
        self.dropout = dropout

        # 构建GAT层
        self.gat_layers = nn.ModuleList()

        # 第一层
        self.gat_layers.append(
            GATConv(in_features, hidden_features, heads=num_heads, concat=True, dropout=dropout)
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_features * num_heads, hidden_features,
                        heads=num_heads, concat=True, dropout=dropout)
            )

        # 最后一层
        self.gat_layers.append(
            GATConv(hidden_features * num_heads, out_features,
                    heads=1, concat=False, dropout=dropout)
        )

    def forward(self, x, edge_index, curvature_weights=None):
        """
        前向传播
        Args:
            x: 节点特征 [N, in_features]
            edge_index: 边索引 [2, E]
            curvature_weights: 曲率权重 [E] (可选)
        Returns:
            x: 输出节点特征 [N, out_features]
        """
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, edge_index)

            # 应用曲率增强 (在注意力计算后)
            if curvature_weights is not None and i < len(self.gat_layers) - 1:
                # 简化的曲率增强：调制特征
                x = x * (1 + self.gamma * curvature_weights.mean())

            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class BiLSTMModule(nn.Module):
    """双向LSTM模块用于捕获全局依赖"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(BiLSTMModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 投影层：将双向LSTM输出映射回原始维度
        self.projection = nn.Linear(hidden_size * 2, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, seq_len, input_size] 或 [seq_len, input_size]
        Returns:
            output: 输出特征，与输入维度相同
        """
        # 确保输入是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_len, input_size]
            squeeze_output = True
        else:
            squeeze_output = False

        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 投影到原始维度
        output = self.projection(lstm_out)
        output = self.dropout(output)

        # 恢复原始维度
        if squeeze_output:
            output = output.squeeze(0)

        return output


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            mask: 注意力掩码 (可选)
        Returns:
            output: 输出特征 [batch_size, seq_len, d_model]
        """
        # 确保输入是3D张量
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_len, d_model]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.size(0)
        seq_len = x.size(1)

        # 残差连接
        residual = x

        # 生成Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 重塑并连接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影
        output = self.W_o(attention_output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        # 恢复原始维度
        if squeeze_output:
            output = output.squeeze(0)

        return output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        Args:
            Q, K, V: 查询、键、值矩阵 [batch_size, num_heads, seq_len, d_k]
            mask: 注意力掩码
        Returns:
            output: 注意力输出 [batch_size, num_heads, seq_len, d_k]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)

        return output


class FeatureProjection(nn.Module):
    """特征投影层"""

    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.projection(x)


class MLPPredictor(nn.Module):
    """MLP预测器"""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(MLPPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x).squeeze(-1)