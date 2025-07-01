"""
model.py
CurvEdgeNet主模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    CurvatureCalculator,
    CurvatureEnhancedGAT,
    BiLSTMModule,
    MultiHeadAttention,
    FeatureProjection,
    MLPPredictor
)


class CurvEdgeNet(nn.Module):
    """
    CurvEdgeNet: 基于曲率增强图注意力网络的circRNA-疾病关联预测模型
    """

    def __init__(self,
                 num_circrna,
                 num_disease,
                 embedding_dim=128,
                 hidden_dim=128,
                 num_heads=8,
                 num_gat_layers=3,
                 gamma=0.3,
                 dropout=0.1):
        """
        初始化CurvEdgeNet模型

        Args:
            num_circrna: circRNA数量
            num_disease: 疾病数量
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力头数
            num_gat_layers: GAT层数
            gamma: 曲率影响参数
            dropout: Dropout比率
        """
        super(CurvEdgeNet, self).__init__()

        self.num_circrna = num_circrna
        self.num_disease = num_disease
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # 1. 特征投影层
        self.circrna_projection = FeatureProjection(num_circrna, embedding_dim, dropout)
        self.disease_projection = FeatureProjection(num_disease, embedding_dim, dropout)
        self.curvature_projection = FeatureProjection(
            num_circrna + num_disease, embedding_dim, dropout
        )

        # 2. 曲率增强的图注意力网络
        self.curvature_gat = CurvatureEnhancedGAT(
            in_features=embedding_dim,
            hidden_features=hidden_dim,
            out_features=embedding_dim,
            num_heads=1,
            num_layers=num_gat_layers,
            gamma=gamma,
            dropout=dropout
        )

        # 3. 双向LSTM模块
        self.bilstm = BiLSTMModule(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            dropout=dropout
        )

        # 4. 多头自注意力机制
        self.multihead_attention = MultiHeadAttention(
            d_model=embedding_dim * 2,
            num_heads=num_heads,
            dropout=dropout
        )

        # 5. 预测器
        self.predictor = MLPPredictor(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
            dropout=0.3
        )

        # 6. 曲率计算器
        self.curvature_calculator = CurvatureCalculator()

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self,
                circrna_sim,
                disease_sim,
                association_matrix,
                edge_index,
                circrna_indices,
                disease_indices,
                return_embeddings=False):
        """
        前向传播

        Args:
            circrna_sim: circRNA相似性矩阵 [num_circrna, num_circrna]
            disease_sim: 疾病相似性矩阵 [num_disease, num_disease]
            association_matrix: 关联矩阵 [num_circrna, num_disease]
            edge_index: 边索引 [2, num_edges]
            circrna_indices: circRNA索引 [batch_size]
            disease_indices: 疾病索引 [batch_size]
            return_embeddings: 是否返回节点嵌入

        Returns:
            predictions: 预测概率 [batch_size]
            embeddings: 节点嵌入 (如果return_embeddings=True)
        """
        device = circrna_sim.device

        # 1. 构建异构网络的邻接矩阵
        hetero_adj = self._build_heterogeneous_adjacency(
            circrna_sim, disease_sim, association_matrix
        )

        # 2. 计算网络曲率
        curvature_matrix = self._compute_network_curvature(
            hetero_adj, circrna_sim, disease_sim, association_matrix
        )
        curvature_matrix = torch.FloatTensor(curvature_matrix).to(device)

        # 3. 特征投影
        h_circrna = self.circrna_projection(circrna_sim)
        h_disease = self.disease_projection(disease_sim)
        h_curvature = self.curvature_projection(curvature_matrix)

        # 4. 构建异构网络节点特征
        h_nodes = torch.cat([h_circrna, h_disease], dim=0)  # [num_circrna + num_disease, embedding_dim]

        # 5. 曲率增强的图注意力网络
        h_gat = self.curvature_gat(h_nodes, edge_index)

        # 6. 双向LSTM全局特征提取
        h_seq = self.bilstm(h_gat)

        # 7. 多尺度特征融合
        h_multi = torch.cat([h_gat, h_seq], dim=-1)  # [num_nodes, embedding_dim * 2]

        # 8. 多头自注意力
        h_fused = self.multihead_attention(h_multi)

        # 9. 提取circRNA-disease对的特征
        circrna_features = h_fused[circrna_indices]  # [batch_size, embedding_dim * 2]
        disease_features = h_fused[self.num_circrna + disease_indices]  # [batch_size, embedding_dim * 2]

        # 10. 组合特征（元素级乘法 + 连接）
        combined_features = circrna_features * disease_features  # [batch_size, embedding_dim * 2]

        # 11. 预测关联概率
        predictions = self.predictor(combined_features)

        if return_embeddings:
            return predictions, h_fused
        else:
            return predictions

    def _build_heterogeneous_adjacency(self, circrna_sim, disease_sim, association_matrix):
        """
        构建异构网络邻接矩阵

        Args:
            circrna_sim: circRNA相似性矩阵
            disease_sim: 疾病相似性矩阵
            association_matrix: 关联矩阵

        Returns:
            hetero_adj: 异构网络邻接矩阵
        """
        num_circrna = circrna_sim.size(0)
        num_disease = disease_sim.size(0)
        total_nodes = num_circrna + num_disease

        # 初始化异构邻接矩阵
        hetero_adj = torch.zeros(total_nodes, total_nodes, device=circrna_sim.device)

        # circRNA-circRNA块
        hetero_adj[:num_circrna, :num_circrna] = circrna_sim

        # 疾病-疾病块
        hetero_adj[num_circrna:, num_circrna:] = disease_sim

        # circRNA-疾病块 (关联矩阵)
        hetero_adj[:num_circrna, num_circrna:] = association_matrix
        hetero_adj[num_circrna:, :num_circrna] = association_matrix.T

        return hetero_adj

    def _compute_network_curvature(self, hetero_adj, circrna_sim, disease_sim, association_matrix):
        """
        计算网络曲率

        Args:
            hetero_adj: 异构网络邻接矩阵
            circrna_sim: circRNA相似性矩阵
            disease_sim: 疾病相似性矩阵
            association_matrix: 关联矩阵

        Returns:
            normalized_curvature: 归一化曲率矩阵
        """
        # 转换为numpy进行曲率计算
        hetero_adj_np = hetero_adj.detach().cpu().numpy()

        # 构建相似性矩阵
        num_circrna = circrna_sim.size(0)
        num_disease = disease_sim.size(0)
        total_nodes = num_circrna + num_disease

        similarity_matrix = torch.zeros(total_nodes, total_nodes, device=circrna_sim.device)
        similarity_matrix[:num_circrna, :num_circrna] = circrna_sim
        similarity_matrix[num_circrna:, num_circrna:] = disease_sim
        similarity_matrix[:num_circrna, num_circrna:] = association_matrix
        similarity_matrix[num_circrna:, :num_circrna] = association_matrix.T

        similarity_matrix_np = similarity_matrix.detach().cpu().numpy()

        # 计算曲率
        curvature_matrix = self.curvature_calculator.compute_edge_curvature(
            hetero_adj_np, similarity_matrix_np
        )

        # 归一化曲率
        normalized_curvature = self.curvature_calculator.normalize_curvature(curvature_matrix)

        return normalized_curvature

    def predict_associations(self, circrna_sim, disease_sim, association_matrix, edge_index, threshold=0.5):
        """
        预测所有可能的circRNA-疾病关联

        Args:
            circrna_sim: circRNA相似性矩阵
            disease_sim: 疾病相似性矩阵
            association_matrix: 已知关联矩阵
            edge_index: 边索引
            threshold: 预测阈值

        Returns:
            predictions: 预测结果矩阵 [num_circrna, num_disease]
        """
        self.eval()
        device = next(self.parameters()).device

        num_circrna = circrna_sim.size(0)
        num_disease = disease_sim.size(0)

        predictions = torch.zeros(num_circrna, num_disease, device=device)

        with torch.no_grad():
            # 批量预测所有可能的组合
            for i in range(num_circrna):
                for j in range(num_disease):
                    circrna_indices = torch.tensor([i], device=device)
                    disease_indices = torch.tensor([j], device=device)

                    pred = self.forward(
                        circrna_sim, disease_sim, association_matrix,
                        edge_index, circrna_indices, disease_indices
                    )

                    predictions[i, j] = pred.item()

        return predictions

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'model_name': 'CurvEdgeNet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_circrna': self.num_circrna,
            'num_disease': self.num_disease,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'gamma': self.gamma
        }

        return info