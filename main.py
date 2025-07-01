"""
main.py
CurvEdgeNet主运行脚本
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch_geometric.utils import dense_to_sparse

# 导入自定义模块
from model import CurvEdgeNet
from opt import CurvEdgeNetTrainer, CircRNADiseaseDataset, CrossValidator, create_negative_samples
from layers import CurvatureCalculator


# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_similarity_matrices(data_dir):
    """
    加载相似性矩阵

    Args:
        data_dir: 数据目录

    Returns:
        circrna_sim: circRNA相似性矩阵
        disease_sim: 疾病相似性矩阵
    """
    print("加载相似性矩阵...")

    # 加载circRNA相似性矩阵
    circrna_sim_path = os.path.join(data_dir, 'circrna_similarity.csv')
    if os.path.exists(circrna_sim_path):
        circrna_sim = pd.read_csv(circrna_sim_path, index_col=0).values
    else:
        # 如果文件不存在，创建示例数据
        print("未找到circRNA相似性矩阵，创建示例数据...")
        num_circrna = 1000
        circrna_sim = np.random.rand(num_circrna, num_circrna)
        circrna_sim = (circrna_sim + circrna_sim.T) / 2  # 确保对称
        np.fill_diagonal(circrna_sim, 1.0)

    # 加载疾病相似性矩阵
    disease_sim_path = os.path.join(data_dir, 'disease_similarity.csv')
    if os.path.exists(disease_sim_path):
        disease_sim = pd.read_csv(disease_sim_path, index_col=0).values
    else:
        # 如果文件不存在，创建示例数据
        print("未找到疾病相似性矩阵，创建示例数据...")
        num_disease = 200
        disease_sim = np.random.rand(num_disease, num_disease)
        disease_sim = (disease_sim + disease_sim.T) / 2  # 确保对称
        np.fill_diagonal(disease_sim, 1.0)

    return torch.FloatTensor(circrna_sim), torch.FloatTensor(disease_sim)


def load_association_matrix(data_dir, num_circrna, num_disease):
    """
    加载关联矩阵

    Args:
        data_dir: 数据目录
        num_circrna: circRNA数量
        num_disease: 疾病数量

    Returns:
        association_matrix: 关联矩阵
        positive_pairs: 正样本对
    """
    print("加载关联矩阵...")

    association_path = os.path.join(data_dir, 'circrna_disease_associations.csv')

    if os.path.exists(association_path):
        # 加载真实关联数据
        associations_df = pd.read_csv(association_path)
        association_matrix = np.zeros((num_circrna, num_disease))

        positive_pairs = []
        for _, row in associations_df.iterrows():
            circrna_idx = int(row['circrna_id'])
            disease_idx = int(row['disease_id'])
            association_matrix[circrna_idx, disease_idx] = 1
            positive_pairs.append((circrna_idx, disease_idx))
    else:
        # 创建示例关联数据
        print("未找到关联矩阵，创建示例数据...")
        association_matrix = np.zeros((num_circrna, num_disease))

        # 随机生成一些正关联
        num_associations = min(5000, num_circrna * num_disease // 20)  # 稀疏关联
        positive_pairs = []

        for _ in range(num_associations):
            circrna_idx = np.random.randint(0, num_circrna)
            disease_idx = np.random.randint(0, num_disease)
            if association_matrix[circrna_idx, disease_idx] == 0:
                association_matrix[circrna_idx, disease_idx] = 1
                positive_pairs.append((circrna_idx, disease_idx))

    return torch.FloatTensor(association_matrix), positive_pairs


def create_edge_index(hetero_adj, threshold=0.1):
    """
    创建边索引

    Args:
        hetero_adj: 异构网络邻接矩阵
        threshold: 边权重阈值

    Returns:
        edge_index: 边索引
    """
    # 过滤小权重的边
    hetero_adj_filtered = hetero_adj.clone()
    hetero_adj_filtered[hetero_adj_filtered < threshold] = 0

    # 转换为稀疏格式
    edge_index, _ = dense_to_sparse(hetero_adj_filtered)

    return edge_index


def prepare_datasets(circrna_sim, disease_sim, association_matrix, positive_pairs,
                     test_size=0.2, val_size=0.1, negative_ratio=1):
    """
    准备训练、验证和测试数据集

    Args:
        circrna_sim: circRNA相似性矩阵
        disease_sim: 疾病相似性矩阵
        association_matrix: 关联矩阵
        positive_pairs: 正样本对
        test_size: 测试集比例
        val_size: 验证集比例
        negative_ratio: 负样本比例

    Returns:
        train_dataset, val_dataset, test_dataset: 数据集
    """
    print(f"准备数据集，正样本数量: {len(positive_pairs)}")

    # 创建负样本
    num_circrna, num_disease = association_matrix.shape
    negative_pairs = create_negative_samples(
        positive_pairs, num_circrna, num_disease, ratio=negative_ratio
    )

    print(f"负样本数量: {len(negative_pairs)}")

    # 分割正样本
    pos_train_val, pos_test = train_test_split(
        positive_pairs, test_size=test_size, random_state=42
    )
    pos_train, pos_val = train_test_split(
        pos_train_val, test_size=val_size / (1 - test_size), random_state=42
    )

    # 分割负样本
    neg_train_val, neg_test = train_test_split(
        negative_pairs, test_size=test_size, random_state=42
    )
    neg_train, neg_val = train_test_split(
        neg_train_val, test_size=val_size / (1 - test_size), random_state=42
    )

    # 创建边索引
    num_total = num_circrna + num_disease
    hetero_adj = torch.zeros(num_total, num_total)
    hetero_adj[:num_circrna, :num_circrna] = circrna_sim
    hetero_adj[num_circrna:, num_circrna:] = disease_sim
    hetero_adj[:num_circrna, num_circrna:] = association_matrix
    hetero_adj[num_circrna:, :num_circrna] = association_matrix.T

    edge_index = create_edge_index(hetero_adj)

    # 创建数据集
    train_dataset = CircRNADiseaseDataset(
        circrna_sim, disease_sim, association_matrix, edge_index, pos_train, neg_train
    )
    val_dataset = CircRNADiseaseDataset(
        circrna_sim, disease_sim, association_matrix, edge_index, pos_val, neg_val
    )
    test_dataset = CircRNADiseaseDataset(
        circrna_sim, disease_sim, association_matrix, edge_index, pos_test, neg_test
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def train_model(args):
    """训练模型"""
    print("=== 开始训练模型 ===")

    # 加载数据
    circrna_sim, disease_sim = load_similarity_matrices(args.data_dir)
    num_circrna, _ = circrna_sim.shape
    num_disease, _ = disease_sim.shape

    association_matrix, positive_pairs = load_association_matrix(
        args.data_dir, num_circrna, num_disease
    )

    # 准备数据集
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        circrna_sim, disease_sim, association_matrix, positive_pairs,
        test_size=args.test_size, val_size=args.val_size, negative_ratio=args.negative_ratio
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = CurvEdgeNet(
        num_circrna=num_circrna,
        num_disease=num_disease,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_gat_layers=args.num_gat_layers,
        gamma=args.gamma,
        dropout=args.dropout
    )

    print(f"模型信息: {model.get_model_info()}")

    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    trainer = CurvEdgeNetTrainer(
        model, device=device, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience
    )

    # 训练
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # 测试
    print("=== 测试模型 ===")
    test_metrics = trainer.evaluate(test_loader)

    print("测试集结果:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # 保存结果
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'model_info': model.get_model_info(),
        'args': vars(args)
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 保存模型
    trainer.save_model(os.path.join(args.output_dir, 'final_model.pth'))

    # 绘制训练曲线
    plot_training_curves(history, args.output_dir)

    return results


def cross_validate(args):
    """执行交叉验证"""
    print("=== 开始交叉验证 ===")

    # 加载数据
    circrna_sim, disease_sim = load_similarity_matrices(args.data_dir)
    num_circrna, _ = circrna_sim.shape
    num_disease, _ = disease_sim.shape

    association_matrix, positive_pairs = load_association_matrix(
        args.data_dir, num_circrna, num_disease
    )

    # 创建负样本
    negative_pairs = create_negative_samples(
        positive_pairs, num_circrna, num_disease, ratio=args.negative_ratio
    )

    # 创建边索引
    num_total = num_circrna + num_disease
    hetero_adj = torch.zeros(num_total, num_total)
    hetero_adj[:num_circrna, :num_circrna] = circrna_sim
    hetero_adj[num_circrna:, num_circrna:] = disease_sim
    hetero_adj[:num_circrna, num_circrna:] = association_matrix
    hetero_adj[num_circrna:, :num_circrna] = association_matrix.T
    edge_index = create_edge_index(hetero_adj)

    # 创建完整数据集
    dataset = CircRNADiseaseDataset(
        circrna_sim, disease_sim, association_matrix, edge_index,
        positive_pairs, negative_pairs
    )

    # 模型参数
    model_params = {
        'num_circrna': num_circrna,
        'num_disease': num_disease,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_gat_layers': args.num_gat_layers,
        'gamma': args.gamma,
        'dropout': args.dropout
    }

    # 训练器参数
    trainer_params = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience
    }

    # 执行交叉验证
    cv = CrossValidator(CurvEdgeNet, model_params, trainer_params, k_folds=args.k_folds)
    avg_results, std_results = cv.run_cross_validation(dataset)

    # 保存交叉验证结果
    cv_results = {
        'avg_results': avg_results,
        'std_results': std_results,
        'fold_results': cv.cv_results,
        'model_params': model_params,
        'trainer_params': trainer_params
    }

    with open(os.path.join(args.output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)

    return cv_results


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # AUC曲线
    axes[0, 1].plot(history['val_auc'], label='Val AUC', color='green')
    axes[0, 1].set_title('AUC Curve')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # AUPR曲线
    axes[1, 0].plot(history['val_aupr'], label='Val AUPR', color='orange')
    axes[1, 0].set_title('AUPR Curve')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUPR')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 综合曲线
    axes[1, 1].plot(history['val_auc'], label='AUC', color='green')
    axes[1, 1].plot(history['val_aupr'], label='AUPR', color='orange')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CurvEdgeNet for circRNA-Disease Association Prediction')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--negative_ratio', type=int, default=1, help='负样本比例')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_heads', type=int, default=8, help='多头注意力头数')
    parser.add_argument('--num_gat_layers', type=int, default=3, help='GAT层数')
    parser.add_argument('--gamma', type=float, default=0.3, help='曲率影响参数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')

    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA')

    # 实验参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'cv'],
                        help='运行模式: train (训练) 或 cv (交叉验证)')
    parser.add_argument('--k_folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行实验
    if args.mode == 'train':
        results = train_model(args)
        print(f"训练完成，结果保存在 {args.output_dir}")
    elif args.mode == 'cv':
        results = cross_validate(args)
        print(f"交叉验证完成，结果保存在 {args.output_dir}")

    return results


if __name__ == "__main__":
    results = main()