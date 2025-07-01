"""
opt.py
CurvEdgeNet的优化器、训练器和评估器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import KFold
import time
import logging
from tqdm import tqdm


class CircRNADiseaseDataset(Dataset):
    """circRNA-疾病关联数据集"""

    def __init__(self, circrna_sim, disease_sim, association_matrix, edge_index,
                 positive_pairs, negative_pairs):
        """
        初始化数据集

        Args:
            circrna_sim: circRNA相似性矩阵
            disease_sim: 疾病相似性矩阵
            association_matrix: 关联矩阵
            edge_index: 边索引
            positive_pairs: 正样本对 [(circrna_idx, disease_idx), ...]
            negative_pairs: 负样本对 [(circrna_idx, disease_idx), ...]
        """
        self.circrna_sim = circrna_sim
        self.disease_sim = disease_sim
        self.association_matrix = association_matrix
        self.edge_index = edge_index

        # 合并正负样本
        self.samples = []

        # 添加正样本
        for circrna_idx, disease_idx in positive_pairs:
            self.samples.append((circrna_idx, disease_idx, 1))

        # 添加负样本
        for circrna_idx, disease_idx in negative_pairs:
            self.samples.append((circrna_idx, disease_idx, 0))

        # 打乱样本
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        circrna_idx, disease_idx, label = self.samples[idx]

        return {
            'circrna_sim': self.circrna_sim,
            'disease_sim': self.disease_sim,
            'association_matrix': self.association_matrix,
            'edge_index': self.edge_index,
            'circrna_indices': torch.tensor([circrna_idx], dtype=torch.long),
            'disease_indices': torch.tensor([disease_idx], dtype=torch.long),
            'labels': torch.tensor([label], dtype=torch.float)
        }


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)

        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class CurvEdgeNetTrainer:
    """CurvEdgeNet训练器"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu',
                 lr=0.001, weight_decay=1e-4, patience=10):
        """
        初始化训练器

        Args:
            model: CurvEdgeNet模型
            device: 计算设备
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # 损失函数
        self.criterion = nn.BCELoss()

        # 早停
        self.early_stopping = EarlyStopping(patience=patience)

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_aupr': []
        }

        # 日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            self.optimizer.zero_grad()

            # 准备数据
            batch_data = self._prepare_batch(batch)

            # 前向传播
            predictions = self.model(
                batch_data['circrna_sim'],
                batch_data['disease_sim'],
                batch_data['association_matrix'],
                batch_data['edge_index'],
                batch_data['circrna_indices'],
                batch_data['disease_indices']
            )

            # 计算损失
            loss = self.criterion(predictions, batch_data['labels'])

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch_data = self._prepare_batch(batch)

                predictions = self.model(
                    batch_data['circrna_sim'],
                    batch_data['disease_sim'],
                    batch_data['association_matrix'],
                    batch_data['edge_index'],
                    batch_data['circrna_indices'],
                    batch_data['disease_indices']
                )

                loss = self.criterion(predictions, batch_data['labels'])
                total_loss += loss.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_data['labels'].cpu().numpy())

        # 计算评估指标
        metrics = self._compute_metrics(all_labels, all_predictions)
        metrics['val_loss'] = total_loss / len(val_loader)

        return metrics

    def fit(self, train_loader, val_loader, epochs=100):
        """训练模型"""
        self.logger.info(f"开始训练，共{epochs}个epoch")

        best_auc = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证
            val_metrics = self.evaluate(val_loader)

            # 学习率调度
            self.scheduler.step(val_metrics['auc'])

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_aupr'].append(val_metrics['aupr'])

            epoch_time = time.time() - epoch_start

            # 打印训练信息
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val AUPR: {val_metrics['aupr']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                self.save_model('best_model.pth')

            # 早停检查
            if self.early_stopping(val_metrics['auc'], self.model):
                self.logger.info(f"早停在第{epoch + 1}个epoch")
                break

        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总时间: {total_time:.2f}s，最佳AUC: {best_auc:.4f}")

        return self.history

    def _prepare_batch(self, batch):
        """准备批次数据"""
        return {
            'circrna_sim': batch['circrna_sim'][0].to(self.device),
            'disease_sim': batch['disease_sim'][0].to(self.device),
            'association_matrix': batch['association_matrix'][0].to(self.device),
            'edge_index': batch['edge_index'][0].to(self.device),
            'circrna_indices': batch['circrna_indices'].squeeze().to(self.device),
            'disease_indices': batch['disease_indices'].squeeze().to(self.device),
            'labels': batch['labels'].squeeze().to(self.device)
        }

    def _compute_metrics(self, y_true, y_pred):
        """计算评估指标"""
        # 二分类预测
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_true, y_pred),
            'aupr': average_precision_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred_binary),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary)
        }

        return metrics

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


class CrossValidator:
    """交叉验证器"""

    def __init__(self, model_class, model_params, trainer_params, k_folds=5):
        """
        初始化交叉验证器

        Args:
            model_class: 模型类
            model_params: 模型参数
            trainer_params: 训练器参数
            k_folds: 折数
        """
        self.model_class = model_class
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.k_folds = k_folds

        self.kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # 交叉验证结果
        self.cv_results = {
            'auc': [],
            'aupr': [],
            'f1': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }

    def run_cross_validation(self, dataset):
        """运行交叉验证"""
        samples = dataset.samples

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(samples)):
            print(f"\n=== Fold {fold + 1}/{self.k_folds} ===")

            # 分割数据
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            # 创建数据加载器
            train_dataset = self._create_fold_dataset(dataset, train_samples)
            val_dataset = self._create_fold_dataset(dataset, val_samples)

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            # 创建模型和训练器
            model = self.model_class(**self.model_params)
            trainer = CurvEdgeNetTrainer(model, **self.trainer_params)

            # 训练
            trainer.fit(train_loader, val_loader, epochs=50)

            # 评估
            fold_metrics = trainer.evaluate(val_loader)

            # 记录结果
            for metric in self.cv_results.keys():
                self.cv_results[metric].append(fold_metrics[metric])

            print(f"Fold {fold + 1} - AUC: {fold_metrics['auc']:.4f}, AUPR: {fold_metrics['aupr']:.4f}")

        # 计算平均结果
        avg_results = {}
        std_results = {}

        for metric in self.cv_results.keys():
            avg_results[metric] = np.mean(self.cv_results[metric])
            std_results[metric] = np.std(self.cv_results[metric])

        print(f"\n=== 交叉验证结果 ===")
        for metric in avg_results.keys():
            print(f"{metric.upper()}: {avg_results[metric]:.4f} ± {std_results[metric]:.4f}")

        return avg_results, std_results

    def _create_fold_dataset(self, original_dataset, samples):
        """为单折创建数据集"""
        # 创建新的数据集实例
        fold_dataset = CircRNADiseaseDataset.__new__(CircRNADiseaseDataset)
        fold_dataset.circrna_sim = original_dataset.circrna_sim
        fold_dataset.disease_sim = original_dataset.disease_sim
        fold_dataset.association_matrix = original_dataset.association_matrix
        fold_dataset.edge_index = original_dataset.edge_index
        fold_dataset.samples = samples

        return fold_dataset


def create_negative_samples(positive_pairs, num_circrna, num_disease, ratio=1):
    """
    创建负样本

    Args:
        positive_pairs: 正样本对
        num_circrna: circRNA数量
        num_disease: 疾病数量
        ratio: 负样本与正样本的比例

    Returns:
        negative_pairs: 负样本对
    """
    positive_set = set(positive_pairs)
    negative_pairs = []

    num_negatives = len(positive_pairs) * ratio

    while len(negative_pairs) < num_negatives:
        circrna_idx = np.random.randint(0, num_circrna)
        disease_idx = np.random.randint(0, num_disease)

        if (circrna_idx, disease_idx) not in positive_set:
            negative_pairs.append((circrna_idx, disease_idx))

    return negative_pairs