
import os
import numpy as np
import scipy.sparse as sp
import torch
from tadc.utils import set_seed


def load_original_data(dataset_name, attacked_root='attacked_graphs'):

    data_path = os.path.join(attacked_root, dataset_name, 'original_data.npz')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"原始数据文件不存在: {data_path}")

    print(f"加载原始数据: {data_path}")

    with np.load(data_path, allow_pickle=True) as data:
        # 加载数据
        adj = data['adj'].item() if isinstance(data['adj'], np.ndarray) else data['adj']
        features = data['features'].item() if isinstance(data['features'], np.ndarray) else data['features']
        labels = data['labels']
        idx_train = data['idx_train']
        idx_val = data['idx_val']
        idx_test = data.get('idx_test', data.get('idx_unlabeled', data['idx_val']))

        # 确保邻接矩阵是scipy稀疏矩阵
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        else:
            adj = adj.tocsr()

        # 确保特征矩阵是numpy数组
        if sp.issparse(features):
            features = features.toarray()
        elif isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        # 确保标签是numpy数组
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        print(f"数据形状: adj={adj.shape}, features={features.shape}, labels={labels.shape}")
        print(f"训练集大小: {len(idx_train)}, 验证集大小: {len(idx_val)}, 测试集大小: {len(idx_test)}")

        return labels, features, idx_train, idx_val, idx_test, adj


def load_attacked_data(dataset_name, perturbation_rate, attacked_root='attacked_graphs'):

    # 首先加载原始数据获取特征和标签
    labels, features, idx_train, idx_val, idx_test, _ = load_original_data(dataset_name, attacked_root)

    # 加载攻击后的邻接矩阵
    attacked_file = os.path.join(attacked_root, dataset_name, f'attacked_p{perturbation_rate:.2f}.npz')

    if not os.path.exists(attacked_file):
        raise FileNotFoundError(f"攻击数据文件不存在: {attacked_file}")

    print(f"加载攻击数据: {attacked_file}")

    with np.load(attacked_file, allow_pickle=True) as data:
        adj_attacked = data['adj_attack'].item() if isinstance(data['adj_attack'], np.ndarray) else data['adj_attack']

        # 确保邻接矩阵是scipy稀疏矩阵
        if not sp.issparse(adj_attacked):
            adj_attacked = sp.csr_matrix(adj_attacked)
        else:
            adj_attacked = adj_attacked.tocsr()

        print(f"攻击后邻接矩阵形状: {adj_attacked.shape}, 非零元素: {adj_attacked.nnz}")

        return labels, features, idx_train, idx_val, idx_test, adj_attacked


def get_available_perturbation_rates(dataset_name, attacked_root='attacked_graphs'):

    dataset_dir = os.path.join(attacked_root, dataset_name)

    if not os.path.exists(dataset_dir):
        return []

    perturbation_rates = []

    # 检查标准扰动率文件
    standard_rates = [0.05, 0.10, 0.15, 0.20, 0.25]

    for rate in standard_rates:
        attacked_file = os.path.join(dataset_dir, f'attacked_p{rate:.2f}.npz')
        if os.path.exists(attacked_file):
            perturbation_rates.append(rate)

    return sorted(perturbation_rates)


def get_available_datasets(attacked_root='attacked_graphs'):

    if not os.path.exists(attacked_root):
        return []

    datasets = []

    for item in os.listdir(attacked_root):
        item_path = os.path.join(attacked_root, item)
        if os.path.isdir(item_path):
            # 检查是否有原始数据文件
            original_file = os.path.join(item_path, 'original_data.npz')
            if os.path.exists(original_file):
                datasets.append(item)

    return sorted(datasets)


def load_attack_summary(dataset_name, attacked_root='attacked_graphs'):

    summary_file = os.path.join(attacked_root, dataset_name, 'attack_summary.npy')

    if not os.path.exists(summary_file):
        return None

    try:
        summary = np.load(summary_file, allow_pickle=True)
        if hasattr(summary, 'item'):
            summary = summary.item()
        return summary
    except Exception as e:
        print(f"加载攻击摘要失败: {e}")
        return None


def compare_adjacency_matrices(adj_original, adj_attacked):

    # 确保都是稀疏矩阵
    if not sp.issparse(adj_original):
        adj_original = sp.csr_matrix(adj_original)
    if not sp.issparse(adj_attacked):
        adj_attacked = sp.csr_matrix(adj_attacked)

    # 计算差异
    diff = adj_attacked - adj_original

    # 统计信息
    original_edges = adj_original.nnz // 2  # 无向图，除以2
    attacked_edges = adj_attacked.nnz // 2
    added_edges = (diff > 0).nnz // 2
    removed_edges = (diff < 0).nnz // 2

    perturbation_rate = (added_edges + removed_edges) / original_edges if original_edges > 0 else 0

    return {
        'original_edges': original_edges,
        'attacked_edges': attacked_edges,
        'added_edges': added_edges,
        'removed_edges': removed_edges,
        'total_changes': added_edges + removed_edges,
        'perturbation_rate': perturbation_rate
    }


if __name__ == "__main__":
    # 测试数据加载功能
    print("=== 测试扰动数据加载功能 ===")

    # 获取可用数据集
    datasets = get_available_datasets()
    print(f"可用数据集: {datasets}")

    for dataset in datasets[:1]:  # 只测试第一个数据集
        print(f"\n--- 测试数据集: {dataset} ---")

        # 获取可用扰动率
        rates = get_available_perturbation_rates(dataset)
        print(f"可用扰动率: {rates}")

        # 加载原始数据
        try:
            labels, features, idx_train, idx_val, idx_test, adj_orig = load_original_data(dataset)
            print("原始数据加载成功")

            # 测试一个扰动率
            if rates:
                rate = rates[0]
                labels_att, features_att, idx_train_att, idx_val_att, idx_test_att, adj_att = load_attacked_data(
                    dataset, rate)
                print(f"扰动数据 (p={rate}) 加载成功")

                # 比较邻接矩阵
                comparison = compare_adjacency_matrices(adj_orig, adj_att)
                print(f"邻接矩阵比较结果: {comparison}")

        except Exception as e:
            print(f"测试失败: {e}")