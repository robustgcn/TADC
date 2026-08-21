"""
数据集配置文件
每个数据集独立的参数设置，运行 main.py 时自动读取
"""

# ============================================================
# 全局设置
# ============================================================
ATTACKED_DIR = 'attacked_graphs'
OUT_DIR = 'results_rramgcn'
RANDOM_SEED = 42

# ============================================================
# 数据集划分比例：训练集 / 验证集 / 测试集
# ============================================================
TRAIN_RATIO = 0.1
VAL_RATIO = 0.1
TEST_RATIO = 0.8

# ============================================================
# 攻击类型
# ============================================================
ATTACK_TYPE = 'metattack'

# ============================================================
# 各数据集独立参数
# ============================================================

# --- Cora ---
cora = {
    'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.8,
    'structure_diff_weight': 0.8,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000,
    'attention_tau': 0.6,
    'clamp_sim_min': True,
    'attn_drop': 0.25,
    'importance_alpha': 0.2,
    'importance_beta': 0.6,
    'max_changes_per_iter': 2,
    'recompute_every': 10,
}

# --- CiteSeer ---
citeseer = {
    'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.8,
    'structure_diff_weight': 0.8,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000,
    'attention_tau': 0.6,
    'clamp_sim_min': True,
    'attn_drop': 0.25,
    'importance_alpha': 0.2,
    'importance_beta': 0.6,
    'max_changes_per_iter': 2,
    'recompute_every': 10,
}

# --- PubMed ---
pubmed = {
    'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.8,
    'structure_diff_weight': 0.8,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000,
    'attention_tau': 0.6,
    'clamp_sim_min': True,
    'attn_drop': 0.25,
    'importance_alpha': 0.2,
    'importance_beta': 0.6,
    'max_changes_per_iter': 2,
    'recompute_every': 10,
}

# --- Cora-ML ---
cora_ml = {
    'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.8,
    'structure_diff_weight': 0.8,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000,
    'attention_tau': 0.6,
    'clamp_sim_min': True,
    'attn_drop': 0.25,
    'importance_alpha': 0.2,
    'importance_beta': 0.6,
    'max_changes_per_iter': 2,
    'recompute_every': 10,
}

# --- BlogCatalog ---
BlogCatalog = {
    'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.6,
    'lr': 0.005,
    'weight_decay': 0.002,
    'epochs': 200,
    'patience': 40,
    'sim_threshold': 0.8,
    'structure_diff_weight': 0.8,
    'inner_finetune_steps': 3,
    'candidate_k': 15,
    'max_candidates': 6000,
    'attention_tau': 0.6,
    'clamp_sim_min': True,
    'attn_drop': 0.25,
    'importance_alpha': 0.2,
    'importance_beta': 0.6,
    'max_changes_per_iter': 2,
    'recompute_every': 10,
}

# ============================================================
# 数据集列表（按顺序运行）
# ============================================================
DATASETS = ['cora', 'citeseer', 'pubmed', 'cora_ml', 'BlogCatalog']


def get_dataset_params(dataset_name):
    """
    根据数据集名称获取对应参数。
    如果数据集没有独立配置，则返回默认参数。
    """
    default_params = {
        'perturb_ratios': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.6,
        'lr': 0.005,
        'weight_decay': 0.002,
        'epochs': 200,
        'patience': 40,
        'sim_threshold': 0.8,
        'structure_diff_weight': 0.8,
        'inner_finetune_steps': 3,
        'candidate_k': 15,
        'max_candidates': 6000,
        'attention_tau': 0.6,
        'clamp_sim_min': True,
        'attn_drop': 0.25,
        'importance_alpha': 0.2,
        'importance_beta': 0.6,
        'max_changes_per_iter': 2,
        'recompute_every': 10,
    }

    # 动态获取模块中的变量
    import sys
    module = sys.modules[__name__]
    if hasattr(module, dataset_name):
        params = getattr(module, dataset_name)
        if isinstance(params, dict):
            return params
    return default_params