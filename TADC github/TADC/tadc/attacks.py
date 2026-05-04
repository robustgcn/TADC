'''run_experiments.py
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool

from deeprobust.graph.global_attack import Metattack, MetaApprox, PGDAttack, DICE
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as GCN_defense

import torch
import scipy.sparse as sp


############################################
# 核心工具函数：数据格式统一与内存隔离（增强稀疏张量处理）
############################################
def _to_sparse_csr(x):
    """
    将输入统一转换为scipy.sparse.csr_matrix（支持三种输入类型）
    - 输入1：scipy稀疏矩阵 → 转csr格式并复制
    - 输入2：numpy.ndarray → 转csr矩阵（确保内存独立）
    - 输入3：torch.Tensor → 先转numpy再转csr（断开计算图+CPU迁移）
    增强：处理稀疏张量，先转为稠密格式
    """
    # 关键修复：处理稀疏张量，先转为稠密格式
    if isinstance(x, torch.Tensor) and x.is_sparse:
        x = x.to_dense().detach().cpu().numpy().copy()

    if sp.issparse(x):
        return x.tocsr().copy()
    elif isinstance(x, np.ndarray):
        return sp.csr_matrix(x.copy())
    elif torch.is_tensor(x):
        # 关键：torch张量转numpy需先detach（断计算图）、cpu（避GPU错误）、再复制
        x_np = x.detach().cpu().numpy().copy()
        return sp.csr_matrix(x_np)
    else:
        raise ValueError(
            f"不支持的数据类型: {type(x)}，需为以下类型之一：\n"
            "- numpy.ndarray（稠密数组）\n"
            "- scipy.sparse.*（如csr_matrix、lil_matrix等稀疏矩阵）\n"
            "- torch.Tensor（PyTorch张量，CPU/GPU均可）"
        )


def _to_numpy_labels(labels):
    """将标签统一转换为numpy数组（兼容torch张量/列表/ndarray）"""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy().copy()
    elif isinstance(labels, np.ndarray):
        return labels.copy()
    elif isinstance(labels, list):
        return np.array(labels).copy()
    else:
        raise ValueError(f"不支持的标签类型: {type(labels)}")


def _ensure_numpy_index(idx):
    """确保索引为numpy数组（兼容torch张量/列表/ndarray，避免内存共享）"""
    if idx is None:
        return None
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().copy()
    elif isinstance(idx, list):
        return np.array(idx).copy()
    elif isinstance(idx, np.ndarray):
        return idx.copy()
    else:
        raise ValueError(f"不支持的索引类型: {type(idx)}")


############################################
# 全局攻击实现：Metattack（内存隔离+格式兼容）
############################################
def run_metattack(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations):
    print(f"\n正在执行 Metattack 攻击 (扰动数量: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # 代理模型强制CPU训练（规避GPU内存共享问题）
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            dropout=0.5,
            device='cpu'
        ).to('cpu')

        # 训练用复制后的输入，避免修改原始数据
        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # 初始化攻击器（CPU执行，与代理模型一致）
        attacker = Metattack(
            surrogate,
            nnodes=adj_sp.shape[0],
            feature_shape=features_sp.shape,
            attack_structure=True,
            attack_features=False,
            device='cpu'
        ).to('cpu')

        # 执行攻击（扰动数量>0时才攻击，否则返回原始矩阵副本）
        if n_perturbations > 0:
            attacker.attack(
                features_sp,
                adj_sp,
                labels_np,
                idx_train,
                idx_unlabeled=idx_unlabeled,
                n_perturbations=n_perturbations,
                ll_constraint=False
            )
            adj_attack = attacker.modified_adj  # 可能为torch.Tensor
        else:
            adj_attack = adj_sp.copy()

        # 统一输出为csr矩阵（自动处理torch/numpy/scipy格式）
        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"Metattack 错误: {str(e)[:120]}...")  # 截断长错误信息
        return adj_sp.copy()  # 出错时返回原始矩阵副本，避免中断流程


############################################
# 全局攻击实现：PGD
############################################
'''
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
    print(f"\n正在执行 PGD 攻击 (扰动数量: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            device='cpu'
        ).to('cpu')

        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # 初始化PGD攻击器
        atk = PGDAttack(
            model=surrogate,
            nnodes=adj_sp.shape[0],
            loss_type="CE",
            device='cpu'
        )

        if n_perturbations > 0:
            atk.attack(
                features_sp,
                adj_sp,
                labels_np,
                idx_train,
                n_perturbations=n_perturbations
            )
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"PGD 错误: {str(e)[:120]}...")
        return adj_sp.copy()
'''
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
    print(f"\n正在执行 PGD 攻击 (扰动数量: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # ---- 确保输入是 numpy dense ----
        if not isinstance(features_sp, np.ndarray):
            if sp.issparse(features_sp):
                features_sp = features_sp.toarray()
            else:
                features_sp = features_sp.cpu().numpy()
        if not isinstance(adj_sp, np.ndarray):
            if sp.issparse(adj_sp):
                adj_sp = adj_sp.toarray()
            else:
                adj_sp = adj_sp.cpu().numpy()

        # surrogate GCN
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            device='cpu'
        ).to('cpu')

        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # 初始化 PGD 攻击器
        atk = PGDAttack(
            model=surrogate,
            nnodes=adj_sp.shape[0],
            loss_type="CE",
            device='cpu'
        )

        if n_perturbations > 0:
            atk.attack(
                features_sp,
                adj_sp,
                labels_np,
                idx_train,
                n_perturbations=n_perturbations
            )
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"PGD 错误: {str(e)[:120]}...")
        return _to_sparse_csr(adj_sp)




############################################
# 全局攻击实现：DICE
############################################
def run_dice(adj_sp, labels_np, n_perturbations):
    print(f"\n正在执行 DICE 攻击 (扰动数量: {n_perturbations})...")
    try:
        atk = DICE()
        if n_perturbations > 0:
            # DICE支持稀疏矩阵直接输入，无需转换
            atk.attack(adj_sp, labels_np, n_perturbations=n_perturbations)
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"DICE 错误: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# 全局攻击实现：Random
############################################
def run_random(adj_sp, features_sp, labels_np, perturb_ratio, dataset_name):
    print(f"\n正在执行 Random 攻击 (扰动比例: {perturb_ratio})...")
    try:
        n_edges = adj_sp.sum() // 2
        n_perturbations = int(perturb_ratio * n_edges)
        adj_mod = adj_sp.tolil(copy=True)  # 转为LIL格式便于修改边
        num_nodes = adj_sp.shape[0]

        # 随机扰动指定数量的边（加/删各50%概率）
        for _ in range(n_perturbations):
            i, j = np.random.randint(0, num_nodes, 2)
            if i == j:  # 跳过自环
                continue
            # 翻转边状态：0→1（加边），1→0（删边）
            if adj_mod[i, j] == 0:
                adj_mod[i, j] = 1
                adj_mod[j, i] = 1  # 保持无向图对称性
            else:
                adj_mod[i, j] = 0
                adj_mod[j, i] = 0

        return adj_mod.tocsr()  # 转回CSR格式（高效存储）

    except Exception as e:
        print(f"Random 错误: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# 目标攻击实现：Nettack（针对指定节点的攻击）
############################################
def run_nettack_targets(adj, features, labels, idx_train, target_nodes, perturbations,
                        device="cpu", return_adjs=False):
    print(f"\n正在执行 Nettack 目标攻击 (每节点扰动数量: {perturbations})...")
    results = {}
    attacked_adjs = {} if return_adjs else None

    # 统一输入格式（支持torch/numpy/scipy）
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)

    # 训练代理GCN（CPU执行，避免内存问题）
    nclass = int(labels_np.max() + 1)
    surrogate = GCN_defense(
        nfeat=features_sp.shape[1],
        nclass=nclass,
        nhid=16,
        device='cpu'
    ).to('cpu')

    surrogate.fit(
        features_sp,
        adj_sp,
        labels_np,
        idx_train,
        train_iters=200,
        verbose=False
    )

    # 逐个攻击目标节点
    for target in target_nodes:
        try:
            target = int(target)  # 确保节点ID为整数（避免numpy类型问题）
            print(f"攻击目标节点: {target}")

            # 初始化Nettack攻击器
            atk = Nettack(surrogate, nnodes=adj_sp.shape[0], device='cpu')
            atk.attack(
                features_sp,
                adj_sp.copy(),  # 用副本攻击，不影响原始矩阵
                labels_np,
                target,
                n_perturbations=perturbations,
                verbose=False
            )

            # 记录攻击结果（成功状态+修改的边）
            results[target] = {
                'success': atk.check_attack_success(),  # 检查攻击是否成功
                'modified_edges': atk.modified_edges  # 记录被修改的边
            }

            # 若需要返回攻击后的邻接矩阵，统一格式为CSR
            if return_adjs:
                attacked_adjs[target] = _to_sparse_csr(atk.modified_adj)

            print(f"目标节点 {target} 攻击完成（成功: {results[target]['success']}）")

        except Exception as e:
            error_msg = str(e)[:80]  # 截断长错误
            print(f"目标节点 {target} 攻击失败: {error_msg}...")
            results[target] = {'success': False, 'error': error_msg}

    print("Nettack 目标攻击流程结束")
    return attacked_adjs if return_adjs else results
'''

############################################
# 攻击调度函数：全局攻击统一入口
############################################
def run_global_attack(attack_type, adj, features, labels, perturb_ratio, dataset_name,
                      idx_train=None, idx_unlabeled=None, device="cpu"):
    print(f"\n=== 执行全局攻击 ===")
    print(f"攻击类型: {attack_type} | 扰动比例: {perturb_ratio} | 数据集: {dataset_name}")

    # 1. 统一所有输入格式（支持torch/numpy/scipy）
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)
    idx_unlabeled = _ensure_numpy_index(idx_unlabeled)

    # 2. 计算扰动数量（基于原始边数）
    n_edges = adj_sp.sum() // 2  # 无向图边数=总边数//2
    n_perturbations = int(perturb_ratio * n_edges)
    print(f"原始边数: {n_edges} | 扰动边数: {n_perturbations}")

    # 3. 调度对应攻击方法（返回统一的CSR矩阵）
    if attack_type == "metattack":
        return run_metattack(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations)
    elif attack_type == "meta_train":
        return run_meta_train(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations)
    elif attack_type == "pgd":
        return run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations)
    elif attack_type == "dice":
        return run_dice(adj_sp, labels_np, n_perturbations)
    elif attack_type == "random":
        return run_random(adj_sp, features_sp, labels_np, perturb_ratio, dataset_name)
    else:
        raise ValueError(
            f"未知的全局攻击类型: {attack_type}\n"
            f"支持的攻击类型: metattack, meta_train, pgd, dice, random"
        )





# attacks.py - 修复后的攻击模块
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool

from deeprobust.graph.global_attack import Metattack, MetaApprox, PGDAttack, DICE
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as GCN_defense

import torch
import scipy.sparse as sp


############################################
# 核心工具函数
############################################
def _to_sparse_csr(x):
    # 处理稀疏张量，先转为稠密格式
    if isinstance(x, torch.Tensor) and x.is_sparse:
        x = x.to_dense().detach().cpu().numpy().copy()

    if sp.issparse(x):
        return x.tocsr().copy()
    elif isinstance(x, np.ndarray):
        return sp.csr_matrix(x.copy())
    elif torch.is_tensor(x):
        # torch张量转numpy需先detach（断计算图）、cpu（避GPU错误）、再复制
        x_np = x.detach().cpu().numpy().copy()
        return sp.csr_matrix(x_np)
    else:
        raise ValueError(
            f"不支持的数据类型: {type(x)}，需为以下类型之一：\n"
            "- numpy.ndarray（稠密数组）\n"
            "- scipy.sparse.*（如csr_matrix、lil_matrix等稀疏矩阵）\n"
            "- torch.Tensor（PyTorch张量，CPU/GPU均可）"
        )


def _to_numpy_labels(labels):
    """将标签统一转换为numpy数组（兼容torch张量/列表/ndarray）"""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy().copy()
    elif isinstance(labels, np.ndarray):
        return labels.copy()
    elif isinstance(labels, list):
        return np.array(labels).copy()
    else:
        raise ValueError(f"不支持的标签类型: {type(labels)}")


def _ensure_numpy_index(idx):
    """确保索引为numpy数组（兼容torch张量/列表/ndarray，避免内存共享）"""
    if idx is None:
        return None
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().copy()
    elif isinstance(idx, list):
        return np.array(idx).copy()
    elif isinstance(idx, np.ndarray):
        return idx.copy()
    else:
        raise ValueError(f"不支持的索引类型: {type(idx)}")


def _to_tensor(x, dtype=torch.float, device="cpu"):
    """将输入转换为指定类型的torch张量"""
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    elif sp.issparse(x):
        return torch.tensor(x.toarray(), dtype=dtype, device=device)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError(f"不支持的数据类型: {type(x)}")


def _to_scipy_adj(adj):
    """将邻接矩阵转换为scipy稀疏矩阵"""
    if sp.issparse(adj):
        return adj.tocsr()
    elif torch.is_tensor(adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        return sp.csr_matrix(adj.detach().cpu().numpy())
    elif isinstance(adj, np.ndarray):
        return sp.csr_matrix(adj)
    else:
        raise ValueError(f"不支持的邻接矩阵类型: {type(adj)}")


############################################
# 全局攻击实现：Metattack
############################################
def run_metattack(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations):
    print(f"\n正在执行 Metattack 攻击 (扰动数量: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # 代理模型强制CPU训练（规避GPU内存共享问题）
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            dropout=0.5,
            device='cpu'
        ).to('cpu')

        # 训练用复制后的输入，避免修改原始数据
        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # 初始化攻击器（CPU执行，与代理模型一致）
        attacker = Metattack(
            surrogate,
            nnodes=adj_sp.shape[0],
            feature_shape=features_sp.shape,
            attack_structure=True,
            attack_features=False,
            device='cpu'
        ).to('cpu')

        # 执行攻击（扰动数量>0时才攻击，否则返回原始矩阵副本）
        if n_perturbations > 0:
            attacker.attack(
                features_sp,
                adj_sp,
                labels_np,
                idx_train,
                idx_unlabeled=idx_unlabeled,
                n_perturbations=n_perturbations,
                ll_constraint=False
            )
            adj_attack = attacker.modified_adj  # 可能为torch.Tensor
        else:
            adj_attack = adj_sp.copy()

        # 统一输出为csr矩阵（自动处理torch/numpy/scipy格式）
        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"Metattack 错误: {str(e)[:120]}...")  # 截断长错误信息
        return adj_sp.copy()  # 出错时返回原始矩阵副本，避免中断流程



############################################
# 全局攻击实现：PGD
############################################
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
    print(f"\n正在执行 PGD 攻击 (扰动数量: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # 确保输入为numpy数组格式（PGD要求）
        if not isinstance(features_sp, np.ndarray):
            if sp.issparse(features_sp):
                features_sp = features_sp.toarray()
            else:
                features_sp = features_sp.cpu().numpy()
        if not isinstance(adj_sp, np.ndarray):
            if sp.issparse(adj_sp):
                adj_sp = adj_sp.toarray()
            else:
                adj_sp = adj_sp.cpu().numpy()

        # 代理模型（CPU执行）
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            device='cpu'
        ).to('cpu')

        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # PGD攻击器
        atk = PGDAttack(
            model=surrogate,
            nnodes=adj_sp.shape[0],
            loss_type="CE",
            device='cpu'
        )

        if n_perturbations > 0:
            atk.attack(
                features_sp,
                adj_sp,
                labels_np,
                idx_train,
                n_perturbations=n_perturbations
            )
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"PGD 错误: {str(e)[:120]}...")
        return _to_sparse_csr(adj_sp)


############################################
# 全局攻击实现：DICE
############################################
def run_dice(adj_sp, labels_np, n_perturbations):
    print(f"\n正在执行 DICE 攻击 (扰动数量: {n_perturbations})...")
    try:
        atk = DICE()
        if n_perturbations > 0:
            # DICE攻击（仅需邻接矩阵和标签）
            atk.attack(adj_sp, labels_np, n_perturbations=n_perturbations)
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"DICE 错误: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# 全局攻击实现：Random
############################################
def run_random(adj_sp, features_sp, labels_np, perturb_ratio, dataset_name):
    print(f"\n正在执行 Random 攻击 (扰动比例: {perturb_ratio})...")
    try:
        n_edges = adj_sp.sum() // 2
        n_perturbations = int(perturb_ratio * n_edges)
        adj_mod = adj_sp.tolil(copy=True)  # 转为LIL格式便于修改边
        num_nodes = adj_sp.shape[0]

        # 随机添加/删除边
        for _ in range(n_perturbations):
            i, j = np.random.randint(0, num_nodes, 2)
            if i == j:  # 跳过自环
                continue
            # 翻转边状态（0→1或1→0）
            if adj_mod[i, j] == 0:
                adj_mod[i, j] = 1
                adj_mod[j, i] = 1  # 保持无向图对称性
            else:
                adj_mod[i, j] = 0
                adj_mod[j, i] = 0

        return adj_mod.tocsr()  # 转回CSR格式（高效存储）

    except Exception as e:
        print(f"Random 错误: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# 目标攻击实现：Nettack（针对指定节点的攻击）
############################################
def run_nettack_targets(adj, features, labels, idx_train, target_nodes, perturbations,
                        device="cpu", return_adjs=False):
    print(f"\n正在执行 Nettack 目标攻击 (每节点扰动数量: {perturbations})...")
    results = {}
    attacked_adjs = {} if return_adjs else None

    # 统一输入格式（支持torch/numpy/scipy）
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)

    # 训练代理GCN（CPU执行，避免内存问题）
    nclass = int(labels_np.max() + 1)
    surrogate = GCN_defense(
        nfeat=features_sp.shape[1],
        nclass=nclass,
        nhid=16,
        device='cpu'
    ).to('cpu')

    surrogate.fit(
        features_sp,
        adj_sp,
        labels_np,
        idx_train,
        train_iters=200,
        verbose=False
    )

    # 逐个攻击目标节点
    for target in target_nodes:
        try:
            target = int(target)  # 确保节点ID为整数（避免numpy类型问题）
            print(f"攻击目标节点: {target}")

            # 初始化Nettack攻击器
            atk = Nettack(surrogate, nnodes=adj_sp.shape[0], device='cpu')
            atk.attack(
                features_sp,
                adj_sp.copy(),  # 用副本攻击，不影响原始矩阵
                labels_np,
                target,
                n_perturbations=perturbations,
                verbose=False
            )

            # 记录攻击结果（成功状态+修改的边）
            results[target] = {
                'success': atk.check_attack_success(),  # 检查攻击是否成功
                'modified_edges': atk.modified_edges  # 记录被修改的边
            }

            # 若需要返回攻击后的邻接矩阵，统一格式为CSR
            if return_adjs:
                attacked_adjs[target] = _to_sparse_csr(atk.modified_adj)

            print(f"目标节点 {target} 攻击完成（成功: {results[target]['success']}）")

        except Exception as e:
            error_msg = str(e)[:80]  # 截断长错误
            print(f"目标节点 {target} 攻击失败: {error_msg}...")
            results[target] = {'success': False, 'error': error_msg}

    print("Nettack 目标攻击流程结束")
    return attacked_adjs if return_adjs else results


















