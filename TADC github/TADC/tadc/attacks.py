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
# Core Utility Functions: Data Format Unification & Memory Isolation (Enhanced Sparse Tensor Processing)
############################################
def _to_sparse_csr(x):
    """
    Convert input uniformly to scipy.sparse.csr_matrix (supports three input types)
    - Input 1: scipy sparse matrix -> convert to CSR format and copy
    - Input 2: numpy.ndarray -> convert to CSR matrix (ensure independent memory)
    - Input 3: torch.Tensor -> convert to numpy first then CSR (detach computation graph + migrate to CPU)
    Enhancement: handle sparse tensors by converting to dense format first
    """
    # Critical fix: convert sparse tensor to dense firstly
    if isinstance(x, torch.Tensor) and x.is_sparse:
        x = x.to_dense().detach().cpu().numpy().copy()

    if sp.issparse(x):
        return x.tocsr().copy()
    elif isinstance(x, np.ndarray):
        return sp.csr_matrix(x.copy())
    elif torch.is_tensor(x):
        # Key: detach computation graph, move to CPU then copy when converting torch tensor to numpy
        x_np = x.detach().cpu().numpy().copy()
        return sp.csr_matrix(x_np)
    else:
        raise ValueError(
            f"Unsupported data type: {type(x)}, must be one of the following:\n"
            "- numpy.ndarray (dense array)\n"
            "- scipy.sparse.* sparse matrix (csr_matrix, lil_matrix, etc.)\n"
            "- torch.Tensor (PyTorch tensor, CPU/GPU acceptable)"
        )


def _to_numpy_labels(labels):
    """Convert labels to numpy array uniformly (compatible with torch tensor/list/ndarray)"""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy().copy()
    elif isinstance(labels, np.ndarray):
        return labels.copy()
    elif isinstance(labels, list):
        return np.array(labels).copy()
    else:
        raise ValueError(f"Unsupported label type: {type(labels)}")


def _ensure_numpy_index(idx):
    """Ensure index is numpy array (compatible with torch tensor/list/ndarray, avoid memory sharing)"""
    if idx is None:
        return None
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().copy()
    elif isinstance(idx, list):
        return np.array(idx).copy()
    elif isinstance(idx, np.ndarray):
        return idx.copy()
    else:
        raise ValueError(f"Unsupported index type: {type(idx)}")


############################################
# Global Attack Implementation: Metattack (Memory Isolation & Format Compatibility)
############################################
def run_metattack(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations):
    print(f"\nExecuting Metattack attack (number of perturbations: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # Force surrogate model to train on CPU (avoid GPU memory sharing issues)
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            dropout=0.5,
            device='cpu'
        ).to('cpu')

        # Train with copied inputs to prevent modifying original data
        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # Initialize attacker (run on CPU, consistent with surrogate model)
        attacker = Metattack(
            surrogate,
            nnodes=adj_sp.shape[0],
            feature_shape=features_sp.shape,
            attack_structure=True,
            attack_features=False,
            device='cpu'
        ).to('cpu')

        # Perform attack only when perturbation count > 0, otherwise return copy of original matrix
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
            adj_attack = attacker.modified_adj  # May be torch.Tensor
        else:
            adj_attack = adj_sp.copy()

        # Unify output as CSR matrix (automatically handle torch/numpy/scipy formats)
        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"Metattack error: {str(e)[:120]}...")  # Truncate long error messages
        return adj_sp.copy()  # Return original matrix copy on error to avoid process interruption


############################################
# Global Attack Implementation: PGD
############################################
'''
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
        return adj_sp.copy()
'''
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
    print(f"\nExecuting PGD attack (number of perturbations: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # ---- Ensure inputs are numpy dense arrays ----
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

        # Initialize PGD attacker
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
        print(f"PGD error: {str(e)[:120]}...")
        return _to_sparse_csr(adj_sp)




############################################
# Global Attack Implementation: DICE
############################################
def run_dice(adj_sp, labels_np, n_perturbations):
    print(f"\nExecuting DICE attack (number of perturbations: {n_perturbations})...")
    try:
        atk = DICE()
        if n_perturbations > 0:
            # DICE supports direct sparse matrix input without conversion
            atk.attack(adj_sp, labels_np, n_perturbations=n_perturbations)
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"DICE error: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# Global Attack Implementation: Random
############################################
def run_random(adj_sp, features_sp, labels_np, perturb_ratio, dataset_name):
    print(f"\nExecuting Random attack (perturbation ratio: {perturb_ratio})...")
    try:
        n_edges = adj_sp.sum() // 2
        n_perturbations = int(perturb_ratio * n_edges)
        adj_mod = adj_sp.tolil(copy=True)  # Convert to LIL format for convenient edge modification
        num_nodes = adj_sp.shape[0]

        # Randomly perturb specified number of edges (50% probability for add/delete each)
        for _ in range(n_perturbations):
            i, j = np.random.randint(0, num_nodes, 2)
            if i == j:  # Skip self-loops
                continue
            # Flip edge status: 0→1 (add edge), 1→0 (remove edge)
            if adj_mod[i, j] == 0:
                adj_mod[i, j] = 1
                adj_mod[j, i] = 1  # Maintain symmetry of undirected graph
            else:
                adj_mod[i, j] = 0
                adj_mod[j, i] = 0

        return adj_mod.tocsr()  # Convert back to CSR format for efficient storage

    except Exception as e:
        print(f"Random error: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# Targeted Attack Implementation: Nettack (attack specified nodes)
############################################
def run_nettack_targets(adj, features, labels, idx_train, target_nodes, perturbations,
                        device="cpu", return_adjs=False):
    print(f"\nExecuting Nettack targeted attack (perturbations per node: {perturbations})...")
    results = {}
    attacked_adjs = {} if return_adjs else None

    # Unify input format (supports torch/numpy/scipy)
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)

    # Train surrogate GCN (run on CPU to avoid memory issues)
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

    # Attack each target node one by one
    for target in target_nodes:
        try:
            target = int(target)  # Ensure node ID is integer (avoid numpy type issues)
            print(f"Attacking target node: {target}")

            # Initialize Nettack attacker
            atk = Nettack(surrogate, nnodes=adj_sp.shape[0], device='cpu')
            atk.attack(
                features_sp,
                adj_sp.copy(),  # Attack with copy without modifying original matrix
                labels_np,
                target,
                n_perturbations=perturbations,
                verbose=False
            )

            # Record attack results (success status + modified edges)
            results[target] = {
                'success': atk.check_attack_success(),  # Check whether attack succeeds
                'modified_edges': atk.modified_edges  # Record modified edges
            }

            # Convert attacked adjacency matrix to CSR format if return is required
            if return_adjs:
                attacked_adjs[target] = _to_sparse_csr(atk.modified_adj)

            print(f"Attack on target node {target} finished (success: {results[target]['success']})")

        except Exception as e:
            error_msg = str(e)[:80]  # Truncate long error message
            print(f"Attack on target node {target} failed: {error_msg}...")
            results[target] = {'success': False, 'error': error_msg}

    print("Nettack targeted attack process finished")
    return attacked_adjs if return_adjs else results
'''

############################################
# Attack Scheduler: Unified Entry for Global Attacks
############################################
def run_global_attack(attack_type, adj, features, labels, perturb_ratio, dataset_name,
                      idx_train=None, idx_unlabeled=None, device="cpu"):
    print(f"\n=== Executing global attack ===")
    print(f"Attack type: {attack_type} | Perturbation ratio: {perturb_ratio} | Dataset: {dataset_name}")

    # 1. Unify format of all inputs (supports torch/numpy/scipy)
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)
    idx_unlabeled = _ensure_numpy_index(idx_unlabeled)

    # 2. Calculate perturbation quantity based on original edge count
    n_edges = adj_sp.sum() // 2  # Edge count of undirected graph = total edges // 2
    n_perturbations = int(perturb_ratio * n_edges)
    print(f"Original edges: {n_edges} | Perturbed edges: {n_perturbations}")

    # 3. Dispatch corresponding attack method (return unified CSR matrix)
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
            f"Unknown global attack type: {attack_type}\n"
            f"Supported attack types: metattack, meta_train, pgd, dice, random"
        )





# attacks.py - Fixed attack module
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool

from deeprobust.graph.global_attack import Metattack, MetaApprox, PGDAttack, DICE
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN as GCN_defense

import torch
import scipy.sparse as sp


############################################
# Core Utility Functions
############################################
def _to_sparse_csr(x):
    # Convert sparse tensor to dense first
    if isinstance(x, torch.Tensor) and x.is_sparse:
        x = x.to_dense().detach().cpu().numpy().copy()

    if sp.issparse(x):
        return x.tocsr().copy()
    elif isinstance(x, np.ndarray):
        return sp.csr_matrix(x.copy())
    elif torch.is_tensor(x):
        # Detach graph, move to CPU and copy when converting torch tensor to numpy
        x_np = x.detach().cpu().numpy().copy()
        return sp.csr_matrix(x_np)
    else:
        raise ValueError(
            f"Unsupported data type: {type(x)}, must be one of the following:\n"
            "- numpy.ndarray (dense array)\n"
            "- scipy.sparse.* sparse matrix (csr_matrix, lil_matrix, etc.)\n"
            "- torch.Tensor (PyTorch tensor, CPU/GPU acceptable)"
        )


def _to_numpy_labels(labels):
    """Convert labels to numpy array uniformly (compatible with torch tensor/list/ndarray)"""
    if torch.is_tensor(labels):
        return labels.detach().cpu().numpy().copy()
    elif isinstance(labels, np.ndarray):
        return labels.copy()
    elif isinstance(labels, list):
        return np.array(labels).copy()
    else:
        raise ValueError(f"Unsupported label type: {type(labels)}")


def _ensure_numpy_index(idx):
    """Ensure index is numpy array (compatible with torch tensor/list/ndarray, avoid memory sharing)"""
    if idx is None:
        return None
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().copy()
    elif isinstance(idx, list):
        return np.array(idx).copy()
    elif isinstance(idx, np.ndarray):
        return idx.copy()
    else:
        raise ValueError(f"Unsupported index type: {type(idx)}")


def _to_tensor(x, dtype=torch.float, device="cpu"):
    """Convert input to torch tensor with specified dtype"""
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    elif sp.issparse(x):
        return torch.tensor(x.toarray(), dtype=dtype, device=device)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported data type: {type(x)}")


def _to_scipy_adj(adj):
    """Convert adjacency matrix to scipy sparse matrix"""
    if sp.issparse(adj):
        return adj.tocsr()
    elif torch.is_tensor(adj):
        if adj.is_sparse:
            adj = adj.to_dense()
        return sp.csr_matrix(adj.detach().cpu().numpy())
    elif isinstance(adj, np.ndarray):
        return sp.csr_matrix(adj)
    else:
        raise ValueError(f"Unsupported adjacency matrix type: {type(adj)}")


############################################
# Global Attack Implementation: Metattack
############################################
def run_metattack(features_sp, adj_sp, labels_np, idx_train, idx_unlabeled, n_perturbations):
    print(f"\nExecuting Metattack attack (number of perturbations: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # Force surrogate model to train on CPU (avoid GPU memory sharing issues)
        surrogate = GCN_defense(
            nfeat=features_sp.shape[1],
            nclass=nclass,
            nhid=16,
            dropout=0.5,
            device='cpu'
        ).to('cpu')

        # Train with copied inputs to prevent modifying original data
        surrogate.fit(
            features_sp,
            adj_sp,
            labels_np,
            idx_train,
            train_iters=200,
            verbose=False
        )

        # Initialize attacker (run on CPU, consistent with surrogate model)
        attacker = Metattack(
            surrogate,
            nnodes=adj_sp.shape[0],
            feature_shape=features_sp.shape,
            attack_structure=True,
            attack_features=False,
            device='cpu'
        ).to('cpu')

        # Perform attack only when perturbation count > 0, otherwise return copy of original matrix
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
            adj_attack = attacker.modified_adj  # May be torch.Tensor
        else:
            adj_attack = adj_sp.copy()

        # Unify output as CSR matrix (automatically handle torch/numpy/scipy formats)
        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"Metattack error: {str(e)[:120]}...")  # Truncate long error messages
        return adj_sp.copy()  # Return original matrix copy on error to avoid process interruption



############################################
# Global Attack Implementation: PGD
############################################
def run_pgd(features_sp, adj_sp, labels_np, idx_train, n_perturbations):
    print(f"\nExecuting PGD attack (number of perturbations: {n_perturbations})...")
    try:
        nclass = int(labels_np.max() + 1)

        # Ensure inputs are numpy arrays (required by PGD)
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

        # Surrogate model (CPU execution)
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

        # PGD attacker
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
        print(f"PGD error: {str(e)[:120]}...")
        return _to_sparse_csr(adj_sp)


############################################
# Global Attack Implementation: DICE
############################################
def run_dice(adj_sp, labels_np, n_perturbations):
    print(f"\nExecuting DICE attack (number of perturbations: {n_perturbations})...")
    try:
        atk = DICE()
        if n_perturbations > 0:
            # DICE attack only requires adjacency matrix and labels
            atk.attack(adj_sp, labels_np, n_perturbations=n_perturbations)
            adj_attack = atk.modified_adj
        else:
            adj_attack = adj_sp.copy()

        return _to_sparse_csr(adj_attack)

    except Exception as e:
        print(f"DICE error: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# Global Attack Implementation: Random
############################################
def run_random(adj_sp, features_sp, labels_np, perturb_ratio, dataset_name):
    print(f"\nExecuting Random attack (perturbation ratio: {perturb_ratio})...")
    try:
        n_edges = adj_sp.sum() // 2
        n_perturbations = int(perturb_ratio * n_edges)
        adj_mod = adj_sp.tolil(copy=True)  # Convert to LIL format for convenient edge modification
        num_nodes = adj_sp.shape[0]

        # Randomly add/remove edges
        for _ in range(n_perturbations):
            i, j = np.random.randint(0, num_nodes, 2)
            if i == j:  # Skip self-loops
                continue
            # Flip edge state (0→1 or 1→0)
            if adj_mod[i, j] == 0:
                adj_mod[i, j] = 1
                adj_mod[j, i] = 1  # Maintain symmetry of undirected graph
            else:
                adj_mod[i, j] = 0
                adj_mod[j, i] = 0

        return adj_mod.tocsr()  # Convert back to CSR format for efficient storage

    except Exception as e:
        print(f"Random error: {str(e)[:120]}...")
        return adj_sp.copy()


############################################
# Targeted Attack Implementation: Nettack (attack specified nodes)
############################################
def run_nettack_targets(adj, features, labels, idx_train, target_nodes, perturbations,
                        device="cpu", return_adjs=False):
    print(f"\nExecuting Nettack targeted attack (perturbations per node: {perturbations})...")
    results = {}
    attacked_adjs = {} if return_adjs else None

    # Unify input format (supports torch/numpy/scipy)
    adj_sp = _to_sparse_csr(adj)
    features_sp = _to_sparse_csr(features)
    labels_np = _to_numpy_labels(labels)
    idx_train = _ensure_numpy_index(idx_train)

    # Train surrogate GCN (run on CPU to avoid memory issues)
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

    # Attack each target node one by one
    for target in target_nodes:
        try:
            target = int(target)  # Ensure node ID is integer (avoid numpy type issues)
            print(f"Attacking target node: {target}")

            # Initialize Nettack attacker
            atk = Nettack(surrogate, nnodes=adj_sp.shape[0], device='cpu')
            atk.attack(
                features_sp,
                adj_sp.copy(),  # Attack with copy without modifying original matrix
                labels_np,
                target,
                n_perturbations=perturbations,
                verbose=False
            )

            # Record attack results (success status + modified edges)
            results[target] = {
                'success': atk.check_attack_success(),  # Check whether attack succeeds
                'modified_edges': atk.modified_edges  # Record modified edges
            }

            # Convert attacked adjacency matrix to CSR format if return is required
            if return_adjs:
                attacked_adjs[target] = _to_sparse_csr(atk.modified_adj)

            print(f"Attack on target node {target} finished (success: {results[target]['success']})")

        except Exception as e:
            error_msg = str(e)[:80]  # Truncate long error message
            print(f"Attack on target node {target} failed: {error_msg}...")
            results[target] = {'success': False, 'error': error_msg}

    print("Nettack targeted attack process finished")
    return attacked_adjs if return_adjs else results
