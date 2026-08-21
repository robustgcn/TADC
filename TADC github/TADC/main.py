import os
import torch
import numpy as np
import time
import json
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

from tadc.utils import set_seed
from tadc.attacks import run_nettack_targets
from tadc.model import TADCTrainer

from dataset_config import (
    DATASETS, ATTACKED_DIR, OUT_DIR, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, ATTACK_TYPE,
    get_dataset_params,
)


######################################################
# Utility Functions
######################################################
def _to_scipy_adj(adj):
    """Ensure return scipy.sparse.csr_matrix"""
    if sp.issparse(adj):
        return adj.tocsr()
    if isinstance(adj, np.ndarray):
        return sp.csr_matrix(adj)
    if isinstance(adj, torch.Tensor):
        return sp.csr_matrix(adj.cpu().numpy())
    raise ValueError(f"Unsupported adj format: {type(adj)}")


def save_results(results, out_dir):
    """Save results to JSON file"""
    for dataset, data_res in results.items():
        result_file = os.path.join(out_dir, f"{dataset}_results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                try:
                    existing_res = json.load(f)
                except Exception:
                    existing_res = {}
            existing_res.update(data_res)
            data_res = existing_res
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data_res, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_dir}")


def split_dataset(labels_np, train_ratio=0.1, val_ratio=0.1, test_ratio=0.8, random_seed=42):
    """
    按比例划分数据集：训练集 train_ratio / 验证集 val_ratio / 测试集 test_ratio
    使用分层采样保证类别分布一致
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    n = len(labels_np)
    idx = np.arange(n)

    # 第一步：分出训练集 (train_ratio)
    idx_train, idx_temp, _, y_temp = train_test_split(
        idx, labels_np,
        train_size=train_ratio,
        stratify=labels_np,
        random_state=random_seed,
    )

    # 第二步：从剩余中分出验证集和测试集
    # val_ratio / (val_ratio + test_ratio) 即验证集占剩余部分的比例
    val_frac = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp,
        train_size=val_frac,
        stratify=y_temp,
        random_state=random_seed,
    )

    return idx_train, idx_val, idx_test


def _load_original_data(dataset, attacked_root='attacked_graphs'):
    """Load original data from attacked_graphs/<dataset>/original_data.npz"""
    data_path = os.path.join(attacked_root, dataset, 'original_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Original data file missing: {data_path}")

    with np.load(data_path, allow_pickle=True) as f:
        A_orig = f['adj'].item() if isinstance(f['adj'], np.ndarray) else f['adj']
        features = f['features'].item() if isinstance(f['features'], np.ndarray) else f['features']
        labels_np = f['labels']

    A_orig = _to_scipy_adj(A_orig)
    return labels_np, features, A_orig


def _load_perturbed_adj(dataset, perturb_ratio, attacked_root='attacked_graphs'):
    """Load attacked adjacency matrix"""
    ratio_str = f"{perturb_ratio:.2f}"
    data_path = os.path.join(attacked_root, dataset, f'attacked_p{ratio_str}.npz')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Attack file missing: {data_path}")

    with np.load(data_path, allow_pickle=True) as f:
        adj_attack = f['adj_attack'].item() if isinstance(f['adj_attack'], np.ndarray) else f['adj_attack']

    return _to_scipy_adj(adj_attack)


######################################################
# Global Settings (from config)
######################################################
os.makedirs(OUT_DIR, exist_ok=True)

use_cuda = torch.cuda.is_available()
model_device = torch.device("cuda" if use_cuda else "cpu")
attack_device = torch.device("cpu")
print(f"Device configuration: model training={model_device}, attack execution={attack_device}")
print(f"Root directory for attack files: {ATTACKED_DIR}")
print(f"Data split: train={TRAIN_RATIO*100:.0f}%, val={VAL_RATIO*100:.0f}%, test={TEST_RATIO*100:.0f}%")


def run_all():
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}

        # 从配置文件读取该数据集的参数（copy 避免修改原配置）
        params = get_dataset_params(dataset).copy()
        perturb_ratios = params.pop('perturb_ratios')  # 单独提取 perturb_ratios

        print(f"\n{'='*60}")
        print(f"=== Start processing dataset: {dataset} ===")
        print(f"=== Parameters: {params}")
        print(f"=== Perturbation ratios: {perturb_ratios}")
        print(f"{'='*60}")

        # 1. Load original data (without pre-saved splits)
        try:
            labels_np, features, A_orig = _load_original_data(
                dataset=dataset, attacked_root=ATTACKED_DIR
            )

            # --- 重新划分数据集：训练集10% / 验证集10% / 测试集80% ---
            split_train, split_val, split_test = split_dataset(
                labels_np,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO,
                random_seed=RANDOM_SEED,
            )
            # split_unlabeled 即测试集
            split_unlabeled = split_test

            if sp.issparse(features):
                feats_np = features.toarray()
                features = torch.from_numpy(feats_np).float().to(model_device)
            elif isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float().to(model_device)
            elif isinstance(features, torch.Tensor):
                features = features.to(model_device)
            else:
                raise TypeError(f"Unsupported feature format: {type(features)}")

            labels = torch.tensor(labels_np, dtype=torch.long, device=model_device)

            split_train = np.array(split_train, dtype=int)
            split_val = np.array(split_val, dtype=int)
            split_unlabeled = np.array(split_unlabeled, dtype=int)

            print(f"Original data loaded successfully: nodes={A_orig.shape[0]}, edges={int(A_orig.sum()//2)}, feature dimension={features.shape[1]}")
            print(f"split sizes -> train:{len(split_train)} val:{len(split_val)} test:{len(split_unlabeled)}")
            print("overlaps -> train∩val:", len(set(split_train) & set(split_val)),
                  "train∩test:", len(set(split_train) & set(split_unlabeled)),
                  "val∩test:", len(set(split_val) & set(split_unlabeled)))
        except Exception as e:
            print(f" Failed to load dataset {dataset}: {str(e)}")
            results[dataset]['error'] = f"Data loading failed: {str(e)}"
            save_results(results, OUT_DIR)
            continue

        # 2. Perturbation experiment
        for ratio in perturb_ratios:
            ratio_key = str(ratio)
            results[dataset][ratio_key] = {}
            print(f"\n-> Current perturbation ratio: {ratio} (attack type: {ATTACK_TYPE})")

            try:
                start_time = time.time()
                if ratio == 0.0:
                    perturbed_adj = A_orig
                    print(f"  Using original adjacency matrix (no perturbation)")
                else:
                    perturbed_adj = _load_perturbed_adj(dataset, ratio, ATTACKED_DIR)
                    print(f"  Perturbed adjacency matrix loaded")

                trainer = TADCTrainer(**params, device=str(model_device))
                def_acc = trainer.fit_and_eval(
                    features=features,
                    labels=labels,
                    perturbed_adj=perturbed_adj,
                    split_train=split_train,
                    split_val=split_val,
                    split_unlabeled=split_unlabeled,
                    dataset_name=dataset
                )

                elapsed_time = time.time() - start_time
                if isinstance(def_acc, (list, tuple)):
                    def_acc_val = def_acc[0]
                else:
                    def_acc_val = def_acc
                def_acc_val = float(def_acc_val)
                def_acc_val = round(def_acc_val, 4)
                results[dataset][ratio_key][ATTACK_TYPE] = def_acc_val
                print(f" {ATTACK_TYPE} @ perturbation {ratio}: defense accuracy={def_acc_val:.4f}, time cost={elapsed_time:.2f}s")
            except Exception as e:
                error_msg = str(e)[:200]
                print(f" Execution failed: {error_msg}")
                results[dataset][ratio_key][ATTACK_TYPE] = f"Failed: {error_msg}"
                continue

        # 3. Targeted attack (Nettack)
        try:
            print(f"\n-> Start targeted attack: Nettack")
            degrees = np.array(A_orig.sum(axis=1)).flatten()
            candidate_nodes = np.where((degrees > 10) & (degrees > 0))[0]
            if len(candidate_nodes) == 0:
                results[dataset]['nettack'] = "No eligible target nodes"
                save_results(results, OUT_DIR)
                continue
            target_nodes = np.random.choice(candidate_nodes, size=min(10, len(candidate_nodes)), replace=False)
            target_nodes = [int(node) for node in target_nodes]
            print(f"Selected target nodes: {target_nodes}")

            nettack_results = {}
            for perturb_num in [0, 1, 2, 3, 4, 5]:
                perturb_key = str(perturb_num)
                nettack_results[perturb_key] = {}
                attacked_adjs = run_nettack_targets(
                    adj=A_orig,
                    features=features,
                    labels=labels,
                    idx_train=split_train,
                    target_nodes=target_nodes,
                    perturbations=perturb_num,
                    device=attack_device,
                    return_adjs=True
                )

                target_acc_list = []
                for target_node, modified_adj in attacked_adjs.items():
                    try:
                        trainer = TADCTrainer(**params, device=str(model_device))
                        target_acc = trainer.fit_and_eval(
                            features=features,
                            labels=labels,
                            perturbed_adj=_to_scipy_adj(modified_adj),
                            split_train=split_train,
                            split_val=split_val,
                            split_unlabeled=split_unlabeled,
                            dataset_name=dataset,
                            target_nodes=[target_node]
                        )
                        if isinstance(target_acc, (list, tuple)):
                            target_acc_val = float(target_acc[0])
                        else:
                            target_acc_val = float(target_acc)
                        target_acc_val = round(target_acc_val, 4)
                        target_acc_list.append(target_acc_val)
                        nettack_results[perturb_key][f"node_{target_node}"] = target_acc_val
                    except Exception as e:
                        print(f"     Target node {target_node} failed: {str(e)[:200]}...")
                        nettack_results[perturb_key][f"node_{target_node}"] = f"Failed: {str(e)[:200]}"
                        continue

                valid_accs = [acc for acc in target_acc_list if isinstance(acc, (float, int))]
                avg_acc = round(np.mean(valid_accs), 4) if valid_accs else 0.0
                nettack_results[perturb_key]['average_acc'] = avg_acc

            results[dataset]['nettack'] = nettack_results
        except Exception as e:
            print(f" Targeted attack execution failed: {str(e)}")
            results[dataset]['nettack'] = f"Execution failed: {str(e)}"
            save_results(results, OUT_DIR)
            continue

        # Save intermediate results for each dataset
        save_results(results, OUT_DIR)
        print(f"=== Dataset {dataset} processing finished ===")

    print("\n All dataset experiments completed!")
    return results


if __name__ == "__main__":
    set_seed(RANDOM_SEED, use_cuda)
    run_all()