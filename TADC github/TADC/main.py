import os
import torch
import numpy as np
import time
import json
import scipy.sparse as sp

from tadc.utils import set_seed
from tadc.attacks import run_nettack_targets
from tadc.model import TADCTrainer

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


def _load_original_data(dataset, attacked_root='attacked_graphs'):
    """Load original data from attacked_graphs/<dataset>/original_data.npz"""
    data_path = os.path.join(attacked_root, dataset, 'original_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Original data file missing: {data_path}")

    with np.load(data_path, allow_pickle=True) as f:
        A_orig = f['adj'].item() if isinstance(f['adj'], np.ndarray) else f['adj']
        features = f['features'].item() if isinstance(f['features'], np.ndarray) else f['features']
        labels_np = f['labels']
        # Key: split_unlabeled corresponds to idx_test (do not assign idx_val to it)
        split_train = f['idx_train']
        split_val = f['idx_val']
        split_unlabeled = f.get('idx_test', f.get('idx_unlabeled', f['idx_val']))

    A_orig = _to_scipy_adj(A_orig)
    return labels_np, features, split_train, split_val, split_unlabeled, A_orig


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
# Global Settings
######################################################

DATASETS = ['citeseer']
perturb_ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

RAMGCN_PARAMS = {
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
    'max_candidates': 6000
}


ATTACKED_DIR = 'attacked_graphs'
OUT_DIR = 'results_rramgcn'
os.makedirs(OUT_DIR, exist_ok=True)

use_cuda = torch.cuda.is_available()
model_device = torch.device("cuda" if use_cuda else "cpu")
attack_device = torch.device("cpu")
print(f"Device configuration: model training={model_device}, attack execution={attack_device}")
print(f"RAM-GCN hyperparameters: {RAMGCN_PARAMS}")
print(f"Root directory for attack files: {ATTACKED_DIR}")

def run_all():
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        print(f"\n=== Start processing dataset: {dataset} ===")
        # 1. Load original data
        try:
            labels_np, features, split_train, split_val, split_unlabeled, A_orig = _load_original_data(
                dataset=dataset, attacked_root=ATTACKED_DIR
            )

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
        attack_type = 'metattack'
        for ratio in perturb_ratios:
            ratio_key = str(ratio)
            results[dataset][ratio_key] = {}
            print(f"\n-> Current perturbation ratio: {ratio} (attack type: {attack_type})")

            try:
                start_time = time.time()
                if ratio == 0.0:
                    perturbed_adj = A_orig
                    print(f"  Using original adjacency matrix (no perturbation)")
                else:
                    perturbed_adj = _load_perturbed_adj(dataset, ratio, ATTACKED_DIR)
                    print(f"  Perturbed adjacency matrix loaded")

                trainer = TADCTrainer(**RAMGCN_PARAMS, device=str(model_device))
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
                results[dataset][ratio_key][attack_type] = def_acc_val
                print(f" {attack_type} @ perturbation {ratio}: defense accuracy={def_acc_val:.4f}, time cost={elapsed_time:.2f}s")
            except Exception as e:
                error_msg = str(e)[:200]
                print(f" Execution failed: {error_msg}")
                results[dataset][ratio_key][attack_type] = f"Failed: {error_msg}"
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
                        trainer = TADCTrainer(**RAMGCN_PARAMS, device=str(model_device))
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
    set_seed(42, use_cuda)
    run_all()
