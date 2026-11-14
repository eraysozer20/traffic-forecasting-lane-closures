# utils/cli.py
import argparse
import torch

def build_args():
    p = argparse.ArgumentParser()

    p.add_argument("--enable-cuda", action="store_true")
    p.add_argument("--root", type=str, default="data/")
    p.add_argument("--model_variant", type=str, default="wz", choices=["wz", "no"])
    p.add_argument(
        "--dataset_choice",
        type=str,
        default="Tyson",
        choices=[
            "Tyson", "PEMSD8", "PEMSD3", "PEMSD4", "PEMSD5",
            "PEMSD6", "PEMSD10", "PEMSD11", "PEMSD12", "SIM_GRID", "PEMSD3SUB", "PEMSD4SUB", "PEMSD8SUB", "PEMSD11SUB", "PEMSD12SUB"
        ],
    )

    p.add_argument(
        "--base_model",
        type=str,
        default="stgcn",
        choices=["stgcn", "gwnet", "lstm"],
    )

    p.add_argument("--adj", type=str, default=None)
    p.add_argument("--adjdata", type=str, default=None)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--con_data", type=str, default=None)
    p.add_argument("--dis_data", type=str, default=None)
    p.add_argument("--num_nodes", type=int, default=None)
    p.add_argument("--lambda_", type=float, default=None)

    p.add_argument("--T", type=int, default=12)
    p.add_argument("--P", type=int, default=6)

    # Baseline / generic batch size
    p.add_argument("--batch_size", type=int, default=16)

    # New: separate ResCAL batch sizes
    p.add_argument("--rescal_batch_size", type=int, default=8)
    p.add_argument("--rescal_eval_batch_size", type=int, default=8)

    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=1)

    p.add_argument("--feature", type=int, default=3)
    p.add_argument("--in_channels", type=int, default=1)

    p.add_argument("--hyp1_a_ratio", type=float, default=0.8)
    p.add_argument("--hyp1_b_ratio", type=float, default=0.5)

    p.add_argument("--ablation_modes", nargs="+", default=[
        "baseline",
        "baseline_no",
        "rescal_base_only",
        "rescal_base_plus_fold",
        "rescal_base_plus_fold_no_closure",
    ])
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--ablation_csv", type=str, default="ablation_results.csv")

    p.add_argument("--baseline_clean_start", type=int, default=0)
    p.add_argument("--rescal_clean_start", type=int, default=0)
    p.add_argument("--reuse_residual_cache", type=int, default=1)
    p.add_argument("--num_folds", type=int, default=5)

    p.add_argument(
        "--adj_type",
        type=str,
        default="binary",
        choices=["binary", "weighted"],
    )

    p.add_argument(
        "--weighted_adj_threshold",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--closure_representation",
        type=str,
        default="weighted",
        choices=["binary", "weighted"],
    )

    p.add_argument("--rescal_use_time_features", type=int, default=1)

    p.add_argument(
        "--rescal_residual_history",
        type=int,
        default=None,
        help="Number of past residual horizon channels used by ResCAL. If None, uses args.P."
    )

    args = p.parse_args()

    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    return args