import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .graph_utils import erdos_renyi_connected

from .dynamics import run_sync_dynamics
from .metrics import largest_cluster_fraction, mean_pair_agreement

def parse_args():
    parser = argparse.ArgumentParser(description="Synchronous Footrule-median dynamics on random graphs")
    parser.add_argument("--n", type=int, default=3, help="number of agents")
    parser.add_argument("--m", type=int, default=3, help="number of alternatives")
    parser.add_argument("--p", type=float, default=0.5, help="edge probability for G(n,p)")
    parser.add_argument("--trials", type=int, default=10000, help="number of trials to simulate")
    parser.add_argument("--max-steps", type=int, default=500, help="maximum synchronous steps")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--self-inclusive", action="store_true", help="include agent's own ranking in median"
    )
    parser.add_argument("--out-csv", type=Path, default=Path("results.csv"), help="per-trial results file")
    parser.add_argument("--summary-csv", type=Path, default=Path("summary.csv"), help="summary results file")
    return parser.parse_args()


def append_row(path: Path, data: dict):
    df = pd.DataFrame([data])
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(["n", "m", "p"])
    rows = []
    for (n, m, p), g in groups:
        cycle_counts = g.loc[g["converged"] == 0, "cycle_length"].value_counts().to_dict()
        rows.append(
            {
                "n": n,
                "m": m,
                "p": p,
                "converged_count": int(g["converged"].sum()),
                "non_converged_count": int((1 - g["converged"]).sum()),
                "phi_initial_mean": g["phi_initial"].mean(),
                "phi_initial_std": g["phi_initial"].std(ddof=0),
                "phi_final_mean": g["phi_final"].mean(),
                "phi_final_std": g["phi_final"].std(ddof=0),
                "largest_cluster_frac_mean": g["largest_cluster_frac"].mean(),
                "largest_cluster_frac_std": g["largest_cluster_frac"].std(ddof=0),
                "mean_pair_agreement_mean": g["mean_pair_agreement"].mean(),
                "mean_pair_agreement_std": g["mean_pair_agreement"].std(ddof=0),
                "convergence_rate": g["converged"].mean(),
                "cycle_length_distribution": json.dumps(cycle_counts),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_path = args.out_csv
    summary_path = args.summary_csv

    for trial_id in range(args.trials):
        seed_used = args.seed + trial_id
        rng = np.random.default_rng(seed_used)
        G, _ = erdos_renyi_connected(args.n, args.p, rng)
        initial_perms = np.array([rng.permutation(args.m) for _ in range(args.n)], dtype=int)
        sim = run_sync_dynamics(
            G,
            initial_perms,
            max_steps=args.max_steps,
            self_inclusive=args.self_inclusive,
        )
        perms_final = sim["perms"]
        phi_initial = sim["phi_history"][0]
        phi_final = sim["phi_history"][-1]
        phi_norm_initial = (
            phi_initial / (G.number_of_edges() * (args.m * args.m / 2))
            if G.number_of_edges() > 0
            else 0.0
        )
        phi_norm_final = (
            phi_final / (G.number_of_edges() * (args.m * args.m / 2))
            if G.number_of_edges() > 0
            else 0.0
        )
        row = {
            "trial_id": trial_id,
            "n": args.n,
            "m": args.m,
            "p": args.p,
            "seed_used": seed_used,
            "converged": int(sim["converged"]),
            "cycle_length": sim["cycle_length"],
            "steps_run": sim["steps_run"],
            "phi_initial": phi_initial,
            "phi_final": phi_final,
            "phi_normalised_initial": phi_norm_initial,
            "phi_normalised_final": phi_norm_final,
            "largest_cluster_frac": largest_cluster_fraction(perms_final),
            "mean_pair_agreement": mean_pair_agreement(perms_final, G),
        }
        append_row(results_path, row)

    df = pd.read_csv(results_path)
    summary_df = aggregate_summary(df)
    summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
