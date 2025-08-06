#!/usr/bin/env python3
"""
Post-processing and plot generation for the empirical study.
Creates all 8 plots as specified in README_empirical.md.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pathlib import Path

# Set up paths
RAW = pathlib.Path("results/raw")
FIGS = pathlib.Path("results/figs")
FIGS.mkdir(parents=True, exist_ok=True)

def load_all_data():
    """Load consolidated results from the single CSV file."""
    consolidated_file = RAW / "all_results.csv"
    
    if not consolidated_file.exists():
        print(f"Consolidated file {consolidated_file} not found!")
        print("Run consolidate_results.py first or batch_runner.py with updated code.")
        return pd.DataFrame()
    
    print(f"Loading consolidated data from {consolidated_file}...")
    df = pd.read_csv(consolidated_file)
    print(f"Loaded {len(df)} simulation results")
    return df

def main():
    # Load all data
    df = load_all_data()
    if df.empty:
        return
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Modes: {df['mode'].unique()}")
    print(f"Parameter ranges: n={df['n'].unique()}, m={df['m'].unique()}, p={df['p'].unique()}")
    
    # Set up matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # ---------- 1. Async convergence CDFs ----------
    print("Generating async convergence CDFs...")
    async_df = df[df['mode'] == "async"]
    
    if not async_df.empty:
        for (n, p), g in async_df.groupby(["n", "p"]):
            plt.figure(figsize=(10, 7))
            sns.ecdfplot(g, x="T_stop")
            plt.title(f"Asynchronous Convergence Time CDF\nn={n} agents, p={p} edge probability, m∈{sorted(g['m'].unique())}\n{len(g)} simulation runs", 
                     fontsize=14, pad=20)
            plt.xlabel("Number of updates to convergence (T_stop)", fontsize=12)
            plt.ylabel("Cumulative probability", fontsize=12)
            plt.grid(True, alpha=0.3)
            # Add summary stats as text
            median_t = g["T_stop"].median()
            plt.text(0.7, 0.2, f"Median: {median_t:.0f} updates", transform=plt.gca().transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            plt.savefig(FIGS / f"async_cdf_n{n}_p{p}.png", bbox_inches='tight', dpi=300)
            plt.close()
    
    # ---------- 2. Φ_final vs. m (async) -----------
    print("Generating Φ_final box plot...")
    if not async_df.empty:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=async_df, x="m", y="phi_f")
        plt.yscale("log")
        plt.xlabel("Number of items in rankings (m)", fontsize=12)
        plt.ylabel("Final potential Φ_f (log scale)", fontsize=12)
        plt.title(f"Final Potential Distribution - Asynchronous Dynamics\nn∈{sorted(async_df['n'].unique())}, p∈{sorted(async_df['p'].unique())}\n{len(async_df)} simulation runs", 
                 fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        # Add sample size annotations
        for i, m_val in enumerate(sorted(async_df['m'].unique())):
            count = len(async_df[async_df['m'] == m_val])
            plt.text(i, plt.ylim()[0]*1.5, f"n={count}", ha='center', fontsize=10)
        plt.savefig(FIGS / "phi_f_box_async.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # ---------- 3. Sync convergence heat-map -------
    print("Generating sync convergence heatmap...")
    sync_df = df[df['mode'] == "sync"]
    
    if not sync_df.empty:
        heat = sync_df.pivot_table(values="converged", index="n", columns="p", aggfunc="mean")
        plt.figure(figsize=(12, 10))
        sns.heatmap(heat, annot=True, cmap="viridis", vmin=0, vmax=1, 
                   cbar_kws={'label': 'Convergence fraction'}, fmt='.2f', annot_kws={'fontsize': 11})
        plt.title(f"Synchronous Convergence Rate by Network Parameters\nm∈{sorted(sync_df['m'].unique())}, {len(sync_df)} total runs\nDark = Low convergence, Light = High convergence", 
                 fontsize=14, pad=20)
        plt.xlabel("Edge probability (p) - Network density", fontsize=12)
        plt.ylabel("Number of agents (n) - Network size", fontsize=12)
        plt.savefig(FIGS / "sync_heatmap.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # ---------- 4. Cycle length histogram ----------
    print("Generating cycle length histogram...")
    sync_cycles = sync_df[sync_df['converged'] == 0]
    
    if not sync_cycles.empty and sync_cycles['cycle_len'].max() > 0:
        plt.figure(figsize=(12, 8))
        sns.histplot(sync_cycles, x="cycle_len", bins=20)
        plt.xlabel("Cycle length (number of states in periodic orbit)", fontsize=12)
        plt.ylabel("Number of non-converged runs", fontsize=12)
        plt.title(f"Cycle Length Distribution - Synchronous Non-Converged Runs\n{len(sync_cycles)} non-converged runs out of {len(sync_df)} total sync runs\nn∈{sorted(sync_cycles['n'].unique())}, m∈{sorted(sync_cycles['m'].unique())}, p∈{sorted(sync_cycles['p'].unique())}", 
                 fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        # Add stats
        mean_len = sync_cycles['cycle_len'].mean()
        plt.axvline(mean_len, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_len:.1f}')
        plt.legend()
        plt.savefig(FIGS / "cycle_len_hist.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # ---------- 5. Consensus histograms -----------
    print("Generating consensus histograms...")
    for mode in ["async", "sync"]:
        mode_df = df[df['mode'] == mode]
        if not mode_df.empty:
            plt.figure(figsize=(12, 8))
            sns.histplot(mode_df, x="consensus_frac", bins=20, alpha=0.7)
            plt.xlabel("Consensus fraction (proportion agreeing on plurality winner)", fontsize=12)
            plt.ylabel("Number of simulation runs", fontsize=12)
            plt.title(f"Consensus Level Distribution - {mode.capitalize()} Dynamics\nn∈{sorted(mode_df['n'].unique())}, m∈{sorted(mode_df['m'].unique())}, p∈{sorted(mode_df['p'].unique())}\n{len(mode_df)} simulation runs", 
                     fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            # Add summary stats
            mean_cons = mode_df['consensus_frac'].mean()
            median_cons = mode_df['consensus_frac'].median()
            plt.axvline(mean_cons, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_cons:.3f}')
            plt.axvline(median_cons, color='blue', linestyle='--', alpha=0.8, label=f'Median: {median_cons:.3f}')
            plt.legend()
            plt.savefig(FIGS / f"consensus_{mode}.png", bbox_inches='tight', dpi=300)
            plt.close()
    
    # ---------- 6. Consensus heat-map --------------
    print("Generating consensus heatmap...")
    if not df.empty:
        cons = df.pivot_table(values="consensus_frac", index="n", columns="p", aggfunc="mean")
        plt.figure(figsize=(12, 10))
        sns.heatmap(cons, annot=True, cmap="crest", vmin=0, vmax=1,
                   cbar_kws={'label': 'Mean consensus fraction'}, fmt='.3f', annot_kws={'fontsize': 11})
        plt.title(f"Mean Consensus Fraction Across Network Parameters\nBoth async & sync modes, m∈{sorted(df['m'].unique())}, {len(df)} total runs\nDark = High consensus, Light = Low consensus", 
                 fontsize=14, pad=20)
        plt.xlabel("Edge probability (p) - Network density", fontsize=12)
        plt.ylabel("Number of agents (n) - Network size", fontsize=12)
        plt.savefig(FIGS / "consensus_heatmap.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # ---------- 7. Fragmentation vs. consensus -----
    print("Generating consensus vs. components scatter plot...")
    if not df.empty:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x="k_components", y="consensus_frac", hue="mode", alpha=0.6, s=60)
        plt.xlabel("Number of disagreement components\n(Connected components in disagreement graph)", fontsize=12)
        plt.ylabel("Consensus fraction\n(Proportion agreeing on plurality winner)", fontsize=12)
        plt.title(f"Consensus vs. Network Fragmentation\nn∈{sorted(df['n'].unique())}, m∈{sorted(df['m'].unique())}, p∈{sorted(df['p'].unique())}\n{len(df)} simulation runs, colored by dynamics mode", 
                 fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Dynamics Mode", title_fontsize=12, fontsize=11)
        # Add correlation info
        corr = df[['k_components', 'consensus_frac']].corr().iloc[0,1]
        plt.text(0.02, 0.98, f"Overall correlation: r = {corr:.3f}", transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=11, va='top')
        plt.savefig(FIGS / "consensus_vs_clusters.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # ---------- 8. Δ_FR vs. consensus --------------
    print("Generating ΔFR vs. consensus plot...")
    if not df.empty:
        g = sns.lmplot(data=df, x="consensus_frac", y="delta_FR", hue="mode", 
                      scatter_kws=dict(alpha=0.4, s=40), height=8, aspect=1.3)
        g.set_axis_labels("Consensus fraction\n(Proportion agreeing on plurality winner)", 
                         "Average Footrule shift (Δ_FR)\n(Mean ranking change per agent)", fontsize=12)
        plt.suptitle(f"Ranking Distortion vs. Consensus Achievement\nn∈{sorted(df['n'].unique())}, m∈{sorted(df['m'].unique())}, p∈{sorted(df['p'].unique())}\n{len(df)} simulation runs with trend lines by dynamics mode", 
                    fontsize=14, y=1.02)
        plt.grid(True, alpha=0.3)
        # Add correlation info for each mode
        corr_async = df[df['mode']=='async'][['consensus_frac', 'delta_FR']].corr().iloc[0,1]
        corr_sync = df[df['mode']=='sync'][['consensus_frac', 'delta_FR']].corr().iloc[0,1]
        plt.text(0.02, 0.98, f"Correlations:\nAsync: r = {corr_async:.3f}\nSync: r = {corr_sync:.3f}", 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=11, va='top')
        plt.savefig(FIGS / "deltaFR_vs_consensus.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"All plots saved to {FIGS}/")
    
    # Generate summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total simulations: {len(df)}")
    
    if not async_df.empty:
        async_converged = async_df['converged'].mean()
        print(f"Async convergence rate: {async_converged:.3f}")
    
    if not sync_df.empty:
        sync_converged = sync_df['converged'].mean()
        print(f"Sync convergence rate: {sync_converged:.3f}")
    
    for mode in ['async', 'sync']:
        mode_df = df[df['mode'] == mode]
        if not mode_df.empty:
            mean_consensus = mode_df['consensus_frac'].mean()
            print(f"Mean consensus ({mode}): {mean_consensus:.3f}")
    
    print("Plot generation completed!")

if __name__ == "__main__":
    main()