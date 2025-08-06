Below is the **complete `README_empirical.md`**—ready to commit at the root of **`rank-propogation-game`**.
It tells collaborators (human or LLM) exactly how to reproduce every run, build every figure, and auto-draft the “Empirical Results” storyboard that will drop straight into your thesis.

---

# Empirical Study – Rank-Propagation Dynamics

> *“Strong theorems are nice; strong plots are nicer.”*

---

## 0  |  Run matrix (36 000 simulations)

```yaml
# run_matrix.yaml  —  version 2025-08-06
n:      [20, 50, 100, 150, 200]           # number of agents
m:      [5, 8, 10, 12, 15]                # items per ranking
p:      [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]  # ER edge probability
mode:   [async, sync]                     # update schedule
reps:   120                               # repetitions per grid-point
base_seed: 20250806                       # reproducible seeds
max_steps_sync: "4*n*(m-1)"               # upper bound for cycle detection
max_steps_async: "inf"                    # run until fix-point
```

### Run command

```bash
python batch_runner.py \
       --config run_matrix.yaml \
       --outdir results/raw \
       --jobs 16             # adjust parallelism
```

Every simulation must emit **one CSV** named

```
results/raw/run_n{n}_m{m}_p{p}_{mode}_rep{rep}.csv
```

with **all** columns below:

| column                     | meaning                                                                                      |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `n, m, p, mode, rep, seed` | —                                                                                            |
| `T_stop`                   | number of individual updates actually applied (async) or rounds executed (sync)              |
| `converged`                | **1** if reached a fix-point; **0** if a cycle was detected (sync only)                      |
| `cycle_len`                | period length if `converged == 0`; **0** otherwise                                           |
| `phi_0, phi_f`             | potential Φ at start and at the “final snapshot” (see §4)                                    |
| `k_components`             | # connected components in the final disagreement graph                                       |
| `winner_flip`              | 1 if plurality top-1 item differs between t = 0 and the final snapshot                       |
| **`winner_item`**          | ID (0-based) of the plurality winner in the final snapshot                                   |
| **`consensus_frac`**       | fraction of nodes whose personal top choice equals `winner_item` at that snapshot (∈ \[0,1]) |

---

## 1  |  Post-processing & plots

Create `aggregate_and_plot.py` alongside `batch_runner.py`.

```python
import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt, pathlib

RAW   = pathlib.Path("results/raw")
FIGS  = pathlib.Path("results/figs"); FIGS.mkdir(parents=True, exist_ok=True)

df = pd.concat(pd.read_csv(f) for f in RAW.glob("*.csv"))

# ---------- 1. Async convergence CDFs ----------
async_df = df[df.mode == "async"]
for (n, p), g in async_df.groupby(["n", "p"]):
    sns.ecdfplot(g, x="T_stop")
    plt.title(f"Async convergence CDF  —  n={n}, p={p}")
    plt.xlabel("updates"); plt.ylabel("CDF")
    plt.savefig(FIGS / f"async_cdf_n{n}_p{p}.pdf"); plt.clf()

# ---------- 2. Φ_final vs. m (async) -----------
sns.boxplot(async_df, x="m", y="phi_f")
plt.yscale("log"); plt.xlabel("m"); plt.ylabel("Φ_final")
plt.savefig(FIGS / "phi_f_box_async.pdf"); plt.clf()

# ---------- 3. Sync convergence heat-map -------
sync = df[df.mode == "sync"]
heat = sync.pivot_table(values="converged", index="n", columns="p", aggfunc="mean")
sns.heatmap(heat, annot=True, cmap="viridis", vmin=0, vmax=1)
plt.title("Fraction converged (sync)")
plt.savefig(FIGS / "sync_heatmap.pdf"); plt.clf()

# ---------- 4. Cycle length histogram ----------
sns.histplot(sync[sync.converged == 0], x="cycle_len", log_scale=True, bins=20)
plt.xlabel("cycle length"); plt.savefig(FIGS / "cycle_len_hist.pdf"); plt.clf()

# ---------- 5. Consensus histograms -----------
for mode in ["async", "sync"]:
    sns.histplot(df[df.mode == mode], x="consensus_frac", bins=20)
    plt.xlabel("fraction of nodes sharing plurality winner")
    plt.title(f"Consensus — {mode}")
    plt.savefig(FIGS / f"consensus_{mode}.pdf"); plt.clf()

# ---------- 6. Consensus heat-map --------------
cons = df.pivot_table(values="consensus_frac", index="n", columns="p", aggfunc="mean")
sns.heatmap(cons, annot=True, cmap="crest", vmin=0, vmax=1)
plt.title("Mean consensus fraction (all m pooled)")
plt.savefig(FIGS / "consensus_heatmap.pdf"); plt.clf()

# ---------- 7. Fragmentation vs. consensus -----
import numpy as np
sns.scatterplot(df, x="k_components", y="consensus_frac", hue="mode", alpha=.4)
plt.xlabel("# disagreement components"); plt.ylabel("consensus_frac")
plt.savefig(FIGS / "consensus_vs_clusters.pdf"); plt.clf()

# ---------- 8. Δ_FR vs. consensus --------------
sns.lmplot(data=df, x="consensus_frac", y="delta_FR", hue="mode", scatter_kws=dict(alpha=.3))
plt.savefig(FIGS / "deltaFR_vs_consensus.pdf")
```

---

## 2  |  Storyboard auto-draft

Create `build_storyboard.py`.
It answers **eight** focused questions and dumps them as JSON.

```python
import json, math, numpy as np, pandas as pd, pathlib, statsmodels.api as sm

RAW = pathlib.Path("results/raw")
df  = pd.concat(pd.read_csv(f) for f in RAW.glob("*.csv"))
story = {}

# ---------- Q1  Async certainty ----------
async_fail = df[(df.mode=="async") & (df.converged==0)]
story["Q1"] = ("All asynchronous runs converged."
               if async_fail.empty else
               f"{len(async_fail)} async runs cycled; worst case: "
               f"{async_fail[['n','m','p']].drop_duplicates().values.tolist()}")

# ---------- Q2  Convergence speed law ----------
tbl = {}
for p, g in df[df.mode=="async"].groupby("p"):
    X = np.log(g["n"]); y = np.log(g["T_stop"])
    alpha = sm.OLS(y, sm.add_constant(X)).fit().params[1]
    tbl[p] = round(alpha, 3)
story["Q2"] = tbl  # {p: α}

# ---------- Q3  Sync fragility ----------
frag = (df[df.mode=="sync"]
        .groupby(["n","m","p"])
        .converged.mean()
        .reset_index())
weak   = frag[frag.converged < 0.5]
story["Q3"] = weak.to_dict(orient="records")

# ---------- Q4  Cycle anatomy ----------
f = df[(df.mode=="sync") & (df.converged==0)]["cycle_len"]
story["Q4"] = dict(min=int(f.min()), median=float(f.median()),
                   pct95=float(f.quantile(.95)))

# ---------- Q5  Consensus level ----------
def mean_cons(mode): return round(df[df.mode==mode].consensus_frac.mean(),3)
story["Q5"] = {"async": mean_cons("async"), "sync": mean_cons("sync")}

# ---------- Q6  Consensus vs. clusters ----------
r = df[["k_components","consensus_frac"]].corr().iloc[0,1]
story["Q6"] = f"Pearson r = {r:.3f}"

# ---------- Q7  Distortion vs. consensus ----------
r2 = df[["consensus_frac","delta_FR"]].corr().iloc[0,1]
story["Q7"] = f"Pearson r = {r2:.3f} (negative ⇒ more consensus, less distortion)"

# ---------- Q8  Winner-flip risk ----------
flip_tbl = (df.groupby(["m","p"])
              .winner_flip.mean()
              .reset_index()
              .pivot(index="m", columns="p", values="winner_flip")
              .round(3))
story["Q8"] = flip_tbl.to_dict()

json.dump(story, open("story.json","w"), indent=2, ensure_ascii=False)
print("story.json written.")
```

---

## 3  |  Eight empirical questions (for the thesis)

1. **Async certainty** – Does asynchronous updating ever fail to converge?
2. **Convergence speed law** – How does median convergence time scale with *n* (per density *p*)?
3. **Sync fragility** – For which network sizes and densities does synchronous updating converge less than 50 % of the time?
4. **Cycle anatomy** – When sync fails, what cycle lengths arise (min / median / 95-th percentile)?
5. **Consensus level** – On average, what fraction of agents share the plurality winner under async vs. sync?  How does density *p* affect this?
6. **Consensus vs. clusters** – Does a higher number of disagreement components predict lower consensus?
7. **Distortion vs. consensus** – Do larger Footrule shifts correlate with weaker consensus?
8. **Winner-flip risk** – What share of runs change the plurality winner, and how does this depend on *m* and *p*?

The JSON produced by `build_storyboard.py` gives one succinct answer per question—copy-paste each bullet into the “Empirical Results” narrative.

---

## 4  |  Cycle snapshots (definition)

If a synchronous run enters a cycle:

1. Detect the minimal period (`cycle_len`).
2. Treat the **first profile** of that period as the “final snapshot” for all measurements (`winner_item`, `consensus_frac`, `phi_f`, etc.).

This deterministic rule avoids ambiguity and requires no averaging across states.

---

## 5  |  Quick sanity targets

| Mode  | Dense (*p ≥ 0.5*) consensus\_frac | Sparse (*p ≤ 0.1*) consensus\_frac |
| ----- | --------------------------------- | ---------------------------------- |
| async | 0.85 – 1.00                       | 0.40 – 0.70                        |
| sync  | 0.70 – 0.95 (if converged)        | 0.30 – 0.60 (cycles common)        |

If your numbers diverge greatly, double-check the plurality-winner and disagreement-graph routines.

---

Happy crunching!