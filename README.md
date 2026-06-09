# Ad Click-Through Rate (CTR) User Segmentation

> Discover hidden user segments in ad-impression data using unsupervised learning, then recommend a concrete budget reallocation that lifts campaign CTR — with an honest disclosure of what the data can and can't tell us.

**Headline result:** A PyTorch autoencoder + K-Means at K = 6 on a 4-dimensional latent space outperforms baseline K-Means on the raw 16-dimensional feature space — **Silhouette 0.226 → 0.480** (2.1×) and **Davies-Bouldin Index 1.83 → 0.74** (2.5× better). The autoencoder surfaces a high-value segment (*Female + Tablet + Education browsers*, 74.05% CTR) that the raw-feature baseline could not isolate. Recommended budget reallocation lifts overall CTR from **65.0% → 66.1% (+1.1 pp absolute, +1.7% relative)** under the dataset's existing exposure pattern.

---

## Problem

A digital advertising company displays ads to all users indiscriminately, resulting in low Click-Through Rate (CTR) and wasted ad spend. The ad operations team needs **describable, stable user segments** — not per-user click probabilities — to inform creative strategy and budget allocation.

**Business question:** *"Which user segments should receive the most ad spend, and which should receive the least, to maximize overall Click-Through Rate?"*

- **Optimization target:** CTR (sole north star).
- **Stated limitation:** ROAS is the true business objective but is not measurable from this dataset (no conversion or revenue columns). CTR is the measurable proxy under the assumption that conversion rate is roughly constant across segments — an assumption to validate in production A/B testing.
- **Recommendation unit:** percentage reallocation of the existing ad budget from low-CTR to high-CTR segments.

---

## Dataset

| Field | Value |
|---|---|
| Source | `ad_click_dataset.csv` (synthetic / educational) |
| Rows | 10,000 ad impressions |
| Unique users | 4,000 |
| Impressions per user (avg) | 2.5 |
| Columns | `id`, `full_name`, `age`, `gender`, `device_type`, `ad_position`, `browsing_history`, `time_of_day`, `click` |
| Missingness | 47% on `age` / `gender` / `browsing_history`; 20% on the other three |
| Overall impression-CTR | 65.0% |
| Overall user-CTR | 12.5% |

**Important data property uncovered during EDA:** clicker users appear in the impression log ~13× more often than non-clicker users — the dataset reflects exposure under an *already-optimized* targeting policy, not the random targeting assumed in the original problem framing. This is acknowledged as a methodology limitation throughout.

---

## Approach

Two-stage comparison on **identical preprocessing, evaluation metrics, K-selection methodology, and cluster-profiling pipeline.** The only thing that differs between stages is the source of the feature matrix fed to K-Means.

| Stage | Pipeline | Feature space |
|---|---|---|
| **1 (baseline)** | Preprocess → K-Means | Raw 16-dim `X` |
| **2 (advanced)** | Preprocess → Autoencoder → encode → K-Means | 4-dim learned latents |

The preprocessing pipeline produces a user-level frame `X` (4,000 rows × 16 columns, all in `[0, 1]`) with hard assertions guarding against target leakage, missing values, scaling regressions, and index misalignment. Every step is documented inside `ad_click_segmentation.ipynb` with the *why*, not just the *what*.

### Key methodology decisions

| Decision | Choice | Rationale |
|---|---|---|
| Granularity collapse | `df.groupby("id").first()` (preserves first non-null per column) | Catches users whose NaN pattern varies across impressions — `drop_duplicates` would silently drop known demographic info |
| Missing `age` | Median impute + `is_age_missing` binary flag | Median is robust to skew; flag retains the *fact* of missingness as a feature |
| Missing categoricals | `"MISSING"` as its own category | Preserves all 4,000 users without fabricating values; ~47% missingness is too severe for mode imputation |
| Encoding | One-hot (`dtype=float`) | Categories have no natural order; target-encoding would risk leakage |
| Scaling | `MinMaxScaler` on `age` only (not StandardScaler) | Matches the `[0, 1]` range of the one-hot columns so no feature dominates Euclidean distance in K-Means |
| K selection (both stages) | Filter (`min_cluster_pct ≥ 5%`) → Silhouette (primary) → DB Index (tiebreaker) → WCSS elbow → robustness → interpretability | Single-metric K selection is brittle; filter eliminates operationally invalid Ks before metric comparison |
| Autoencoder architecture | `16 → 12 → 8 → 4 → 8 → 12 → 16` symmetric, ReLU hidden / linear latent / sigmoid output | Depth chosen empirically — shallower variants produced lower downstream cluster quality; ~700 params on 3,200 training users is well below overfit risk |
| Loss | `MSE(age) + BCE(15 binary columns)` | Matches variable types: regression on continuous, classification on binary |
| Training | Adam(lr=1e-3), batch=64, max_epochs=200, EarlyStopping(patience=15, restore_best), ReduceLROnPlateau | Best-val-weights restoration is the part most people forget |

---

## Results

### Stage 1 — K-Means at K = 4 on raw features

K-Means mechanically partitioned users by `device_type` (the only categorical with four nearly-equal-sized groups):

| Cluster | Device | n_users | Impressions | CTR | Lift |
|---:|---|---:|---:|---:|---:|
| 0 | Desktop | 1,099 | 3,183 | **70.94%** | 1.09× |
| 1 | Tablet | 1,077 | 3,046 | 69.99% | 1.08× |
| 2 | Mobile | 1,134 | 3,081 | 68.48% | 1.05× |
| 3 | (none recorded) | 690 | 690 | **0.00%** | 0.00× |

**Cluster quality:** Silhouette 0.226, DB Index 1.83.
**Limitation:** every non-device feature came out indistinguishable across clusters (`gender_mode = MISSING`, `browsing_mode = MISSING` in every cluster) — the segmentation captured the data's strongest available structure but did not surface joint demographic patterns. Cluster 3's 0% CTR is partly a synthetic-data artifact (exactly 1 impression per user; the generator pre-deprioritized this group).

### Stage 2 — Autoencoder + K-Means at K = 6 on 4-dim latents

The autoencoder surfaced **demographic and behavioral joint structure** the raw K-Means missed:

| Cluster | n_users | CTR | Lift | Identity (one sentence) |
|---:|---:|---:|---:|---|
| **4** | 1,201 | **74.05%** | 1.14× | **Female-skewing (36%), Tablet (29%), Education browsers (21%)** — the only cluster where `browsing_history` surfaced as a non-MISSING mode |
| 1 | 562 | 67.10% | 1.03× | Female (39%), Tablet (45%), browsing undisclosed |
| 2 | 463 | 61.72% | 0.95× | Gender + browsing both 100% undisclosed, Tablet (51%) |
| 5 | 980 | 60.71% | 0.93× | Gender 98% undisclosed, Mobile, Social Media |
| 0 | 410 | 56.63% | 0.87× | Male (40%), Mobile (74%), browsing undisclosed |
| **3** | 384 | **35.38%** | 0.54× | **Anonymous low-information mobile users** — gender + browsing 100% undisclosed, Mobile (59%) |

**Cluster quality:** Silhouette 0.480, DB Index 0.74.

### Stage 1 vs Stage 2 — head to head

| Metric | Stage 1 (raw K-Means) | Stage 2 (AE + K-Means) | Winner |
|---|---:|---:|---|
| Silhouette Score | 0.226 | **0.480** | Stage 2 (2.1×) |
| Davies-Bouldin Index | 1.83 | **0.74** | Stage 2 (2.5× lower) |
| Best cluster CTR | 70.94% | **74.05%** | Stage 2 (+3.11 pp) |
| Worst cluster CTR | 0.00% (artifact) | 35.38% (realistic) | — (Stage 1's was inflated) |
| Browsing surfaced as discriminator | No | Yes | Stage 2 |
| Min cluster size (% users) | 17.25% | 9.60% | Stage 1 (larger floor) |
| Projected CTR lift | +4.82 pp | **+1.11 pp** | — (Stage 2's is more honest) |

**Verdict: Stage 2 (Autoencoder + K-Means at K = 6) is the recommended model.** Both cluster quality metrics improve substantially, the segments are demographically interpretable rather than mechanically device-defined, and the smaller projected CTR lift is a *feature*, not a bug — Stage 1's larger number was inflated by a synthetic-data artifact.

### Business recommendation

Withdraw the **5.71%** of impression budget currently spent on cluster 3 (anonymous low-information mobile users, 35.38% CTR — well below the 65% baseline) and redistribute toward the two above-baseline clusters: **cluster 4** (74.05% CTR, *Female + Tablet + Education*) and **cluster 1** (67.10% CTR). Tilt redistribution toward cluster 4 since it leads on both CTR and segment size.

**Projected overall CTR: 65.00% → 66.11% (+1.11 pp absolute, +1.7% relative)** under the dataset's existing exposure pattern.

---

## Limitations (honest disclosure)

1. **Synthetic-data caveat.** Observed CTRs reflect the dataset's past targeting policy, not random exposure (clickers received ~13× more impressions than non-clickers). The projected +1.11 pp lift should be treated as an **upper bound** for production rollout under truly random targeting, not a forecast.
2. **Latent space is non-interpretable by construction.** Stage 2's four latent dimensions have no semantic meaning; cluster identities are reverse-engineered from the user-level demographic profile in Section 6.10 of the notebook.
3. **Seed sensitivity.** The autoencoder's latents depend on initialization. A robust deployment would average over multiple seeded training runs; this project fixes a single seed for reproducibility.
4. **CTR ≠ ROAS.** CTR is a measurable proxy. The true business goal is return on ad spend, which requires conversion data this dataset does not contain.
5. **Methodological choice flagged in writeup:** K-Means uses Euclidean distance on a feature space with 15 of 16 columns being one-hot binaries. K-Prototypes or Gower-distance K-Medoids would be more theoretically appropriate; this was an in-scope project limitation.

---

## How to reproduce

### 1. Clone and set up the environment

```bash
git clone https://github.com/ManishDiddi/Ad-Targeting-Optimization.git
cd Ad-Targeting-Optimization

python3 -m venv .venv
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn jupyter torch
```

(The pinned `requirements.txt` in the repo may lag behind the live notebook's runtime versions; the command above installs the libraries the notebook actually imports.)

### 2. Run the notebook

```bash
jupyter notebook ad_click_segmentation.ipynb
```

Then **Kernel → Restart and Run All**. Expected runtime: under 2 minutes on a modern laptop CPU (no GPU required).

### 3. What you should see

- Section 3 (EDA) — distributions, missingness, baseline CTR
- Section 4 (preprocessing) — final validation block prints `X shape: (4000, 16)`, leakage check passes, all values in `[0, 1]`
- Section 5 (Stage 1) — K-selection plot, K = 4 chosen, cluster profiles, Stage 1 verdict
- Section 6 (Stage 2) — autoencoder training curves, K-selection on latents, K = 6 chosen, cluster profiles, Stage 2 verdict
- Section 7 (comparison) — side-by-side table and bar chart, final deployment recommendation

### 4. Reproducibility notes

- Seed `42` is set for `random`, `numpy`, `PYTHONHASHSEED`, `torch`.
- The CSV is committed to `data/` so no download step is required.
- K-Means uses `n_init=10`, pinned explicitly (sklearn's `"auto"` default has shifted across versions).
- Notebook runs top-to-bottom on a fresh kernel restart.

---

## Project structure

```
.
├── ad_click_segmentation.ipynb   # the project — single notebook, top-to-bottom
├── data/
│   └── ad_click_dataset.csv      # 10,000 impressions, committed for reproducibility
├── requirements.txt              # pinned dependencies (may lag; see step 1 above)
├── README.md                     # this file
└── .gitignore
```