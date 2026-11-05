# Personalized Yet the Same? A Multi-Model Audit of Generative AI Attraction Recommendations Across Chinese and International Urban Hierarchies

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![Reproducibility: 100% Verified](https://img.shields.io/badge/Reproducibility-9%2F9%20Checks%20PASS-2ea44f.svg)](code/08_validate.py)

> **Double-Anonymous Replication Repository** for *Telematics and Informatics*  
> This repository contains the complete, self-contained empirical pipeline, 600-attraction municipal ground-truth catalog, 30-city open-benchmark stratification data, 10,191 SHA-256 verified live API query receipts (`101,910` recommendation slots across 7 experimental batches), econometric estimation scripts, publication figures, and Online Supplementary Materials (`Tables S1–S8`).

---

## 1. Quickstart (One-Command End-to-End Replication)

Every individual file in this repository is `< 48 MB` (fully compliant with standard GitHub `< 50 MB` file limits without requiring Git LFS). No API keys or external network calls are required to reproduce all tables, figures, and statistical tests from the archived raw JSONL receipts.

```bash
# 1. Install standard scientific Python dependencies
pip install -r requirements.txt

# 2. Run the complete 8-stage replication pipeline + 9-assertion validation gatekeeper
python run_all.py
```

Or execute individual pipeline stages step-by-step:

```bash
python code/01_ground_truth.py    # Build & verify the 600-POI catalog across 10 cities
python code/02_collect.py         # Verify 10,191 live HTTP-200 JSONL receipts & SHA-256 digests
python code/03_parse_geocode.py   # Multilingual entity resolution, 10% two-pipeline cross-val & geocoding
python code/04_metrics.py         # Compute RBO, NPI/NNPI, Delta_K10, Gi*/SDE, S1-S7 & MMR simulation
python code/05_models.py          # Fit City-Cluster (df=9) & 2-Way Dyadic MixedLM, PERMANOVA & NB GLMs
python code/06_figures.py         # Render 600-DPI publication Figures 1-6 (PNG, PDF, TIF, JPG)
python code/07_tables.py          # Export Tables 1-5 CSVs
python code/08_validate.py        # Run 9-assertion forensic & mathematical validation gatekeeper
```

---

## 2. Repository Structure

```text
.
├── README.md                          # Repository documentation & replication guide
├── LICENSE                            # MIT License (Anonymous Author(s))
├── requirements.txt                   # Python package dependencies
├── .gitignore                         # Excludes .env.local secrets, __pycache__, and OS metadata
├── .env.example                       # Sanitized template for optional live API re-collection
├── run_all.py                         # Top-level one-command replication orchestrator
├── Supplementary_Materials.md         # Online Supplementary Tables S1-S8 (Markdown)
├── Supplementary_Materials.docx       # Online Supplementary Tables S1-S8 (Formatted Word DOCX)
├── code/
│   ├── 00_config.py                   # Experimental grid, 10-city metadata, 48 personas, seeds & Haversine
│   ├── 01_ground_truth.py             # 600-POI catalog builder & exogenous salience rank verifier
│   ├── 02_collect.py                  # Raw JSONL receipt loader, deduplicator & SHA-256 integrity auditor
│   ├── 03_parse_geocode.py            # CJK/Latin entity resolution, 10% rule cross-val & 50-row manual audit
│   ├── 04_metrics.py                  # RBO, NNPI, collective monoculture, spatial SDE/Gi* & MMR simulation
│   ├── 05_models.py                   # Dyadic MixedLM (df=9 & 2-way dyadic SE), 999-perm PERMANOVA & NB GLMs
│   ├── 06_figures.py                  # Reproducible 600-DPI generation of Figures 1-6
│   ├── 07_tables.py                   # Reproducible export of Tables 1-5 CSVs
│   ├── 08_validate.py                 # Standalone 9-assertion empirical & repository hygiene validator
│   ├── collect_v2.py                  # Stateless multi-endpoint live API query runner (Batches B1-B6, B3b)
│   ├── collect_claude_arm.py          # Dedicated live API runner for the claude-sonnet-4-6 endpoint arm
│   ├── fetch_30city_benchmarks.py     # Live Overpass, Wikimedia Pageviews & Wikidata 30-city stratification
│   ├── fetch_wikimedia_benchmarks.py  # Live Wikimedia/Wikidata POI-level salience retriever
│   ├── repair_wikimedia_collisions.py # Batched Wikidata QID disambiguation & P625 coordinate verifier
│   └── run_all.py                     # Pipeline orchestrator
├── data/
│   ├── raw_responses_v2final/         # 7 raw JSONL batch files (batch_B1..B6, B3b; 10,431 raw lines -> 10,191 valid)
│   ├── ground_truth_pois_10cities.csv # 600 verified municipal attractions with WGS-84 & Wikimedia salience
│   ├── candidate_cities_30pool_distribution.csv # 30-city sampling pool & composite Z_info stratification (Table S1)
│   ├── audit_queries_corpus.csv       # 10,191 query-level records with parsed Top-10 POI IDs & latencies
│   ├── all_recommendations.csv        # 101,910 slot-level recommendation records with rationales & distances
│   ├── city_model_metrics.csv         # 40 city x endpoint cells with RBO, NPI, NNPI, Gini, SDE, Gi* & CIs
│   ├── dyadic_evaluation_pairs.csv    # Balanced N = 9,600 orthogonal dyadic pairs across 10 cities
│   ├── dyadic_se_comparison_s7.csv    # 4-estimator SE comparison: Naive, 1-Way, 2-Way Dyadic, City-Cluster (Table S7)
│   ├── permanova_2way_crossed_10cities.csv # 999-permutation Freedman-Lane two-way crossed PERMANOVA
│   ├── permanova_3endpoints_noflash.csv    # 3-endpoint PERMANOVA robustness excluding qwen3.7-flash
│   ├── collective_monoculture_omission_audit.csv # 4-endpoint collective zero-exposure permutation test
│   ├── rank_stratified_overlap_b1.csv # Top-k (k=1, 3, 5, 10) rank-stratified anchor vs. fringe overlap
│   ├── discursive_tailoring_audit.csv # Surface rationale tailoring vs. Same-POI rationale divergence
│   ├── language_x_origin_2x2_audit.csv # 2x2 Prompt Language (ZH/EN) x Endpoint Origin decomposition (Table S6)
│   ├── mmr_reranking_simulation.csv   # Pure empirical propensity MMR post-processing simulation
│   ├── mmr_lambda_sensitivity_frontier.csv # 16-grid (lambda_1, lambda_2) MMR Pareto frontier (Table S8)
│   ├── rule_cross_validation_1019rows.csv  # 10% two-pipeline rule cross-validation (N = 1,019 queries)
│   ├── entity_resolution_50sample_audit.csv # Stratified 50-slot manual entity-resolution audit
│   ├── Table_3_Full_Supplementary_25Cols.csv # Complete 25-column city x model empirical matrix (Table S5)
│   └── accessibility_coding_rubric.md # Three-tier step-free accessibility rubric & bilingual dictionary (Table S2)
├── tables/
│   ├── Table_1_City_Sample_and_Baselines.csv
│   ├── Table_2_Prompt_Matrix_and_Batches.csv
│   ├── Table_3_Personalization_and_Convergence_Audit.csv
│   ├── Table_4_Dyadic_Multilevel_Regression.csv
│   └── Table_5_Spatial_Concentration_and_NB_Regression.csv
└── figures/
    ├── Figure_1_Audit_Framework.{png,pdf}
    ├── Figure_2_Personalization_vs_Noise.{png,pdf}
    ├── Figure_3_InterModel_Convergence_PERMANOVA.{png,pdf}
    ├── Figure_4_Lorenz_and_LongTail_Erasure.{png,pdf}
    ├── Figure_5_Spatial_Hotspots_SDE.{png,pdf}
    └── Figure_6_Amplification_Elasticity_and_Robustness.{png,pdf}
```

---

## 3. Script-to-Manuscript Mapping

| Script | Manuscript Section | Outputs Produced |
| :--- | :--- | :--- |
| `code/fetch_30city_benchmarks.py` | §3.1 | `data/candidate_cities_30pool_distribution.csv`, **Table 1**, **Supplementary Table S1** |
| `code/01_ground_truth.py` | §3.1, §3.3 | `data/ground_truth_pois_10cities.csv`, **Supplementary Table S2**, **Supplementary Table S3** |
| `code/02_collect.py` | §3.2 | `data/endpoint_version_audit.csv`, **Table 2**, **Supplementary Table S4** |
| `code/03_parse_geocode.py` | §3.3 | `data/audit_queries_corpus.csv`, `data/all_recommendations.csv`, `data/rule_cross_validation_1019rows.csv` |
| `code/04_metrics.py` | §3.4, §4.1–§4.5 | `data/city_model_metrics.csv`, `data/collective_monoculture_omission_audit.csv`, `data/mmr_reranking_simulation.csv`, **Table 3**, **Supplementary Tables S5, S6, S8** |
| `code/05_models.py` | §3.5, §4.1–§4.4 | `data/dyadic_se_comparison_s7.csv`, `data/permanova_2way_crossed_10cities.csv`, **Table 4**, **Table 5**, **Table 6 (NB GLMs)**, **Supplementary Table S7** |
| `code/06_figures.py` | §3.1–§4.5 | **Figures 1–6** (`figures/Figure_1..6.png` & `.pdf`) |
| `code/07_tables.py` | §3.1–§4.4 | `tables/Table_1..5.csv` |
| `code/08_validate.py` | All Sections | `logs/validation_audit_report.json` (9/9 automated forensic & statistical checks) |

---

## 4. Optional Live API Query Collection

To issue new live queries against commercial LLM endpoints rather than re-running from the archived `data/raw_responses_v2final/batch_*.jsonl` receipts:

1. Copy `.env.example` to `.env.local` and fill in your API credentials (`POMO_BASE_URL`, `POMO_API_KEY`, `RELAY_BASE_URL`, `RELAY_API_KEY`).
2. Run a diagnostic probe or batch collection:
   ```bash
   python code/collect_v2.py --probe
   python code/collect_v2.py --batch B1 --limit 5
   python code/collect_claude_arm.py --batch B1 --limit 5
   ```
