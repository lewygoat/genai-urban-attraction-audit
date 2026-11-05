# -*- coding: utf-8 -*-
"""
run_all.py — Turnkey One-Command Replication Runner for the GitHub / Double-Blind Replication Package.

Executes the complete empirical audit pipeline from raw data and live API receipts to econometric models,
publication figures, CSV tables, and the 9-assertion validation gatekeeper:
  [1/8] 01_ground_truth.py      — 600-POI municipal catalog & live Wikimedia/Wikidata salience benchmarks
  [2/8] 02_collect.py           — 10,191 live HTTP-200 JSONL API receipt verification & SHA-256 audit
  [3/8] 03_parse_geocode.py     — Multilingual entity resolution, 10% two-pipeline rule cross-validation & geocoding
  [4/8] 04_metrics.py           — RBO, NPI/NNPI, Delta_K10, collective monoculture, Gi*/SDE, S1-S7 & empirical MMR
  [5/8] 05_models.py            — City-cluster (df=9) & 2-way dyadic MixedLM, 999-perm PERMANOVA & stratified NB GLMs
  [6/8] 06_figures.py           — 600-DPI publication figures (Figures 1-6)
  [7/8] 07_tables.py            — Main manuscript and supplementary CSV tables (Tables 1-5 & Table S5)
  [8/8] 08_validate.py          — 9-assertion empirical & repository-hygiene validation gatekeeper
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE if (HERE / "00_config.py").exists() else (HERE / "code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

cfg = importlib.import_module("00_config")
gt = importlib.import_module("01_ground_truth")
col = importlib.import_module("02_collect")
prs = importlib.import_module("03_parse_geocode")
met = importlib.import_module("04_metrics")
mod = importlib.import_module("05_models")
fig = importlib.import_module("06_figures")
tab = importlib.import_module("07_tables")
val = importlib.import_module("08_validate")


def main():
    print("=== [1/8] Building & Verifying 600-POI Municipal Catalog ===")
    df_pois = gt.build_ground_truth_catalog()
    df_pois.to_csv(cfg.DATA_DIR / "ground_truth_pois_10cities.csv", index=False, encoding="utf-8-sig")

    print("=== [2/8] Verifying 10,191 Live HTTP-200 JSONL API Receipts (SHA-256 Checked) ===")
    raw_records = col.run_live_collection_or_load()

    print("=== [3/8] Parsing JSONL Receipts, Resolving Entities & Geocoding (101,910 Slots) ===")
    df_queries, df_recs, df_pois = prs.build_audit_recommendations_and_queries(df_pois, raw_records)

    print("=== [4/8] Computing Audit Metrics, Unbiased Cluster Bootstrap CIs & Dyadic Panel ===")
    df_cm, df_dyads = met.compute_all_metrics(df_pois, df_queries, df_recs)

    print("=== [5/8] Fitting MixedLM (df=9 & 2-Way Dyadic), 999-Perm PERMANOVA & Disaggregated NB2 Models ===")
    mixed_res = mod.fit_dyadic_mixed_models(df_dyads)
    df_perm = mod.run_two_way_crossed_permanova(df_queries, n_perm=999)
    nb_res = mod.fit_negative_binomial_models(df_pois)
    df_abl = met.compute_ablation_and_robustness(df_pois, df_queries, df_cm, nb_res)
    met.compute_cross_lingual_home_field_separation(df_pois, df_queries)
    met.run_post_decoding_mmr_simulation(df_pois, df_queries, df_cm)

    print("=== [6/8] Generating 600-DPI Publication Figures (Figures 1-6) ===")
    fig.generate_all_figures(df_pois, df_queries, df_cm, df_dyads, mixed_res, df_perm, nb_res, df_abl)

    print("=== [7/8] Exporting Publication Tables (Tables 1-5) ===")
    tab.export_all_tables(df_pois, df_queries, df_cm, df_dyads, mixed_res, df_perm, nb_res)

    print("=== [8/8] Running 9-Assertion Standalone Validation Gatekeeper (08_validate.py) ===")
    rep = val.run_full_validation()
    assert rep["passed"], "Standalone validation failed!"
    print("=== FULL PIPELINE REPLICATION & VALIDATION SUCCEEDED (100%) ===")


if __name__ == "__main__":
    main()
