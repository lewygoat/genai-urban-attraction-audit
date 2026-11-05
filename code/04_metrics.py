# -*- coding: utf-8 -*-
"""
04_metrics.py — Core Quantitative, Discursive, Cross-Lingual & Spatial Audit Metrics (`v2final` Live Corpus):
  - Rank-Biased Overlap (RBO at p = 0.85, plus sensitivity p = 0.80 and p = 0.90)
  - Replicate-agnostic Unbiased Off-Diagonal Two-Stage Stratified Cluster Bootstrap 95% CIs for NPI_RBO
  - Normalized Net Personalization Index (NNPI_RBO = NPI_RBO / Noise_RBO) & Exact Top-10 Slot Turnover (Delta_K10_Slots)
  - Scale-Free Discursive Tailoring Metrics:
      * Ceiling-normalized Rationale Turnover (`Discursive_NNPI` on entity-decontaminated bilingual tokens)
      * Same-POI Controlled Rationale Rewriting & Persona Cue Precision (`discursive_tailoring_audit.csv`)
  - Single-Endpoint Concentration & Coverage across 40 (city, model) cells:
      * Hill Numbers (N0, N1, N2), HHI, Gini_OSM (60-POI catalog), Gini_EqualCap_M15, Gini_AllOSM
      * Zero-Visibility Long-Tail Erasure (%) & Unweighted Physical `SDE_OSM` vs. Mention-Weighted `SDE_AI`
  - Complete 2x2 Language x Destination-Origin Orthogonal Audit (`B1 + B3 + B3b` -> `language_x_origin_2x2_audit.csv`)
  - Empirical Greedy Decoding Noise Floor (`B5`, tau=0.0, 3 replicates/cell, N=1,440 -> `b1_vs_b5_matched12ctx_comparison.csv`)
  - Blank Baseline Anchoring (`B2`), Prompt Paraphrase (`B4`), Temporal Retest (`B6`), 10-City LOCO, & Deterministic MMR Reranking
"""
import importlib
import itertools
import math
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, ttest_ind, ttest_rel, wilcoxon

cfg = importlib.import_module("00_config")


def rbo_score(list1: list, list2: list, p: float = 0.85) -> float:
    """Exact finite-depth Rank-Biased Overlap (Webber et al., 2010) at depth k = min(len1, len2)."""
    if not list1 or not list2:
        return 0.0
    k = min(len(list1), len(list2))
    s1, s2 = set(), set()
    weight_sum = 0.0
    norm_sum = 0.0
    for d in range(1, k + 1):
        s1.add(list1[d - 1])
        s2.add(list2[d - 1])
        agreement = len(s1 & s2) / float(d)
        w = p ** (d - 1)
        weight_sum += w * agreement
        norm_sum += w
    return float(weight_sum / norm_sum) if norm_sum > 0 else 0.0


def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    u = len(set1 | set2)
    return float(len(set1 & set2) / u) if u > 0 else 0.0


def gini_coefficient(counts: np.ndarray) -> float:
    arr = np.sort(np.maximum(0.0, np.asarray(counts, dtype=float)))
    n = len(arr)
    if n == 0 or np.sum(arr) <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1.0) / n)


def hill_numbers(counts: np.ndarray) -> tuple[float, float, float, float]:
    arr = np.asarray(counts, dtype=float)
    pos = arr[arr > 0]
    if len(pos) == 0:
        return 0.0, 0.0, 0.0, 1.0
    p = pos / np.sum(pos)
    n0 = float(len(pos))
    shannon_h = -float(np.sum(p * np.log(p)))
    n1 = float(np.exp(shannon_h))
    hhi = float(np.sum(p ** 2))
    n2 = float(1.0 / hhi) if hhi > 0 else 0.0
    return n0, n1, n2, hhi


def jensen_shannon_divergence(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    p = np.maximum(0.0, np.asarray(p_counts, dtype=float)) + 1e-9
    q = np.maximum(0.0, np.asarray(q_counts, dtype=float)) + 1e-9
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


def compute_weighted_sde_radius(lats: np.ndarray, lons: np.ndarray, weights: np.ndarray) -> float:
    """Computes the equivalent circular radius (km) of the Weighted Standard Deviational Ellipse (SDE)."""
    w = np.maximum(0.0, np.asarray(weights, dtype=float))
    if np.sum(w) <= 0:
        w = np.ones_like(lats, dtype=float)
    w = w / np.sum(w)
    lat_c = float(np.sum(w * lats))
    lon_c = float(np.sum(w * lons))
    dy = (lats - lat_c) * 111.32
    dx = (lons - lon_c) * 111.32 * math.cos(math.radians(lat_c))
    var_x = float(np.sum(w * (dx ** 2)))
    var_y = float(np.sum(w * (dy ** 2)))
    return float(math.sqrt(var_x + var_y))


def compute_getis_ord_gi_star(lats: np.ndarray, lons: np.ndarray, values: np.ndarray, d_thresh_km: float = 5.0) -> np.ndarray:
    """Exact Getis-Ord Gi*(d) local spatial autocorrelation z-score with binary distance threshold d_thresh_km."""
    n = len(values)
    x = np.asarray(values, dtype=float)
    x_bar = float(np.mean(x))
    s = float(np.sqrt(np.mean((x - x_bar) ** 2)))
    if s < 1e-9 or n <= 2:
        return np.zeros(n, dtype=float)
    z_scores = np.zeros(n, dtype=float)
    for i in range(n):
        w_i = np.zeros(n, dtype=float)
        for j in range(n):
            d_ij = cfg.haversine_km(float(lats[i]), float(lons[i]), float(lats[j]), float(lons[j]))
            if d_ij <= d_thresh_km:
                w_i[j] = 1.0
        sum_w = float(np.sum(w_i))
        num = float(np.sum(w_i * x)) - x_bar * sum_w
        den = s * math.sqrt(max(1e-9, (n * sum_w - sum_w ** 2) / (n - 1.0)))
        z_scores[i] = num / den if den > 1e-9 else 0.0
    return z_scores


def benjamini_hochberg_fdr(p_vals: np.ndarray) -> np.ndarray:
    """Computes Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(p_vals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]
    q = np.zeros(m, dtype=float)
    prev_q = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev_q, ranked_p[i] * m / rank)
        q[i] = val
        prev_q = val
    out = np.zeros(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def compute_discursive_tailoring_audit(df_queries: pd.DataFrame, df_recs: pd.DataFrame) -> pd.DataFrame:
    """
    Computes scale-free discursive vs. entity tailoring diagnostics on Batch B1:
      1. Persona-targeted lexical cue hit rates (Wheelchair accessibility cues, Family/Senior cues, Budget cues).
      2. Same-POI Controlled Rationale Rewriting: holding `(city_id, model, resolved_poi_id)` fixed,
         compares within-persona vs. between-persona rationale bigram Jaccard similarity so entity selection
         cannot confound discursive tailoring.
    """
    b1_q = df_queries[df_queries["batch_id"] == "B1"].copy()
    b1_r = df_recs[(df_recs["batch_id"] == "B1") & (df_recs["resolved"] == 1)].copy()

    def _char_bigrams(s: str) -> set[str]:
        t = re.sub(r"^recommended\s+[^:]{0,120}:\s*", "", str(s).lower().strip(), flags=re.I)
        t = re.sub(r"^推荐[^：:]{0,80}[：:]\s*", "", t)
        t = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 3:
            return set()
        return {t[i:i + 3] for i in range(len(t) - 2)}

    # Group slot rationales by exact (city_id, model, resolved_poi_id)
    poi_groups = defaultdict(list)
    for _, rr in b1_r.iterrows():
        bg = _char_bigrams(rr.get("reason", ""))
        if len(bg) >= 3:
            poi_groups[(rr["city_id"], rr["model"], rr["resolved_poi_id"])].append((rr["context_id"], bg))

    rng = cfg.get_rng(cfg.GLOBAL_SEED + 919)
    cell_same_poi_jw = defaultdict(list)
    cell_same_poi_jb = defaultdict(list)

    for (cid, mod_n, _pid), entries in poi_groups.items():
        if len(entries) < 12:
            continue
        by_ctx = defaultdict(list)
        for cx, bg in entries:
            by_ctx[cx].append(bg)
        w_sims = []
        for bgs in by_ctx.values():
            if len(bgs) >= 2:
                for i in range(len(bgs)):
                    for j in range(i + 1, len(bgs)):
                        w_sims.append(jaccard_similarity(bgs[i], bgs[j]))
        b_sims = []
        for _ in range(min(120, len(entries) * 2)):
            i1, i2 = int(rng.integers(0, len(entries))), int(rng.integers(0, len(entries)))
            if entries[i1][0] != entries[i2][0]:
                b_sims.append(jaccard_similarity(entries[i1][1], entries[i2][1]))
        if w_sims and b_sims:
            cell_same_poi_jw[(cid, mod_n)].append(float(np.mean(w_sims)))
            cell_same_poi_jb[(cid, mod_n)].append(float(np.mean(b_sims)))

    rows = []
    for (cid, mod_n), gq in b1_q.groupby(["city_id", "model"]):
        w_mob = gq[gq["mobility"] == "Wheelchair"]["has_acc_cue"].mean() * 100.0
        u_mob = gq[gq["mobility"] == "Unrestricted"]["has_acc_cue"].mean() * 100.0
        f_cmp = gq[gq["companion"].isin(["Family_Child", "Senior_70plus"])]["has_fam_cue"].mean() * 100.0
        nf_cmp = gq[~gq["companion"].isin(["Family_Child", "Senior_70plus"])]["has_fam_cue"].mean() * 100.0
        jw_vals = cell_same_poi_jw.get((cid, mod_n), [])
        jb_vals = cell_same_poi_jb.get((cid, mod_n), [])
        jw_m = float(np.mean(jw_vals)) if jw_vals else float("nan")
        jb_m = float(np.mean(jb_vals)) if jb_vals else float("nan")
        nnpi_same_poi = (jw_m - jb_m) / max(1e-6, jw_m) if (not np.isnan(jw_m) and not np.isnan(jb_m)) else float("nan")
        rows.append({
            "city_id": cid,
            "model": mod_n,
            "Wheelchair_Cue_Hit_pct": round(float(w_mob), 2),
            "Unrestricted_Cue_Hit_pct": round(float(u_mob), 2),
            "Delta_Acc_Cue_pp": round(float(w_mob - u_mob), 2),
            "FamilySenior_Cue_Hit_pct": round(float(f_cmp), 2),
            "SoloCouple_Cue_Hit_pct": round(float(nf_cmp), 2),
            "Delta_Fam_Cue_pp": round(float(f_cmp - nf_cmp), 2),
            "SamePOI_Within_Jaccard": round(jw_m, 4),
            "SamePOI_Between_Jaccard": round(jb_m, 4),
            "SamePOI_Rationale_NNPI": round(nnpi_same_poi, 4),
        })
    df_disc = pd.DataFrame(rows)
    n_missing_samepoi = int(df_disc["SamePOI_Rationale_NNPI"].isna().sum())
    assert n_missing_samepoi == 0, f"Same-POI rationale audit has {n_missing_samepoi} missing cells!"
    df_disc["Missing_SamePOI_Flag"] = df_disc["SamePOI_Rationale_NNPI"].isna().astype(int)
    df_disc.to_csv(cfg.DATA_DIR / "discursive_tailoring_audit.csv", index=False, encoding="utf-8-sig")
    return df_disc




def compute_collective_monoculture_and_rank_audits(
    df_pois: pd.DataFrame, df_queries: pd.DataFrame, df_recs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes three reviewer-requested empirical diagnostics on existing live v2final data:
      1) R-10: Cross-Endpoint Collective Zero-Visibility Overlap (`collective_monoculture_omission_audit.csv`)
         testing Kleinberg & Raghavan (2021) / Bommasani et al. (2022) correlated omission across all 4 endpoints
         against both exact marginal independence prod_m p_{0,c,m} and B = 2,000 marginal-preserving permutations.
      2) Opportunity Analysis: Rank-stratified within- vs. between-persona overlap at Top-1, Top-3, Top-5, Top-10
         (`rank_stratified_overlap_b1.csv`), reporting both Jaccard@k and SlotAgreement@k.
      3) R-07: Out-of-catalog verified municipal attraction (`Class B1`) recall diversity across strata
         (`out_of_catalog_municipal_recall_audit.csv`), testing extensive-margin narrowing outside the 60-POI catalog.
    """
    b1_q = df_queries[df_queries["batch_id"] == "B1"].copy()
    rng_mono = cfg.get_rng(cfg.GLOBAL_SEED + 1010)
    n_perm_mono = 2000

    mono_rows = []
    for c_meta in cfg.CITIES_META:
        cid = c_meta["city_id"]
        c_pois = df_pois[df_pois["city_id"] == cid].sort_values("poi_id").reset_index(drop=True)
        pids = list(c_pois["poi_id"].astype(str))
        pid_to_i = {p: idx for idx, p in enumerate(pids)}
        mat = np.zeros((len(cfg.MODELS), len(pids)), dtype=int)
        for m_idx, m_obj in enumerate(cfg.MODELS):
            sub_m = b1_q[(b1_q["city_id"] == cid) & (b1_q["model"] == m_obj["model"])]
            for s in sub_m["top10_poi_ids"]:
                for p in str(s).split("|"):
                    if p in pid_to_i:
                        mat[m_idx, pid_to_i[p]] += 1
        zero_mat = (mat == 0)
        p0_vec = zero_mat.mean(axis=1)
        obs_all4_cnt = int(np.all(zero_mat, axis=0).sum())
        obs_all4_pct = float(obs_all4_cnt / len(pids) * 100.0)
        exp_indep_pct = float(np.prod(p0_vec) * 100.0)

        perm_vals = np.zeros(n_perm_mono, dtype=float)
        for b_it in range(n_perm_mono):
            p_mat = np.array([rng_mono.permutation(zero_mat[mi]) for mi in range(len(cfg.MODELS))])
            perm_vals[b_it] = float(np.all(p_mat, axis=0).mean() * 100.0)
        p_perm = float((np.sum(perm_vals >= obs_all4_pct - 1e-9) + 1.0) / (n_perm_mono + 1.0))

        mono_rows.append({
            "city_id": cid,
            "city": c_meta["city"],
            "origin": c_meta["origin"],
            "richness": c_meta["richness"],
            "Mean_Single_ZeroVis_pct": round(float(p0_vec.mean() * 100.0), 2),
            "All4_Omitted_Count": obs_all4_cnt,
            "All4_Omitted_pct": round(obs_all4_pct, 2),
            "Expected_Indep_pct": round(exp_indep_pct, 2),
            "Perm_Null_Mean_pct": round(float(perm_vals.mean()), 2),
            "Excess_Omission_pp": round(obs_all4_pct - exp_indep_pct, 2),
            "Excess_Ratio_Obs_over_Exp": round(obs_all4_pct / max(1e-6, exp_indep_pct), 2),
            "Perm_p_val": round(p_perm, 4),
        })
    df_mono = pd.DataFrame(mono_rows)
    df_mono.to_csv(cfg.DATA_DIR / "collective_monoculture_omission_audit.csv", index=False, encoding="utf-8-sig")
    df_mono.to_csv(cfg.TAB_DIR / "collective_monoculture_omission_audit.csv", index=False, encoding="utf-8-sig")

    # 2) Rank-stratified overlap at Top-1, Top-3, Top-5, Top-10 across all 40 (city, model) cells
    rk_rows = []
    for k in [1, 3, 5, 10]:
        w_jac_cells, b_jac_cells, w_agr_cells, b_agr_cells = [], [], [], []
        for (_cid, _mn), grp in b1_q.groupby(["city_id", "model"]):
            by_ctx = defaultdict(list)
            for _, rq in grp.sort_values("rep_id").iterrows():
                by_ctx[rq["context_id"]].append(str(rq["top10_poi_ids"]).split("|")[:k])
            cks = sorted(by_ctx.keys())
            wj, bj, wa, ba = [], [], [], []
            for cx in cks:
                ls = by_ctx[cx]
                for r1 in range(len(ls)):
                    for r2 in range(r1 + 1, len(ls)):
                        s1, s2 = set(ls[r1]), set(ls[r2])
                        wj.append(jaccard_similarity(s1, s2))
                        wa.append(len(s1 & s2) / float(k))
            for i in range(len(cks)):
                for j in range(i + 1, len(cks)):
                    n_al = min(len(by_ctx[cks[i]]), len(by_ctx[cks[j]]))
                    bj.append(float(np.mean([jaccard_similarity(set(by_ctx[cks[i]][r]), set(by_ctx[cks[j]][r])) for r in range(n_al)])))
                    ba.append(float(np.mean([len(set(by_ctx[cks[i]][r]) & set(by_ctx[cks[j]][r])) / float(k) for r in range(n_al)])))
            w_jac_cells.append(float(np.mean(wj)))
            b_jac_cells.append(float(np.mean(bj)))
            w_agr_cells.append(float(np.mean(wa)))
            b_agr_cells.append(float(np.mean(ba)))
        mw_j, mb_j = float(np.mean(w_jac_cells)), float(np.mean(b_jac_cells))
        mw_a, mb_a = float(np.mean(w_agr_cells)), float(np.mean(b_agr_cells))
        rk_rows.append({
            "Depth_k": f"Top-{k}",
            "k": k,
            "Within_Jaccard_at_k": round(mw_j, 4),
            "Between_Jaccard_at_k": round(mb_j, 4),
            "Jaccard_Ratio_Bet_over_Within": round(mb_j / max(1e-6, mw_j), 4),
            "Within_SlotAgreement_at_k": round(mw_a, 4),
            "Between_SlotAgreement_at_k": round(mb_a, 4),
            "Net_Slot_Turnover_at_k": round((mw_a - mb_a) * k, 3),
        })
    df_rk = pd.DataFrame(rk_rows)
    df_rk.to_csv(cfg.DATA_DIR / "rank_stratified_overlap_b1.csv", index=False, encoding="utf-8-sig")

    # 3) R-07: Out-of-catalog verified municipal POI (`Class B1`) recall & catalog-free unique venue diversity
    b1_r = df_recs[df_recs["batch_id"] == "B1"].copy()
    ooc_rows = []
    for c_meta in cfg.CITIES_META:
        cid = c_meta["city_id"]
        sub_r = b1_r[b1_r["city_id"] == cid]
        for m_obj in cfg.MODELS:
            mn = m_obj["model"]
            sm = sub_r[sub_r["model"] == mn]
            in_cat_u = int(sm[sm["slot_class"] == "A_in_catalog"]["resolved_poi_id"].nunique())
            ooc_muni = sm[(sm["slot_class"] != "A_in_catalog") & (sm["is_hallucinated"] == 0) & (sm["is_misattributed"] == 0)]
            ooc_u = int(ooc_muni["resolved_poi_id"].nunique())
            ooc_rows.append({
                "city_id": cid,
                "city": c_meta["city"],
                "richness": c_meta["richness"],
                "model": mn,
                "InCatalog_Unique_N0": in_cat_u,
                "OutOfCatalog_Municipal_Unique": ooc_u,
                "Total_Municipal_Unique_All": in_cat_u + ooc_u,
                "OutOfCatalog_Slot_Share_pct": round(float(len(ooc_muni) / max(1, len(sm)) * 100.0), 2),
            })
    df_ooc = pd.DataFrame(ooc_rows)
    df_ooc.to_csv(cfg.DATA_DIR / "out_of_catalog_municipal_recall_audit.csv", index=False, encoding="utf-8-sig")
    return df_mono, df_rk, df_ooc


def compute_all_metrics(df_pois: pd.DataFrame, df_queries: pd.DataFrame, df_recs: pd.DataFrame):
    """
    Computes:
      1) 40-row `city_model_metrics.csv` (single-endpoint coverage, concentration, bootstrap NPI CIs,
         NNPI, slot turnover, scale-free Discursive_NNPI, and Blank-Baseline B2 overlap).
      2) `discursive_tailoring_audit.csv` (Same-POI controlled rationale rewriting & persona cue precision).
      3) 9,600-row `dyadic_evaluation_pairs.csv` (balanced 2x2 orthogonal dyadic panel).
      4) `stratum_hypothesis_tests.csv` (both 40-cell and 10-city cluster/exact-permutation tests).
    """
    df_disc_audit = compute_discursive_tailoring_audit(df_queries, df_recs)
    compute_collective_monoculture_and_rank_audits(df_pois, df_queries, df_recs)
    disc_lookup = {
        (r["city_id"], r["model"]): r for _, r in df_disc_audit.iterrows()
    }

    df_main = df_queries[df_queries["batch_id"] == "B1"].copy()
    df_b2 = df_queries[df_queries["batch_id"] == "B2"].copy()
    city_meta_map = {c["city_id"]: c for c in cfg.CITIES_META}
    model_meta_map = {m["model"]: m for m in cfg.MODELS}

    # Empirical Public-Attention Top-10 per city (sorted by live 2025 Wikimedia annual pageviews descending)
    city_human_top10 = {}
    city_pois_sorted = {}
    for cid, grp in df_pois.groupby("city_id"):
        s_grp = grp.sort_values(["wiki_views_annual", "wikidata_sitelinks"], ascending=False).reset_index(drop=True)
        city_human_top10[cid] = list(s_grp["poi_id"].iloc[:10].astype(str))
        city_pois_sorted[cid] = grp.sort_values("poi_id").reset_index(drop=True)

    cm_rows = []
    rng_boot = cfg.get_rng(cfg.GLOBAL_SEED + 404)

    for c_meta in cfg.CITIES_META:
        cid = c_meta["city_id"]
        c_pois = city_pois_sorted[cid]
        catalog_pids = list(c_pois["poi_id"].astype(str))
        pid_to_idx = {pid: idx for idx, pid in enumerate(catalog_pids)}
        pub_vec = np.maximum(1.0, c_pois["wiki_views_annual"].values.astype(float))
        human_top10 = city_human_top10[cid]
        full_osm_denom = int(c_meta["osm_pois"])

        for m_obj in cfg.MODELS:
            m_name = m_obj["model"]
            sub = df_main[(df_main["city_id"] == cid) & (df_main["model"] == m_name)].sort_values(
                ["context_id", "rep_id"]
            ).reset_index(drop=True)
            sub_b2 = df_b2[(df_b2["city_id"] == cid) & (df_b2["model"] == m_name)]
            b2_lists = [str(s).split("|") for s in sub_b2["top10_poi_ids"]]

            ctx_ids = sorted(sub["context_id"].unique())
            n_ctx = len(ctx_ids)
            ctx_lists = []
            ctx_sets = []
            ctx_reasons = []

            poi_counts = np.zeros(len(catalog_pids), dtype=float)
            human_rbo_vals = []
            b2_rbo_vals = []
            b2_jac_vals = []
            b2_sh_vals = []

            for cx_id in ctx_ids:
                c_sub = sub[sub["context_id"] == cx_id].sort_values("rep_id")
                l_reps, s_reps, r_reps = [], [], []
                for _, row_q in c_sub.iterrows():
                    pids = str(row_q["top10_poi_ids"]).split("|")
                    r_toks = set(str(row_q.get("reason_tokens", "")).split("|")) - {""}
                    l_reps.append(pids)
                    s_reps.append(set(pids))
                    r_reps.append(r_toks)
                    human_rbo_vals.append(rbo_score(pids, human_top10, p=0.85))
                    for b2_l in b2_lists:
                        b2_rbo_vals.append(rbo_score(pids, b2_l, p=0.85))
                        b2_jac_vals.append(jaccard_similarity(set(pids), set(b2_l)))
                        b2_sh_vals.append(float(len(set(pids) & set(b2_l))))
                    for p in pids:
                        if p in pid_to_idx:
                            poi_counts[pid_to_idx[p]] += 1.0
                ctx_lists.append(l_reps)
                ctx_sets.append(s_reps)
                ctx_reasons.append(r_reps)

            # 1) Within-context replicate similarity (replicate-count agnostic: supports n_r = 2 or 3 cleanly!)
            ctx_w_rbo85 = []
            ctx_w_rbo80 = []
            ctx_w_rbo90 = []
            ctx_w_jac = []
            ctx_w_sh = []
            ctx_w_rsn_sim = []
            ctx_pair_rbo85_lists = []

            for c_i in range(n_ctx):
                n_r = len(ctx_lists[c_i])
                p85_i, p80_i, p90_i, pj_i, psh_i, pr_i = [], [], [], [], [], []
                for r1 in range(n_r):
                    for r2 in range(r1 + 1, n_r):
                        p85_i.append(rbo_score(ctx_lists[c_i][r1], ctx_lists[c_i][r2], p=0.85))
                        p80_i.append(rbo_score(ctx_lists[c_i][r1], ctx_lists[c_i][r2], p=0.80))
                        p90_i.append(rbo_score(ctx_lists[c_i][r1], ctx_lists[c_i][r2], p=0.90))
                        pj_i.append(jaccard_similarity(ctx_sets[c_i][r1], ctx_sets[c_i][r2]))
                        psh_i.append(float(len(ctx_sets[c_i][r1] & ctx_sets[c_i][r2])))
                        pr_i.append(jaccard_similarity(ctx_reasons[c_i][r1], ctx_reasons[c_i][r2]))
                ctx_pair_rbo85_lists.append(p85_i)
                ctx_w_rbo85.append(float(np.mean(p85_i)) if p85_i else np.nan)
                ctx_w_rbo80.append(float(np.mean(p80_i)) if p80_i else np.nan)
                ctx_w_rbo90.append(float(np.mean(p90_i)) if p90_i else np.nan)
                ctx_w_jac.append(float(np.mean(pj_i)) if pj_i else np.nan)
                ctx_w_sh.append(float(np.mean(psh_i)) if psh_i else np.nan)
                ctx_w_rsn_sim.append(float(np.mean(pr_i)) if pr_i else np.nan)

            noise_rbo = float(np.nanmean(ctx_w_rbo85))
            noise_rbo_sd = float(np.nanstd(ctx_w_rbo85, ddof=1))
            noise_rbo_p80 = float(np.nanmean(ctx_w_rbo80))
            noise_rbo_p90 = float(np.nanmean(ctx_w_rbo90))
            noise_jac = float(np.nanmean(ctx_w_jac))
            k10_within = float(np.nanmean(ctx_w_sh))
            reason_within_sim = float(np.nanmean(ctx_w_rsn_sim))
            reason_noise_turn = 1.0 - reason_within_sim

            # 2) Between-context similarity across all C(n_ctx, 2) distinct context pairs
            btw_rbo_mat = np.zeros((n_ctx, n_ctx), dtype=float)
            btw_rbo_p80_mat = np.zeros((n_ctx, n_ctx), dtype=float)
            btw_rbo_p90_mat = np.zeros((n_ctx, n_ctx), dtype=float)
            btw_jac_mat = np.zeros((n_ctx, n_ctx), dtype=float)
            btw_sh_mat = np.zeros((n_ctx, n_ctx), dtype=float)
            btw_rsn_sim_mat = np.zeros((n_ctx, n_ctx), dtype=float)

            for c1 in range(n_ctx):
                for c2 in range(c1 + 1, n_ctx):
                    n_align = min(len(ctx_lists[c1]), len(ctx_lists[c2]))
                    v_85 = float(np.mean([rbo_score(ctx_lists[c1][r], ctx_lists[c2][r], p=0.85) for r in range(n_align)]))
                    v_80 = float(np.mean([rbo_score(ctx_lists[c1][r], ctx_lists[c2][r], p=0.80) for r in range(n_align)]))
                    v_90 = float(np.mean([rbo_score(ctx_lists[c1][r], ctx_lists[c2][r], p=0.90) for r in range(n_align)]))
                    v_jc = float(np.mean([jaccard_similarity(ctx_sets[c1][r], ctx_sets[c2][r]) for r in range(n_align)]))
                    v_sh = float(np.mean([float(len(ctx_sets[c1][r] & ctx_sets[c2][r])) for r in range(n_align)]))
                    v_rs = float(np.mean([jaccard_similarity(ctx_reasons[c1][r], ctx_reasons[c2][r]) for r in range(n_align)]))
                    btw_rbo_mat[c1, c2] = btw_rbo_mat[c2, c1] = v_85
                    btw_rbo_p80_mat[c1, c2] = btw_rbo_p80_mat[c2, c1] = v_80
                    btw_rbo_p90_mat[c1, c2] = btw_rbo_p90_mat[c2, c1] = v_90
                    btw_jac_mat[c1, c2] = btw_jac_mat[c2, c1] = v_jc
                    btw_sh_mat[c1, c2] = btw_sh_mat[c2, c1] = v_sh
                    btw_rsn_sim_mat[c1, c2] = btw_rsn_sim_mat[c2, c1] = v_rs

            triu_i, triu_j = np.triu_indices(n_ctx, k=1)
            between_rbo = float(np.mean(btw_rbo_mat[triu_i, triu_j]))
            between_rbo_p80 = float(np.mean(btw_rbo_p80_mat[triu_i, triu_j]))
            between_rbo_p90 = float(np.mean(btw_rbo_p90_mat[triu_i, triu_j]))
            between_jac = float(np.mean(btw_jac_mat[triu_i, triu_j]))
            k10_between = float(np.mean(btw_sh_mat[triu_i, triu_j]))
            reason_between_sim = float(np.mean(btw_rsn_sim_mat[triu_i, triu_j]))
            reason_btw_turn = 1.0 - reason_between_sim

            npi_rbo = noise_rbo - between_rbo
            npi_rbo_p80 = noise_rbo_p80 - between_rbo_p80
            npi_rbo_p90 = noise_rbo_p90 - between_rbo_p90
            npi_jac = noise_jac - between_jac

            # Ceiling-normalized entity & discursive personalization indices + exact pair-level slot turnover
            nnpi_rbo = npi_rbo / max(1e-6, noise_rbo)
            delta_k10_slots = k10_within - k10_between

            # Discursive_NPI: ceiling-normalized discursive divergence `(J_within - J_between) / J_within`
            # (also storing raw Jaccard difference `Discursive_Raw_Diff` for complete transparency)
            discursive_raw_diff = reason_within_sim - reason_between_sim
            discursive_nnpi = discursive_raw_diff / max(1e-6, reason_within_sim)
            same_poi_nnpi = float(disc_lookup[(cid, m_name)]["SamePOI_Rationale_NNPI"])

            # 3) Unbiased Off-Diagonal Two-Stage Stratified Cluster Bootstrap (B = 1,000 draws)
            n_boot = 1000
            boot_npis = np.zeros(n_boot, dtype=float)
            for b_iter in range(n_boot):
                b_idx = rng_boot.integers(0, n_ctx, size=n_ctx)
                w_draws = []
                for c_sel in b_idx:
                    pairs_avail = ctx_pair_rbo85_lists[int(c_sel)]
                    if pairs_avail:
                        sel_p = rng_boot.choice(pairs_avail, size=len(pairs_avail), replace=True)
                        w_draws.append(float(np.mean(sel_p)))
                b_noise = float(np.mean(w_draws))
                c_u = b_idx[triu_i]
                c_v = b_idx[triu_j]
                off_diag_mask = (c_u != c_v)
                b_bet = float(np.mean(btw_rbo_mat[c_u[off_diag_mask], c_v[off_diag_mask]]))
                boot_npis[b_iter] = b_noise - b_bet

            npi_ci_low = float(np.percentile(boot_npis, 2.5))
            npi_ci_high = float(np.percentile(boot_npis, 97.5))
            npi_se = float(np.std(boot_npis, ddof=1))
            z_npi = npi_rbo / max(1e-6, npi_se)
            p_npi_zero = float(2.0 * norm.sf(abs(z_npi)))

            # 4) Concentration & Diversity Metrics (Single-Endpoint 60-POI Catalog & Full-OSM Denominator)
            n0, n1, n2, hhi = hill_numbers(poi_counts)
            gini_osm = gini_coefficient(poi_counts)
            pos_counts = poi_counts[poi_counts > 0]
            gini_sample = gini_coefficient(pos_counts) if len(pos_counts) > 0 else 0.0

            rng_eq = cfg.get_rng(cfg.stable_seed("EQ_CAP_M15", cid, m_name))
            top30_idx = np.argsort(-pub_vec)[:30]
            eq_ginis = [
                gini_coefficient(poi_counts[rng_eq.choice(top30_idx, size=15, replace=False)])
                for _ in range(200)
            ]
            gini_eq_m15 = float(np.mean(eq_ginis))

            full_osm_vec = np.zeros(full_osm_denom, dtype=float)
            full_osm_vec[:len(poi_counts)] = poi_counts
            gini_all_osm = gini_coefficient(full_osm_vec)

            sorted_cnts = np.sort(poi_counts)[::-1]
            top10_share_pct = float(np.sum(sorted_cnts[:10]) / max(1.0, np.sum(poi_counts)) * 100.0)
            zero_vis_pct = float(np.sum(poi_counts == 0) / len(poi_counts) * 100.0)
            jsd_rev = jensen_shannon_divergence(poi_counts, pub_vec)

            total_slots = max(1, len(sub) * 10)
            in_cat_pct = float(sub["in_catalog_pois"].sum() / total_slots * 100.0)
            out_cat_pct = float(sub["out_of_catalog_pois"].sum() / total_slots * 100.0)
            halluc_pct = float(sub["hallucinated_pois"].sum() / total_slots * 100.0)
            misattr_pct = float(sub["misattributed_pois"].sum() / total_slots * 100.0)

            b2_shared_k10 = float(np.mean(b2_sh_vals)) if b2_sh_vals else k10_between

            cm_rows.append({
                "city_id": cid,
                "city": c_meta["city"],
                "origin": c_meta["origin"],
                "richness": c_meta["richness"],
                "model": m_name,
                "model_origin": m_obj["model_origin"],
                "queries_n": len(sub),
                "N0_unique": round(n0, 2),
                "N1_effective": round(n1, 2),
                "N2_simpson": round(n2, 2),
                "HHI": round(hhi, 4),
                "Gini_OSM": round(gini_osm, 4),
                "Gini_Sample": round(gini_sample, 4),
                "Gini_EqualCap_M15": round(gini_eq_m15, 4),
                "Gini_AllOSM": round(gini_all_osm, 4),
                "Top10_share_pct": round(top10_share_pct, 2),
                "Zero_visibility_pct": round(zero_vis_pct, 2),
                "Noise_RBO": round(noise_rbo, 4),
                "Noise_RBO_SD": round(noise_rbo_sd, 4),
                "Between_Ctx_RBO": round(between_rbo, 4),
                "NPI_RBO": round(npi_rbo, 4),
                "NPI_CI_Low": round(npi_ci_low, 4),
                "NPI_CI_High": round(npi_ci_high, 4),
                "NPI_SE": round(npi_se, 4),
                "NPI_pval": p_npi_zero,
                "NNPI_RBO": round(nnpi_rbo, 4),
                "Shared_Within_K10": round(k10_within, 3),
                "Shared_Between_K10": round(k10_between, 3),
                "Delta_K10_Slots": round(delta_k10_slots, 3),
                "Blank_B2_RBO": round(float(np.mean(b2_rbo_vals)) if b2_rbo_vals else 0.0, 4),
                "Blank_B2_Shared_K10": round(b2_shared_k10, 3),
                "NPI_RBO_p08": round(npi_rbo_p80, 4),
                "NPI_RBO_p09": round(npi_rbo_p90, 4),
                "Noise_Jaccard": round(noise_jac, 4),
                "Between_Ctx_Jaccard": round(between_jac, 4),
                "NPI_Jaccard": round(npi_jac, 4),
                "Reason_Noise_Turnover": round(reason_noise_turn, 4),
                "Reason_Between_Turnover": round(reason_btw_turn, 4),
                "Discursive_Raw_Diff": round(discursive_raw_diff, 4),
                "Discursive_NNPI": round(discursive_nnpi, 4),
                "Discursive_NPI": round(discursive_nnpi, 4),
                "SamePOI_Rationale_NNPI": round(same_poi_nnpi, 4),
                "Human_Top10_RBO": round(float(np.mean(human_rbo_vals)), 4),
                "JSD_vs_Reviews": round(jsd_rev, 4),
                "In_Catalog_pct": round(in_cat_pct, 2),
                "Out_Of_Catalog_Real_pct": round(out_cat_pct, 2),
                "Hallucination_pct": round(halluc_pct, 2),
                "Misattribution_pct": round(misattr_pct, 2),
            })

    df_cm = pd.DataFrame(cm_rows)
    df_cm["BH_FDR_qval"] = benjamini_hochberg_fdr(df_cm["NPI_pval"].values)
    df_cm.to_csv(cfg.DATA_DIR / "city_model_metrics.csv", index=False, encoding="utf-8-sig")

    # 5) Formal Two-Group Hypothesis Tests: Both Cell-Level (n=24 vs 16) AND Independent City-Level (n=6 vs 4)
    rich_cm = df_cm[df_cm["richness"] == "Info-Rich"]
    scarce_cm = df_cm[df_cm["richness"] == "Info-Scarce"]
    df_city_agg = df_cm.groupby(["city_id", "richness"], as_index=False).mean(numeric_only=True)
    rich_city = df_city_agg[df_city_agg["richness"] == "Info-Rich"]
    scarce_city = df_city_agg[df_city_agg["richness"] == "Info-Scarce"]

    htest_rows = []
    for col_m, lbl_m in [
        ("NPI_RBO", "Net Personalization Index (NPI_RBO)"),
        ("NNPI_RBO", "Normalized NPI (NNPI_RBO = NPI / Noise)"),
        ("Delta_K10_Slots", "Net Top-10 Slot Turnover (Delta_K10)"),
        ("Discursive_NNPI", "Normalized Discursive Tailoring Index (Discursive_NNPI)"),
        ("SamePOI_Rationale_NNPI", "Same-POI Controlled Rationale Tailoring (NNPI)"),
        ("Gini_OSM", "AI Mention Concentration (Gini_OSM)"),
        ("Gini_AllOSM", "Full-OSM Denominator Concentration (Gini_AllOSM)"),
        ("Zero_visibility_pct", "Zero-Visibility Long-Tail Erasure (%)"),
        ("N0_unique", "Unique Recalled Attractions (Hill N0)"),
        ("Human_Top10_RBO", "Alignment with Public Attention Top-10 (RBO)"),
    ]:
        v_r = rich_cm[col_m].values
        v_s = scarce_cm[col_m].values
        diff_mean = float(np.mean(v_r) - np.mean(v_s))
        t_stat, p_val = ttest_ind(v_r, v_s, equal_var=False)
        u_stat, p_mwu = mannwhitneyu(v_r, v_s, alternative="two-sided")
        se_diff = math.sqrt(np.var(v_r, ddof=1) / len(v_r) + np.var(v_s, ddof=1) / len(v_s))

        c_r = rich_city[col_m].values
        c_s = scarce_city[col_m].values
        c_t_stat, c_p_val = ttest_ind(c_r, c_s, equal_var=False)
        c_u_stat, c_p_mwu = mannwhitneyu(c_r, c_s, alternative="two-sided")
        all_c_vals = np.concatenate([c_r, c_s])
        obs_abs_diff = abs(float(np.mean(c_r) - np.mean(c_s)))
        extreme_cnt = 0
        for idx_comb in itertools.combinations(range(10), 6):
            mask_r = np.zeros(10, dtype=bool)
            mask_r[list(idx_comb)] = True
            p_diff = abs(float(np.mean(all_c_vals[mask_r]) - np.mean(all_c_vals[~mask_r])))
            if p_diff >= obs_abs_diff - 1e-12:
                extreme_cnt += 1
        c_p_perm = float(extreme_cnt / 210.0)

        htest_rows.append({
            "Metric": col_m,
            "Description": lbl_m,
            "Info_Rich_Mean": round(float(np.mean(v_r)), 4),
            "Info_Rich_SD": round(float(np.std(v_r, ddof=1)), 4),
            "Info_Scarce_Mean": round(float(np.mean(v_s)), 4),
            "Info_Scarce_SD": round(float(np.std(v_s, ddof=1)), 4),
            "Diff_Rich_minus_Scarce": round(diff_mean, 4),
            "Diff_95CI_Low": round(diff_mean - 1.96 * se_diff, 4),
            "Diff_95CI_High": round(diff_mean + 1.96 * se_diff, 4),
            "Welch_t": round(float(t_stat), 3),
            "Welch_p": float(p_val),
            "MannWhitney_p": float(p_mwu),
            "City_Welch_t": round(float(c_t_stat), 3),
            "City_Welch_p": float(c_p_val),
            "City_MannWhitney_U": round(float(c_u_stat), 1),
            "City_MannWhitney_p": float(c_p_mwu),
            "City_ExactPerm_p": round(c_p_perm, 5),
        })
    df_htest = pd.DataFrame(htest_rows)
    df_htest["BH_FDR_qval"] = benjamini_hochberg_fdr(df_htest["Welch_p"].values)
    df_htest["City_BH_FDR_qval"] = benjamini_hochberg_fdr(df_htest["City_Welch_p"].values)
    df_htest.to_csv(cfg.DATA_DIR / "stratum_hypothesis_tests.csv", index=False, encoding="utf-8-sig")

    # 6) Construct Balanced 2x2 Orthogonal Dyadic Evaluation Panel (N = 9,600: 960 dyads/city x 10 cities)
    rng_dyad = cfg.get_rng(cfg.GLOBAL_SEED + 808)
    dyad_rows = []
    ctx_meta_map = {cx["context_id"]: cx for cx in cfg.CONTEXTS}
    all_ctx_ids = [cx["context_id"] for cx in cfg.CONTEXTS]
    model_names = [m["model"] for m in cfg.MODELS]

    for c_meta in cfg.CITIES_META:
        cid = c_meta["city_id"]
        sub_c = df_main[df_main["city_id"] == cid]
        lookup_cell = defaultdict(list)
        for _, r_q in sub_c.sort_values("rep_id").iterrows():
            lookup_cell[(r_q["model"], r_q["context_id"])].append(str(r_q["top10_poi_ids"]).split("|"))

        same_ctx_pairs = [(all_ctx_ids[i % 48], all_ctx_ids[i % 48]) for i in range(240)]
        diff_ctx_pairs = []
        while len(diff_ctx_pairs) < 240:
            ca, cb = rng_dyad.choice(all_ctx_ids, size=2, replace=False)
            diff_ctx_pairs.append((str(ca), str(cb)))

        same_mod_pairs = [(model_names[i % 4], model_names[i % 4]) for i in range(240)]
        distinct_m_pairs = [
            (model_names[i_m], model_names[j_m])
            for i_m in range(4) for j_m in range(i_m + 1, 4)
        ]
        diff_mod_pairs = [distinct_m_pairs[i % 6] for i in range(240)]

        quadrants = [
            (1, 1, same_mod_pairs, same_ctx_pairs),
            (1, 0, same_mod_pairs, diff_ctx_pairs),
            (0, 1, diff_mod_pairs, same_ctx_pairs),
            (0, 0, diff_mod_pairs, diff_ctx_pairs),
        ]

        for sm_flag, sc_flag, m_pairs, c_pairs in quadrants:
            for idx_p in range(240):
                ma, mb = m_pairs[idx_p]
                cxa, cxb = c_pairs[idx_p]
                reps_a = lookup_cell[(ma, cxa)]
                reps_b = lookup_cell[(mb, cxb)]
                if sm_flag == 1 and sc_flag == 1:
                    la = reps_a[0]
                    lb = reps_a[1] if len(reps_a) >= 2 else reps_a[0]
                else:
                    ia = int(rng_dyad.integers(0, len(reps_a)))
                    ib = int(rng_dyad.integers(0, len(reps_b)))
                    if sm_flag == 1 and cxa == cxb and ia == ib and len(reps_a) >= 2:
                        ib = (ia + 1) % len(reps_a)
                    la = reps_a[ia]
                    lb = reps_b[ib]

                rbo_v = rbo_score(la, lb, p=0.85)
                jac_v = jaccard_similarity(set(la), set(lb))

                ca_obj = ctx_meta_map[cxa]
                cb_obj = ctx_meta_map[cxb]
                orig_a = model_meta_map[ma]["model_origin"]
                orig_b = model_meta_map[mb]["model_origin"]
                same_orig = int(orig_a == orig_b)
                same_eco_diff_mod = int(same_orig == 1 and sm_flag == 0)
                home_m = int((orig_a == c_meta["origin"]) and (orig_b == c_meta["origin"]))

                dyad_rows.append({
                    "city_id": cid,
                    "city": c_meta["city"],
                    "origin": c_meta["origin"],
                    "richness": c_meta["richness"],
                    "info_rich_num": int(c_meta["richness"] == "Info-Rich"),
                    "model_a": ma,
                    "model_b": mb,
                    "model_pair": f"{ma} vs {mb}",
                    "context_a": cxa,
                    "context_b": cxb,
                    "list_cluster_a": f"{cid}_{ma}_{cxa}",
                    "list_cluster_b": f"{cid}_{mb}_{cxb}",
                    "same_model": int(sm_flag),
                    "same_origin": int(same_orig),
                    "same_ecology_diff_model": int(same_eco_diff_mod),
                    "same_context": int(sc_flag),
                    "diff_companion": int(ca_obj["companion"] != cb_obj["companion"]),
                    "diff_budget": int(ca_obj["budget"] != cb_obj["budget"]),
                    "diff_season": int(ca_obj["season"] != cb_obj["season"]),
                    "diff_mobility": int(ca_obj["mobility"] != cb_obj["mobility"]),
                    "home_match": int(home_m),
                    "rbo": round(rbo_v, 5),
                    "jaccard": round(jac_v, 5),
                })

    df_dyads = pd.DataFrame(dyad_rows)
    df_dyads.to_csv(cfg.DATA_DIR / "dyadic_evaluation_pairs.csv", index=False, encoding="utf-8-sig")
    return df_cm, df_dyads


def compute_ablation_and_robustness(
    df_pois: pd.DataFrame,
    df_queries: pd.DataFrame,
    df_cm_stats: pd.DataFrame,
    nb_res: dict,
) -> pd.DataFrame:
    """
    Computes:
      1) 7-Specification Robustness Suite (`ablation_robustness_7specs.csv`) across B1, B3, B3b, B4, B5, B6.
      2) 10-City Leave-One-City-Out (`loco_10cities_sensitivity.csv`) sensitivity audit.
      3) Matched 12-Context comparison between Batch B1 (tau=0.7) and Batch B5 (tau=0.0, 3 empirical reps/cell,
         N=1,440 -> `b1_vs_b5_matched12ctx_comparison.csv`).
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df_enc = nb_res["df_encoded"].copy()

    def fit_nb_beta(sub_df: pd.DataFrame, col_y: str = "mentions_b1_unres", col_x: str = "log_reviews_c") -> float:
        f_str = f"{col_y} ~ {col_x} + {col_x}:info_scarce + dist_center_km + wheelchair_accessible + C(category) + C(city_id)"
        pm = smf.glm(f_str, data=sub_df, family=sm.families.Poisson()).fit()
        mu = np.maximum(1e-3, pm.mu)
        y = sub_df[col_y].values.astype(float)
        alpha_e = float(np.clip(np.sum((y - mu) ** 2 - y) / max(1e-6, np.sum(mu ** 2)), 0.15, 3.5))
        nbm = smf.glm(
            f_str, data=sub_df, family=sm.families.NegativeBinomial(alpha=alpha_e)
        ).fit(start_params=pm.params, method="lbfgs", maxiter=600)
        return float(nbm.params[col_x])

    def fit_batch_nb_beta(sub_q: pd.DataFrame, col_x: str = "log_reviews_c") -> float:
        q_use = sub_q[sub_q["mobility"] == "Unrestricted"] if "mobility" in sub_q.columns else sub_q
        if len(q_use) == 0:
            q_use = sub_q
        cities_present = sorted(q_use["city_id"].unique())
        tmp_df = df_enc[df_enc["city_id"].isin(cities_present)].copy()
        cnt_map = {str(p): 0 for p in tmp_df["poi_id"]}
        for _, rq in q_use.iterrows():
            for p in str(rq["top10_poi_ids"]).split("|"):
                if p in cnt_map:
                    cnt_map[p] += 1
        tmp_df["batch_mentions"] = [cnt_map[str(p)] for p in tmp_df["poi_id"]]
        return fit_nb_beta(tmp_df, col_y="batch_mentions", col_x=col_x)

    # 1) Leave-One-City-Out (LOCO) across all 10 cities
    loco_rows = []
    for c_drop in cfg.CITIES_META:
        drop_id = c_drop["city_id"]
        sub_cm = df_cm_stats[df_cm_stats["city_id"] != drop_id]
        sub_p = df_enc[df_enc["city_id"] != drop_id].copy()
        beta_loco = fit_nb_beta(sub_p)
        loco_rows.append({
            "dropped_city_id": drop_id,
            "dropped_city": c_drop["city"],
            "dropped_richness": c_drop["richness"],
            "Mean_NPI_RBO": round(float(sub_cm["NPI_RBO"].mean()), 4),
            "Mean_Gini_OSM": round(float(sub_cm["Gini_OSM"].mean()), 4),
            "Scarce_minus_Rich_Gini": round(
                float(sub_cm[sub_cm["richness"] == "Info-Scarce"]["Gini_OSM"].mean() - sub_cm[sub_cm["richness"] == "Info-Rich"]["Gini_OSM"].mean()), 4
            ),
            "NB_Amplification_Beta": round(beta_loco, 4),
        })
    df_loco = pd.DataFrame(loco_rows)
    df_loco.to_csv(cfg.DATA_DIR / "loco_10cities_sensitivity.csv", index=False, encoding="utf-8-sig")

    # 2) Batch-specific metrics using genuine empirical replicates (ZERO hardcoded 1.0 ceilings!)
    def eval_subset_detailed(sub_q: pd.DataFrame, p_val: float = 0.85) -> dict:
        npi_list, nnpi_list, gini_list = [], [], []
        btw_rbo_list, btw_jac_list, btw_sh_list, w_rbo_list, w_jac_list, dk10_list = [], [], [], [], [], []
        for (cid, m_name), grp in sub_q.groupby(["city_id", "model"]):
            c_pois = df_pois[df_pois["city_id"] == cid]
            pid_idx = {str(p): i for i, p in enumerate(c_pois["poi_id"])}
            cnts = np.zeros(len(c_pois), dtype=float)
            lists_by_ctx = defaultdict(list)
            for _, rq in grp.sort_values(["template_id", "rep_id"]).iterrows():
                pids = str(rq["top10_poi_ids"]).split("|")
                lists_by_ctx[rq["context_id"]].append(pids)
                for p in pids:
                    if p in pid_idx:
                        cnts[pid_idx[p]] += 1.0
            gini_list.append(gini_coefficient(cnts))
            c_keys = sorted(lists_by_ctx.keys())
            if len(c_keys) > 1:
                bet_v, bet_j, bet_sh = [], [], []
                for i in range(len(c_keys)):
                    for j in range(i + 1, len(c_keys)):
                        n_al = min(len(lists_by_ctx[c_keys[i]]), len(lists_by_ctx[c_keys[j]]))
                        bet_v.append(float(np.mean([rbo_score(lists_by_ctx[c_keys[i]][r], lists_by_ctx[c_keys[j]][r], p=p_val) for r in range(n_al)])))
                        bet_j.append(float(np.mean([jaccard_similarity(set(lists_by_ctx[c_keys[i]][r]), set(lists_by_ctx[c_keys[j]][r])) for r in range(n_al)])))
                        bet_sh.append(float(np.mean([float(len(set(lists_by_ctx[c_keys[i]][r]) & set(lists_by_ctx[c_keys[j]][r]))) for r in range(n_al)])))
                w_v, w_j, w_sh = [], [], []
                for k in c_keys:
                    l_k = lists_by_ctx[k]
                    if len(l_k) >= 2:
                        for r1 in range(len(l_k)):
                            for r2 in range(r1 + 1, len(l_k)):
                                w_v.append(rbo_score(l_k[r1], l_k[r2], p=p_val))
                                w_j.append(jaccard_similarity(set(l_k[r1]), set(l_k[r2])))
                                w_sh.append(float(len(set(l_k[r1]) & set(l_k[r2]))))
                if len(w_v) > 0:
                    noise_est = float(np.mean(w_v))
                    noise_j = float(np.mean(w_j))
                    k10_w = float(np.mean(w_sh))
                else:
                    # Single-replicate temporal retest B6 pairs each B6 list against its matched B1 cell list
                    q_b1_cell = df_queries[
                        (df_queries["batch_id"] == "B1")
                        & (df_queries["city_id"] == cid)
                        & (df_queries["model"] == m_name)
                        & (df_queries["context_id"].isin(c_keys))
                    ]
                    retest_rbo, retest_jac, retest_sh = [], [], []
                    for k in c_keys:
                        b6_l = lists_by_ctx[k][0]
                        b1_matches = q_b1_cell[q_b1_cell["context_id"] == k]
                        for _, b1_r in b1_matches.iterrows():
                            b1_l = str(b1_r["top10_poi_ids"]).split("|")
                            retest_rbo.append(rbo_score(b6_l, b1_l, p=p_val))
                            retest_jac.append(jaccard_similarity(set(b6_l), set(b1_l)))
                            retest_sh.append(float(len(set(b6_l) & set(b1_l))))
                    noise_est = float(np.mean(retest_rbo)) if retest_rbo else float(df_cm_stats[(df_cm_stats["city_id"] == cid) & (df_cm_stats["model"] == m_name)]["Noise_RBO"].iloc[0])
                    noise_j = float(np.mean(retest_jac)) if retest_jac else float(df_cm_stats[(df_cm_stats["city_id"] == cid) & (df_cm_stats["model"] == m_name)]["Noise_Jaccard"].iloc[0])
                    k10_w = float(np.mean(retest_sh)) if retest_sh else float(df_cm_stats[(df_cm_stats["city_id"] == cid) & (df_cm_stats["model"] == m_name)]["Shared_Within_K10"].iloc[0])

                b_rbo_mean = float(np.mean(bet_v))
                b_jac_mean = float(np.mean(bet_j))
                k10_b = float(np.mean(bet_sh))
                npi_cell = noise_est - b_rbo_mean
                btw_rbo_list.append(b_rbo_mean)
                btw_jac_list.append(b_jac_mean)
                btw_sh_list.append(k10_b)
                w_rbo_list.append(noise_est)
                w_jac_list.append(noise_j)
                npi_list.append(npi_cell)
                nnpi_list.append(npi_cell / max(1e-6, noise_est))
                dk10_list.append(k10_w - k10_b)

        mean_b_jac = float(np.mean(btw_jac_list))
        shared_k10 = float(np.mean(btw_sh_list))
        return {
            "NPI_RBO": float(np.mean(npi_list)),
            "NNPI_RBO": float(np.mean(nnpi_list)),
            "Gini_OSM": float(np.mean(gini_list)),
            "Within_Noise_RBO": float(np.mean(w_rbo_list)),
            "Between_Persona_RBO": float(np.mean(btw_rbo_list)),
            "Between_Persona_Jaccard": mean_b_jac,
            "Shared_POIs_k10": shared_k10,
            "Delta_K10_Slots": float(np.mean(dk10_list)),
        }

    q_b1_full = df_queries[df_queries["batch_id"] == "B1"]
    q_b1_12ctx = q_b1_full[q_b1_full["context_id"].isin(cfg.STRATIFIED_12_CTX_IDS)]
    q_b3 = df_queries[df_queries["batch_id"] == "B3"]
    q_b3b = df_queries[df_queries["batch_id"] == "B3b"]
    q_b4 = df_queries[df_queries["batch_id"] == "B4"]
    q_b5 = df_queries[df_queries["batch_id"] == "B5"]
    q_b6 = df_queries[df_queries["batch_id"] == "B6"]

    res_b1_12 = eval_subset_detailed(q_b1_12ctx, p_val=0.85)
    res_b3 = eval_subset_detailed(q_b3, p_val=0.85)
    res_b3b = eval_subset_detailed(q_b3b, p_val=0.85)
    res_b4 = eval_subset_detailed(q_b4, p_val=0.85)
    res_b5 = eval_subset_detailed(q_b5, p_val=0.85)
    res_b6 = eval_subset_detailed(q_b6, p_val=0.85)

    # Save Matched 12-Context Comparison (Batch B1 tau=0.7 vs. Batch B5 tau=0.0 with 3 empirical replicates/cell!)
    b1_mean_b_jac = float(df_cm_stats["Between_Ctx_Jaccard"].mean())
    b1_shared_k10 = float(df_cm_stats["Shared_Between_K10"].mean())
    df_b1_b5_comp = pd.DataFrame([
        {
            "Regime": f"Batch_B1_Full_48Contexts (tau=0.7, N={len(q_b1_full):,})",
            "Contexts_Evaluated": 48,
            "Replicates_Per_Cell": 3,
            "Within_Noise_RBO": round(float(df_cm_stats["Noise_RBO"].mean()), 4),
            "Between_Persona_RBO": round(float(df_cm_stats["Between_Ctx_RBO"].mean()), 4),
            "Between_Persona_Jaccard": round(b1_mean_b_jac, 4),
            "Shared_POIs_Out_Of_10": round(b1_shared_k10, 2),
            "NPI_RBO": round(float(df_cm_stats["NPI_RBO"].mean()), 4),
            "NNPI_RBO": round(float(df_cm_stats["NNPI_RBO"].mean()), 4),
            "Delta_K10_Slots": round(float(df_cm_stats["Delta_K10_Slots"].mean()), 2),
            "Mean_Gini_OSM": round(float(df_cm_stats["Gini_OSM"].mean()), 4),
        },
        {
            "Regime": f"Batch_B1_Matched_12Contexts (tau=0.7, N={len(q_b1_12ctx):,})",
            "Contexts_Evaluated": 12,
            "Replicates_Per_Cell": 3,
            "Within_Noise_RBO": round(res_b1_12["Within_Noise_RBO"], 4),
            "Between_Persona_RBO": round(res_b1_12["Between_Persona_RBO"], 4),
            "Between_Persona_Jaccard": round(res_b1_12["Between_Persona_Jaccard"], 4),
            "Shared_POIs_Out_Of_10": round(res_b1_12["Shared_POIs_k10"], 2),
            "NPI_RBO": round(res_b1_12["NPI_RBO"], 4),
            "NNPI_RBO": round(res_b1_12["NNPI_RBO"], 4),
            "Delta_K10_Slots": round(res_b1_12["Delta_K10_Slots"], 2),
            "Mean_Gini_OSM": round(res_b1_12["Gini_OSM"], 4),
        },
        {
            "Regime": f"Batch_B5_Greedy_12Contexts (tau=0.0, N={len(q_b5):,})",
            "Contexts_Evaluated": 12,
            "Replicates_Per_Cell": 3,
            "Within_Noise_RBO": round(res_b5["Within_Noise_RBO"], 4),
            "Between_Persona_RBO": round(res_b5["Between_Persona_RBO"], 4),
            "Between_Persona_Jaccard": round(res_b5["Between_Persona_Jaccard"], 4),
            "Shared_POIs_Out_Of_10": round(res_b5["Shared_POIs_k10"], 2),
            "NPI_RBO": round(res_b5["NPI_RBO"], 4),
            "NNPI_RBO": round(res_b5["NNPI_RBO"], 4),
            "Delta_K10_Slots": round(res_b5["Delta_K10_Slots"], 2),
            "Mean_Gini_OSM": round(res_b5["Gini_OSM"], 4),
        },
    ])
    df_b1_b5_comp.to_csv(cfg.DATA_DIR / "b1_vs_b5_matched12ctx_comparison.csv", index=False, encoding="utf-8-sig")
    df_b1_b5_comp.to_csv(cfg.TAB_DIR / "b1_vs_b5_matched12ctx_comparison.csv", index=False, encoding="utf-8-sig")

    # Fit batch-specific NB2 amplification slopes across independent salience benchmarks (Unrestricted persona arm)
    df_enc_all = df_enc.copy()
    beta_sitelinks = fit_nb_beta(df_enc_all, col_y="mentions_b1_unres", col_x="log_wiki_c")
    beta_docbytes = fit_nb_beta(df_enc_all, col_y="mentions_b1_unres", col_x="log_search_c")
    beta_b3 = fit_batch_nb_beta(pd.concat([q_b3, q_b3b], ignore_index=True))
    beta_b4 = fit_batch_nb_beta(q_b4)
    beta_b5 = fit_batch_nb_beta(q_b5)
    beta_b6 = fit_batch_nb_beta(q_b6)

    specs = [
        {
            "Spec_ID": "S1_Baseline_B1",
            "Specification": f"S1: Main Factorial Baseline (Batch B1, p=0.85, T=0.7, N={len(q_b1_full):,})",
            "Mean_NPI_RBO": round(float(df_cm_stats["NPI_RBO"].mean()), 4),
            "Mean_Gini_OSM": round(float(df_cm_stats["Gini_OSM"].mean()), 4),
            "NB_Amplification_Beta": round(float(nb_res["nb_unres_catfe"].params["log_reviews_c"]), 4),
        },
        {
            "Spec_ID": "S2_RBO_p080",
            "Specification": "S2: Top-Heavy Persistence (RBO p=0.80 + Equal-Cap M=15 Gini + Wikidata Sitelinks NB2)",
            "Mean_NPI_RBO": round(float(df_cm_stats["NPI_RBO_p08"].mean()), 4),
            "Mean_Gini_OSM": round(float(df_cm_stats["Gini_EqualCap_M15"].mean()), 4),
            "NB_Amplification_Beta": round(beta_sitelinks, 4),
        },
        {
            "Spec_ID": "S3_RBO_p090",
            "Specification": "S3: Deep-Rank Persistence (RBO p=0.90 + Full-OSM Gini + Wiki Doc-Bytes NB3)",
            "Mean_NPI_RBO": round(float(df_cm_stats["NPI_RBO_p09"].mean()), 4),
            "Mean_Gini_OSM": round(float(df_cm_stats["Gini_AllOSM"].mean()), 4),
            "NB_Amplification_Beta": round(beta_docbytes, 4),
        },
        {
            "Spec_ID": "S4_CrossLingual_B3_B3b",
            "Specification": f"S4: Cross-Lingual 2x2 Control (Batches B3+B3b, N={len(q_b3)+len(q_b3b):,})",
            "Mean_NPI_RBO": round(0.67 * res_b3["NPI_RBO"] + 0.33 * res_b3b["NPI_RBO"], 4),
            "Mean_Gini_OSM": round(0.67 * res_b3["Gini_OSM"] + 0.33 * res_b3b["Gini_OSM"], 4),
            "NB_Amplification_Beta": round(beta_b3, 4),
        },
        {
            "Spec_ID": "S5_Paraphrase_B4",
            "Specification": f"S5: Paraphrased Prompt Templates T2/T3 (Batch B4, N={len(q_b4):,})",
            "Mean_NPI_RBO": round(res_b4["NPI_RBO"], 4),
            "Mean_Gini_OSM": round(res_b4["Gini_OSM"], 4),
            "NB_Amplification_Beta": round(beta_b4, 4),
        },
        {
            "Spec_ID": "S6_TempZero_B5",
            "Specification": f"S6: Greedy Decoding Temperature=0.0 Control (Batch B5, 3 reps/cell, N={len(q_b5):,})",
            "Mean_NPI_RBO": round(res_b5["NPI_RBO"], 4),
            "Mean_Gini_OSM": round(res_b5["Gini_OSM"], 4),
            "NB_Amplification_Beta": round(beta_b5, 4),
        },
        {
            "Spec_ID": "S7_IntraDay_Retest_B6",
            "Specification": f"S7: Intra-Day Temporal Stability Retest (Batch B6, median lag 7.8h, max 11.4h, N={len(q_b6):,})",
            "Mean_NPI_RBO": round(res_b6["NPI_RBO"], 4),
            "Mean_Gini_OSM": round(res_b6["Gini_OSM"], 4),
            "NB_Amplification_Beta": round(beta_b6, 4),
        },
    ]
    df_abl = pd.DataFrame(specs)
    df_abl.to_csv(cfg.DATA_DIR / "ablation_robustness_7specs.csv", index=False, encoding="utf-8-sig")
    return df_abl


def compute_cross_lingual_home_field_separation(df_pois: pd.DataFrame, df_queries: pd.DataFrame):
    """
    Evaluates:
      1) Within-ecosystem cross-model overlap on the 5 Domestic cities comparing Batch B1 (Chinese) vs.
         Batch B3 (English) across matched 12 contexts (`cross_lingual_separation_b1_vs_b3.csv`,
         `cross_lingual_paired_tests.csv`).
      2) Complete Closed 2x2 Language (zh vs. en) x Destination Origin (Domestic vs. International)
         Orthogonal Design combining B1 (matched 12 ctx), B3, and B3b (`language_x_origin_2x2_audit.csv`),
         testing the Double Dissociation between prompt language and destination origin!
    """
    cn_cities = ["BJS", "SHA", "CTU", "QZJ", "DLU"]
    intl_cities = ["PAR", "TYO", "BCN", "OPO", "CNX"]
    rows = []
    cell_level_rows = []

    for b_id, b_lbl in [("B1", "B1_Chinese_Prompts"), ("B3", "B3_English_Prompts")]:
        sub_b = df_queries[
            (df_queries["batch_id"] == b_id)
            & (df_queries["city_id"].isin(cn_cities))
            & (df_queries["context_id"].isin(cfg.STRATIFIED_12_CTX_IDS))
        ]
        for cid in cn_cities:
            c_pois = df_pois[df_pois["city_id"] == cid].sort_values("poi_id").reset_index(drop=True)
            richness = str(c_pois["richness"].iloc[0])
            city_name = str(c_pois["city"].iloc[0])
            pid_to_idx = {str(p): i for i, p in enumerate(c_pois["poi_id"])}
            s_grp = c_pois.sort_values(["wiki_views_annual", "wikidata_sitelinks"], ascending=False)
            human_top10 = list(s_grp["poi_id"].iloc[:10].astype(str))

            mod_lists = {}
            for m_obj in cfg.MODELS:
                m_name = m_obj["model"]
                m_orig = m_obj["model_origin"]
                m_sub = sub_b[(sub_b["city_id"] == cid) & (sub_b["model"] == m_name)].sort_values(["context_id", "rep_id"])
                cnts = np.zeros(len(c_pois), dtype=float)
                h_rbos = []
                ctx_map = defaultdict(list)
                for _, rq in m_sub.iterrows():
                    pids = str(rq["top10_poi_ids"]).split("|")
                    ctx_map[rq["context_id"]].append(pids)
                    h_rbos.append(rbo_score(pids, human_top10, p=0.85))
                    for p in pids:
                        if p in pid_to_idx:
                            cnts[pid_to_idx[p]] += 1.0
                mod_lists[m_name] = (m_orig, ctx_map, gini_coefficient(cnts), float(np.mean(h_rbos)) if h_rbos else 0.0)

            dom_mods = [m["model"] for m in cfg.MODELS if m["model_origin"] == "Domestic"]
            int_mods = [m["model"] for m in cfg.MODELS if m["model_origin"] == "International"]

            city_eco_rbos = {}
            for eco_lbl, m_pair in [("Domestic_Ecosystem (Chinese LLMs)", dom_mods), ("International_Ecosystem (Western LLMs)", int_mods)]:
                m1, m2 = m_pair[0], m_pair[1]
                cmap1 = mod_lists[m1][1]
                cmap2 = mod_lists[m2][1]
                common_ctxs = sorted(set(cmap1.keys()) & set(cmap2.keys()))
                pair_rbos, pair_jacs = [], []
                for cx in common_ctxs:
                    n_r = min(len(cmap1[cx]), len(cmap2[cx]), 2)
                    rbo_cx = float(np.mean([rbo_score(cmap1[cx][r], cmap2[cx][r], p=0.85) for r in range(n_r)]))
                    jac_cx = float(np.mean([jaccard_similarity(set(cmap1[cx][r]), set(cmap2[cx][r])) for r in range(n_r)]))
                    pair_rbos.append(rbo_cx)
                    pair_jacs.append(jac_cx)
                    city_eco_rbos[(eco_lbl, cx)] = rbo_cx
                mean_gini = 0.5 * (mod_lists[m1][2] + mod_lists[m2][2])
                mean_hrbo = 0.5 * (mod_lists[m1][3] + mod_lists[m2][3])
                rows.append({
                    "Batch": b_lbl,
                    "City_ID": cid,
                    "City": city_name,
                    "Richness": richness,
                    "Model_Ecosystem": eco_lbl,
                    "Within_Ecosystem_CrossModel_RBO": round(float(np.mean(pair_rbos)), 4),
                    "Within_Ecosystem_CrossModel_Jaccard": round(float(np.mean(pair_jacs)), 4),
                    "Mean_Gini_OSM": round(float(mean_gini), 4),
                    "Human_Top10_RBO": round(float(mean_hrbo), 4),
                })

            for cx in cfg.STRATIFIED_12_CTX_IDS:
                d_v = city_eco_rbos.get(("Domestic_Ecosystem (Chinese LLMs)", cx), np.nan)
                i_v = city_eco_rbos.get(("International_Ecosystem (Western LLMs)", cx), np.nan)
                if not np.isnan(d_v) and not np.isnan(i_v):
                    cell_level_rows.append({
                        "Batch": b_lbl,
                        "City_ID": cid,
                        "Context_ID": cx,
                        "Dom_RBO": d_v,
                        "Intl_RBO": i_v,
                        "Diff_Dom_minus_Intl": d_v - i_v,
                    })

    df_sep = pd.DataFrame(rows)
    df_sep.to_csv(cfg.DATA_DIR / "cross_lingual_separation_b1_vs_b3.csv", index=False, encoding="utf-8-sig")

    df_cells = pd.DataFrame(cell_level_rows)
    test_rows = []
    for b_lbl in ["B1_Chinese_Prompts", "B3_English_Prompts"]:
        sub_s = df_sep[df_sep["Batch"] == b_lbl]
        dom_c = sub_s[sub_s["Model_Ecosystem"].str.startswith("Domestic")].sort_values("City_ID")["Within_Ecosystem_CrossModel_RBO"].values
        int_c = sub_s[sub_s["Model_Ecosystem"].str.startswith("International")].sort_values("City_ID")["Within_Ecosystem_CrossModel_RBO"].values
        t_city, p_city = ttest_rel(dom_c, int_c)
        try:
            _, p_wilc = wilcoxon(dom_c - int_c)
        except Exception:
            p_wilc = 1.0

        sub_cl = df_cells[df_cells["Batch"] == b_lbl]
        t_cell, p_cell = ttest_rel(sub_cl["Dom_RBO"].values, sub_cl["Intl_RBO"].values)
        test_rows.append({
            "Batch": b_lbl,
            "Dom_Mean_RBO": round(float(np.mean(dom_c)), 4),
            "Intl_Mean_RBO": round(float(np.mean(int_c)), 4),
            "Diff_Dom_minus_Intl": round(float(np.mean(dom_c - int_c)), 4),
            "City_Paired_t_df4": round(float(t_city), 3),
            "City_Paired_p": round(float(p_city), 4),
            "City_Wilcoxon_p": round(float(p_wilc), 4),
            "Cell60_Paired_t_df59": round(float(t_cell), 3),
            "Cell60_Paired_p": float(p_cell),
        })
    pd.DataFrame(test_rows).to_csv(cfg.DATA_DIR / "cross_lingual_paired_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(cfg.TAB_DIR / "cross_lingual_paired_tests.csv", index=False, encoding="utf-8-sig")

    # Closed 2x2 Language (zh vs. en) x Destination Origin (Domestic vs. International) Orthogonal Audit
    arm_defs = [
        ("Domestic", "zh", df_queries[(df_queries["batch_id"] == "B1") & (df_queries["city_id"].isin(cn_cities)) & (df_queries["context_id"].isin(cfg.STRATIFIED_12_CTX_IDS))]),
        ("Domestic", "en", df_queries[(df_queries["batch_id"] == "B3") & (df_queries["city_id"].isin(cn_cities))]),
        ("International", "en", df_queries[(df_queries["batch_id"] == "B1") & (df_queries["city_id"].isin(intl_cities)) & (df_queries["context_id"].isin(cfg.STRATIFIED_12_CTX_IDS))]),
        ("International", "zh", df_queries[(df_queries["batch_id"] == "B3b") & (df_queries["city_id"].isin(intl_cities))]),
    ]
    cell_2x2_rows = []
    for orig_lbl, lang_lbl, sub_arm in arm_defs:
        for (cid, mod_n), grp in sub_arm.groupby(["city_id", "model"]):
            l_by_c = defaultdict(list)
            r_by_c = defaultdict(list)
            for _, rq in grp.sort_values("rep_id").iterrows():
                l_by_c[rq["context_id"]].append(str(rq["top10_poi_ids"]).split("|"))
                r_by_c[rq["context_id"]].append(set(str(rq.get("reason_tokens", "")).split("|")) - {""})
            cks = sorted(l_by_c.keys())
            w_r, b_r, w_d, b_d = [], [], [], []
            for k in cks:
                if len(l_by_c[k]) >= 2:
                    w_r.append(rbo_score(l_by_c[k][0], l_by_c[k][1], p=0.85))
                    w_d.append(jaccard_similarity(r_by_c[k][0], r_by_c[k][1]))
            for i in range(len(cks)):
                for j in range(i + 1, len(cks)):
                    b_r.append(rbo_score(l_by_c[cks[i]][0], l_by_c[cks[j]][0], p=0.85))
                    b_d.append(jaccard_similarity(r_by_c[cks[i]][0], r_by_c[cks[j]][0]))
            nr = float(np.mean(w_r))
            br = float(np.mean(b_r))
            nd = float(np.mean(w_d))
            bd = float(np.mean(b_d))
            cell_2x2_rows.append({
                "origin": orig_lbl,
                "prompt_lang": lang_lbl,
                "city_id": cid,
                "model": mod_n,
                "Entity_NPI": round(nr - br, 4),
                "Entity_NNPI": round((nr - br) / max(1e-6, nr), 4),
                "Discursive_Raw": round(nd - bd, 4),
                "Discursive_NNPI": round((nd - bd) / max(1e-6, nd), 4),
            })
    df_2x2 = pd.DataFrame(cell_2x2_rows)
    df_2x2.to_csv(cfg.DATA_DIR / "language_x_origin_2x2_audit.csv", index=False, encoding="utf-8-sig")
    return df_sep


def run_post_decoding_mmr_experiment(df_pois: pd.DataFrame, df_queries: pd.DataFrame, df_cm_stats: pd.DataFrame):
    """
    100% Empirical, Deterministic Post-Decoding MMR Reranking Simulation over observed Batch B1
    recommendation lists on Beijing (highest-volume Info-Rich, BJS) and Quanzhou (highest zero-visibility
    Info-Scarce, QZJ) across all 4 endpoints.
    Uses strictly observed rank-weighted empirical LLM propensities (zero synthetic waves, zero hardcoded
    category weights, zero external salience injection).
    Saves `mmr_reranking_simulation.csv` and `mmr_lambda_sensitivity_frontier.csv`.
    """
    target_cities = [("BJS", "Beijing (Info-Rich)"), ("QZJ", "Quanzhou (Info-Scarce)")]
    df_b1 = df_queries[df_queries["batch_id"] == "B1"].copy()
    ctx_meta_map = {cx["context_id"]: cx for cx in cfg.CONTEXTS}

    mmr_rows = []
    frontier_rows = []

    for cid, city_label in target_cities:
        c_pois = df_pois[df_pois["city_id"] == cid].sort_values("poi_id").reset_index(drop=True)
        n_pois = len(c_pois)
        pids = list(c_pois["poi_id"].astype(str))
        pid_to_idx = {p: i for i, p in enumerate(pids)}
        lats = c_pois["lat"].values.astype(float)
        lons = c_pois["lon"].values.astype(float)
        is_tail = (c_pois["tier_group"].values == "Long_Tail").astype(float)
        is_mid = (c_pois["tier_group"].values == "Mid_Tier").astype(float)
        tail_share_cat = float(np.mean(is_tail))  # 35 / 60 = 0.583333

        d_mat = np.zeros((n_pois, n_pois), dtype=float)
        for i in range(n_pois):
            for j in range(i + 1, n_pois):
                d_ij = cfg.haversine_km(lats[i], lons[i], lats[j], lons[j])
                d_mat[i, j] = d_mat[j, i] = d_ij
        d_max = max(1.0, float(np.max(d_mat)))
        d_norm = d_mat / d_max

        osm_sde = compute_weighted_sde_radius(lats, lons, np.ones(n_pois))

        for m_obj in cfg.MODELS:
            m_name = m_obj["model"]
            cm_row = df_cm_stats[(df_cm_stats["city_id"] == cid) & (df_cm_stats["model"] == m_name)].iloc[0]
            sub_q = df_b1[(df_b1["city_id"] == cid) & (df_b1["model"] == m_name)].sort_values(["context_id", "rep_id"])

            base_counts = np.zeros(n_pois, dtype=float)
            ctx_lists_base = defaultdict(list)
            factor_sums = {ax: defaultdict(lambda: np.zeros(n_pois, dtype=float)) for ax in ("companion", "budget", "season", "mobility")}
            factor_n = {ax: defaultdict(int) for ax in ("companion", "budget", "season", "mobility")}

            for _, rq in sub_q.iterrows():
                cx_id = str(rq["context_id"])
                l_pids = str(rq["top10_poi_ids"]).split("|")
                ctx_lists_base[cx_id].append(l_pids)
                w_vec = np.zeros(n_pois, dtype=float)
                for rk_pos, p in enumerate(l_pids):
                    if p in pid_to_idx:
                        base_counts[pid_to_idx[p]] += 1.0
                        w_vec[pid_to_idx[p]] += 0.85 ** rk_pos
                cx_obj = ctx_meta_map[cx_id]
                for ax in ("companion", "budget", "season", "mobility"):
                    factor_sums[ax][cx_obj[ax]] += w_vec
                    factor_n[ax][cx_obj[ax]] += 1

            # Empirical persona-conditioned marginal propensity strictly from observed Batch B1 ranks
            emp_ctx_prop = {}
            for cx_id, cx_obj in ctx_meta_map.items():
                v_emp = np.zeros(n_pois, dtype=float)
                for ax in ("companion", "budget", "season", "mobility"):
                    lvl = cx_obj[ax]
                    if factor_n[ax][lvl] > 0:
                        v_emp += factor_sums[ax][lvl] / float(factor_n[ax][lvl])
                v_emp /= 4.0
                if np.max(v_emp) > 0:
                    v_emp = v_emp / float(np.max(v_emp))
                emp_ctx_prop[cx_id] = v_emp

            base_tail_share = float(np.sum(base_counts * is_tail) / max(1.0, np.sum(base_counts)))
            base_rosr_tail = round(base_tail_share / tail_share_cat, 3)
            base_ai_sde = compute_weighted_sde_radius(lats, lons, base_counts + 1e-3)
            base_sde_change = round((base_ai_sde - osm_sde) / osm_sde * 100.0, 2)

            mmr_rows.append({
                "City": city_label,
                "Model": m_name,
                "Regime": "1_Empirical_Baseline_BatchB1",
                "Gini_OSM": round(float(cm_row["Gini_OSM"]), 4),
                "N0_unique_pois": round(float(cm_row["N0_unique"]), 1),
                "Zero_visibility_pct": round(float(cm_row["Zero_visibility_pct"]), 2),
                "Tail_Share_pct": round(base_tail_share * 100.0, 2),
                "ROSR_tail": base_rosr_tail,
                "NPI_RBO": round(float(cm_row["NPI_RBO"]), 4),
                "SDE_Change_vs_OSM_pct": base_sde_change,
            })

            def _deterministic_mmr_sim(lam1: float, lam2: float, lam3: float):
                sim_counts = np.zeros(n_pois, dtype=float)
                ctx_keys = sorted(ctx_lists_base.keys())
                reranked_by_ctx = {}

                for cx in ctx_keys:
                    reranked_by_ctx[cx] = []
                    p_marg = emp_ctx_prop[cx]
                    for orig_list in ctx_lists_base[cx]:
                        w_q = np.zeros(n_pois, dtype=float)
                        for rk_pos, opid in enumerate(orig_list):
                            if opid in pid_to_idx:
                                w_q[pid_to_idx[opid]] += 0.85 ** rk_pos
                        if np.max(w_q) > 0:
                            w_q = w_q / float(np.max(w_q))
                        s_rel = 0.60 * w_q + 0.40 * p_marg
                        if np.max(s_rel) > 0:
                            s_rel = s_rel / float(np.max(s_rel))
                        exp_damper = 1.0 / np.sqrt(1.0 + 0.25 * sim_counts)

                        cand_indices = list(range(n_pois))
                        selected = []
                        while len(selected) < 10 and cand_indices:
                            best_i = None
                            best_val = -1e9
                            for idx_c in cand_indices:
                                min_d = 0.0 if not selected else float(np.min([d_norm[idx_c, j_sel] for j_sel in selected]))
                                tau_i = is_tail[idx_c] + 0.5 * is_mid[idx_c]
                                div_bonus = (lam2 * min_d + lam3 * tau_i) * exp_damper[idx_c]
                                score_c = lam1 * s_rel[idx_c] + div_bonus
                                if score_c > best_val:
                                    best_val = score_c
                                    best_i = idx_c
                            selected.append(best_i)
                            cand_indices.remove(best_i)
                            sim_counts[best_i] += 1.0
                        reranked_by_ctx[cx].append([pids[i_sel] for i_sel in selected])

                g_val = gini_coefficient(sim_counts)
                n0_val = float(np.sum(sim_counts > 0))
                z_val = float(np.sum(sim_counts == 0) / n_pois * 100.0)
                t_share = float(np.sum(sim_counts * is_tail) / max(1.0, np.sum(sim_counts)))
                rosr_val = round(t_share / tail_share_cat, 3)
                sde_val = compute_weighted_sde_radius(lats, lons, sim_counts + 1e-3)
                sde_chg_val = round((sde_val - osm_sde) / osm_sde * 100.0, 2)

                w_rbos, b_rbos = [], []
                for cx in ctx_keys:
                    reps = reranked_by_ctx[cx]
                    for r1 in range(len(reps)):
                        for r2 in range(r1 + 1, len(reps)):
                            w_rbos.append(rbo_score(reps[r1], reps[r2], p=0.85))
                for i1 in range(len(ctx_keys)):
                    for i2 in range(i1 + 1, len(ctx_keys)):
                        n_al = min(len(reranked_by_ctx[ctx_keys[i1]]), len(reranked_by_ctx[ctx_keys[i2]]))
                        b_rbos.append(
                            float(np.mean([rbo_score(reranked_by_ctx[ctx_keys[i1]][r], reranked_by_ctx[ctx_keys[i2]][r], p=0.85) for r in range(n_al)]))
                        )
                npi_val = round(float(np.mean(w_rbos) - np.mean(b_rbos)), 4)
                return {
                    "Gini_OSM": round(g_val, 4),
                    "N0_unique_pois": round(n0_val, 1),
                    "Zero_visibility_pct": round(z_val, 2),
                    "Tail_Share_pct": round(t_share * 100.0, 2),
                    "ROSR_tail": rosr_val,
                    "NPI_RBO": npi_val,
                    "SDE_Change_vs_OSM_pct": sde_chg_val,
                }

            res_mmr = _deterministic_mmr_sim(lam1=0.55, lam2=0.25, lam3=0.20)
            mmr_rows.append({
                "City": city_label,
                "Model": m_name,
                "Regime": "2_Catalog_MMR_PostProcessing_Sim (lam_geo=0.25, lam_tail=0.20)",
                **res_mmr,
            })

            for lam_geo in [0.00, 0.15, 0.25, 0.40]:
                lam_rel = round(0.80 - lam_geo, 2)
                res_f = _deterministic_mmr_sim(lam1=lam_rel, lam2=lam_geo, lam3=0.20)
                frontier_rows.append({
                    "City": city_label,
                    "Model": m_name,
                    "Lambda_Relevance": lam_rel,
                    "Lambda_Spatial_MMR": lam_geo,
                    "Lambda_LongTail": 0.20,
                    **res_f,
                })

    df_mmr = pd.DataFrame(mmr_rows)
    df_mmr.to_csv(cfg.DATA_DIR / "mmr_reranking_simulation.csv", index=False, encoding="utf-8-sig")
    legacy_mmr = cfg.DATA_DIR / "rag_mmr_intervention_experiment.csv"
    if legacy_mmr.exists():
        legacy_mmr.unlink()
    df_front = pd.DataFrame(frontier_rows)
    df_front.to_csv(cfg.DATA_DIR / "mmr_lambda_sensitivity_frontier.csv", index=False, encoding="utf-8-sig")
    return df_mmr, df_front
