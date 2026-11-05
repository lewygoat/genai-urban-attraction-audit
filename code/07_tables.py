# -*- coding: utf-8 -*-
"""
07_tables.py — 100% Data-Driven Publication Table Exporter (Tables 1 to 5 + Table 6).
Every cell is dynamically extracted from the live `raw_responses_v2final/` dataframes
and fitted statsmodels econometric objects.
"""
import importlib
import numpy as np
import pandas as pd

cfg = importlib.import_module("00_config")
met = importlib.import_module("04_metrics")


def fmt_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_coding_reliability(y1: np.ndarray, y2: np.ndarray) -> tuple[int, float, float, float]:
    """Returns (positive_cases, raw_agreement_pct, pabak, cohens_kappa) on the 10% dual-verified sample."""
    mask = (~pd.isna(y1)) & (~pd.isna(y2))
    a1 = np.asarray(y1[mask], dtype=int)
    a2 = np.asarray(y2[mask], dtype=int)
    valid = (a1 >= 0) & (a2 >= 0)
    a1, a2 = a1[valid], a2[valid]
    n = len(a1)
    if n == 0:
        return 0, 100.0, 1.0, np.nan
    n_pos = int(np.sum((a1 == 1) | (a2 == 1)))
    po = float(np.mean(a1 == a2))
    pabak = 2.0 * po - 1.0
    p1_1 = float(np.mean(a1 == 1))
    p2_1 = float(np.mean(a2 == 1))
    pe = p1_1 * p2_1 + (1.0 - p1_1) * (1.0 - p2_1)
    if n_pos == 0 or abs(1.0 - pe) < 1e-12:
        kappa = np.nan
    else:
        kappa = float((po - pe) / (1.0 - pe))
    return n_pos, po * 100.0, pabak, kappa


def export_all_tables(
    df_pois: pd.DataFrame,
    df_queries: pd.DataFrame,
    df_cm_stats: pd.DataFrame,
    df_dyads: pd.DataFrame,
    mixed_res: dict,
    df_permanova: pd.DataFrame,
    nb_res: dict,
):
    # ==========================================================================
    # TABLE 1: 10 Cities Baseline, Public Salience Gini, Unwarped SDE Radii (km) & AI Concentration
    # ==========================================================================
    t1_rows = []
    t1_num_cache = []
    for c in cfg.CITIES_META:
        cid = c["city_id"]
        sub = df_cm_stats[df_cm_stats["city_id"] == cid]
        c_p = df_pois[df_pois["city_id"] == cid]
        pub_gini = met.gini_coefficient(c_p["wiki_views_annual"].values)
        lats = c_p["lat"].values
        lons = c_p["lon"].values
        mentions = c_p["total_ai_mentions"].values.astype(float)
        osm_sde = met.compute_weighted_sde_radius(lats, lons, np.ones(len(lats)))
        ai_sde = met.compute_weighted_sde_radius(lats, lons, mentions + 1e-3)
        sde_chg_pct = (ai_sde - osm_sde) / max(1e-6, osm_sde) * 100.0

        inner_mask = c_p["dist_center_km"].values <= 20.0
        if np.sum(inner_mask) >= 5:
            osm_sde_20 = met.compute_weighted_sde_radius(lats[inner_mask], lons[inner_mask], np.ones(int(np.sum(inner_mask))))
            ai_sde_20 = met.compute_weighted_sde_radius(lats[inner_mask], lons[inner_mask], mentions[inner_mask] + 1e-3)
            sde_chg_20_pct = (ai_sde_20 - osm_sde_20) / max(1e-6, osm_sde_20) * 100.0
        else:
            osm_sde_20, ai_sde_20, sde_chg_20_pct = osm_sde, ai_sde, sde_chg_pct

        gini_osm_raw = float(sub["Gini_OSM"].mean())
        gini_all_raw = float(sub["Gini_AllOSM"].mean())
        n0_raw = float(sub["N0_unique"].mean())
        zvis_raw = float(sub["Zero_visibility_pct"].mean())
        wiki_k_raw = float(c["wiki_views_k"])
        sitelinks_raw = int(c["wikidata_sitelinks"])

        row_dict = {
            "City": f"{c['city']} ({c['country']})",
            "Group": f"{c['origin']} / {c['richness']}",
            "OSM Tourism POIs": int(c["osm_pois"]),
            "Wiki Views (k/yr)": round(wiki_k_raw, 1),
            "Wikidata Sitelinks": sitelinks_raw,
            "Public Salience Gini": round(pub_gini, 3),
            "AI Mention Gini (Mean)": round(gini_osm_raw, 3),
            "Full-OSM Gini": round(gini_all_raw, 3),
            "AI Unique POIs (N0)": round(n0_raw, 1),
            "Zero-Visibility Tail (%)": round(zvis_raw, 1),
            "OSM SDE (km)": round(osm_sde, 2),
            "AI SDE (km)": round(ai_sde, 2),
            "SDE Change (%)": f"{sde_chg_pct:+.1f}%",
            "Inner-Core <=20km SDE Change (%)": f"{sde_chg_20_pct:+.1f}%",
        }
        t1_rows.append(row_dict)
        t1_num_cache.append({
            **row_dict,
            "_wiki_k": wiki_k_raw,
            "_sitelinks": sitelinks_raw,
            "_pub_gini": pub_gini,
            "_gini_osm": gini_osm_raw,
            "_gini_all": gini_all_raw,
            "_n0": n0_raw,
            "_zvis": zvis_raw,
            "_osm_sde": osm_sde,
            "_ai_sde": ai_sde,
            "_sde_chg": sde_chg_pct,
            "_sde20_chg": sde_chg_20_pct,
        })

    def _make_t1_summary(rows_sub: list, label: str, grp_tag: str):
        osm_m = float(np.mean([r["_osm_sde"] for r in rows_sub]))
        ai_m = float(np.mean([r["_ai_sde"] for r in rows_sub]))
        chg_m = float(np.mean([r["_sde_chg"] for r in rows_sub]))
        chg20_m = float(np.mean([r["_sde20_chg"] for r in rows_sub]))
        return {
            "City": label,
            "Group": grp_tag,
            "OSM Tourism POIs": int(round(float(np.mean([r["OSM Tourism POIs"] for r in rows_sub])))),
            "Wiki Views (k/yr)": round(float(np.mean([r["_wiki_k"] for r in rows_sub])), 1),
            "Wikidata Sitelinks": int(round(float(np.mean([r["_sitelinks"] for r in rows_sub])))),
            "Public Salience Gini": round(float(np.mean([r["_pub_gini"] for r in rows_sub])), 3),
            "AI Mention Gini (Mean)": round(float(np.mean([r["_gini_osm"] for r in rows_sub])), 3),
            "Full-OSM Gini": round(float(np.mean([r["_gini_all"] for r in rows_sub])), 3),
            "AI Unique POIs (N0)": round(float(np.mean([r["_n0"] for r in rows_sub])), 1),
            "Zero-Visibility Tail (%)": round(float(np.mean([r["_zvis"] for r in rows_sub])), 1),
            "OSM SDE (km)": round(osm_m, 2),
            "AI SDE (km)": round(ai_m, 2),
            "SDE Change (%)": f"{chg_m:+.1f}%",
            "Inner-Core <=20km SDE Change (%)": f"{chg20_m:+.1f}%",
        }

    rich_rows = [r for r in t1_num_cache if "Info-Rich" in r["Group"]]
    scarce_rows = [r for r in t1_num_cache if "Info-Scarce" in r["Group"]]
    t1_rows.append(_make_t1_summary(rich_rows, "Mean / Summary (Info-Rich, n = 6)", "Info-Rich Stratum Mean"))
    t1_rows.append(_make_t1_summary(scarce_rows, "Mean / Summary (Info-Scarce, n = 4)", "Info-Scarce Stratum Mean"))
    t1_rows.append(_make_t1_summary(rich_rows + scarce_rows, "Overall Mean (All 10 Cities)", "Grand Mean (N = 10)"))
    df_t1 = pd.DataFrame(t1_rows)
    df_t1.to_csv(cfg.TAB_DIR / "Table_1_Cities_Baseline.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # TABLE 2: Audit Batches (B1, B2, B3, B3b, B4, B5, B6), JSON Validity, Taxonomy & Dual-Verifier Reliability
    # ==========================================================================
    t2_rows = []
    for b_key, b_label in cfg.BATCH_SPECS:
        sub_b = df_queries[df_queries["batch_id"] == b_key]
        if len(sub_b) == 0:
            continue
        n_calls = len(sub_b)
        n_slots = n_calls * 10
        n_sampled = int((sub_b["sampled_10pct_coding"] == 1).sum())
        valid_pct = float(sub_b["valid_json"].mean() * 100.0)
        in_cat_cnt = int(sub_b["in_catalog_pois"].sum()) if "in_catalog_pois" in sub_b.columns else 0
        halluc_cnt = int(sub_b["hallucinated_pois"].sum())
        misattr_cnt = int(sub_b["misattributed_pois"].sum())
        halluc_rate = float(halluc_cnt / n_slots * 100.0)
        misattr_rate = float(misattr_cnt / n_slots * 100.0)
        retained_rate = 100.0 - halluc_rate - misattr_rate
        in_cat_rate = float(in_cat_cnt / n_slots * 100.0)
        sub_coded = sub_b[sub_b["sampled_10pct_coding"] == 1]
        n_pos, raw_agr, pabak, kappa = compute_coding_reliability(
            sub_coded["coder1_flag"].values, sub_coded["coder2_flag"].values
        )
        t2_rows.append({
            "Sampling Batch": b_label,
            "API Queries (N)": n_calls,
            "Total POI Slots": n_slots,
            "10% Rule Cross-Val. (n)": n_sampled,
            "Anomaly Cases": n_pos,
            "Valid JSON (%)": round(valid_pct, 2),
            "In-Catalog 60-POI (%)": f"{in_cat_rate:.2f}%",
            "Unverified / Generic (%)": f"{halluc_rate:.2f}% ({halluc_cnt})",
            "Exurban Spillover (%)": f"{misattr_rate:.2f}% ({misattr_cnt})",
            "Valid Municipal POIs (%)": f"{retained_rate:.2f}%",
            "2-Pipeline Agree (%)": round(raw_agr, 2),
            "PABAK": round(pabak, 3),
            "Cohen's Kappa": "N/A (0 pos)" if np.isnan(kappa) else f"{kappa:.3f}",
        })

    n_all = len(df_queries)
    slots_all = n_all * 10
    samp_all = int((df_queries["sampled_10pct_coding"] == 1).sum())
    val_all = float(df_queries["valid_json"].mean() * 100.0)
    inc_all = int(df_queries["in_catalog_pois"].sum()) if "in_catalog_pois" in df_queries.columns else 0
    h_all = int(df_queries["hallucinated_pois"].sum())
    m_all = int(df_queries["misattributed_pois"].sum())
    hr_all = float(h_all / slots_all * 100.0)
    mr_all = float(m_all / slots_all * 100.0)
    inc_r_all = float(inc_all / slots_all * 100.0)
    coded_all = df_queries[df_queries["sampled_10pct_coding"] == 1]
    npos_all, ragr_all, pabak_all, kap_all = compute_coding_reliability(
        coded_all["coder1_flag"].values, coded_all["coder2_flag"].values
    )
    t2_rows.append({
        "Sampling Batch": "Total / Overall Corpus (7 Batches: B1-B6 & B3b)",
        "API Queries (N)": n_all,
        "Total POI Slots": slots_all,
        "10% Rule Cross-Val. (n)": samp_all,
        "Anomaly Cases": npos_all,
        "Valid JSON (%)": round(val_all, 2),
        "In-Catalog 60-POI (%)": f"{inc_r_all:.2f}%",
        "Unverified / Generic (%)": f"{hr_all:.2f}% ({h_all})",
        "Exurban Spillover (%)": f"{mr_all:.2f}% ({m_all})",
        "Valid Municipal POIs (%)": f"{100.0 - hr_all - mr_all:.2f}%",
        "2-Pipeline Agree (%)": round(ragr_all, 2),
        "PABAK": round(pabak_all, 3),
        "Cohen's Kappa": "N/A (0 pos)" if np.isnan(kap_all) else f"{kap_all:.3f}",
    })
    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(cfg.TAB_DIR / "Table_2_Audit_Sampling_and_Hallucinations.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # TABLE 3: Diversity, Gini, Unbiased Bootstrap NPI, NNPI, Slot Turnover, Blank Anchoring & Discursive NPI
    # ==========================================================================
    grp_cols = ["model", "origin", "richness"]
    num_cols = [
        "N0_unique", "N1_effective", "HHI", "Gini_OSM", "Gini_Sample", "Gini_EqualCap_M15", "Gini_AllOSM",
        "Top10_share_pct", "Noise_RBO", "Noise_RBO_SD", "Between_Ctx_RBO", "NPI_RBO",
        "NPI_CI_Low", "NPI_CI_High", "NNPI_RBO", "Delta_K10_Slots", "Discursive_NNPI",
        "Blank_B2_RBO", "Blank_B2_Shared_K10", "Human_Top10_RBO", "JSD_vs_Reviews", "Hallucination_pct",
    ]
    avail_cols = [c for c in num_cols if c in df_cm_stats.columns]
    df_t3_full = df_cm_stats.groupby(grp_cols, as_index=False, sort=False)[avail_cols].mean().round(3)
    assert df_t3_full.shape[1] == 25, f"Expected 25 columns in Table 3 Full Supplementary, got {df_t3_full.shape[1]}"
    old_23 = cfg.DATA_DIR / "Table_3_Full_Supplementary_23Cols.csv"
    if old_23.exists():
        old_23.unlink()
    df_t3_full.to_csv(cfg.DATA_DIR / "Table_3_Full_Supplementary_25Cols.csv", index=False, encoding="utf-8-sig")
    df_t3 = df_t3_full.copy()
    df_t3.to_csv(cfg.TAB_DIR / "Table_3_Diversity_Personalization_Summary.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # TABLE 4: Multilevel Linear Mixed-Effects Regression (REML) + Cluster-Robust SEs & Orthogonalized Ecology Check
    # ==========================================================================
    mod_rbo = mixed_res["rbo"]
    mod_jac = mixed_res["jaccard"]
    mod_orth = mixed_res["rbo_orth"]
    rbo_st = mixed_res["rbo_stats"]
    jac_st = mixed_res["jac_stats"]
    cl_se_rbo = mixed_res["rbo_cluster_se"]
    dy_se_rbo = mixed_res["rbo_dyad_se"]
    cl_z_rbo = mixed_res["rbo_cluster_z"]
    cl_p_rbo = mixed_res["rbo_cluster_p"]
    cl_q_rbo = mixed_res["rbo_cluster_q"]

    cl_se_jac = mixed_res["jac_cluster_se"]
    dy_se_jac = mixed_res["jac_dyad_se"]
    cl_z_jac = mixed_res["jac_cluster_z"]
    cl_p_jac = mixed_res["jac_cluster_p"]
    cl_q_jac = mixed_res["jac_cluster_q"]

    orth_cl_se = mixed_res["orth_cluster_se"]
    orth_dy_se = mixed_res["orth_dyad_se"]
    orth_cl_z = mixed_res["orth_cluster_z"]
    orth_cl_p = mixed_res["orth_cluster_p"]

    var_map = [
        ("Intercept", "Intercept (Baseline Reference)"),
        ("diff_companion", "Diff Companion (0/1)"),
        ("diff_budget", "Diff Budget (0/1)"),
        ("diff_season", "Diff Season (0/1)"),
        ("diff_mobility", "Diff Mobility / Wheelchair (0/1)"),
        ("same_model", "Same Endpoint-Tier Configuration (0/1)"),
        ("same_origin", "Same Geo-Bloc / Origin (Raw Collinear Spec)"),
        ("home_match", "Home-Field Match (Origin == City, Info-Scarce)"),
        ("info_rich_num", "Info-Rich City (1 = Info-Rich, 0 = Info-Scarce)"),
        ("home_match:info_rich_num", "Home-Field Match x Info-Rich City"),
    ]
    t4_rows = []
    for k, label in var_map:
        b1 = float(mod_rbo.fe_params[k])
        cl1 = float(cl_se_rbo[k])
        dy1 = float(dy_se_rbo[k])
        z1 = float(cl_z_rbo[k])
        p1 = float(cl_p_rbo[k])
        q1 = float(cl_q_rbo[k])

        b2 = float(mod_jac.fe_params[k])
        cl2 = float(cl_se_jac[k])
        dy2 = float(dy_se_jac[k])
        z2 = float(cl_z_jac[k])
        p2 = float(cl_p_jac[k])
        q2 = float(cl_q_jac[k])

        pq1_str = f"{p1:.4f} (-)" if np.isnan(q1) else f"{p1:.4f} ({q1:.4f})"
        pq2_str = f"{p2:.4f} (-)" if np.isnan(q2) else f"{p2:.4f} ({q2:.4f})"

        t4_rows.append({
            "Variable": label,
            "Model 1 (RBO) Coef": f"{b1:+.4f}{fmt_star(p1)}",
            "Model 1 City-Cluster SE (2-Way Dyadic SE)": f"{cl1:.4f} ({dy1:.4f})",
            "Model 1 (RBO) t-stat (df=9)": f"{z1:+.2f}",
            "Model 1 (RBO) p-val (q)": pq1_str,
            "Model 2 (Jaccard) Coef": f"{b2:+.4f}{fmt_star(p2)}",
            "Model 2 City-Cluster SE (2-Way Dyadic SE)": f"{cl2:.4f} ({dy2:.4f})",
            "Model 2 (Jaccard) t-stat (df=9)": f"{z2:+.2f}",
            "Model 2 (Jaccard) p-val (q)": pq2_str,
        })

    b_orth_eco = float(mod_orth.fe_params["same_ecology_diff_model"])
    dy_orth_eco = float(orth_dy_se["same_ecology_diff_model"])
    cl_orth_eco = float(orth_cl_se["same_ecology_diff_model"])
    z_orth_eco = float(orth_cl_z["same_ecology_diff_model"])
    p_orth_eco = float(orth_cl_p["same_ecology_diff_model"])
    b_orth_sm = float(mod_orth.fe_params["same_model"])
    cl_orth_sm = float(orth_cl_se["same_model"])
    p_orth_sm = float(orth_cl_p["same_model"])
    t4_rows.append({
        "Variable": "Orthogonalized Spec: Same Geo-Bloc x Cross-Endpoint, I(SameOrigin=1 & SameEndpoint=0)",
        "Model 1 (RBO) Coef": f"{b_orth_eco:+.4f}{fmt_star(p_orth_eco)} (SameEndpoint={b_orth_sm:+.4f}{fmt_star(p_orth_sm)})",
        "Model 1 City-Cluster SE (2-Way Dyadic SE)": f"{cl_orth_eco:.4f} ({dy_orth_eco:.4f})",
        "Model 1 (RBO) t-stat (df=9)": f"{z_orth_eco:+.2f}",
        "Model 1 (RBO) p-val (q)": f"{p_orth_eco:.4f} (SameEp SE={cl_orth_sm:.4f})",
        "Model 2 (Jaccard) Coef": "-",
        "Model 2 City-Cluster SE (2-Way Dyadic SE)": "-",
        "Model 2 (Jaccard) t-stat (df=9)": "-",
        "Model 2 (Jaccard) p-val (q)": "-",
    })

    re_sd_rbo = float(np.sqrt(max(1e-9, mod_rbo.cov_re.iloc[0, 0])))
    re_sd_jac = float(np.sqrt(max(1e-9, mod_jac.cov_re.iloc[0, 0])))
    max_corr = float(
        max(
            abs(np.corrcoef(df_dyads["same_model"], df_dyads[col_c])[0, 1])
            for col_c in ["diff_companion", "diff_budget", "diff_season", "diff_mobility", "same_context"]
        )
    )
    max_vif = max(mixed_res["vif"].values())
    corr_sm_so = mixed_res["corr_same_model_origin"]

    t4_rows.extend([
        {
            "Variable": "City Random Intercept SD",
            "Model 1 (RBO) Coef": f"{re_sd_rbo:.4f}", "Model 1 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 1 (RBO) t-stat (df=9)": "-", "Model 1 (RBO) p-val (q)": "-",
            "Model 2 (Jaccard) Coef": f"{re_sd_jac:.4f}", "Model 2 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 2 (Jaccard) t-stat (df=9)": "-", "Model 2 (Jaccard) p-val (q)": "-",
        },
        {
            "Variable": "Intraclass Correlation (ICC)",
            "Model 1 (RBO) Coef": f"{rbo_st['icc']:.4f}", "Model 1 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 1 (RBO) t-stat (df=9)": "-", "Model 1 (RBO) p-val (q)": "-",
            "Model 2 (Jaccard) Coef": f"{jac_st['icc']:.4f}", "Model 2 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 2 (Jaccard) t-stat (df=9)": "-", "Model 2 (Jaccard) p-val (q)": "-",
        },
        {
            "Variable": "Marginal R2 / Conditional R2",
            "Model 1 (RBO) Coef": f"{rbo_st['r2_marginal']:.3f} / {rbo_st['r2_conditional']:.3f}", "Model 1 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 1 (RBO) t-stat (df=9)": "-", "Model 1 (RBO) p-val (q)": "-",
            "Model 2 (Jaccard) Coef": f"{jac_st['r2_marginal']:.3f} / {jac_st['r2_conditional']:.3f}", "Model 2 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 2 (Jaccard) t-stat (df=9)": "-", "Model 2 (Jaccard) p-val (q)": "-",
        },
        {
            "Variable": "Max |r(SameEndpoint, ContextDiff)| & Collinearity Diagnostics",
            "Model 1 (RBO) Coef": f"|r|={max_corr:.4f}", "Model 1 City-Cluster SE (2-Way Dyadic SE)": f"r(SM,SO)={corr_sm_so:.4f}", "Model 1 (RBO) t-stat (df=9)": f"MaxVIF={max_vif:.2f}", "Model 1 (RBO) p-val (q)": "-",
            "Model 2 (Jaccard) Coef": f"|r|={max_corr:.4f}", "Model 2 City-Cluster SE (2-Way Dyadic SE)": f"r(SM,SO)={corr_sm_so:.4f}", "Model 2 (Jaccard) t-stat (df=9)": f"MaxVIF={max_vif:.2f}", "Model 2 (Jaccard) p-val (q)": "-",
        },
        {
            "Variable": "Observations (Balanced 2x2 Dyadic Pairs N)",
            "Model 1 (RBO) Coef": f"{len(df_dyads)}", "Model 1 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 1 (RBO) t-stat (df=9)": "-", "Model 1 (RBO) p-val (q)": "-",
            "Model 2 (Jaccard) Coef": f"{len(df_dyads)}", "Model 2 City-Cluster SE (2-Way Dyadic SE)": "-", "Model 2 (Jaccard) t-stat (df=9)": "-", "Model 2 (Jaccard) p-val (q)": "-",
        },
    ])
    df_t4 = pd.DataFrame(t4_rows)
    df_t4.to_csv(cfg.TAB_DIR / "Table_4_Dyadic_Multilevel_Regression.csv", index=False, encoding="utf-8-sig")

    # ==========================================================================
    # TABLE 5: Two-Way Crossed PERMANOVA & Negative Binomial Salience Gradient Models
    # ==========================================================================
    p_rich = df_permanova[df_permanova["richness"] == "Info-Rich"]
    p_scarce = df_permanova[df_permanova["richness"] == "Info-Scarce"]
    df_res_min = int(df_permanova["df_Residual"].min())
    df_res_max = int(df_permanova["df_Residual"].max())
    df_res_lbl = f"df={df_res_max}" if df_res_min == df_res_max else f"df={df_res_min}..{df_res_max}"

    per_city_ratio_m = float(df_permanova["F_Model_to_Context_Ratio"].mean())
    per_city_ratio_sd = float(df_permanova["F_Model_to_Context_Ratio"].std())
    pooled_ratio = float(df_permanova["F_Model"].mean() / df_permanova["F_Context"].mean())
    t5_rows = [
        {
            "Panel": "Panel A: Two-Way Crossed PERMANOVA (999 perms)",
            "Source / Predictor": "Composite Endpoint-Tier-Gateway Factor (df=3)",
            "All Cities R2 (%) / Coef": f"{df_permanova['R2_Model_pct'].mean():.2f}% (SD={df_permanova['R2_Model_pct'].std():.2f}%)",
            "Info-Rich R2 (%) / SE": f"{p_rich['R2_Model_pct'].mean():.2f}%",
            "Info-Scarce R2 (%) / IRR": f"{p_scarce['R2_Model_pct'].mean():.2f}%",
            "F-stat / z-stat": f"{df_permanova['F_Model'].mean():.2f} (SD={df_permanova['F_Model'].std():.2f})",
            "p-value": f"{df_permanova['p_Model'].mean():.3f}",
        },
        {
            "Panel": "Panel A: Two-Way Crossed PERMANOVA (999 perms)",
            "Source / Predictor": "Traveler Context (df=47)",
            "All Cities R2 (%) / Coef": f"{df_permanova['R2_Context_pct'].mean():.2f}% (SD={df_permanova['R2_Context_pct'].std():.2f}%)",
            "Info-Rich R2 (%) / SE": f"{p_rich['R2_Context_pct'].mean():.2f}%",
            "Info-Scarce R2 (%) / IRR": f"{p_scarce['R2_Context_pct'].mean():.2f}%",
            "F-stat / z-stat": f"{df_permanova['F_Context'].mean():.2f} (SD={df_permanova['F_Context'].std():.2f})",
            "p-value": f"{df_permanova['p_Context'].mean():.3f}",
        },
        {
            "Panel": "Panel A: Two-Way Crossed PERMANOVA (999 perms)",
            "Source / Predictor": "Endpoint x Context Inter. (df=141)",
            "All Cities R2 (%) / Coef": f"{df_permanova['R2_Interaction_pct'].mean():.2f}% (SD={df_permanova['R2_Interaction_pct'].std():.2f}%)",
            "Info-Rich R2 (%) / SE": f"{p_rich['R2_Interaction_pct'].mean():.2f}%",
            "Info-Scarce R2 (%) / IRR": f"{p_scarce['R2_Interaction_pct'].mean():.2f}%",
            "F-stat / z-stat": f"{df_permanova['F_Interaction'].mean():.2f} (SD={df_permanova['F_Interaction'].std():.2f})",
            "p-value": f"{df_permanova['p_Interaction'].mean():.3f}",
        },
        {
            "Panel": "Panel A: Two-Way Crossed PERMANOVA (999 perms)",
            "Source / Predictor": f"Within-Cell Residual ({df_res_lbl})",
            "All Cities R2 (%) / Coef": f"{df_permanova['R2_Residual_pct'].mean():.2f}% (SD={df_permanova['R2_Residual_pct'].std():.2f}%)",
            "Info-Rich R2 (%) / SE": f"{p_rich['R2_Residual_pct'].mean():.2f}%",
            "Info-Scarce R2 (%) / IRR": f"{p_scarce['R2_Residual_pct'].mean():.2f}%",
            "F-stat / z-stat": "-",
            "p-value": "-",
        },
        {
            "Panel": "Panel A: Two-Way Crossed PERMANOVA (999 perms)",
            "Source / Predictor": "Total Orthogonal Variance Sum & Per-City F-Ratio (Mean-F Ratio)",
            "All Cities R2 (%) / Coef": "100.00%",
            "Info-Rich R2 (%) / SE": "100.00%",
            "Info-Scarce R2 (%) / IRR": "100.00%",
            "F-stat / z-stat": f"{per_city_ratio_m:.2f}x±{per_city_ratio_sd:.2f} ({pooled_ratio:.2f}x)",
            "p-value": "-",
        },
    ]

    nb_unres_nocat = nb_res["nb_unres_nocatfe"]
    nb_unres_cat = nb_res["nb_unres_catfe"]
    nb_unres_strat = nb_res["nb_unres_stratum"]
    nb_wheel_nocat = nb_res["nb_wheel_nocatfe"]
    nb_wheel_cat = nb_res["nb_wheel_catfe"]
    nb_blank_nocat = nb_res["nb_blank_nocatfe"]
    nb_blank_cat = nb_res["nb_blank_catfe"]
    nb_pooled_cat = nb_res["nb_reviews_catfe"]
    nb2 = nb_res["nb_wiki"]
    nb3 = nb_res["nb_search"]

    def _add_nb_row(panel_lbl: str, mod_obj, var_key: str, var_lbl: str):
        b = float(mod_obj.params[var_key])
        se = float(mod_obj.bse[var_key])
        z = float(mod_obj.tvalues[var_key])
        p = float(mod_obj.pvalues[var_key])
        irr = float(np.exp(np.clip(b, -10.0, 10.0)))
        t5_rows.append({
            "Panel": panel_lbl,
            "Source / Predictor": var_lbl,
            "All Cities R2 (%) / Coef": f"{b:+.4f}{fmt_star(p)}",
            "Info-Rich R2 (%) / SE": f"({se:.4f})",
            "Info-Scarce R2 (%) / IRR": f"{irr:.3f}",
            "F-stat / z-stat": f"{z:+.2f}",
            "p-value": f"{p:.4f}",
        })

    b_strat = float(nb_unres_strat.params["info_scarce"])
    se_strat = float(nb_unres_strat.bse["info_scarce"])

    # Primary Arm 1: NB1-Unrestricted (24 Non-Wheelchair Personas, N=2,874 queries)
    p_unres = "Panel B1: NB1-Unrestricted (24 Non-Wheelchair Personas, Cat + City FE)"
    for vk, vl in [
        ("Intercept", "NB1-Unrestricted (2025 Pageviews + Category FE + City FE): Intercept"),
        ("log_reviews_c", "NB1-Unrestricted: ln(1 + 2025 Pageviews) (Info-Rich Gradient γ1)"),
    ]:
        _add_nb_row(p_unres, nb_unres_cat, vk, vl)

    t5_rows.append({
        "Panel": p_unres,
        "Source / Predictor": "NB1-Unrestricted: Info-Scarce Stratum Main Effect (γ2)",
        "All Cities R2 (%) / Coef": "Absorbed by City FE",
        "Info-Rich R2 (%) / SE": f"(No-City-FE: {b_strat:+.4f})",
        "Info-Scarce R2 (%) / IRR": f"(SE={se_strat:.4f})",
        "F-stat / z-stat": "Collinear w/ 10 City FE",
        "p-value": "-",
    })

    for vk, vl in [
        ("log_reviews_c:info_scarce", "NB1-Unrestricted: ln(1 + 2025 Pageviews) x Info-Scarce (γ3)"),
        ("dist_center_km", "NB1-Unrestricted: Distance to City Center (km, γ4)"),
        ("wheelchair_accessible", "NB1-Unrestricted: Step-Free Tier 2 (with Category FE, γ5)"),
    ]:
        _add_nb_row(p_unres, nb_unres_cat, vk, vl)

    _add_nb_row(
        "Panel B1: NB1-Unrestricted (No-Category-FE Comparison)",
        nb_unres_nocat,
        "wheelchair_accessible",
        "NB1-Unrestricted (No Cat FE): Step-Free Tier 2 (without Category FE)",
    )

    # Arm 2: NB1-Wheelchair (24 Wheelchair Personas, N=2,878 queries)
    p_wheel = "Panel B2: NB1-Wheelchair (24 Wheelchair Personas, Cat + City FE)"
    for vk, vl in [
        ("log_reviews_c", "NB1-Wheelchair: ln(1 + 2025 Pageviews) (Info-Rich Gradient γ1)"),
        ("log_reviews_c:info_scarce", "NB1-Wheelchair: ln(1 + 2025 Pageviews) x Info-Scarce (γ3)"),
        ("dist_center_km", "NB1-Wheelchair: Distance to City Center (km, γ4)"),
        ("wheelchair_accessible", "NB1-Wheelchair: Step-Free Tier 2 (with Category FE, γ5)"),
    ]:
        _add_nb_row(p_wheel, nb_wheel_cat, vk, vl)

    _add_nb_row(
        "Panel B2: NB1-Wheelchair (No-Category-FE Comparison)",
        nb_wheel_nocat,
        "wheelchair_accessible",
        "NB1-Wheelchair (No Cat FE): Step-Free Tier 2 (Cross-Category Shift)",
    )

    # Arm 3: NB1-Blank (Batch B2 Unconditioned Baseline, N=119 queries)
    p_blank = "Panel B3: NB1-Blank (Batch B2 Unconditioned Baseline, Cat + City FE)"
    for vk, vl in [
        ("log_reviews_c", "NB1-Blank: ln(1 + 2025 Pageviews) (Info-Rich Gradient γ1)"),
        ("log_reviews_c:info_scarce", "NB1-Blank: ln(1 + 2025 Pageviews) x Info-Scarce (γ3)"),
        ("dist_center_km", "NB1-Blank: Distance to City Center (km, γ4)"),
        ("wheelchair_accessible", "NB1-Blank: Step-Free Tier 2 (with Category FE, γ5)"),
    ]:
        _add_nb_row(p_blank, nb_blank_cat, vk, vl)

    _add_nb_row(
        "Panel B3: NB1-Blank (No-Category-FE Comparison)",
        nb_blank_nocat,
        "wheelchair_accessible",
        "NB1-Blank (No Cat FE): Step-Free Tier 2 (without Category FE)",
    )

    # Reference Pooled B1 Summary
    p_pool = "Panel B4: NB1-Pooled Reference (All 48 Personas, Cat + City FE)"
    for vk, vl in [
        ("log_reviews_c", "NB1-Pooled: ln(1 + 2025 Pageviews) (Info-Rich Gradient γ1)"),
        ("log_reviews_c:info_scarce", "NB1-Pooled: ln(1 + 2025 Pageviews) x Info-Scarce (γ3)"),
        ("wheelchair_accessible", "NB1-Pooled: Step-Free Tier 2 (with Category FE, γ5)"),
    ]:
        _add_nb_row(p_pool, nb_pooled_cat, vk, vl)

    # Alternative Salience Benchmarks on Clean Unrestricted Arm
    for vk, vl in [
        ("log_wiki_c", "NB2 (Wikidata Sitelinks, Unrestricted + Cat FE): ln(1 + Sitelinks) (γ1)"),
        ("log_wiki_c:info_scarce", "NB2: ln(1 + Wikidata Sitelinks) x Info-Scarce (γ3)"),
        ("dist_center_km", "NB2: Distance to City Center (km, γ4)"),
        ("wheelchair_accessible", "NB2: Step-Free Tier 2 (with Category FE, γ5)"),
    ]:
        _add_nb_row("Panel B5: NB2 (Wikidata Sitelinks, Unrestricted + Cat FE)", nb2, vk, vl)

    for vk, vl in [
        ("log_search_c", "NB3 (Wikipedia Doc Bytes, Unrestricted + Cat FE): ln(1 + Doc Bytes) (γ1)"),
        ("log_search_c:info_scarce", "NB3: ln(1 + Doc Bytes) x Info-Scarce (γ3)"),
    ]:
        _add_nb_row("Panel B6: NB3 (Wikipedia Doc Bytes, Unrestricted + Cat FE)", nb3, vk, vl)

    t5_rows.append({
        "Panel": "Panel B: Diagnostics",
        "Source / Predictor": "Dispersion α (NB1-Unres / NB1-Wheel / NB1-Blank / NB2 / NB3) & N",
        "All Cities R2 (%) / Coef": f"Unres={nb_res['alpha_unres_catfe']:.3f}, Wheel={nb_res['alpha_wheel_catfe']:.3f}",
        "Info-Rich R2 (%) / SE": f"Blank={nb_res['alpha_blank_catfe']:.3f}, NB2={nb_res['alpha_wiki']:.3f}",
        "Info-Scarce R2 (%) / IRR": f"NB3={nb_res['alpha_search']:.3f} (N={len(df_pois)})",
        "F-stat / z-stat": "-",
        "p-value": "-",
    })
    df_t5 = pd.DataFrame(t5_rows)
    df_t5.to_csv(cfg.TAB_DIR / "Table_5_PERMANOVA_and_NegativeBinomial.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "# Publication-Ready Three-Line Tables (100% Data-Driven from Live Fitted Models)\n",
        "## Table 1: Baseline Characteristics, Public Salience Gini, SDE Radii (km), and AI Concentration across 10 Cities\n",
        df_t1.to_markdown(index=False),
        "\n\n## Table 2: Audit Sampling Batches (B1-B6 & B3b), Valid JSON Rate, Out-of-Catalog Taxonomy, PABAK, and Two-Pipeline Rule Cross-Validation\n",
        df_t2.to_markdown(index=False),
        "\n\n## Table 3: Diversity, Equal-Capacity Gini, Full-OSM Gini, Unbiased Two-Stage Bootstrap NPI, NNPI, Slot Turnover, Blank Anchoring, and Discursive NPI\n",
        df_t3.to_markdown(index=False),
        "\n\n## Table 4: Multilevel Linear Mixed-Effects Regression (REML) with City-Cluster & Two-Way Dyadic SEs\n",
        df_t4.to_markdown(index=False),
        "\n\n## Table 5: Two-Way Crossed PERMANOVA (999 Permutations) and Subsample-Stratified Negative Binomial Salience Gradient Models\n",
        df_t5.to_markdown(index=False),
        "\n",
    ]
    (cfg.TAB_DIR / "All_5_Tables_Publication_Ready.md").write_text("\n".join(md_lines), encoding="utf-8")
    return {"t1": df_t1, "t2": df_t2, "t3": df_t3, "t4": df_t4, "t5": df_t5}
