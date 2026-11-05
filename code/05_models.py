# -*- coding: utf-8 -*-
"""
05_models.py — Rigorous Statistical Modeling on `v2final` Live Corpus:
  1. Multilevel Linear Mixed-Effects Models (REML MixedLM with city random intercepts),
     Hierarchical City-Cluster-Robust SEs (G = 10 cities, df = 9 small-cluster t reference,
     subsuming all within-city two-way dyadic dependence), Symmetric Two-Way Dyadic Cluster SEs
     (`list_cluster_a` + `list_cluster_b` unordered cell sharing), m = 9 BH-FDR q-values
     (strictly excluding Intercept), VIF Collinearity Diagnostics, and Orthogonalized
     `same_ecology_diff_model`.
  2. Exact QR-Projection Two-Way Crossed Gower-Centered PERMANOVA (Anderson, 2001) with n_perm = 999
     Freedman-Lane residual permutations.
  3. Negative Binomial Regression (NB2 with Cameron-Trivedi dispersion alpha) across:
       - Subsample-stratified Batch B1 models (`NB1-Unrestricted` vs. `NB1-Wheelchair`) + Blank `Batch B2`
         (`NB1-Blank`) + Pooled `Batch B1` (`NB1-Pooled`), eliminating the 50% wheelchair persona confound;
       - Alternative live Wikimedia/Wikidata public salience benchmarks (`NB2` Wikidata Sitelinks,
         `NB3` Wikipedia Documentation Byte Volume).
"""
from collections import Counter, defaultdict
import importlib
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import f as f_dist, t as t_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

cfg = importlib.import_module("00_config")


def _symmetric_twoway_dyadic_se(ols_fit, list_a: pd.Series, list_b: pd.Series) -> dict:
    """
    Cameron-Gelbach-Miller (2011) / Aronow-Samii-Assenova (2015) symmetric dyadic cluster-robust SEs
    accounting for any shared endpoint-context cell across unordered dyad sides (a_i == a_j, b_i == b_j,
    a_i == b_j, or b_i == a_j).
    """
    X = ols_fit.model.exog
    e = ols_fit.resid.values
    Xe = X * e[:, None]
    n, k = X.shape

    cell_to_rows = defaultdict(list)
    for idx, (ca, cb) in enumerate(zip(list_a.astype(str), list_b.astype(str))):
        cell_to_rows[ca].append(idx)
        if cb != ca:
            cell_to_rows[cb].append(idx)

    meat_sum = np.zeros((k, k), dtype=float)
    n_distinct_sides = np.zeros(n, dtype=float)
    for _cell, r_idxs in cell_to_rows.items():
        s = Xe[r_idxs].sum(axis=0)
        meat_sum += np.outer(s, s)
        for ri in r_idxs:
            n_distinct_sides[ri] += 1.0

    # Subtract diagonal over-counting: if pair i had 2 distinct cells (ca != cb), (i,i) was added twice;
    # if ca == cb (within-cell replicate pair), (i,i) was added once (n_distinct_sides[i] - 1 == 0).
    overcount_w = np.maximum(0.0, n_distinct_sides - 1.0)
    meat_over = (Xe * overcount_w[:, None]).T @ Xe
    meat_dyad = meat_sum - meat_over

    bread = np.linalg.inv(X.T @ X)
    dof_scale = float(n) / max(1.0, float(n - k))
    cov_dyad = bread @ meat_dyad @ bread * dof_scale
    se_vec = np.sqrt(np.maximum(1e-16, np.diag(cov_dyad)))
    return {nm: float(se_vec[j]) for j, nm in enumerate(ols_fit.model.exog_names)}


def fit_dyadic_mixed_models(df_dyads: pd.DataFrame):
    formula_rbo = (
        "rbo ~ diff_companion + diff_budget + diff_season + diff_mobility "
        "+ same_model + same_origin + home_match * info_rich_num"
    )
    formula_jac = (
        "jaccard ~ diff_companion + diff_budget + diff_season + diff_mobility "
        "+ same_model + same_origin + home_match * info_rich_num"
    )
    formula_rbo_orth = (
        "rbo ~ diff_companion + diff_budget + diff_season + diff_mobility "
        "+ same_model + same_ecology_diff_model + home_match * info_rich_num"
    )

    mod_rbo = smf.mixedlm(formula_rbo, data=df_dyads, groups=df_dyads["city_id"]).fit(reml=True, method=["powell", "lbfgs"])
    mod_jac = smf.mixedlm(formula_jac, data=df_dyads, groups=df_dyads["city_id"]).fit(reml=True, method=["powell", "lbfgs"])
    mod_rbo_orth = smf.mixedlm(formula_rbo_orth, data=df_dyads, groups=df_dyads["city_id"]).fit(reml=True, method=["powell", "lbfgs"])

    # Unclustered OLS for exact marginal composite residuals + symmetric two-way dyadic SEs
    ols_rbo_raw = smf.ols(formula_rbo, data=df_dyads).fit()
    ols_jac_raw = smf.ols(formula_jac, data=df_dyads).fit()
    ols_orth_raw = smf.ols(formula_rbo_orth, data=df_dyads).fit()

    # Hierarchical City-Cluster SE (G = 10 cities, df = 9): subsumes all within-city two-way dyadic correlation
    n_cities = int(df_dyads["city_id"].nunique())
    df_city = max(1, n_cities - 1)  # df = 9
    ols_rbo_city = smf.ols(formula_rbo, data=df_dyads).fit(
        cov_type="cluster", cov_kwds={"groups": df_dyads["city_id"]}
    )
    ols_jac_city = smf.ols(formula_jac, data=df_dyads).fit(
        cov_type="cluster", cov_kwds={"groups": df_dyads["city_id"]}
    )
    ols_orth_city = smf.ols(formula_rbo_orth, data=df_dyads).fit(
        cov_type="cluster", cov_kwds={"groups": df_dyads["city_id"]}
    )

    rbo_dyad_se = _symmetric_twoway_dyadic_se(ols_rbo_raw, df_dyads["list_cluster_a"], df_dyads["list_cluster_b"])
    jac_dyad_se = _symmetric_twoway_dyadic_se(ols_jac_raw, df_dyads["list_cluster_a"], df_dyads["list_cluster_b"])
    orth_dyad_se = _symmetric_twoway_dyadic_se(ols_orth_raw, df_dyads["list_cluster_a"], df_dyads["list_cluster_b"])

    df_x = df_dyads[[
        "diff_companion", "diff_budget", "diff_season", "diff_mobility",
        "same_model", "same_origin", "home_match", "info_rich_num"
    ]].copy()
    df_x["home_x_rich"] = df_x["home_match"] * df_x["info_rich_num"]
    df_x_const = sm.add_constant(df_x)
    vif_dict = {}
    for idx_col, col_name in enumerate(df_x_const.columns):
        if col_name != "const":
            vif_dict[col_name] = round(float(variance_inflation_factor(df_x_const.values, idx_col)), 3)

    corr_sm_so = float(np.corrcoef(df_dyads["same_model"].values, df_dyads["same_origin"].values)[0, 1])

    def compute_mixed_r2_icc(res):
        fe_names = [k for k in res.fe_params.index]
        xb = np.zeros(len(df_dyads), dtype=float)
        for k in fe_names:
            if k == "Intercept":
                xb += float(res.fe_params[k])
            elif ":" in k:
                c1, c2 = k.split(":")
                xb += float(res.fe_params[k]) * (df_dyads[c1].values * df_dyads[c2].values)
            else:
                xb += float(res.fe_params[k]) * df_dyads[k].values
        var_f = float(np.var(xb))
        var_u = max(1e-8, float(res.cov_re.iloc[0, 0]))
        var_e = max(1e-8, float(res.scale))
        var_tot = var_f + var_u + var_e
        return var_f / var_tot, (var_f + var_u) / var_tot, var_u / (var_u + var_e)

    r2m_rbo, r2c_rbo, icc_rbo = compute_mixed_r2_icc(mod_rbo)
    r2m_jac, r2c_jac, icc_jac = compute_mixed_r2_icc(mod_jac)

    met_mod = importlib.import_module("04_metrics")

    def _build_inf_dicts(mod_reml, ols_city):
        fe_keys = [k for k in mod_reml.fe_params.index if k in ols_city.bse]
        cl_se = {k: float(ols_city.bse[k]) for k in fe_keys}
        cl_t = {k: float(mod_reml.fe_params[k]) / cl_se[k] for k in fe_keys}
        cl_p = {k: float(2.0 * t_dist.sf(abs(cl_t[k]), df=df_city)) for k in fe_keys}
        non_int_keys = [k for k in fe_keys if k != "Intercept"]
        assert len(non_int_keys) == 9, f"Expected m=9 non-intercept predictors for BH-FDR, got {len(non_int_keys)}"
        q_arr = met_mod.benjamini_hochberg_fdr(np.array([cl_p[k] for k in non_int_keys]))
        cl_q = {"Intercept": np.nan}
        for k, qv in zip(non_int_keys, q_arr):
            cl_q[k] = float(qv)
        return cl_se, cl_t, cl_p, cl_q

    rbo_cl_se, rbo_cl_z, rbo_cl_p, rbo_cl_q = _build_inf_dicts(mod_rbo, ols_rbo_city)
    jac_cl_se, jac_cl_z, jac_cl_p, jac_cl_q = _build_inf_dicts(mod_jac, ols_jac_city)
    orth_cl_se, orth_cl_z, orth_cl_p, _ = _build_inf_dicts(mod_rbo_orth, ols_orth_city)

    return {
        "rbo": mod_rbo,
        "jaccard": mod_jac,
        "rbo_orth": mod_rbo_orth,
        "city_df": df_city,
        "t_crit_95": float(t_dist.ppf(0.975, df=df_city)),
        "rbo_cluster_se": rbo_cl_se,
        "rbo_dyad_se": rbo_dyad_se,
        "rbo_cluster_z": rbo_cl_z,
        "rbo_cluster_p": rbo_cl_p,
        "rbo_cluster_q": rbo_cl_q,
        "jac_cluster_se": jac_cl_se,
        "jac_dyad_se": jac_dyad_se,
        "jac_cluster_z": jac_cl_z,
        "jac_cluster_p": jac_cl_p,
        "jac_cluster_q": jac_cl_q,
        "orth_cluster_se": orth_cl_se,
        "orth_dyad_se": orth_dyad_se,
        "orth_cluster_z": orth_cl_z,
        "orth_cluster_p": orth_cl_p,
        "vif": vif_dict,
        "corr_same_model_origin": round(corr_sm_so, 4),
        "rbo_stats": {"r2_marginal": r2m_rbo, "r2_conditional": r2c_rbo, "icc": icc_rbo},
        "jac_stats": {"r2_marginal": r2m_jac, "r2_conditional": r2c_jac, "icc": icc_jac},
    }


def _orthonormal_basis(design_mat: np.ndarray) -> np.ndarray:
    """Returns thin orthonormal column basis Q (N x rank) via SVD with rank tolerance."""
    u, s, _vt = np.linalg.svd(design_mat, full_matrices=False)
    rank = int(np.sum(s > 1e-9))
    return u[:, :rank]


def run_two_way_crossed_permanova(df_queries: pd.DataFrame, n_perm: int = 999, save_filename: str = "permanova_2way_crossed_10cities.csv"):
    """
    Exact Anderson (2001) Two-Way Crossed PERMANOVA via Orthonormal Subspace Projections
    on Gower's centered inner-product matrix `G = -0.5 * H * D^2 * H` (`n_perm = 999`).
    """
    rng = cfg.get_rng(cfg.GLOBAL_SEED + 505)
    df_main = df_queries[df_queries["batch_id"] == "B1"].copy()
    city_permanova_rows = []

    for c in cfg.CITIES_META:
        cid = c["city_id"]
        sub = df_main[df_main["city_id"] == cid].sort_values(
            ["model", "context_id", "rep_id"]
        ).reset_index(drop=True)

        lists = [set(str(s).split("|")) for s in sub["top10_poi_ids"]]
        N = len(lists)
        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            si = lists[i]
            for j in range(i + 1, N):
                sj = lists[j]
                u = len(si | sj)
                D[i, j] = D[j, i] = 1.0 - (len(si & sj) / u if u > 0 else 0.0)

        D2 = D ** 2
        H = np.eye(N) - np.ones((N, N), dtype=float) / N
        G = -0.5 * (H @ D2 @ H)
        ss_total = float(np.trace(G))

        m_lbls = sub["model"].values
        c_lbls = sub["context_id"].values
        cell_lbls = np.array([f"{m}__{cx}" for m, cx in zip(m_lbls, c_lbls)])

        uniq_m = np.unique(m_lbls)
        uniq_c = np.unique(c_lbls)
        uniq_cell = np.unique(cell_lbls)

        X_m = H @ (m_lbls[:, None] == uniq_m[None, :]).astype(float)
        X_c = H @ (c_lbls[:, None] == uniq_c[None, :]).astype(float)
        X_mc = H @ np.hstack([X_m, X_c])
        X_cell = H @ (cell_lbls[:, None] == uniq_cell[None, :]).astype(float)

        Q_m = _orthonormal_basis(X_m)
        Q_c = _orthonormal_basis(X_c)
        Q_mc = _orthonormal_basis(X_mc)
        Q_cell = _orthonormal_basis(X_cell)

        ss_model = float(np.sum(Q_m * (G @ Q_m)))
        ss_mc = float(np.sum(Q_mc * (G @ Q_mc)))
        ss_context = max(0.0, ss_mc - ss_model)
        ss_cells = float(np.sum(Q_cell * (G @ Q_cell)))
        ss_inter = max(0.0, ss_cells - ss_mc)
        ss_resid = max(1e-9, ss_total - ss_cells)

        df_m = Q_m.shape[1]
        df_c = Q_mc.shape[1] - Q_m.shape[1]
        df_i = Q_cell.shape[1] - Q_mc.shape[1]
        df_r = N - 1 - Q_cell.shape[1]

        ms_m = ss_model / df_m
        ms_c = ss_context / df_c
        ms_i = ss_inter / df_i
        ms_r = ss_resid / df_r

        f_m = ms_m / ms_r
        f_c = ms_c / ms_r
        f_i = ms_i / ms_r

        P_c = Q_c @ Q_c.T
        P_m = Q_m @ Q_m.T
        P_mc = Q_mc @ Q_mc.T

        G_noC = (np.eye(N) - P_c) @ G @ (np.eye(N) - P_c)
        G_noM = (np.eye(N) - P_m) @ G @ (np.eye(N) - P_m)
        G_red = (np.eye(N) - P_mc) @ G @ (np.eye(N) - P_mc)

        ss_noC_tot = float(np.trace(G_noC))
        ss_noM_tot = float(np.trace(G_noM))
        ss_red_tot = float(np.trace(G_red))

        cnt_m, cnt_c, cnt_i = 0, 0, 0
        for _ in range(n_perm):
            p_idx = rng.permutation(N)
            Qm_p = Q_m[p_idx]
            Qc_p = Q_c[p_idx]
            Qmc_p = Q_mc[p_idx]
            Qcell_p = Q_cell[p_idx]

            pm_ss = float(np.sum(Qm_p * (G_noC @ Qm_p)))
            pcell_noC = float(np.sum(Qcell_p * (G_noC @ Qcell_p)))
            pm_res = max(1e-9, ss_noC_tot - pcell_noC)
            if (pm_ss / df_m) / (pm_res / df_r) >= f_m:
                cnt_m += 1

            pc_ss = float(np.sum(Qc_p * (G_noM @ Qc_p)))
            pcell_noM = float(np.sum(Qcell_p * (G_noM @ Qcell_p)))
            pc_res = max(1e-9, ss_noM_tot - pcell_noM)
            if (pc_ss / df_c) / (pc_res / df_r) >= f_c:
                cnt_c += 1

            pcell_red = float(np.sum(Qcell_p * (G_red @ Qcell_p)))
            pmc_red = float(np.sum(Qmc_p * (G_red @ Qmc_p)))
            pi_ss = max(0.0, pcell_red - pmc_red)
            pi_res = max(1e-9, ss_red_tot - pcell_red)
            if (pi_ss / df_i) / (pi_res / df_r) >= f_i:
                cnt_i += 1

        p_m = (cnt_m + 1.0) / (n_perm + 1.0)
        p_c = (cnt_c + 1.0) / (n_perm + 1.0)
        p_i = (cnt_i + 1.0) / (n_perm + 1.0)

        r2_m = ss_model / ss_total
        r2_c = ss_context / ss_total
        r2_i = ss_inter / ss_total
        r2_r = ss_resid / ss_total
        assert abs((r2_m + r2_c + r2_i + r2_r) - 1.0) < 1e-6, f"PERMANOVA sum != 1.0 in {cid}"

        city_permanova_rows.append({
            "city_id": cid,
            "city": c["city"],
            "origin": c["origin"],
            "richness": c["richness"],
            "N_obs": N,
            "n_perm": n_perm,
            "R2_Model_pct": r2_m * 100.0,
            "R2_Context_pct": r2_c * 100.0,
            "R2_Interaction_pct": r2_i * 100.0,
            "R2_Residual_pct": r2_r * 100.0,
            "MS_Model_pct": (r2_m * 100.0) / df_m,
            "MS_Context_pct": (r2_c * 100.0) / df_c,
            "F_Model": f_m,
            "F_Context": f_c,
            "F_Model_to_Context_Ratio": f_m / max(1e-9, f_c),
            "F_Interaction": f_i,
            "p_Model": p_m,
            "p_Context": p_c,
            "p_Interaction": p_i,
            "p_Model_Parametric": float(f_dist.sf(f_m, df_m, df_r)),
            "p_Context_Parametric": float(f_dist.sf(f_c, df_c, df_r)),
            "p_Interaction_Parametric": float(f_dist.sf(f_i, df_i, df_r)),
            "df_Model": df_m,
            "df_Context": df_c,
            "df_Interaction": df_i,
            "df_Residual": df_r,
        })

    df_perm = pd.DataFrame(city_permanova_rows)
    df_perm.to_csv(cfg.DATA_DIR / save_filename, index=False, encoding="utf-8-sig")
    if save_filename == "permanova_2way_crossed_10cities.csv":
        q_noflash = df_queries[df_queries["model"] != "qwen3.7-flash"].copy()
        if q_noflash["model"].nunique() == 3:
            run_two_way_crossed_permanova(
                q_noflash, n_perm=n_perm, save_filename="permanova_3endpoints_noflash.csv"
            )
    return df_perm


def fit_negative_binomial_models(df_pois: pd.DataFrame, df_queries: pd.DataFrame = None):
    """
    Fits Negative Binomial (NB2) models on POI recommendation counts (`N = 600`) across:
      1. Persona-mobility disaggregated arms (`mentions_b1_unres` [24 non-wheelchair personas, N=2,874],
         `mentions_b1_wheel` [24 wheelchair personas, N=2,878], and `mentions_b2_blank` [unconditioned
         blank prompts, N=119]) alongside pooled Batch B1 (`total_ai_mentions`, N=5,752);
      2. Three live-fetched Wikimedia & Wikidata public digital-salience benchmarks (`wiki_views_annual`,
         `wikidata_sitelinks`, `wiki_doc_bytes`), globally grand-mean centered across the 600 POIs.
    """
    if df_queries is None:
        df_queries = pd.read_csv(cfg.DATA_DIR / "audit_queries_corpus.csv")

    df = df_pois.copy()
    valid_pids = set(df["poi_id"].astype(str))

    def _count_mentions(q_sub: pd.DataFrame) -> list[int]:
        c = Counter()
        for s in q_sub["top10_poi_ids"]:
            for p in str(s).split("|"):
                if p in valid_pids:
                    c[p] += 1
        return [int(c.get(str(pid), 0)) for pid in df["poi_id"]]

    q_b1 = df_queries[df_queries["batch_id"] == "B1"]
    q_b2 = df_queries[df_queries["batch_id"] == "B2"]
    df["mentions_b1_unres"] = _count_mentions(q_b1[q_b1["mobility"] == "Unrestricted"])
    df["mentions_b1_wheel"] = _count_mentions(q_b1[q_b1["mobility"] == "Wheelchair"])
    df["mentions_b2_blank"] = _count_mentions(q_b2)

    df["log_reviews"] = np.log1p(np.maximum(0.0, df["wiki_views_annual"].values.astype(float)))
    df["log_reviews_c"] = df["log_reviews"] - df["log_reviews"].mean()

    df["log_wiki"] = np.log1p(np.maximum(0.0, df["wikidata_sitelinks"].values.astype(float)))
    df["log_wiki_c"] = df["log_wiki"] - df["log_wiki"].mean()

    df["log_search"] = np.log1p(np.maximum(0.0, df["wiki_doc_bytes"].values.astype(float)))
    df["log_search_c"] = df["log_search"] - df["log_search"].mean()

    df["info_scarce"] = (df["richness"] == "Info-Scarce").astype(int)

    def _fit_nb(dep_col: str, rhs_str: str):
        formula_str = f"{dep_col} ~ {rhs_str}"
        pois_m = smf.glm(formula_str, data=df, family=sm.families.Poisson()).fit()
        mu = np.maximum(1e-3, pois_m.mu)
        y = df[dep_col].values.astype(float)
        alpha_est = float(np.clip(np.sum((y - mu) ** 2 - y) / max(1e-6, np.sum(mu ** 2)), 0.15, 3.5))
        nb_m = smf.glm(
            formula_str, data=df, family=sm.families.NegativeBinomial(alpha=alpha_est)
        ).fit(start_params=pois_m.params, method="lbfgs", maxiter=600)
        return nb_m, alpha_est

    rhs_rev_nocat = "log_reviews_c + log_reviews_c:info_scarce + dist_center_km + wheelchair_accessible + C(city_id)"
    rhs_rev_cat = "log_reviews_c + log_reviews_c:info_scarce + dist_center_km + wheelchair_accessible + C(category) + C(city_id)"
    rhs_rev_strat = "log_reviews_c * info_scarce + dist_center_km + wheelchair_accessible + C(category)"
    rhs_wiki_cat = "log_wiki_c + log_wiki_c:info_scarce + dist_center_km + wheelchair_accessible + C(category) + C(city_id)"
    rhs_bytes_cat = "log_search_c + log_search_c:info_scarce + dist_center_km + wheelchair_accessible + C(category) + C(city_id)"

    # Pooled B1 (backward-compatible keys)
    nb_mod1, alpha1 = _fit_nb("total_ai_mentions", rhs_rev_nocat)
    nb_mod1_cat, alpha1_cat = _fit_nb("total_ai_mentions", rhs_rev_cat)
    nb_mod1_stratum, _ = _fit_nb("total_ai_mentions", rhs_rev_strat)

    # Disaggregated mobility subsamples + Blank B2 baseline (Must-Fix #6)
    nb_unres_nocat, alpha_unres_nocat = _fit_nb("mentions_b1_unres", rhs_rev_nocat)
    nb_unres_cat, alpha_unres_cat = _fit_nb("mentions_b1_unres", rhs_rev_cat)
    nb_unres_strat, _ = _fit_nb("mentions_b1_unres", rhs_rev_strat)

    nb_wheel_nocat, alpha_wheel_nocat = _fit_nb("mentions_b1_wheel", rhs_rev_nocat)
    nb_wheel_cat, alpha_wheel_cat = _fit_nb("mentions_b1_wheel", rhs_rev_cat)

    nb_blank_nocat, alpha_blank_nocat = _fit_nb("mentions_b2_blank", rhs_rev_nocat)
    nb_blank_cat, alpha_blank_cat = _fit_nb("mentions_b2_blank", rhs_rev_cat)

    # Alternative salience benchmarks on clean Unrestricted arm (and Pooled check)
    nb_mod2_unres, alpha2_unres = _fit_nb("mentions_b1_unres", rhs_wiki_cat)
    nb_mod3_unres, alpha3_unres = _fit_nb("mentions_b1_unres", rhs_bytes_cat)
    nb_mod2_pooled, alpha2_pooled = _fit_nb("total_ai_mentions", rhs_wiki_cat)
    nb_mod3_pooled, alpha3_pooled = _fit_nb("total_ai_mentions", rhs_bytes_cat)

    return {
        "nb_reviews": nb_mod1,
        "nb_reviews_catfe": nb_mod1_cat,
        "nb_reviews_stratum": nb_mod1_stratum,
        "nb_unres_nocatfe": nb_unres_nocat,
        "nb_unres_catfe": nb_unres_cat,
        "nb_unres_stratum": nb_unres_strat,
        "nb_wheel_nocatfe": nb_wheel_nocat,
        "nb_wheel_catfe": nb_wheel_cat,
        "nb_blank_nocatfe": nb_blank_nocat,
        "nb_blank_catfe": nb_blank_cat,
        "nb_wiki": nb_mod2_unres,
        "nb_search": nb_mod3_unres,
        "nb_wiki_pooled": nb_mod2_pooled,
        "nb_search_pooled": nb_mod3_pooled,
        "alpha_reviews": alpha1,
        "alpha_reviews_catfe": alpha1_cat,
        "alpha_unres_catfe": alpha_unres_cat,
        "alpha_wheel_catfe": alpha_wheel_cat,
        "alpha_blank_catfe": alpha_blank_cat,
        "alpha_wiki": alpha2_unres,
        "alpha_search": alpha3_unres,
        "df_encoded": df,
    }
