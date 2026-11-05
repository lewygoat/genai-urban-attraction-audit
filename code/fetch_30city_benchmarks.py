# -*- coding: utf-8 -*-
"""
fetch_30city_benchmarks.py — Live Open-API Stratification Benchmark Fetcher for the 30-City Pool.
Queries MediaWiki/Wikidata APIs and the Wikimedia REST v1 Pageviews API (20250101–20251231)
across all 30 candidate cities (15 Domestic Chinese + 15 International) to compute verifiable
within-origin standardized digital representation scores `composite_Z_info`.

Outputs:
  - `data/wikimedia_30cities_stratification.json` (complete API URLs, QIDs, and timestamps)
  - `data/candidate_cities_30pool_distribution.csv` (30-city stratification table)
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import json
import time
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

cfg = importlib.import_module("00_config")

CANDIDATE_30_POOL = [
    # 15 Domestic Chinese Cities
    ("BJS", "Beijing",    "China", "Domestic", 4820, True,  "Info-Rich"),
    ("SHA", "Shanghai",   "China", "Domestic", 4310, True,  "Info-Rich"),
    ("CTU", "Chengdu",    "China", "Domestic", 3190, True,  "Info-Rich"),
    ("CAN", "Guangzhou",  "China", "Domestic", 3580, False, ""),
    ("HGH", "Hangzhou",   "China", "Domestic", 3490, False, ""),
    ("SIA", "Xi'an",      "China", "Domestic", 3120, False, ""),
    ("CKG", "Chongqing",  "China", "Domestic", 2980, False, ""),
    ("NKG", "Nanjing",    "China", "Domestic", 2490, False, ""),
    ("WUH", "Wuhan",      "China", "Domestic", 2240, False, ""),
    ("XMN", "Xiamen",     "China", "Domestic", 1850, False, ""),
    ("QZJ", "Quanzhou",   "China", "Domestic",  680, True,  "Info-Scarce"),
    ("DLU", "Dali",       "China", "Domestic",  540, True,  "Info-Scarce"),
    ("CHZ", "Chaozhou",   "China", "Domestic",  590, False, ""),
    ("JDZ", "Jingdezhen", "China", "Domestic",  510, False, ""),
    ("DTG", "Datong",     "China", "Domestic",  620, False, ""),
    # 15 International Cities
    ("PAR", "Paris",      "France",         "International", 6450, True,  "Info-Rich"),
    ("TYO", "Tokyo",      "Japan",          "International", 5890, True,  "Info-Rich"),
    ("BCN", "Barcelona",  "Spain",          "International", 4120, True,  "Info-Rich"),
    ("LON", "London",     "United Kingdom", "International", 6120, False, ""),
    ("NYC", "New York",   "United States",  "International", 5520, False, ""),
    ("ROM", "Rome",       "Italy",          "International", 4980, False, ""),
    ("SIN", "Singapore",  "Singapore",      "International", 3820, False, ""),
    ("SEL", "Seoul",      "South Korea",    "International", 3250, False, ""),
    ("VIE", "Vienna",     "Austria",        "International", 2720, False, ""),
    ("PRG", "Prague",     "Czechia",        "International", 2440, False, ""),
    ("OPO", "Porto",      "Portugal",       "International",  920, True,  "Info-Scarce"),
    ("CNX", "Chiang Mai", "Thailand",       "International",  810, True,  "Info-Scarce"),
    ("KYO", "Kyoto",      "Japan",          "International", 1180, False, ""),
    ("BRU", "Bruges",     "Belgium",        "International",  790, False, ""),
    ("GRX", "Granada",    "Spain",          "International",  860, False, ""),
]

CITY_WIKI_MAP = {
    "BJS": ("Beijing", "zh", "北京市"),
    "SHA": ("Shanghai", "zh", "上海市"),
    "CTU": ("Chengdu", "zh", "成都市"),
    "CAN": ("Guangzhou", "zh", "广州市"),
    "HGH": ("Hangzhou", "zh", "杭州市"),
    "SIA": ("Xi'an", "zh", "西安市"),
    "CKG": ("Chongqing", "zh", "重庆市"),
    "NKG": ("Nanjing", "zh", "南京市"),
    "WUH": ("Wuhan", "zh", "武汉市"),
    "XMN": ("Xiamen", "zh", "厦门市"),
    "QZJ": ("Quanzhou", "zh", "泉州市"),
    "DLU": ("Dali City", "zh", "大理市"),
    "CHZ": ("Chaozhou", "zh", "潮州市"),
    "JDZ": ("Jingdezhen", "zh", "景德镇市"),
    "DTG": ("Datong", "zh", "大同市"),
    "PAR": ("Paris", "fr", "Paris"),
    "TYO": ("Tokyo", "ja", "東京都"),
    "BCN": ("Barcelona", "es", "Barcelona"),
    "LON": ("London", "en", ""),
    "NYC": ("New York City", "en", ""),
    "ROM": ("Rome", "it", "Roma"),
    "SIN": ("Singapore", "zh", "新加坡"),
    "SEL": ("Seoul", "ko", "서울특별시"),
    "VIE": ("Vienna", "de", "Wien"),
    "PRG": ("Prague", "cs", "Praha"),
    "OPO": ("Porto", "pt", "Porto"),
    "CNX": ("Chiang Mai", "th", "เทศบาลนครเชียงใหม่"),
    "KYO": ("Kyoto", "ja", "京都市"),
    "BRU": ("Bruges", "nl", "Brugge"),
    "GRX": ("Granada", "es", "Granada"),
}

HEADERS = {
    "User-Agent": "UrbanTourismGeoAudit/2.0 (https://www.wikidata.org) Python-urllib/3.12",
    "Accept": "application/json",
}


def _get_json(url: str) -> dict:
    for att in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            time.sleep(0.8 * (att + 1))
    return {}


def _fetch_city_record(entry: tuple) -> dict:
    cid, city_name, country, origin, osm_pois, is_sel, sel_tier = entry
    en_t, nat_lang, nat_t = CITY_WIKI_MAP[cid]
    pm = urllib.parse.urlencode({
        "action": "query",
        "prop": "pageprops|langlinks|info",
        "ppprop": "wikibase_item",
        "lllimit": "500",
        "redirects": "1",
        "titles": en_t,
        "format": "json",
    })
    mw_url = f"https://en.wikipedia.org/w/api.php?{pm}"
    dm = _get_json(mw_url)
    qid = ""
    sitelinks = 0
    en_bytes = 0
    en_resolved = en_t
    for pgid, pg in dm.get("query", {}).get("pages", {}).items():
        if str(pgid) != "-1":
            en_resolved = pg.get("title", en_t)
            qid = pg.get("pageprops", {}).get("wikibase_item", "")
            sitelinks = 1 + len(pg.get("langlinks", []))
            en_bytes = int(pg.get("length", 0))

    enc_en = urllib.parse.quote(en_resolved.replace(" ", "_"), safe="")
    u_en = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/user/{enc_en}/monthly/2025010100/2025123100"
    )
    d_en = _get_json(u_en)
    v_en = sum(int(x.get("views", 0)) for x in d_en.get("items", []))

    v_nat = 0
    u_nat = ""
    if nat_t and nat_lang != "en":
        enc_nat = urllib.parse.quote(nat_t.replace(" ", "_"), safe="")
        u_nat = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{nat_lang}.wikipedia/all-access/user/{enc_nat}/monthly/2025010100/2025123100"
        )
        d_nat = _get_json(u_nat)
        v_nat = sum(int(x.get("views", 0)) for x in d_nat.get("items", []))

    v_tot = int(v_en + v_nat)
    return {
        "city_id": cid,
        "city": city_name,
        "country": country,
        "origin": origin,
        "osm_pois": int(osm_pois),
        "is_selected": bool(is_sel),
        "selected_stratum": sel_tier,
        "wikidata_qid": qid,
        "enwiki_title": en_resolved,
        "native_lang": nat_lang,
        "native_wiki_title": nat_t,
        "en_views_2025": int(v_en),
        "native_views_2025": int(v_nat),
        "wiki_views_2025_total": v_tot,
        "wiki_views_k": round(v_tot / 1000.0, 1),
        "wikidata_sitelinks": int(sitelinks),
        "en_doc_bytes": int(en_bytes),
        "mediawiki_api_url": mw_url,
        "en_pageviews_api_url": u_en,
        "native_pageviews_api_url": u_nat,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    with ThreadPoolExecutor(max_workers=5) as ex:
        recs = list(ex.map(_fetch_city_record, CANDIDATE_30_POOL))

    df = pd.DataFrame(recs)
    for orig, idxs in df.groupby("origin").groups.items():
        sub = df.loc[idxs]
        l_osm = np.log(sub["osm_pois"].values.astype(float))
        l_vw = np.log(sub["wiki_views_2025_total"].values.astype(float))
        l_sl = np.log(sub["wikidata_sitelinks"].values.astype(float))
        z_osm = (l_osm - np.mean(l_osm)) / np.std(l_osm, ddof=0)
        z_vw = (l_vw - np.mean(l_vw)) / np.std(l_vw, ddof=0)
        z_sl = (l_sl - np.mean(l_sl)) / np.std(l_sl, ddof=0)
        z_comp = (z_osm + z_vw + z_sl) / 3.0
        pct = pd.Series(z_comp, index=idxs).rank(method="min") / float(len(sub)) * 100.0
        df.loc[idxs, "composite_Z_info"] = np.round(z_comp, 3)
        df.loc[idxs, "within_origin_percentile"] = np.round(pct.values, 1)

    statuses = []
    for _, r in df.iterrows():
        if r["is_selected"]:
            statuses.append(f"Selected: {r['selected_stratum']}")
        else:
            p = float(r["within_origin_percentile"])
            if p >= 60.0:
                statuses.append("Unselected Candidate (Rich Stratum)")
            elif p <= 33.4:
                statuses.append("Unselected Candidate (Scarce Stratum)")
            else:
                statuses.append("Unselected Candidate (Middle Buffer)")
    df["selection_status"] = statuses

    json_payload = {
        "description": "Live 2025 Wikimedia bilingual pageviews and Wikidata cross-lingual sitelinks for the 30-city stratification pool",
        "formula": "composite_Z_info = (Z_ln(osm_pois) + Z_ln(wiki_views_2025_total) + Z_ln(wikidata_sitelinks)) / 3 within each 15-city origin bloc",
        "records": df.to_dict(orient="records"),
    }
    (cfg.DATA_DIR / "wikimedia_30cities_stratification.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_cols = [
        "city_id", "city", "country", "origin", "osm_pois",
        "wiki_views_k", "wikidata_sitelinks", "selection_status",
        "composite_Z_info", "within_origin_percentile",
    ]
    df_csv = df[csv_cols].copy()
    df_csv.to_csv(cfg.DATA_DIR / "candidate_cities_30pool_distribution.csv", index=False, encoding="utf-8-sig")
    print(df_csv.to_string(index=False))


if __name__ == "__main__":
    main()
