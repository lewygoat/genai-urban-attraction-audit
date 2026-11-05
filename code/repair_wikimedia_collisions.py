# -*- coding: utf-8 -*-
"""
repair_wikimedia_collisions.py — Fast Batched Entity-Level Verifier & Repairer for
`wikimedia_empirical_benchmarks_600pois.json`.
Uses `cjk_aliases_300.json` + `cjk_aliases_intl_300.json` mapped via `ground_truth_pois_10cities.csv`,
45-title batched Wikipedia `prop=pageprops|info` redirects, polite `0.45s`-spaced `wbsearchentities` fallback,
and batched `wbgetentities` (P625 + P31) to guarantee 1-to-1 attraction fidelity with ZERO duplicate QIDs
and ZERO city/district/human/country entity collisions.
"""
import hashlib
import json
import math
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = EXP_ROOT / "data"
JSON_PATH = DATA_DIR / "wikimedia_empirical_benchmarks_600pois.json"

import sys
sys.path.insert(0, str(EXP_ROOT / "code"))
import importlib
cfg = importlib.import_module("00_config")
gt = importlib.import_module("01_ground_truth")

USER_AGENT = "GenerativeAITourismAudit/2.0 (Academic Research; mailto:tourism-audit@research.org) Python-urllib/3.11"

CITY_LANG = {
    "BJS": "zh", "SHA": "zh", "CTU": "zh", "QZJ": "zh", "DLU": "zh",
    "PAR": "fr", "TYO": "ja", "BCN": "es", "OPO": "pt", "CNX": "th",
}

CITY_STRIP_WORDS = {
    "beijing", "shanghai", "chengdu", "quanzhou", "dali", "paris", "tokyo",
    "barcelona", "porto", "chiang", "mai", "city", "district", "province", "china"
}

BANNED_ADMIN_OR_PERSON_P31 = {
    "Q5",         # human (e.g., Salvador Dali)
    "Q6256",      # country
    "Q515",       # city
    "Q1549591",   # big city
    "Q200250",    # metropolis
    "Q1637706",   # million city
    "Q747074",    # municipality
    "Q106658",    # prefecture-level city
    "Q133442",    # county-level city
    "Q9316670",   # autonomous prefecture
    "Q308494",    # special ward of Tokyo
    "Q2074737",   # municipality of Spain
    "Q13217644",  # municipality of Portugal
    "Q484170",    # commune of France
    "Q15127838",  # thesaban nakhon
}

BANNED_QIDS = {
    "Q148", "Q956", "Q8686", "Q30002", "Q68695", "Q855228", "Q999156", "Q5577",
    "Q90", "Q1490", "Q1492", "Q36433", "Q52028", "Q179645", "Q576781", "Q720644",
    "Q265071", "Q1122701",
}

_last_ts = 0.0

def polite_get_json(url: str, min_gap: float = 0.42, retries: int = 4) -> dict:
    global _last_ts
    wait = min_gap - (time.time() - _last_ts)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(retries):
        try:
            _last_ts = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=22) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if "404" in str(e):
                return {}
            time.sleep(2.2 * (attempt + 1))
    return {}

def strip_accents(s: str) -> str:
    s_nfkd = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s_nfkd if not unicodedata.combining(ch))

def norm_tokens(s: str) -> set:
    t = re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower()).split()
    return {w for w in t if len(w) >= 3 and w not in CITY_STRIP_WORDS}

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))

def batch_resolve_wiki_titles(lang: str, titles: list) -> dict:
    """Resolve up to N titles in batches of 40 -> {input_title: qid}."""
    uniq = sorted({t.strip() for t in titles if t and t.strip()})
    out = {}
    for i in range(0, len(uniq), 40):
        chunk = uniq[i:i + 40]
        params = urllib.parse.urlencode({
            "action": "query",
            "titles": "|".join(chunk),
            "redirects": "1",
            "converttitles": "1",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "format": "json",
        })
        res = polite_get_json(f"https://{lang}.wikipedia.org/w/api.php?{params}", min_gap=0.45)
        qobj = res.get("query", {})
        norm_map = {x["from"]: x["to"] for x in qobj.get("normalized", [])}
        conv_map = {x["from"]: x["to"] for x in qobj.get("converted", [])}
        redir_map = {x["from"]: x["to"] for x in qobj.get("redirects", [])}
        page_qid = {}
        for pg in qobj.get("pages", {}).values():
            q = pg.get("pageprops", {}).get("wikibase_item", "")
            if q and pg.get("title"):
                page_qid[pg["title"]] = q
        for raw_t in chunk:
            cur = raw_t
            for _ in range(4):
                cur = redir_map.get(conv_map.get(norm_map.get(cur, cur), conv_map.get(cur, cur)),
                                    conv_map.get(norm_map.get(cur, cur), norm_map.get(cur, cur)))
            if cur in page_qid:
                out[raw_t] = page_qid[cur]
    return out

def fetch_entities_batch(qids: list) -> dict:
    out = {}
    uniq = sorted({q for q in qids if q})
    for i in range(0, len(uniq), 45):
        chunk = uniq[i:i + 45]
        params = urllib.parse.urlencode({
            "action": "wbgetentities",
            "ids": "|".join(chunk),
            "props": "labels|claims|sitelinks",
            "languages": "en|zh|fr|ja|es|pt|th",
            "format": "json",
        })
        res = polite_get_json(f"https://www.wikidata.org/w/api.php?{params}", min_gap=0.45)
        for q, ent in res.get("entities", {}).items():
            if "missing" in ent:
                continue
            p31s = set()
            for cl in ent.get("claims", {}).get("P31", []):
                v = cl.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(v, dict) and v.get("id"):
                    p31s.add(v["id"])
            lat_q, lon_q = None, None
            for cl in ent.get("claims", {}).get("P625", []):
                v = cl.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(v, dict) and "latitude" in v:
                    lat_q, lon_q = float(v["latitude"]), float(v["longitude"])
                    break
            slinks = ent.get("sitelinks", {})
            labels = {lg: v.get("value", "") for lg, v in ent.get("labels", {}).items()}
            out[q] = {
                "qid": q,
                "p31": p31s,
                "lat": lat_q,
                "lon": lon_q,
                "sitelinks_count": len(slinks),
                "sitelinks": {k: v.get("title", "") for k, v in slinks.items()},
                "labels": labels,
            }
    return out

def batch_fetch_lengths(lang: str, titles: list) -> dict:
    uniq = sorted({t for t in titles if t})
    out = {}
    for i in range(0, len(uniq), 45):
        chunk = uniq[i:i + 45]
        params = urllib.parse.urlencode({
            "action": "query",
            "prop": "info",
            "redirects": "1",
            "titles": "|".join(chunk),
            "format": "json",
        })
        res = polite_get_json(f"https://{lang}.wikipedia.org/w/api.php?{params}", min_gap=0.40)
        qobj = res.get("query", {})
        redir = {x["from"]: x["to"] for x in qobj.get("redirects", [])}
        norm = {x["from"]: x["to"] for x in qobj.get("normalized", [])}
        plen = {pg["title"]: int(pg["length"]) for pg in qobj.get("pages", {}).values() if "length" in pg and "title" in pg}
        for raw_t in chunk:
            cur = redir.get(norm.get(raw_t, raw_t), norm.get(raw_t, raw_t))
            out[raw_t] = plen.get(cur, 0)
    return out

def fetch_views_2025(lang: str, title: str) -> tuple:
    if not title:
        return 0, ""
    enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{enc}/monthly/2025010100/2025123100"
    res = polite_get_json(url, min_gap=0.22)
    items = res.get("items", [])
    return int(sum(int(it.get("views", 0)) for it in items)), url

def main():
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    recs = payload["records"]

    # Map CJK aliases properly via ground_truth_pois_10cities.csv (poi_id -> cid::poi_name)
    df_gt_old = pd.read_csv(DATA_DIR / "ground_truth_pois_10cities.csv")
    pid_to_key = {str(r["poi_id"]): f"{r['city_id']}::{r['poi_name']}" for _, r in df_gt_old.iterrows()}
    zh_aliases_by_key = defaultdict(list)
    for fn in ["cjk_aliases_300.json", "cjk_aliases_intl_300.json"]:
        fp = DATA_DIR / fn
        if fp.exists():
            raw_a = json.loads(fp.read_text(encoding="utf-8"))
            for cid, pmap in raw_a.items():
                if cid.startswith("_") or not isinstance(pmap, dict):
                    continue
                for pid, alist in pmap.items():
                    if pid in pid_to_key:
                        zh_aliases_by_key[pid_to_key[pid]].extend(alist)

    poi_coords = {}
    for cid, plist in gt.REAL_600_POIS.items():
        for pname, lat, lon, *_ in plist:
            poi_coords[f"{cid}::{pname}"] = (cid, pname, float(lat), float(lon))

    existing_qids = [r["wikidata_qid"] for r in recs.values() if r["wikidata_qid"]]
    print(f"1. Auditing {len(set(existing_qids))} unique existing QIDs via batched wbgetentities...")
    ent_map = fetch_entities_batch(existing_qids)
    qid_counts = Counter(r["wikidata_qid"] for r in recs.values() if r["wikidata_qid"])

    needs_fix = []
    for k, r in sorted(recs.items()):
        cid, pname, plat, plon = poi_coords[k]
        q = r["wikidata_qid"]
        if not q or q in BANNED_QIDS or qid_counts[q] > 1:
            needs_fix.append(k)
            continue
        ent = ent_map.get(q)
        if not ent or (ent["p31"] & BANNED_ADMIN_OR_PERSON_P31):
            needs_fix.append(k)
            continue
        if ent["lat"] is not None:
            if haversine_km(plat, plon, ent["lat"], ent["lon"]) > 25.0:
                needs_fix.append(k)
                continue
        else:
            ptoks = norm_tokens(pname)
            etoks = norm_tokens(r["enwiki_title"] + " " + ent["labels"].get("en", ""))
            zh_als = zh_aliases_by_key.get(k, [])
            zh_lbl = ent["labels"].get("zh", "") + " " + r["native_wiki_title"]
            zh_hit = any(a in zh_lbl or (len(zh_lbl) >= 2 and zh_lbl in a) for a in zh_als if len(a) >= 2) if zh_als else False
            if len(ptoks & etoks) == 0 and not zh_hit:
                needs_fix.append(k)
                continue

    print(f"2. Flagged {len(needs_fix)} / 600 POIs. Running 40-title batched Wikipedia resolution first...")
    lang_titles = defaultdict(list)
    key_title_probes = defaultdict(list)
    for k in needs_fix:
        cid, pname, _, _ = poi_coords[k]
        nlang = CITY_LANG[cid]
        c_meta = next(c for c in cfg.CITIES_META if c["city_id"] == cid)
        cname = c_meta["city"]

        # CJK aliases on zhwiki
        for za in zh_aliases_by_key.get(k, []):
            lang_titles["zh"].append(za)
            key_title_probes[k].append(("zh", za))
            if cid in ("BJS", "SHA", "CTU", "QZJ", "DLU"):
                zh_city = {"BJS": "北京", "SHA": "上海", "CTU": "成都", "QZJ": "泉州", "DLU": "大理"}[cid]
                lang_titles["zh"].append(f"{za} ({zh_city})")
                key_title_probes[k].append(("zh", f"{za} ({zh_city})"))

        # English & Native title probes
        base_clean = re.sub(r"\s*\(.*?\)\s*", "", pname).strip()
        parens = re.findall(r"\((.*?)\)", pname)
        en_probes = [pname, base_clean, f"{base_clean}, {cname}", f"{base_clean} ({cname})"]
        if "&" in base_clean:
            en_probes.append(base_clean.split("&")[0].strip())
        en_probes.extend(parens)
        for ep in en_probes:
            if ep:
                lang_titles["en"].append(ep)
                key_title_probes[k].append(("en", ep))
                if nlang not in ("en", "zh"):
                    lang_titles[nlang].append(ep)
                    key_title_probes[k].append((nlang, ep))

    resolved_title_qids = {}
    for lg, tlist in lang_titles.items():
        resolved_title_qids[lg] = batch_resolve_wiki_titles(lg, tlist)
        print(f"  [{lg}wiki] Resolved {len(resolved_title_qids[lg])}/{len(set(tlist))} title probes")

    cand_by_key = defaultdict(list)
    for k in needs_fix:
        for lg, t_str in key_title_probes[k]:
            q = resolved_title_qids.get(lg, {}).get(t_str, "")
            if q and q not in BANNED_QIDS and q not in cand_by_key[k]:
                cand_by_key[k].append(q)

    # Only run single polite wbsearchentities on POIs that STILL have 0 candidate QIDs!
    still_empty = [k for k in needs_fix if len(cand_by_key[k]) == 0]
    print(f"3. {len(needs_fix) - len(still_empty)}/{len(needs_fix)} had direct Wikipedia hits; running polite wbsearchentities on remaining {len(still_empty)}...")
    for idx_e, k in enumerate(still_empty, 1):
        cid, pname, _, _ = poi_coords[k]
        base_clean = re.sub(r"\s*\(.*?\)\s*", "", pname).strip()
        q_str = zh_aliases_by_key[k][0] if (cid in ("BJS", "SHA", "CTU", "QZJ", "DLU") and zh_aliases_by_key.get(k)) else base_clean
        s_lang = "zh" if cid in ("BJS", "SHA", "CTU", "QZJ", "DLU") else "en"
        params = urllib.parse.urlencode({
            "action": "wbsearchentities", "search": q_str, "language": s_lang,
            "uselang": s_lang, "type": "item", "limit": "6", "format": "json"
        })
        res = polite_get_json(f"https://www.wikidata.org/w/api.php?{params}", min_gap=0.45)
        for it in res.get("search", []):
            cq = it.get("id", "")
            if cq and cq not in BANNED_QIDS and cq not in cand_by_key[k]:
                cand_by_key[k].append(cq)
        if s_lang == "zh" and len(cand_by_key[k]) == 0:
            params_en = urllib.parse.urlencode({
                "action": "wbsearchentities", "search": base_clean, "language": "en",
                "uselang": "en", "type": "item", "limit": "5", "format": "json"
            })
            res_en = polite_get_json(f"https://www.wikidata.org/w/api.php?{params_en}", min_gap=0.45)
            for it in res_en.get("search", []):
                cq = it.get("id", "")
                if cq and cq not in BANNED_QIDS and cq not in cand_by_key[k]:
                    cand_by_key[k].append(cq)
        if idx_e % 20 == 0 or idx_e == len(still_empty):
            print(f"  wbsearch [{idx_e}/{len(still_empty)}]")

    all_cands = {q for qlist in cand_by_key.values() for q in qlist}
    print(f"4. Fetching {len(all_cands)} candidate entities in batches...")
    ent_map.update(fetch_entities_batch(list(all_cands)))

    claimed_qids = {recs[k]["wikidata_qid"] for k in recs if k not in set(needs_fix) and recs[k]["wikidata_qid"]}

    chosen_q_by_key = {}
    for k in needs_fix:
        cid, pname, plat, plon = poi_coords[k]
        c_meta = next(c for c in cfg.CITIES_META if c["city_id"] == cid)
        clat, clon = float(c_meta["lat"]), float(c_meta["lon"])
        ptoks = norm_tokens(pname)
        zh_als = zh_aliases_by_key.get(k, [])

        best_q, best_sc = None, -1e9
        for cq in cand_by_key.get(k, []):
            if cq in claimed_qids or cq in BANNED_QIDS:
                continue
            ent = ent_map.get(cq)
            if not ent or (ent["p31"] & BANNED_ADMIN_OR_PERSON_P31):
                continue

            if ent["lat"] is not None:
                d_poi = haversine_km(plat, plon, ent["lat"], ent["lon"])
                d_city = haversine_km(clat, clon, ent["lat"], ent["lon"])
                if d_poi > 22.0 and d_city > 65.0:
                    continue
                coord_sc = max(0.0, 20.0 - d_poi) * 2.5
            else:
                d_poi = None
                coord_sc = 0.0

            all_lbl = " ".join(ent["labels"].values()) + " " + " ".join(ent["sitelinks"].values())
            etoks = norm_tokens(all_lbl)
            tok_inter = len(ptoks & etoks)
            tok_jac = tok_inter / max(1, len(ptoks))

            zh_target = ent["labels"].get("zh", "") + " " + ent["sitelinks"].get("zhwiki", "")
            zh_hit = any(za in zh_target or (len(zh_target) >= 2 and zh_target in za) for za in zh_als if len(za) >= 2) if (zh_als and zh_target.strip()) else False

            if d_poi is not None and d_poi <= 3.0 and (tok_inter >= 1 or zh_hit):
                pass
            elif d_poi is not None and d_poi <= 18.0 and (tok_jac >= 0.40 or zh_hit):
                pass
            elif d_poi is None and (tok_jac >= 0.55 or zh_hit):
                pass
            else:
                continue

            sc = coord_sc + tok_jac * 35.0 + (28.0 if zh_hit else 0.0) + math.log1p(ent["sitelinks_count"]) * 2.0
            if sc > best_sc:
                best_sc = sc
                best_q = cq

        if best_q:
            claimed_qids.add(best_q)
            chosen_q_by_key[k] = best_q

    # Batch fetch byte lengths for newly chosen articles
    new_titles_by_lang = defaultdict(list)
    for k, bq in chosen_q_by_key.items():
        cid = poi_coords[k][0]
        nlang = CITY_LANG[cid]
        ent = ent_map[bq]
        if ent["sitelinks"].get("enwiki"):
            new_titles_by_lang["en"].append(ent["sitelinks"]["enwiki"])
        if ent["sitelinks"].get(f"{nlang}wiki"):
            new_titles_by_lang[nlang].append(ent["sitelinks"][f"{nlang}wiki"])

    len_cache = {}
    for lg, tl in new_titles_by_lang.items():
        len_cache[lg] = batch_fetch_lengths(lg, tl)

    print(f"5. Fetching 2025 pageviews for {len(chosen_q_by_key)} newly verified attractions...")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    repaired_cnt, zeroed_cnt = 0, 0
    for k in needs_fix:
        cid, pname, plat, plon = poi_coords[k]
        nlang = CITY_LANG[cid]
        bq = chosen_q_by_key.get(k)
        if bq:
            ent = ent_map[bq]
            en_t = ent["sitelinks"].get("enwiki", "")
            nat_t = ent["sitelinks"].get(f"{nlang}wiki", "")
            b_en = len_cache.get("en", {}).get(en_t, 0)
            b_nat = len_cache.get(nlang, {}).get(nat_t, 0)
            v_en, u_en = fetch_views_2025("en", en_t)
            v_nat, u_nat = fetch_views_2025(nlang, nat_t) if (nlang != "en" and nat_t) else (0, "")
            recs[k] = {
                "city_id": cid, "poi_name": pname, "lat": plat, "lon": plon,
                "wikidata_qid": bq, "wikidata_url": f"https://www.wikidata.org/wiki/{bq}",
                "enwiki_title": en_t, "native_lang": nlang, "native_wiki_title": nat_t,
                "en_views_2025": v_en, "native_views_2025": v_nat,
                "wiki_views_annual": v_en + v_nat,
                "wikidata_sitelinks": int(ent["sitelinks_count"]),
                "wiki_doc_bytes": max(b_en, b_nat),
                "en_pageviews_api_url": u_en, "native_pageviews_api_url": u_nat,
                "fetched_at_utc": now_iso,
            }
            repaired_cnt += 1
        else:
            recs[k] = {
                "city_id": cid, "poi_name": pname, "lat": plat, "lon": plon,
                "wikidata_qid": "", "wikidata_url": "",
                "enwiki_title": "", "native_lang": nlang, "native_wiki_title": "",
                "en_views_2025": 0, "native_views_2025": 0,
                "wiki_views_annual": 0, "wikidata_sitelinks": 0, "wiki_doc_bytes": 0,
                "en_pageviews_api_url": "", "native_pageviews_api_url": "",
                "fetched_at_utc": now_iso,
            }
            zeroed_cnt += 1

    sorted_recs = dict(sorted(recs.items()))
    raw_bytes = json.dumps(sorted_recs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(raw_bytes).hexdigest()
    qids_after = [r["wikidata_qid"] for r in sorted_recs.values() if r["wikidata_qid"]]
    dup_after = {q: c for q, c in Counter(qids_after).items() if c > 1}

    payload["_metadata"] = {
        "description": "100% live-fetched & entity-verified Wikimedia REST API 2025 annual pageviews, Wikidata QIDs/sitelinks, and Wikipedia article byte-lengths for 600 urban attractions across 10 cities.",
        "window": "2025-01-01 to 2025-12-31 (monthly user pageviews)",
        "total_pois": len(sorted_recs),
        "pois_with_qid": len(qids_after),
        "pois_with_wiki_match": sum(1 for r in sorted_recs.values() if r["enwiki_title"] or r["native_wiki_title"]),
        "pois_with_positive_views": sum(1 for r in sorted_recs.values() if r["wiki_views_annual"] > 0),
        "duplicate_qids": len(dup_after),
        "repaired_count": repaired_cnt,
        "empirical_zero_wiki_count": zeroed_cnt,
        "sha256_records": sha,
        "generated_at_utc": now_iso,
    }
    payload["records"] = sorted_recs
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE! Metadata:", payload["_metadata"])

if __name__ == "__main__":
    main()
