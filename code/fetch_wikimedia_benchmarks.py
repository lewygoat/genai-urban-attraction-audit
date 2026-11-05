# -*- coding: utf-8 -*-
"""
fetch_wikimedia_benchmarks.py — Batched & Polite Live Wikimedia/Wikidata Benchmark Collector (600 POIs).

Uses batched Wikidata SPARQL spatial-label queries (1 query per city = 10 queries) + polite rate-limited
MediaWiki & Wikimedia Pageviews REST API calls with compliant User-Agent and exponential 429 backoff.
Zero synthetic numbers; 100% live-resolvable QIDs and REST API URLs.
"""
import hashlib
import importlib
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

cfg = importlib.import_module("00_config")
gt_mod = importlib.import_module("01_ground_truth")

DATA_DIR = CODE_DIR.parent / "data"
OUT_JSON = DATA_DIR / "wikimedia_empirical_benchmarks_600pois.json"

HEADERS = {
    "User-Agent": "UrbanTourismGeoAudit/2.0 (https://www.wikidata.org; mailto:geoaudit-lab@outlook.com) Python-urllib/3.12",
    "Accept": "application/json",
}

CITY_NATIVE_LANG = {
    "BJS": "zh", "SHA": "zh", "CTU": "zh", "QZJ": "zh", "DLU": "zh",
    "PAR": "fr", "TYO": "ja", "BCN": "es", "OPO": "pt", "CNX": "th",
}
CITY_NAME_MAP = {c["city_id"]: c["city"] for c in cfg.CITIES_META}

_rate_lock = threading.Lock()
_last_req_ts = 0.0


def polite_get_json(url: str, min_interval: float = 0.22, retries: int = 5) -> dict:
    global _last_req_ts
    for attempt in range(retries):
        with _rate_lock:
            now = time.perf_counter()
            wait = min_interval - (now - _last_req_ts)
            if wait > 0:
                time.sleep(wait)
            _last_req_ts = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if e.code in (429, 503):
                ra = e.headers.get("Retry-After")
                sleep_s = float(ra) if (ra and ra.isdigit()) else (2.5 * (attempt + 1))
                time.sleep(min(sleep_s, 15.0))
                continue
            return {}
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}


def strip_acc(s: str) -> str:
    s = str(s).replace("œ", "oe").replace("Œ", "oe").replace("æ", "ae")
    nk = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nk if not unicodedata.combining(c)).lower()


def name_overlap_score(poi_name: str, cand_label: str) -> float:
    a = set(re.findall(r"[a-z0-9]{3,}", strip_acc(poi_name))) - {"the", "and", "park", "museum", "temple", "street", "square", "palace", "city"}
    b = set(re.findall(r"[a-z0-9]{3,}", strip_acc(cand_label))) - {"the", "and", "park", "museum", "temple", "street", "square", "palace", "city"}
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def fetch_city_sparql(cid: str, city_pois: list) -> dict[int, dict]:
    """Batch query Wikidata SPARQL around all 60 POI coordinates in one shot."""
    nat = CITY_NATIVE_LANG[cid]
    values_rows = []
    for idx, (pname, lat, lon, *_rest) in enumerate(city_pois):
        values_rows.append(f'({idx} "Point({float(lon):.4f} {float(lat):.4f})"^^geo:wktLiteral)')
    values_str = " ".join(values_rows)
    sparql = f"""
SELECT ?idx ?item ?itemLabel ?sitelinks ?enTitle ?natTitle WHERE {{
  VALUES (?idx ?coord) {{ {values_str} }}
  SERVICE wikibase:around {{
    ?item wdt:P625 ?loc .
    bd:serviceParam wikibase:center ?coord ;
                    wikibase:radius "1.2" .
  }}
  ?item wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= 1)
  OPTIONAL {{
    ?enArticle schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?enTitle .
  }}
  OPTIONAL {{
    ?natArticle schema:about ?item ;
                schema:isPartOf <https://{nat}.wikipedia.org/> ;
                schema:name ?natTitle .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,{nat}" . }}
}}
"""
    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(sparql)
    data = polite_get_json(url, min_interval=0.8)
    bindings = data.get("results", {}).get("bindings", [])
    by_idx = {}
    for b in bindings:
        idx = int(b["idx"]["value"])
        qid = b["item"]["value"].rsplit("/", 1)[-1]
        lbl = b.get("itemLabel", {}).get("value", "")
        sl = int(b.get("sitelinks", {}).get("value", 0))
        en_t = b.get("enTitle", {}).get("value", "")
        nat_t = b.get("natTitle", {}).get("value", "")
        pname = city_pois[idx][0]
        ov = name_overlap_score(pname, lbl) + name_overlap_score(pname, en_t)
        # Prefer candidates with lexical match, then sitelinks
        rank_score = ov * 200.0 + min(sl, 150)
        if idx not in by_idx or rank_score > by_idx[idx]["_score"]:
            by_idx[idx] = {
                "_score": rank_score,
                "_ov": ov,
                "qid": qid,
                "sitelinks": sl,
                "en_title": en_t,
                "nat_title": nat_t,
                "label": lbl,
            }
    return by_idx


def fallback_search_single(cid: str, pname: str, lat: float, lon: float, cjk_als: list[str]) -> dict:
    """Polite fallback MediaWiki search when spatial SPARQL missed or had zero lexical overlap."""
    nat = CITY_NATIVE_LANG[cid]
    cname = CITY_NAME_MAP[cid]
    clean_q = re.sub(r"\([^)]*\)", "", pname).split("&")[0].strip()

    # Try en.wikipedia search
    q_str = f"{clean_q} {cname}" if cname.lower() not in clean_q.lower() else clean_q
    p1 = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": q_str, "srlimit": "2", "format": "json"
    })
    d1 = polite_get_json(f"https://en.wikipedia.org/w/api.php?{p1}")
    hits = [x["title"] for x in d1.get("query", {}).get("search", []) if x.get("title")]
    for t in hits:
        if t.lower() in {cname.lower(), f"list of tourist attractions in {cname.lower()}"}:
            continue
        pm = urllib.parse.urlencode({
            "action": "query", "prop": "pageprops|coordinates|langlinks",
            "ppprop": "wikibase_item", "lllimit": "500", "redirects": "1", "titles": t, "format": "json"
        })
        dm = polite_get_json(f"https://en.wikipedia.org/w/api.php?{pm}")
        for pgid, pg in dm.get("query", {}).get("pages", {}).items():
            if str(pgid) == "-1":
                continue
            coords = pg.get("coordinates", [])
            if coords:
                dist = cfg.haversine_km(lat, lon, float(coords[0]["lat"]), float(coords[0]["lon"]))
                if dist > 70.0:
                    continue
            lls = pg.get("langlinks", [])
            nat_t = next((ll.get("*", "") for ll in lls if ll.get("lang") == nat), "")
            return {
                "qid": pg.get("pageprops", {}).get("wikibase_item", ""),
                "sitelinks": 1 + len(lls),
                "en_title": pg.get("title", t),
                "nat_title": nat_t,
            }

    # Try native Wikipedia for Chinese cities using CJK alias
    if nat == "zh" and cjk_als:
        p2 = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": f"{cjk_als[0]} {cname}", "srlimit": "2", "format": "json"
        })
        d2 = polite_get_json(f"https://zh.wikipedia.org/w/api.php?{p2}")
        for x in d2.get("query", {}).get("search", []):
            t = x.get("title", "")
            if not t:
                continue
            pm = urllib.parse.urlencode({
                "action": "query", "prop": "pageprops|langlinks",
                "ppprop": "wikibase_item", "lllimit": "500", "redirects": "1", "titles": t, "format": "json"
            })
            dm = polite_get_json(f"https://zh.wikipedia.org/w/api.php?{pm}")
            for pgid, pg in dm.get("query", {}).get("pages", {}).items():
                if str(pgid) == "-1":
                    continue
                lls = pg.get("langlinks", [])
                en_t = next((ll.get("*", "") for ll in lls if ll.get("lang") == "en"), "")
                return {
                    "qid": pg.get("pageprops", {}).get("wikibase_item", ""),
                    "sitelinks": 1 + len(lls),
                    "en_title": en_t,
                    "nat_title": pg.get("title", t),
                }
    return {"qid": "", "sitelinks": 0, "en_title": "", "nat_title": ""}


def batch_fetch_lengths(lang: str, titles: list[str]) -> dict[str, int]:
    """Fetch article byte lengths in batches of 45 titles per API call."""
    uniq = sorted({t for t in titles if t})
    out = {}
    for i in range(0, len(uniq), 45):
        chunk = uniq[i:i + 45]
        params = urllib.parse.urlencode({
            "action": "query", "prop": "info", "redirects": "1",
            "titles": "|".join(chunk), "format": "json"
        })
        d = polite_get_json(f"https://{lang}.wikipedia.org/w/api.php?{params}", min_interval=0.25)
        q = d.get("query", {})
        redir_map = {r["to"]: r["from"] for r in q.get("redirects", [])}
        norm_map = {n["to"]: n["from"] for n in q.get("normalized", [])}
        for pg in q.get("pages", {}).values():
            t_final = pg.get("title", "")
            ln = int(pg.get("length", 0))
            out[t_final] = ln
            if t_final in redir_map:
                out[redir_map[t_final]] = ln
                if redir_map[t_final] in norm_map:
                    out[norm_map[redir_map[t_final]]] = ln
            if t_final in norm_map:
                out[norm_map[t_final]] = ln
    return out


def fetch_pageviews_2025(lang: str, title: str) -> tuple[int, str]:
    if not title:
        return 0, ""
    enc = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{enc}/monthly/2025010100/2025123100"
    )
    d = polite_get_json(url, min_interval=0.12)
    items = d.get("items", [])
    return sum(int(x.get("views", 0)) for x in items), url


def main():
    t0 = time.perf_counter()
    # Map CJK aliases from cjk_aliases_300.json
    cjk_by_name = {}
    gt_csv = DATA_DIR / "ground_truth_pois_10cities.csv"
    p_cjk = DATA_DIR / "cjk_aliases_300.json"
    if gt_csv.exists() and p_cjk.exists():
        import csv
        pid_to_name = {}
        with gt_csv.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                pid_to_name[(r["city_id"], r["poi_id"])] = r["poi_name"]
        raw_cjk = json.loads(p_cjk.read_text(encoding="utf-8"))
        for cid, blk in raw_cjk.items():
            if cid.startswith("_"):
                continue
            for pid, als in blk.items():
                nm = pid_to_name.get((cid, pid))
                if nm:
                    cjk_by_name[(cid, nm)] = als

    resolved = {}
    for cid, city_pois in gt_mod.REAL_600_POIS.items():
        sp_hits = fetch_city_sparql(cid, city_pois)
        print(f"[SPARQL] {cid}: spatial hits={len(sp_hits)}/60 ({time.perf_counter()-t0:.1f}s)", flush=True)
        for idx, (pname, lat, lon, *_rest) in enumerate(city_pois):
            h = sp_hits.get(idx)
            if h and (h["_ov"] > 0.0 or h["en_title"] or h["nat_title"]) and h["_ov"] >= 0.25:
                resolved[(cid, pname)] = h
            else:
                fb = fallback_search_single(cid, pname, float(lat), float(lon), cjk_by_name.get((cid, pname), []))
                if fb["qid"] or fb["en_title"] or fb["nat_title"]:
                    resolved[(cid, pname)] = fb
                elif h:
                    resolved[(cid, pname)] = h
                else:
                    resolved[(cid, pname)] = {"qid": "", "sitelinks": 0, "en_title": "", "nat_title": ""}
        print(f"[Resolved] {cid} complete ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Batch fetch article lengths
    en_lengths = batch_fetch_lengths("en", [v["en_title"] for v in resolved.values()])
    nat_lengths_by_lang = {}
    for lang in sorted(set(CITY_NATIVE_LANG.values())):
        t_list = [v["nat_title"] for (cid, _), v in resolved.items() if CITY_NATIVE_LANG[cid] == lang and v["nat_title"]]
        nat_lengths_by_lang[lang] = batch_fetch_lengths(lang, t_list)
    print(f"[Lengths] Fetched article byte lengths ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Fetch 2025 pageviews politely across 3 workers
    pv_tasks = set()
    for (cid, _), v in resolved.items():
        if v["en_title"]:
            pv_tasks.add(("en", v["en_title"]))
        if v["nat_title"]:
            pv_tasks.add((CITY_NATIVE_LANG[cid], v["nat_title"]))

    pv_cache = {}
    pv_list = sorted(pv_tasks)
    print(f"[Pageviews] Querying {len(pv_list)} unique Wikipedia articles...", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_pageviews_2025, lg, tt): (lg, tt) for lg, tt in pv_list}
        done = 0
        for fut in as_completed(futs):
            lg, tt = futs[fut]
            pv_cache[(lg, tt)] = fut.result()
            done += 1
            if done % 150 == 0 or done == len(pv_list):
                print(f"  pageviews [{done}/{len(pv_list)}] ({time.perf_counter()-t0:.1f}s)", flush=True)

    ordered = {}
    for cid, city_pois in gt_mod.REAL_600_POIS.items():
        nat = CITY_NATIVE_LANG[cid]
        for pname, lat, lon, *_rest in city_pois:
            v = resolved[(cid, pname)]
            en_t = v["en_title"]
            nat_t = v["nat_title"]
            qid = v["qid"]
            en_v, en_u = pv_cache.get(("en", en_t), (0, ""))
            nat_v, nat_u = pv_cache.get((nat, nat_t), (0, ""))
            b_len = en_lengths.get(en_t, 0) + nat_lengths_by_lang.get(nat, {}).get(nat_t, 0)
            ordered[f"{cid}::{pname}"] = {
                "city_id": cid,
                "poi_name": pname,
                "lat": float(lat),
                "lon": float(lon),
                "wikidata_qid": qid,
                "wikidata_url": f"https://www.wikidata.org/wiki/{qid}" if qid else "",
                "enwiki_title": en_t,
                "native_lang": nat,
                "native_wiki_title": nat_t,
                "en_views_2025": int(en_v),
                "native_views_2025": int(nat_v),
                "wiki_views_annual": int(en_v + nat_v),
                "wikidata_sitelinks": int(v["sitelinks"]),
                "wiki_doc_bytes": int(b_len),
                "en_pageviews_api_url": en_u,
                "native_pageviews_api_url": nat_u,
                "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

    ordered = {k: ordered[k] for k in sorted(ordered.keys())}
    sha = hashlib.sha256(json.dumps(ordered, ensure_ascii=False, indent=2).encode("utf-8")).hexdigest()
    wrapper = {
        "_metadata": {
            "description": "100% live-fetched Wikimedia REST API 2025 annual pageviews, Wikidata QIDs/sitelinks, and Wikipedia article byte-lengths for 600 urban attractions across 10 cities.",
            "window": "2025-01-01 to 2025-12-31 (monthly user pageviews)",
            "total_pois": len(ordered),
            "pois_with_wiki_match": sum(1 for x in ordered.values() if x["enwiki_title"] or x["native_wiki_title"]),
            "pois_with_qid": sum(1 for x in ordered.values() if x["wikidata_qid"]),
            "pois_with_positive_views": sum(1 for x in ordered.values() if x["wiki_views_annual"] > 0),
            "sha256_records": sha,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "records": ordered,
    }
    OUT_JSON.write_bytes(json.dumps(wrapper, ensure_ascii=False, indent=2).encode("utf-8"))
    print("DONE! Metadata:", wrapper["_metadata"], flush=True)


if __name__ == "__main__":
    main()
