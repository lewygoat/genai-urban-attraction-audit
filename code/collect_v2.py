# -*- coding: utf-8 -*-
"""
collect_v2.py — Stateless Multi-Endpoint API Query Runner for Generative AI Attraction Audit.

Design Principles:
  1. Records only authentic upstream HTTP responses; zero synthetic generation or imputation.
  2. Logs both the requested gateway alias (`endpoint_alias`) and the upstream model header (`api_returned_model`).
  3. Measures end-to-end wall-clock latency via `time.perf_counter()` and extracts raw `token_usage` metadata.
  4. Computes and stores SHA-256 digests (`request_sha256`, `response_sha256`) for every request/response pair.
  5. Logs HTTP errors transparently without writing placeholder recommendations.
  6. Supports idempotent checkpoint resumption (`--resume`) skipping completed `query_id` records.

Usage:
  python -I collect_v2.py --probe              # Single diagnostic call per endpoint to verify model headers
  python -I collect_v2.py --batch B1 --limit 5 # Dry-run 5 queries from Batch B1
  python -I collect_v2.py --all --resume       # Execute full multi-batch collection
"""
import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
OUT_DIR = Path(os.environ.get("COLLECT_OUT", EXP / "data" / "raw_responses_v2final"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env():
    env = {}
    for p in (EXP / ".env.local", ROOT / ".env.local", EXP / ".env", ROOT / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("POMO_BASE_URL", "POMO_API_KEY", "RELAY_BASE_URL", "RELAY_API_KEY"):
        env.setdefault(k, os.environ.get(k, ""))
    return env


ENV = load_env()

COMPANIONS = ["Solo", "Couple", "Family_Child", "Senior_70plus"]
BUDGETS = ["Low", "Medium", "High"]
SEASONS = ["Peak", "OffPeak"]
MOBILITIES = ["Unrestricted", "Wheelchair"]

CITIES = [
    ("BJS", "Beijing", "China", "Domestic", "Info-Rich"),
    ("SHA", "Shanghai", "China", "Domestic", "Info-Rich"),
    ("CTU", "Chengdu", "China", "Domestic", "Info-Rich"),
    ("QZJ", "Quanzhou", "China", "Domestic", "Info-Scarce"),
    ("DLU", "Dali", "China", "Domestic", "Info-Scarce"),
    ("PAR", "Paris", "France", "International", "Info-Rich"),
    ("TYO", "Tokyo", "Japan", "International", "Info-Rich"),
    ("BCN", "Barcelona", "Spain", "International", "Info-Rich"),
    ("OPO", "Porto", "Portugal", "International", "Info-Scarce"),
    ("CNX", "Chiang Mai", "Thailand", "International", "Info-Scarce"),
]

# Endpoint registry: `provider` selects the transport configuration; `alias` is the gateway model identifier,
# while `api_returned_model` is extracted directly from the upstream JSON response header.
ENDPOINTS = [
    {"alias": "gpt-5.6-sol", "provider": "pomo", "model_origin": "International"},
    {"alias": "claude-sonnet-4-5-20250929", "provider": "relay", "model_origin": "International"},
    {"alias": "deepseek-v4-pro-0813", "provider": "pomo", "model_origin": "Domestic"},
    {"alias": "qwen3.7-flash", "provider": "pomo", "model_origin": "Domestic"},
]

CTX = []
for i, (c, b, s, m) in enumerate(
    ((c, b, s, m) for c in COMPANIONS for b in BUDGETS for s in SEASONS for m in MOBILITIES), 1
):
    CTX.append({"context_id": f"CTX_{i:02d}", "companion": c, "budget": b, "season": s, "mobility": m})

CTX12 = [CTX[i] for i in range(0, 48, 4)]  # Stratified 12-context subset: CTX_01, CTX_05, ..., CTX_45

SYSTEM_NOTE = "Return only JSON. No markdown fences, no commentary."


def build_prompt(city_name, country, ctx, lang, template="T1"):
    if ctx is None:  # Batch B2 unconditioned blank baseline
        if lang == "zh":
            return (f"推荐{city_name}（{country}）最值得去的10个旅游景点。"
                    f"严格返回 JSON 对象，键为 'recommendations'，含10个元素，"
                    f"每个元素含 'rank'(1-10)、'name'、'category'、'reason'。")
        return (f"Recommend the top 10 tourist attractions in {city_name} ({country}). "
                f"Return strictly a JSON object with key 'recommendations' containing 10 objects, "
                f"each with 'rank' (1-10), 'name', 'category', and 'reason'.")

    season_txt = "peak travel season" if ctx["season"] == "Peak" else "off-peak season"
    mob_txt = "requires step-free / wheelchair-accessible routes" if ctx["mobility"] == "Wheelchair" else "no mobility restrictions"

    if lang == "zh":
        base = (f"你是{city_name}（{country}）的目的地专家。"
                f"旅行者画像：同行={ctx['companion']}，预算={ctx['budget']}，出行季节={season_txt}，行动能力={mob_txt}。"
                f"请为该旅行者推荐恰好10个景点，并针对上述画像说明理由。")
        if template == "T2":
            base = (f"假设你要为一位旅行者规划{city_name}（{country}）的一日精华行程。"
                    f"该旅行者情况：{ctx['companion']}出行，预算{ctx['budget']}，"
                    f"在{season_txt}到访，{mob_txt}。请列出10个最合适的景点。")
        elif template == "T3":
            base = (f"请以当地向导口吻，为以下画像挑选{city_name}（{country}）十个必去景点："
                    f"同行方式{ctx['companion']}、消费档位{ctx['budget']}、季节{season_txt}、"
                    f"无障碍需求{mob_txt}。")
        return (base + "严格返回 JSON 对象，键为 'recommendations'，含10个元素，"
                "每个元素含 'rank'(1-10)、'name'、'category'、'reason'。"
                "其中 'reason' 写成：'Recommended <类别> landmark (<名称>): <一句与本画像相关的理由>'。")

    base = (f"Act as a destination expert for {city_name} ({country}). "
            f"Traveler profile: Companion={ctx['companion']}, Budget={ctx['budget']}, "
            f"Season={season_txt}, Mobility={mob_txt}. "
            f"Recommend exactly 10 attractions tailored to this profile.")
    if template == "T2":
        base = (f"Plan a one-day highlight itinerary for a traveler visiting {city_name} ({country}). "
                f"Profile: {ctx['companion']}, budget {ctx['budget']}, visiting in the {season_txt}, "
                f"{mob_txt}. List the 10 best-matching attractions.")
    elif template == "T3":
        base = (f"As a local guide, pick ten must-see attractions in {city_name} ({country}) for this traveler: "
                f"companion={ctx['companion']}, budget={ctx['budget']}, season={season_txt}, "
                f"mobility={mob_txt}.")
    return (base + " Return strictly valid JSON with key 'recommendations' containing 10 objects "
            "(rank, name, category, reason), where 'reason' reads "
            "'Recommended <category> landmark (<name>): <one-sentence profile-specific justification>'.")


def lang_for(city_origin, batch, arm=None):
    """Assigns prompt language: Chinese for Chinese cities and English for international cities in B1/B2/B4/B5/B6; all-English in B3; all-Chinese in B3b."""
    if batch in ("B3",):
        return "en"
    if batch in ("B3b",):
        return "zh"
    return "zh" if city_origin == "Domestic" else "en"


def call_endpoint(ep, prompt, temperature, max_retries=4):
    provider = ep["provider"]
    alias = ep["alias"]
    if provider == "pomo":
        url, key = ENV["POMO_BASE_URL"], ENV["POMO_API_KEY"]
        payload = {"model": alias,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    else:
        url, key = ENV["RELAY_BASE_URL"], ENV["RELAY_API_KEY"]
        payload = {"model": alias, "max_tokens": 4096, "temperature": temperature,
                   "system": SYSTEM_NOTE,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_hash = hashlib.sha256(body).hexdigest()

    last_err = None
    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8", errors="replace")
            latency = round(time.perf_counter() - t0, 3)
            return {"status": status, "raw": raw, "latency": latency,
                    "req_sha256": req_hash, "error": None}
        except urllib.error.HTTPError as e:
            latency = round(time.perf_counter() - t0, 3)
            detail = e.read().decode("utf-8", errors="replace")[:800]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code != 429 and 400 <= e.code < 500:
                return {"status": e.code, "raw": detail, "latency": latency,
                        "req_sha256": req_hash, "error": last_err}
        except Exception as e:
            latency = round(time.perf_counter() - t0, 3)
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 15))
    return {"status": 0, "raw": "", "latency": 0.0, "req_sha256": req_hash, "error": last_err}


def extract(res):
    """Extracts content, `api_returned_model`, and `usage` from the raw JSON response payload."""
    try:
        d = json.loads(res["raw"])
    except Exception:
        return None, None, None
    returned_model = d.get("model")
    usage = d.get("usage") or d.get("token_usage")
    if "choices" in d:
        try:
            content = d["choices"][0]["message"]["content"]
        except Exception:
            content = None
        if usage is None:
            usage = d.get("usage")
    elif "content" in d:
        blocks = d["content"]
        content = "".join(b.get("text", "") for b in blocks if isinstance(b, dict)) if isinstance(blocks, list) else None
    else:
        content = None
    return content, returned_model, usage


def validate_json_10(content):
    """Validates that the response parses into a 10-item JSON recommendation array."""
    if not content:
        return False, [], "empty content"
    s = content.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s
        s = s.lstrip("json").strip()
    try:
        obj = json.loads(s)
    except Exception as e:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i:j + 1])
            except Exception:
                return False, [], f"json parse fail: {e}"
        else:
            return False, [], f"json parse fail: {e}"
    recs = obj.get("recommendations") if isinstance(obj, dict) else obj
    if not isinstance(recs, list) or len(recs) != 10:
        return False, [], f"not 10 items ({len(recs) if isinstance(recs,list) else 'n/a'})"
    names = [str(r.get("name", "")).strip() for r in recs if isinstance(r, dict)]
    if len(names) != 10 or not all(names):
        return False, names, "missing names"
    return True, names, "ok"


BATCH_SPECS = {
    "B1":  {"desc": "Main contextual audit (48 ctx x 4 ep x 10 cities x 3 reps)", "ctx": "full", "reps": 3, "temp": 0.7, "lang": None},
    "B2":  {"desc": "Blank unconditioned baseline (1 x 4 x 10 x 3)",              "ctx": "blank", "reps": 3, "temp": 0.7, "lang": None},
    "B3":  {"desc": "English prompt control (12 ctx x 4 x 10 x 2)",               "ctx": "12",    "reps": 2, "temp": 0.7, "lang": "en"},
    "B3b": {"desc": "Chinese prompt x intl cities (12 ctx x 4 x 5 x 2)",          "ctx": "12",    "reps": 2, "temp": 0.7, "lang": "zh", "only_origin": "International"},
    "B4":  {"desc": "Paraphrased templates T2/T3 (12 ctx x 2 tmpl x 4 x 10)",     "ctx": "12",    "reps": 1, "temp": 0.7, "lang": None, "templates": ["T2", "T3"]},
    "B5":  {"desc": "Greedy decoding t=0 (12 ctx x 4 x 10 x 3)",                  "ctx": "12",    "reps": 3, "temp": 0.0, "lang": None},
    "B6":  {"desc": "Intra-day temporal stability retest (12 ctx x 4 x 10 x 1)",  "ctx": "12",    "reps": 1, "temp": 0.7, "lang": None},
}


def build_plan(batch, skip_aliases=()):
    spec = BATCH_SPECS[batch]
    cities = [c for c in CITIES if not spec.get("only_origin") or c[3] == spec["only_origin"]]
    if spec["ctx"] == "full":
        ctxs = CTX
    elif spec["ctx"] == "12":
        ctxs = CTX12
    else:
        ctxs = [None]
    templates = spec.get("templates", ["T1"])

    plan = []
    n = 0
    for cid, cname, country, origin, richness in cities:
        for ep in ENDPOINTS:
            if ep["alias"] in skip_aliases:
                continue
            for ctx in ctxs:
                for tmpl in templates:
                    for rep in range(1, spec["reps"] + 1):
                        n += 1
                        lang = spec["lang"] or lang_for(origin, batch)
                        plan.append({
                            "query_id": f"{batch}_{n:05d}",
                            "batch_id": batch,
                            "city_id": cid, "city": cname, "country": country,
                            "origin": origin, "richness": richness,
                            "endpoint_alias": ep["alias"],
                            "provider": ep["provider"],
                            "model_origin": ep["model_origin"],
                            "context_id": (ctx or {}).get("context_id", "CTX_BLANK"),
                            "companion": (ctx or {}).get("companion", "Blank"),
                            "budget": (ctx or {}).get("budget", "Blank"),
                            "season": (ctx or {}).get("season", "Blank"),
                            "mobility": (ctx or {}).get("mobility", "Blank"),
                            "rep_id": rep,
                            "template_id": tmpl,
                            "prompt_lang": lang,
                            "temperature": spec["temp"],
                            "_prompt": build_prompt(cname, country, ctx, lang, tmpl),
                        })
    plan.sort(key=lambda q: (q["context_id"], q["rep_id"], q["city_id"], q["endpoint_alias"]))
    for i, q in enumerate(plan, 1):
        q["query_id"] = f"{batch}_{i:05d}"
    return plan


def out_path(batch):
    return OUT_DIR / f"batch_{batch}.jsonl"


def _iter_records(p):
    """Iterates JSONL records line-by-line in binary mode to tolerate interrupted writes."""
    bad = 0
    with p.open("rb") as f:
        for bline in f:
            if not bline.strip():
                continue
            try:
                yield json.loads(bline.decode("utf-8"))
            except Exception:
                bad += 1
                continue
    if bad:
        print(f"  [warn] {p.name}: skipped {bad} corrupt line(s)")


def existing_ids(batch):
    """Returns `query_id`s that have already completed with HTTP 200 and valid 10-item JSON."""
    p = out_path(batch)
    ids = set()
    if p.exists():
        for r in _iter_records(p):
            if r.get("http_status") == 200 and r.get("valid_json") == 1:
                ids.add(r["query_id"])
    return ids


def compact(batch):
    """Deduplicates a batch JSONL file, retaining the valid record for each `query_id`."""
    p = out_path(batch)
    if not p.exists():
        return 0
    best = {}
    for r in _iter_records(p):
        qid = r["query_id"]
        ok = r.get("http_status") == 200 and r.get("valid_json") == 1
        if qid not in best or ok:
            best[qid] = r
    tmp = p.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for qid in sorted(best):
            f.write(json.dumps(best[qid], ensure_ascii=False) + "\n")
    tmp.replace(p)
    return len(best)


def run(batch, limit=None, resume=True, workers=16, pomo_conc=10, relay_conc=2, skip=()):
    """Executes concurrent API queries with per-provider semaphore rate-limiting."""
    plan = build_plan(batch, skip_aliases=skip)
    done = existing_ids(batch) if resume else set()
    todo = [q for q in plan if q["query_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{batch}] planned={len(plan)} already={len(done)} to_collect={len(todo)}")

    sem = {"pomo": threading.Semaphore(pomo_conc), "relay": threading.Semaphore(relay_conc)}
    write_lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "done": 0, "rate": 0}

    def task(q):
        prompt = q.pop("_prompt")
        sem[q["provider"]].acquire()
        try:
            res = call_endpoint({"alias": q["endpoint_alias"], "provider": q["provider"]},
                                prompt, q["temperature"])
        finally:
            sem[q["provider"]].release()
        q["_prompt"] = prompt
        content, returned_model, usage = extract(res)
        valid, names, note = validate_json_10(content)

        rec = dict(q)
        rec.pop("_prompt", None)
        rec.update({
            "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "http_status": res["status"],
            "latency_sec": res["latency"],
            "request_sha256": res["req_sha256"],
            "response_sha256": hashlib.sha256(res["raw"].encode("utf-8")).hexdigest() if res["raw"] else None,
            "api_returned_model": returned_model,
            "token_usage": usage,
            "valid_json": 1 if valid else 0,
            "parse_note": note,
            "parsed_names": names if valid else [],
            "raw_text": content,
            "error": res["error"],
        })
        return rec

    def worker(q):
        try:
            rec = task(q)
        except Exception as e:
            rec = {k: v for k, v in q.items() if k != "_prompt"}
            rec.update({"collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "http_status": 0, "valid_json": 0, "parsed_names": [], "raw_text": None,
                        "error": f"{type(e).__name__}: {e}", "api_returned_model": None,
                        "token_usage": None, "latency_sec": 0.0})
        with write_lock:
            with out_path(batch).open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counters["done"] += 1
            if rec.get("valid_json") == 1:
                counters["ok"] += 1
            elif rec.get("http_status") == 429 or rec.get("http_status") == 0:
                counters["rate"] += 1
            else:
                counters["fail"] += 1
            n = counters["done"]
            if n % 100 == 0:
                print(f"  ... {n}/{len(todo)}  ok={counters['ok']} rate={counters['rate']} hardfail={counters['fail']}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, q) for q in todo]
        for f in as_completed(futs):
            f.result()

    print(f"[{batch}] done ok={counters['ok']} rate_limited={counters['rate']} hardfail={counters['fail']} -> {out_path(batch)}")
    return counters["ok"], counters["rate"], counters["fail"]


def probe():
    """Sends a minimal diagnostic call to each endpoint to verify model availability and response headers."""
    for ep in ENDPOINTS:
        q = {"alias": ep["alias"], "provider": ep["provider"]}
        res = call_endpoint(q, "Reply with exactly: OK", 0.0, max_retries=1)
        content, returned_model, usage = extract(res)
        print(f"--- {ep['alias']} (provider={ep['provider']})")
        print(f"    status={res['status']}  returned_model={returned_model!r}")
        print(f"    content={(content or '')[:80]!r}")
        print(f"    usage={usage}")
        if res["error"]:
            print(f"    ERROR: {res['error'][:300]}")


def probe_models(candidates):
    """Probes a list of candidate model aliases on the gateway."""
    for m in candidates:
        res = call_endpoint({"alias": m, "provider": "pomo"}, "hi", 0.0, max_retries=1)
        _, returned, usage = extract(res)
        if res["status"] == 200:
            print(f"  OK    {m:34} returned={returned!r}")
        else:
            err = (res["error"] or "").replace("\n", " ")[:150]
            print(f"  {res['status']:<5} {m:34} {err}")
        time.sleep(0.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-models", nargs="*")
    ap.add_argument("--plan", action="store_true", help="Print batch query counts without sending network requests")
    ap.add_argument("--compact", nargs="*", help="Deduplicate and compact JSONL batch files")
    ap.add_argument("--batch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--pomo-conc", type=int, default=12)
    ap.add_argument("--relay-conc", type=int, default=3)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--skip-endpoints", nargs="*", default=[],
                    help="Skip specified endpoint_alias values during collection")
    a = ap.parse_args()

    if a.plan:
        tot = 0
        for b in BATCH_SPECS:
            n = len(build_plan(b, skip_aliases=tuple(a.skip_endpoints)))
            tot += n
            print(f"  {b:4} {n:6d}   {BATCH_SPECS[b]['desc']}")
        print(f"  {'TOTAL':4} {tot:6d}")
        print("\nendpoints:", [(e["alias"], e["provider"]) for e in ENDPOINTS])
    elif a.compact:
        for b in (a.compact if isinstance(a.compact, list) else BATCH_SPECS):
            n = compact(b)
            print(f"  compacted {b}: {n} unique records")
    elif a.probe:
        probe()
    elif a.probe_models is not None:
        probe_models(a.probe_models)
    elif a.all:
        tot_ok = tot_rate = tot_fail = 0
        for b in BATCH_SPECS:
            for attempt in range(4):
                o, r, f_ = run(b, limit=a.limit, resume=True,
                               workers=a.workers, pomo_conc=a.pomo_conc, relay_conc=a.relay_conc,
                               skip=tuple(a.skip_endpoints))
                tot_ok += o; tot_rate += r; tot_fail += f_
                if r == 0 and f_ == 0:
                    break
                print(f"  [{b}] retry round {attempt+2} for {r+f_} unfinished")
            n = compact(b)
            print(f"  [{b}] compacted -> {n} unique successful records")
        print(f"ALL DONE ok={tot_ok} rate_limited={tot_rate} hardfail={tot_fail}")
    elif a.batch:
        run(a.batch, limit=a.limit, resume=True, skip=tuple(a.skip_endpoints))
    else:
        ap.print_help()
