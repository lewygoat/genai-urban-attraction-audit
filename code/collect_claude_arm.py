# -*- coding: utf-8 -*-
"""
collect_claude_arm.py — Dedicated Collection Runner for the Claude Sonnet 4.6 Endpoint (`claude-sonnet-4-6`).

Overview:
  Executes the `claude-sonnet-4-6` experimental arm across Batches B1, B2, B3, B3b, B4, B5, and B6
  using the identical prompt matrix, temperature schedule, and JSON-10 validation schema defined in `collect_v2.py`.
  Prefixes `query_id` with `CL_` so records append cleanly to `batch_*.jsonl` without colliding with other endpoints.

Usage:
  python -I collect_claude_arm.py --plan
  python -I collect_claude_arm.py --batch B1 --limit 5
  python -I collect_claude_arm.py --all --workers 6
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cv2", ROOT / "collect_v2.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

OUT_DIR = Path(os.environ.get("COLLECT_OUT", ROOT.parent / "data" / "raw_responses_v2final"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLAUDE_ALIAS = os.environ.get("CLAUDE_ARM_MODEL", "claude-sonnet-4-6")
ORIGINAL_ALIAS = "claude-sonnet-4-5-20250929"


def claude_plan(batch):
    """Filters the batch plan for the Claude endpoint arm and prefixes `query_id` with `CL_`."""
    plan = cv.build_plan(batch)
    out = []
    for q in plan:
        if q["endpoint_alias"] != ORIGINAL_ALIAS:
            continue
        q = dict(q)
        q["endpoint_alias"] = CLAUDE_ALIAS
        q["provider"] = "relay"
        q["model_origin"] = "International"
        q["query_id"] = "CL_" + q["query_id"]
        out.append(q)
    return out


def out_path(batch):
    return OUT_DIR / f"batch_{batch}.jsonl"


def done_ids(batch):
    p = out_path(batch)
    ids = set()
    if p.exists():
        for r in cv._iter_records(p):
            if r.get("endpoint_alias") == CLAUDE_ALIAS and r.get("http_status") == 200 \
                    and r.get("valid_json") == 1:
                ids.add(r["query_id"])
    return ids


def run(batch, limit=None, workers=6):
    plan = claude_plan(batch)
    done = done_ids(batch)
    todo = [q for q in plan if q["query_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{batch}] claude planned={len(plan)} already={len(done)} to_collect={len(todo)}", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "rate": 0, "done": 0}

    def task(q):
        prompt = q.pop("_prompt")
        res = cv.call_endpoint({"alias": q["endpoint_alias"], "provider": "relay"},
                               prompt, q["temperature"], max_retries=5)
        content, returned_model, usage = cv.extract(res)
        valid, names, note = cv.validate_json_10(content)
        rec = dict(q)
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

    with open(out_path(batch), "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(task, q): q for q in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                stats["done"] += 1
                if rec["valid_json"] == 1:
                    stats["ok"] += 1
                elif rec["http_status"] == 429:
                    stats["rate"] += 1
                else:
                    stats["fail"] += 1
                if stats["done"] % 20 == 0 or stats["done"] == len(todo):
                    print(f"  ... {stats['done']}/{len(todo)}  ok={stats['ok']} "
                          f"rate={stats['rate']} fail={stats['fail']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    if a.plan:
        tot = 0
        for b in cv.BATCH_SPECS:
            n = len(claude_plan(b))
            tot += n
            print(f"  {b:4} {n:6d}   {cv.BATCH_SPECS[b]['desc']}")
        print(f"  TOTAL {tot}")
        return

    if a.all:
        for b in cv.BATCH_SPECS:
            run(b, limit=a.limit, workers=a.workers)
    elif a.batch:
        run(a.batch, limit=a.limit, workers=a.workers)
    else:
        ap.error("need --batch or --all")


if __name__ == "__main__":
    main()
