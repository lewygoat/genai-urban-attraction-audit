# -*- coding: utf-8 -*-
"""
00_config.py — Deterministic configuration, 10-city unbalanced 2x2 stratified metadata
(6 Info-Rich Megacities vs. 4 Info-Scarce Secondary Cities),
4 live commercial LLM endpoints verified from `raw_responses_v2final/` SHA-256 receipts,
48 full-factorial contextual profiles, 7 audit batches (B1, B2, B3, B3b, B4, B5, B6),
and WGS-84 Haversine utilities.
"""
import itertools
import math
import os
import zlib
from pathlib import Path
import numpy as np

GLOBAL_SEED = 20261022
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)


def get_rng(seed: int = GLOBAL_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_seed(*parts: str) -> int:
    """Process-independent deterministic 32-bit seed using CRC32."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return zlib.crc32(raw) & 0xFFFFFFFF


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS-84 points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(max(0.0, min(1.0, a))))


EXP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = EXP_ROOT / "data"
RAW_API_DIR = DATA_DIR / "raw_responses_v2final"
TAB_DIR = EXP_ROOT / "tables"
FIG_DIR = EXP_ROOT / "figures"
LOG_DIR = EXP_ROOT / "logs"

for d in (DATA_DIR, RAW_API_DIR, TAB_DIR, FIG_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_dotenv():
    candidates = [
        EXP_ROOT.parent / ".env.local",
        EXP_ROOT / ".env.local",
        Path(__file__).resolve().parent / ".env.local",
    ]
    for env_path in candidates:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

POMO_BASE_URL = os.environ.get("POMO_BASE_URL", "https://api.example.com/v1/chat/completions")
POMO_API_KEY = os.environ.get("POMO_API_KEY", "")

RELAY_BASE_URL = os.environ.get("RELAY_BASE_URL", "https://api.example.com/v1/messages")
RELAY_API_KEY = os.environ.get("RELAY_API_KEY", "")

# 10 Cities (Unbalanced 2x2 Stratification: 6 Info-Rich Megacities vs. 4 Info-Scarce Secondary Cities)
# `wiki_views_k` (2025 bilingual municipal Wikipedia pageviews in thousands) and `wikidata_sitelinks`
# are live-verified open-API benchmarks from `wikimedia_30cities_stratification.json`.
CITIES_META = [
    {"city_id": "BJS", "city": "Beijing",    "country": "China",    "origin": "Domestic",      "richness": "Info-Rich",   "lat": 39.9042, "lon": 116.4074, "osm_pois": 4820, "wiki_views_k": 1487.4, "wikidata_sitelinks": 258, "peak_month": "October",  "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "SHA", "city": "Shanghai",   "country": "China",    "origin": "Domestic",      "richness": "Info-Rich",   "lat": 31.2304, "lon": 121.4737, "osm_pois": 4310, "wiki_views_k": 1729.4, "wikidata_sitelinks": 211, "peak_month": "October",  "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "CTU", "city": "Chengdu",    "country": "China",    "origin": "Domestic",      "richness": "Info-Rich",   "lat": 30.5728, "lon": 104.0668, "osm_pois": 3190, "wiki_views_k": 827.0,  "wikidata_sitelinks": 114, "peak_month": "October",  "offpeak_month": "December", "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "QZJ", "city": "Quanzhou",   "country": "China",    "origin": "Domestic",      "richness": "Info-Scarce", "lat": 24.8741, "lon": 118.6757, "osm_pois": 680,  "wiki_views_k": 198.5,  "wikidata_sitelinks": 62,  "peak_month": "October",  "offpeak_month": "February", "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "DLU", "city": "Dali",       "country": "China",    "origin": "Domestic",      "richness": "Info-Scarce", "lat": 25.6065, "lon": 100.2676, "osm_pois": 540,  "wiki_views_k": 118.6,  "wikidata_sitelinks": 41,  "peak_month": "August",   "offpeak_month": "November", "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "PAR", "city": "Paris",      "country": "France",   "origin": "International", "richness": "Info-Rich",   "lat": 48.8566, "lon": 2.3522,   "osm_pois": 6450, "wiki_views_k": 2951.0, "wikidata_sitelinks": 289, "peak_month": "July",     "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "TYO", "city": "Tokyo",      "country": "Japan",    "origin": "International", "richness": "Info-Rich",   "lat": 35.6762, "lon": 139.6503, "osm_pois": 5890, "wiki_views_k": 3118.6, "wikidata_sitelinks": 267, "peak_month": "April",    "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "BCN", "city": "Barcelona",  "country": "Spain",    "origin": "International", "richness": "Info-Rich",   "lat": 41.3874, "lon": 2.1686,   "osm_pois": 4120, "wiki_views_k": 1854.4, "wikidata_sitelinks": 202, "peak_month": "August",   "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "OPO", "city": "Porto",      "country": "Portugal", "origin": "International", "richness": "Info-Scarce", "lat": 41.1456, "lon": -8.6106,  "osm_pois": 920,  "wiki_views_k": 637.8,  "wikidata_sitelinks": 119, "peak_month": "August",   "offpeak_month": "January",  "review_source": "Wikimedia_2025_REST_API"},
    {"city_id": "CNX", "city": "Chiang Mai", "country": "Thailand", "origin": "International", "richness": "Info-Scarce", "lat": 18.7883, "lon": 98.9853,  "osm_pois": 810,  "wiki_views_k": 411.1,  "wikidata_sitelinks": 76,  "peak_month": "December", "offpeak_month": "June",     "review_source": "Wikimedia_2025_REST_API"},
]

# 4 Live Audited Commercial LLM Endpoints (100% Aligned with `raw_responses_v2final` endpoint_alias & api_returned_model)
MODELS = [
    {"model": "gpt-5.6-sol",          "endpoint_tag": "gpt-5.6-sol",          "version": "gpt-5.6-sol",                            "model_origin": "International", "temp": 0.7, "provider": "pomo",  "relay_model": "gpt-5.6-sol"},
    {"model": "claude-sonnet-4-6",    "endpoint_tag": "claude-sonnet-4-6",    "version": "claude-sonnet-4-6",                      "model_origin": "International", "temp": 0.7, "provider": "relay", "relay_model": "claude-sonnet-4-6"},
    {"model": "deepseek-v4-pro-0813", "endpoint_tag": "deepseek-v4-pro-0813", "version": "deepseek-v4-pro-0813 / pro-ga-260813",   "model_origin": "Domestic",      "temp": 0.7, "provider": "pomo",  "relay_model": "deepseek-v4-pro-0813"},
    {"model": "qwen3.7-flash",        "endpoint_tag": "qwen3.7-flash",        "version": "qwen3.7-flash",                          "model_origin": "Domestic",      "temp": 0.7, "provider": "pomo",  "relay_model": "qwen3.7-flash"},
]

COMPANIONS = ["Solo", "Couple", "Family_Child", "Senior_70plus"]
BUDGETS = ["Low", "Medium", "High"]
SEASONS = ["Peak", "OffPeak"]
MOBILITIES = ["Unrestricted", "Wheelchair"]

CONTEXTS = []
for idx, (comp, bud, sea, mob) in enumerate(itertools.product(COMPANIONS, BUDGETS, SEASONS, MOBILITIES), 1):
    CONTEXTS.append({
        "context_id": f"CTX_{idx:02d}",
        "companion": comp,
        "budget": bud,
        "season": sea,
        "mobility": mob,
    })

STRATIFIED_12_CTX_IDS = [f"CTX_{i:02d}" for i in range(1, 49, 4)]

BATCH_SPECS = [
    ("B1",  "B1: Main Contextual Audit (48 Ctx x 4 Endpoints x 10 Cities x 3 Reps; Planned 5,760, Valid 5,752)"),
    ("B2",  "B2: Blank Unconditioned Baseline (1 Ctx x 4 Endpoints x 10 Cities x 3 Reps; Planned 120, Valid 119)"),
    ("B3",  "B3: Cross-Lingual English Prompt Control (12 Ctx x 4 Endpoints x 10 Cities x 2 Reps; N=960)"),
    ("B3b", "B3b: Cross-Lingual Chinese Prompt on Foreign Cities (12 Ctx x 4 Endpoints x 5 Intl Cities x 2 Reps; N=480)"),
    ("B4",  "B4: Paraphrased Prompt Templates T2/T3 (12 Ctx x 2 Tmpl x 4 Endpoints x 10 Cities; N=960)"),
    ("B5",  "B5: Greedy Decoding Temperature=0.0 Control (12 Ctx x 4 Endpoints x 10 Cities x 3 Reps; N=1,440)"),
    ("B6",  "B6: Intra-Day Temporal Stability Retest (12 Ctx x 4 Endpoints x 10 Cities x 1 Rep; Med. 7.8h, N=480)"),
]
