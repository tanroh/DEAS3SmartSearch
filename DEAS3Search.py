"""
DEA S3 Smart Search — Streamlit App
====================================
A portable, self-contained web app for searching Digital Earth Australia
satellite data via natural language questions.

Run with:
    streamlit run app.py

Requires ANTHROPIC_API_KEY in environment (or .env file / Streamlit secrets).
"""

import ast
import os
import re
import json
import time
import requests
import boto3
import pandas as pd
import folium
import streamlit as st
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from streamlit_folium import st_folium
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DEA S3 Search",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
S3_BUCKET     = "dea-public-data"
S3_REGION     = "ap-southeast-2"
STAC_ENDPOINT = "https://explorer.dea.ga.gov.au/stac/"
S3_HTTP_BASE  = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

# ── DEA Collection catalogue ──────────────────────────────────────────────────
DEA_COLLECTIONS = [
    {
        "id": "ga_ls8c_ard_3",
        "sensor": "Landsat 8",
        "title": "Landsat 8 ARD",
        "description": "Surface reflectance ARD. Use for vegetation, land cover, change detection, NDVI, agriculture, drought, bushfire burn severity.",
        "tags": ["vegetation","land cover","change detection","NDVI","agriculture","drought","reflectance","landsat","bushfire","burn","fire"],
        "temporal": "2013–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls9c_ard_3",
        "sensor": "Landsat 9",
        "title": "Landsat 9 ARD",
        "description": "Continuation of Landsat 8 mission from 2022 onward.",
        "tags": ["vegetation","land cover","recent","2022","2023","2024","landsat","bushfire","burn","fire"],
        "temporal": "2022–present",
        "resolution": "30m",
    },
    {
        "id": "ga_s2am_ard_3",
        "sensor": "Sentinel-2A",
        "title": "Sentinel-2A ARD",
        "description": "High-resolution (10m) surface reflectance. Better spatial detail than Landsat.",
        "tags": ["high resolution","urban","coastal","agriculture","detail","sentinel","bushfire","burn","fire"],
        "temporal": "2017–present",
        "resolution": "10m",
    },
    {
        "id": "ga_s2bm_ard_3",
        "sensor": "Sentinel-2B",
        "title": "Sentinel-2B ARD",
        "description": "Paired with 2A for ~5-day revisit at 10m resolution.",
        "tags": ["high resolution","urban","coastal","agriculture","detail","sentinel","bushfire","burn","fire"],
        "temporal": "2017–present",
        "resolution": "10m",
    },
    {
        "id": "ga_ls_wo_3",
        "sensor": "Landsat",
        "title": "Water Observations (WOfS)",
        "description": "Detects water presence. Use for flood mapping, wetland monitoring, reservoir tracking.",
        "tags": ["water","flood","wetland","inundation","reservoir","drought","river","lake"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_fc_3",
        "sensor": "Landsat",
        "title": "Fractional Cover",
        "description": "Estimates ground cover as fractions of green vegetation, non-green vegetation, and bare soil.",
        "tags": ["vegetation","bare soil","ground cover","pasture","grazing","land degradation","agriculture","drought"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_tcw_percentiles_2",
        "sensor": "Landsat",
        "title": "Tasseled Cap Wetness",
        "description": "Summarises landscape wetness over time. Useful for identifying persistently wet areas.",
        "tags": ["wetness","moisture","seasonal","vegetation stress","time series"],
        "temporal": "1986–present",
        "resolution": "30m",
    },
    {
        "id": "ga_srtm_dem1sv1_0",
        "sensor": "SRTM",
        "title": "Digital Elevation Model",
        "description": "Elevation from the Shuttle Radar Topography Mission. Use for terrain, flood modelling, hydrology.",
        "tags": ["elevation","DEM","terrain","flood modelling","slope","topography","hydrology"],
        "temporal": "2000 (static)",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_landcover_class_cyear_2",
        "sensor": "Landsat",
        "title": "DEA Land Cover",
        "description": "Annual land cover classification. Detects changes in land use across Australia.",
        "tags": ["land cover","land use","classification","urban","forest","clearing","change"],
        "temporal": "1988–present",
        "resolution": "30m",
    },
    {
        "id": "ga_ls_mangrove_cover_cyear_3",
        "sensor": "Landsat",
        "title": "Mangrove Cover",
        "description": "Annual mapping of mangrove extent and canopy cover around Australian coastlines.",
        "tags": ["mangrove","coastal","wetland","canopy","intertidal","marine"],
        "temporal": "1987–present",
        "resolution": "30m",
    },
]

EXAMPLE_QUESTIONS = [
    "Which areas in the Riverina had standing water after the 2022 floods?",
    "Show me vegetation stress across the Murray-Darling Basin in summer 2023",
    "Before and after scenes of the 2019-20 Black Summer fires in East Gippsland",
    "How has urban extent changed in Greater Western Sydney since 2018?",
    "Get mangrove cover data along the Gulf of Carpentaria for the last 5 years",
    "Show elevation data for flood-prone areas near Lismore",
    "Detect surface disturbance near mine sites in the Pilbara in 2023",
    "Find areas in the QPRC LGA affected by 2022 bushfires",
]

# ── Cached clients ────────────────────────────────────────────────────────────
@st.cache_resource
def get_s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(signature_version=UNSIGNED),
    )

@st.cache_resource
def get_ai_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

# ── Geocoding ─────────────────────────────────────────────────────────────────
def _build_query_variants(place_name: str) -> list:
    """Return query strings to try, from most to least specific."""
    variants = [place_name]
    if not place_name.lower().endswith("australia"):
        variants.append(place_name + ", Australia")
    # Strip suffixes that confuse geocoders
    stripped = re.sub(
        r'\b(Regional Council|City Council|Shire Council|Council|'
        r'Local Government Area|LGA|Shire|Region|District)\b',
        "", place_name, flags=re.IGNORECASE,
    ).strip(" ,")
    if stripped and stripped.lower() != place_name.lower():
        variants.append(stripped)
        variants.append(stripped + ", Australia")
    return variants


def _geocode_geoapify(place_name: str, api_key: str) -> dict:
    """Try Geoapify geocoding API. Returns result dict or None."""
    for query in _build_query_variants(place_name):
        try:
            resp = requests.get(
                "https://api.geoapify.com/v1/geocode/search",
                params={
                    "text": query,
                    "filter": "countrycode:au",
                    "limit": 1,
                    "apiKey": api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if features:
                f = features[0]
                props = f["properties"]
                lon, lat = f["geometry"]["coordinates"]
                bb = props.get("bbox")
                if bb:
                    bbox = [bb[0], bb[1], bb[2], bb[3]]
                else:
                    bbox = [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]
                return {
                    "place":    props.get("formatted", place_name),
                    "bbox":     bbox,
                    "centroid": [lon, lat],
                }
        except Exception:
            pass
    return None


def _geocode_nominatim(place_name: str) -> dict:
    """Try Nominatim with retries and backoff. Returns result dict or None."""
    for query in _build_query_variants(place_name):
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": 1,
                            "addressdetails": 0, "countrycodes": "au"},
                    headers={"User-Agent": "dea-streamlit-search/1.0 rohan.tankey@gmail.com"},
                    timeout=10,
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if data:
                    g = data[0]
                    lon, lat = float(g["lon"]), float(g["lat"])
                    if "boundingbox" in g:
                        s, n, w, e = [float(v) for v in g["boundingbox"]]
                        bbox = [w, s, e, n]
                    else:
                        bbox = [lon - 0.5, lat - 0.5, lon + 0.5, lat + 0.5]
                    return {
                        "place":    g.get("display_name", place_name),
                        "bbox":     bbox,
                        "centroid": [lon, lat],
                    }
            except requests.RequestException:
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_place(place_name: str) -> dict:
    """
    Convert a place name to a WGS84 bounding box.
    Tries Geoapify first (if GEOAPIFY_API_KEY secret is set), falls back to Nominatim.
    Add GEOAPIFY_API_KEY to Streamlit Secrets for reliable geocoding of Australian LGAs.
    Free tier: https://www.geoapify.com/ (3,000 requests/day, no rate limiting issues).
    """
    geoapify_key = os.environ.get("GEOAPIFY_API_KEY", "")

    result = _geocode_geoapify(place_name, geoapify_key) if geoapify_key else None

    if result is None:
        result = _geocode_nominatim(place_name)

    if result is None:
        raise ValueError(
            f'Could not geocode "{place_name}". '
            "Add GEOAPIFY_API_KEY to Streamlit Secrets for more reliable geocoding "
            "(free at geoapify.com — handles Australian LGAs and place names well)."
        )

    return result

# ── Claude question parser ────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a geospatial assistant for Digital Earth Australia (DEA). "
    "Extract search parameters from the user's question. "
    "Respond ONLY with valid JSON — no markdown fences, no commentary:\n"
    "{\n"
    '  "place_name": "<full resolvable place name — ALWAYS expand acronyms and abbreviations '
    'to their full geocodable form. Examples: QPRC -> Queanbeyan-Palerang, '
    'MDB -> Murray-Darling Basin, SEQ -> South East Queensland, '
    'FNQ -> Far North Queensland, SWWA -> South West Western Australia, '
    'ACT -> Australian Capital Territory, NQ -> North Queensland. '
    'Drop suffixes that confuse geocoders: LGA, Local Government Area, '
    'Regional Council, City Council, Shire Council. '
    'Return null if no location is mentioned.>",\n'
    '  "start_date": "<YYYY-MM-DD>",\n'
    '  "end_date": "<YYYY-MM-DD>",\n'
    '  "collections": ["<collection_id>"],\n'
    '  "max_cloud_pct": <0-100>,\n'
    '  "reasoning": "<one sentence>"\n'
    "}\n"
    "Rules:\n"
    "- ALWAYS expand location acronyms/abbreviations to their full geocodable name\n"
    "- ALWAYS drop administrative suffixes: LGA, Local Government Area, Regional Council, "
    "City Council, Shire Council\n"
    "- If no specific location is mentioned, set place_name to null\n"
    "- Pick 1-2 most relevant collection IDs from the catalogue provided\n"
    "- Default to last 12 months if no date mentioned\n"
    "- For water/flood questions set max_cloud_pct=10\n"
    "- For bushfire/burn questions use Landsat or Sentinel ARD, set max_cloud_pct=20\n"
    "- For DEM/elevation set max_cloud_pct=100\n"
    f"- Today: {datetime.now().strftime('%Y-%m-%d')}"
)


def parse_question_with_claude(question: str) -> dict:
    """Use Claude to parse a natural language question into search params."""
    ai = get_ai_client()
    if not ai:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    catalogue_json = json.dumps(
        [{"id": c["id"], "title": c["title"], "description": c["description"], "tags": c["tags"]}
         for c in DEA_COLLECTIONS],
        indent=2,
    )
    user_msg = f"Question: {question}\n\nAvailable DEA collections:\n{catalogue_json}"

    response = ai.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def parse_question_local(question: str) -> dict:
    """Keyword-based fallback parser (no API key needed)."""
    q = question.lower()
    now = datetime.now()

    m = re.search(r"\b(20\d{2})(?:[–\-](20\d{2}))?\b", q)
    if m:
        y1 = int(m.group(1))
        y2 = int(m.group(2)) if m.group(2) else y1
        start_date, end_date = f"{y1}-01-01", f"{y2}-12-31"
    elif "last 5 year" in q:
        start_date = f"{now.year - 5}-01-01"
        end_date = now.strftime("%Y-%m-%d")
    elif "last year" in q:
        start_date = f"{now.year - 1}-01-01"
        end_date = f"{now.year - 1}-12-31"
    else:
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

    scored = []
    for col in DEA_COLLECTIONS:
        score = sum(2 for tag in col["tags"] if tag.lower() in q)
        if col["sensor"].lower() in q:
            score += 5
        scored.append((score, col["id"]))
    scored.sort(reverse=True)
    collections = [scored[0][1]] if scored[0][0] == 0 else [c for s, c in scored[:2] if s > 0]

    max_cloud = (10 if any(w in q for w in ["water", "flood"])
                 else 100 if any(w in q for w in ["elevation", "dem", "terrain"])
                 else 20)

    # Extract place name: find the longest capitalised multi-word phrase
    stopwords = {"Find","Show","Get","Track","Detect","Which","How","Before","After","Areas","Using"}
    candidates = re.findall(
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:LGA|NSW|VIC|QLD|SA|WA|TAS|NT|ACT))?)\b',
        question
    )
    candidates = [p for p in candidates if p not in stopwords and len(p) > 3]
    place_name = max(candidates, key=len) if candidates else None

    return {
        "place_name":   place_name,
        "start_date":   start_date,
        "end_date":     end_date,
        "collections":  collections,
        "max_cloud_pct": max_cloud,
        "reasoning":    "Matched by keyword analysis (local fallback — no API key).",
    }


def parse_question(question: str) -> dict:
    """Parse question with Claude, falling back to local parser if unavailable."""
    try:
        return parse_question_with_claude(question)
    except Exception:
        return parse_question_local(question)

# ── STAC search ───────────────────────────────────────────────────────────────
def stac_search(collection: str, bbox: list, start_date: str, end_date: str,
                max_results: int = 100) -> list:
    resp = requests.post(
        f"{STAC_ENDPOINT}search",
        json={
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "limit": max_results,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("features", [])


def build_asset_index(features: list, collection_id: str) -> pd.DataFrame:
    rows = []
    for feat in features:
        item_id = feat["id"]
        dt      = feat["properties"].get("datetime", "")
        cloud   = feat["properties"].get("eo:cloud_cover")
        bbox    = feat.get("bbox", [])
        for asset_name, asset in feat.get("assets", {}).items():
            href = asset.get("href", "")
            https_url = (href.replace(f"s3://{S3_BUCKET}/", f"{S3_HTTP_BASE}/")
                         if href.startswith("s3://") else href)
            rows.append({
                "item_id":    item_id,
                "datetime":   dt[:10] if dt else "",
                "collection": collection_id,
                "cloud_pct":  cloud,
                "asset":      asset_name,
                "media_type": asset.get("type", ""),
                "s3_href":    href,
                "https_url":  https_url,
                "bbox_str":   str(bbox),
            })
    return pd.DataFrame(rows)


def run_search(params: dict, bbox: list, max_results: int = 50) -> pd.DataFrame:
    frames = []
    for coll_id in params["collections"]:
        features = stac_search(coll_id, bbox,
                               params["start_date"], params["end_date"],
                               max_results=max_results)
        if not features:
            continue
        df = build_asset_index(features, coll_id)
        if params["max_cloud_pct"] < 100:
            df = df[df["cloud_pct"].isna() | (df["cloud_pct"] <= params["max_cloud_pct"])]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Map helper ────────────────────────────────────────────────────────────────
def make_map(bbox: list, results: pd.DataFrame = None) -> folium.Map:
    west, south, east, north = bbox
    centre = [(south + north) / 2, (west + east) / 2]
    m = folium.Map(location=centre, zoom_start=7, tiles="CartoDB dark_matter")

    folium.Rectangle(
        bounds=[[south, west], [north, east]],
        color="#5fd85f", weight=2, fill=True, fill_opacity=0.05,
    ).add_to(m)

    if results is not None and not results.empty:
        seen = set()
        for _, row in results.iterrows():
            if row["item_id"] in seen:
                continue
            seen.add(row["item_id"])
            try:
                b = ast.literal_eval(row["bbox_str"])
                if len(b) == 4:
                    w2, s2, e2, n2 = b
                    cloud_tip = (f"<br>☁ {row['cloud_pct']:.0f}%"
                                 if row["cloud_pct"] is not None else "")
                    folium.Rectangle(
                        bounds=[[s2, w2], [n2, e2]],
                        color="#3a9e6e", weight=1,
                        fill=True, fill_opacity=0.08,
                        tooltip=f"{row['item_id']}<br>{row['datetime']}{cloud_tip}",
                    ).add_to(m)
            except Exception:
                pass
    return m

# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(active_collection_ids: list = None):
    with st.sidebar:
        st.markdown("## 🛰️ DEA Collections")
        for col in DEA_COLLECTIONS:
            active = bool(active_collection_ids and col["id"] in active_collection_ids)
            bg     = "background-color:#1a3a1a;" if active else ""
            border = "border-left:3px solid #5fd85f;" if active else "border-left:3px solid #2a3a2a;"
            weight = "600" if active else "400"
            colour = "#c8ffc8" if active else "#8aaa8a"
            st.markdown(
                f"""<div style='padding:8px 10px;margin-bottom:6px;border-radius:4px;{bg}{border}'>
                <span style='font-size:13px;font-weight:{weight};color:{colour}'>{col["title"]}</span><br>
                <span style='font-size:10px;color:#5a7a5a;font-family:monospace'>{col["id"]}</span><br>
                <span style='font-size:10px;color:#4a6a4a'>{col["resolution"]} · {col["temporal"]}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("**Links**")
        st.markdown("- [DEA Explorer](https://explorer.dea.ga.gov.au/)")
        st.markdown("- [Knowledge Hub](https://knowledge.dea.ga.gov.au/)")
        st.markdown("- [STAC API](https://explorer.dea.ga.gov.au/stac/)")

# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <style>
    /* Style example buttons as small pill-shaped tags */
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(150,150,150,0.4) !important;
        border-radius: 999px !important;
        font-size: 0.75rem !important;
        padding: 0.2rem 0.6rem !important;
        white-space: normal !important;
        text-align: left !important;
        line-height: 1.3 !important;
        height: auto !important;
        min-height: unset !important;
    }
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        border-color: rgba(150,150,150,0.8) !important;
        background: rgba(150,150,150,0.08) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛰️ DEA S3 Smart Search")
    st.caption("Ask a plain English question about the Australian landscape — returns S3 asset URLs.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "⚠️  `ANTHROPIC_API_KEY` not found. "
            "Using local keyword parser — set the key in Streamlit Secrets for Claude-powered parsing.",
            icon="🔑",
        )

    # ── Input ──
    # Initialise question value from session state so example buttons can populate it
    if "question" not in st.session_state:
        st.session_state["question"] = ""

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Your question",
            value=st.session_state["question"],
            placeholder="e.g. Find areas in the QPRC LGA affected by 2022 bushfires",
            height=100,
            label_visibility="collapsed",
            key="question_input",
        )
        # Keep session state in sync with manual edits
        st.session_state["question"] = question
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")
        max_results = st.slider("Max scenes", 5, 100, 20, step=5)

    # ── Example questions ──
    st.caption("Examples — click to populate the search box:")
    cols = st.columns(4)
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        label = ex[:52] + ("…" if len(ex) > 52 else "")
        if cols[i % 4].button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state["question"] = ex
            st.rerun()

    # ── Run search ──
    if search_clicked and question.strip():
        render_sidebar()

        with st.status("Running search…", expanded=True) as status_box:

            # Step 1: parse
            st.write("🧠 Parsing question…")
            try:
                params = parse_question(question)
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                return

            if not params.get("place_name"):
                st.warning(
                    "⚠️ Couldn't identify a specific location in your question. "
                    "Try adding a place name — e.g. 'in the Pilbara' or 'near Broken Hill'."
                )
                st.stop()

            st.write(f"📍 Location   : **{params['place_name']}**")
            st.write(f"📅 Dates      : **{params['start_date']}** → **{params['end_date']}**")
            st.write(f"🛰  Collections: **{', '.join(params['collections'])}**")
            st.write(f"☁️  Max cloud  : **{params['max_cloud_pct']}%**")
            if params.get("reasoning"):
                st.write(f"💡 _{params['reasoning']}_")

            # Step 2: geocode
            st.write("🗺  Geocoding location…")
            try:
                geo = geocode_place(params["place_name"])
                bbox = geo["bbox"]
                st.write(f"✅ `{geo['place']}`")
                st.write(f"   BBox: `{[round(v, 3) for v in bbox]}`")
            except Exception as e:
                st.error(f"Geocoding failed: {e}")
                return

            # Step 3: STAC search
            st.write("🔍 Querying STAC API…")
            try:
                results = run_search(params, bbox, max_results=max_results)
            except Exception as e:
                st.error(f"STAC search failed: {e}")
                return

            n_scenes = results["item_id"].nunique() if not results.empty else 0
            n_assets = len(results)
            st.write(f"✅ Found **{n_scenes} scenes** / **{n_assets} assets**")
            status_box.update(label="Search complete ✅", state="complete")

        render_sidebar(active_collection_ids=params["collections"])

        if results.empty:
            st.info("No results found. Try adjusting the date range, cloud cover threshold, or location.")
            return

        # ── Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scenes", n_scenes)
        m2.metric("Assets", n_assets)
        avg_cloud = results["cloud_pct"].dropna().mean()
        m3.metric("Avg cloud cover", f"{avg_cloud:.1f}%" if not pd.isna(avg_cloud) else "N/A")
        m4.metric("Collections", results["collection"].nunique())

        tab_map, tab_table, tab_urls, tab_code = st.tabs(
            ["🗺 Map", "📋 Table", "🔗 URLs", "📓 Notebook code"]
        )

        with tab_map:
            st_folium(make_map(bbox, results), width="100%", height=500, returned_objects=[])

        with tab_table:
            display_cols = [c for c in
                            ["item_id","datetime","collection","cloud_pct","asset","media_type"]
                            if c in results.columns]
            st.dataframe(
                results[display_cols].sort_values(["datetime","item_id"]),
                use_container_width=True, height=400,
            )
            st.download_button("⬇️ Download CSV", results.to_csv(index=False),
                               "dea_results.csv", "text/csv")

        with tab_urls:
            st.markdown("**HTTPS-accessible S3 asset URLs** — click to open/download directly.")
            unique_items = results["item_id"].unique()
            for item_id in unique_items[:30]:
                item_rows = results[results["item_id"] == item_id]
                dt = item_rows["datetime"].iloc[0]
                cloud = item_rows["cloud_pct"].iloc[0]
                cloud_str = f" · ☁ {cloud:.0f}%" if pd.notna(cloud) else ""
                with st.expander(f"📦 {item_id}  ({dt}{cloud_str})"):
                    for _, row in item_rows.iterrows():
                        url = row["https_url"]
                        if url:
                            st.markdown(f"`{row['asset']:30s}` [{url.split('/')[-1]}]({url})")
            if len(unique_items) > 30:
                st.caption(f"Showing first 30 of {len(unique_items)} scenes. Download CSV for all.")

        with tab_code:
            st.markdown("**Reproduce this search in Python:**")
            code = f'''import requests, pandas as pd

S3_BUCKET     = "dea-public-data"
S3_REGION     = "ap-southeast-2"
STAC_ENDPOINT = "https://explorer.dea.ga.gov.au/stac/"
S3_HTTP_BASE  = f"https://{{S3_BUCKET}}.s3.{{S3_REGION}}.amazonaws.com"

params = {json.dumps({k: v for k, v in params.items() if k != "reasoning"}, indent=4)}
bbox   = {bbox}

resp = requests.post(
    STAC_ENDPOINT + "search",
    json={{
        "collections": params["collections"],
        "bbox": bbox,
        "datetime": f"{{params['start_date']}}/{{params['end_date']}}",
        "limit": {max_results},
    }}
)
features = resp.json()["features"]

rows = []
for feat in features:
    cloud = feat["properties"].get("eo:cloud_cover")
    if cloud is not None and cloud > params["max_cloud_pct"]:
        continue
    for name, asset in feat["assets"].items():
        href = asset.get("href", "")
        rows.append({{
            "item_id":   feat["id"],
            "datetime":  feat["properties"].get("datetime", "")[:10],
            "asset":     name,
            "s3_href":   href,
            "https_url": href.replace(f"s3://{{S3_BUCKET}}/", f"{{S3_HTTP_BASE}}/"),
            "cloud_pct": cloud,
        }})

df = pd.DataFrame(rows)
print(f"{{df['item_id'].nunique()}} scenes / {{len(df)}} assets")
df[["item_id", "datetime", "asset", "https_url"]].head(10)
'''
            st.code(code, language="python")

    else:
        render_sidebar()
        st.markdown("""
        ### How it works

        1. **Type a question** about the Australian landscape
        2. **Claude parses** the question into a location, date range, and recommended DEA collection(s)
        3. **Nominatim geocodes** the location to a bounding box
        4. **DEA STAC API** returns matching satellite scenes
        5. You get **HTTPS-accessible S3 URLs** for every asset — ready to download or use in analysis

        Results include a map of scene footprints, a filterable table, direct asset URLs, and
        copy-paste Python code to reproduce the search.
        """)


if __name__ == "__main__":
    main()
