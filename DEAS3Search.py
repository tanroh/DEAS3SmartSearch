"""
NASA Astrophysics Data Explorer
Browses JWST and TESS data via the MAST archive (astroquery)
"""

import streamlit as st
import pandas as pd
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image
import requests
from io import BytesIO
import lightkurve as lk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Astrophysics Explorer",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 NASA Astrophysics Explorer")
st.caption("Browse JWST and TESS observations via the MAST archive")

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")

    target_input = st.text_input(
        "Object name or RA Dec",
        value="NGC 628",
        help="Examples:  NGC 628  |  M31  |  261.7 -73.5  |  Trappist-1",
    )

    search_mode = st.radio(
        "Resolve target as",
        ["Object name", "RA / Dec (degrees)"],
        index=0,
    )

    missions = st.multiselect(
        "Missions",
        ["JWST", "TESS"],
        default=["JWST", "TESS"],
    )

    radius_arcmin = st.slider("Search radius (arcmin)", 1, 60, 3)

    max_results = st.slider("Max results per mission", 10, 200, 50)

    run_search = st.button("🔍 Search", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

# Solar system bodies astropy can resolve via ephemeris
SOLAR_SYSTEM_BODIES = {
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune",
    "pluto", "moon", "sun", "io", "europa", "ganymede", "callisto",
    "titan", "enceladus", "triton", "ceres", "eris",
}

@st.cache_data(show_spinner=False)
def resolve_target(target_str: str, mode: str):
    """Return a SkyCoord from a name or 'ra dec' string."""
    if mode == "Object name":
        # Check for solar system bodies first — SIMBAD won't have these
        if target_str.strip().lower() in SOLAR_SYSTEM_BODIES:
            from astropy.coordinates import get_body_barycentric
            from astropy.time import Time
            from astropy.coordinates import solar_system_ephemeris, ICRS
            import astropy.coordinates as coord_module
            with solar_system_ephemeris.set("builtin"):
                body = coord_module.get_body(target_str.strip().lower(), Time.now())
            coord = SkyCoord(ra=body.ra, dec=body.dec, frame="icrs")
            return coord

        # Try CDS Sesame (handles stars, galaxies, nebulae, exoplanet hosts)
        try:
            coord = SkyCoord.from_name(target_str)
            return coord
        except Exception:
            pass

        # Fallback: SIMBAD directly (column names vary by astroquery version)
        from astroquery.simbad import Simbad
        result = Simbad.query_object(target_str)
        if result is None or len(result) == 0:
            raise ValueError(
                f"Could not resolve '{target_str}'. "
                "Try an exact catalogue name (e.g. 'NGC 628', 'TRAPPIST-1') "
                "or switch to RA/Dec mode."
            )
        # Column names differ between astroquery versions: try both cases
        cols = {c.lower(): c for c in result.colnames}
        ra_col  = cols.get("ra",  cols.get("ra_d",  None))
        dec_col = cols.get("dec", cols.get("dec_d", None))
        if ra_col is None or dec_col is None:
            raise ValueError(f"Unexpected SIMBAD columns: {result.colnames}")
        ra_val  = result[ra_col][0]
        dec_val = result[dec_col][0]
        # SIMBAD returns sexagesimal strings for RA/Dec; degree columns end in _d
        if "d" in ra_col.lower():
            coord = SkyCoord(ra=float(ra_val), dec=float(dec_val), unit=u.deg, frame="icrs")
        else:
            coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        parts = target_str.split()
        if len(parts) != 2:
            raise ValueError("RA/Dec mode expects exactly two numbers, e.g. '261.7 -73.5'")
        coord = SkyCoord(ra=float(parts[0]), dec=float(parts[1]), unit=u.deg, frame="icrs")
    return coord


@st.cache_data(show_spinner=False)
def query_mast(ra_deg: float, dec_deg: float, radius_arcmin: float,
               mission: str, max_rows: int) -> pd.DataFrame:
    """Query MAST for observations near a coordinate."""
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg, frame="icrs")
    radius = u.Quantity(radius_arcmin, u.arcmin)
    obs = Observations.query_region(coord, radius=radius)
    if obs is None or len(obs) == 0:
        return pd.DataFrame()
    # Filter to the requested mission
    mask = [str(c).upper() == mission.upper() for c in obs["obs_collection"]]
    obs = obs[mask]
    if len(obs) == 0:
        return pd.DataFrame()

    df = obs.to_pandas()

    # Useful columns (keep only what's available)
    want = [
        "target_name", "obs_collection", "instrument_name", "filters",
        "t_exptime", "t_min", "t_max", "em_min", "em_max",
        "obs_id", "proposal_id", "dataURL", "jpegURL", "s_ra", "s_dec",
        "calib_level", "dataproduct_type",
    ]
    keep = [c for c in want if c in df.columns]
    df = df[keep].head(max_rows)

    # Human-readable date (t_min is MJD — use astropy Time to avoid overflow)
    if "t_min" in df.columns:
        from astropy.time import Time
        def mjd_to_date(val):
            try:
                return Time(float(val), format="mjd").to_datetime().date()
            except Exception:
                return None
        df["obs_date"] = df["t_min"].apply(mjd_to_date)

    return df


@st.cache_data(show_spinner=False)
def fetch_preview(url: str):
    """Download a JPEG preview image from MAST."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def search_lightcurve_meta(target_name: str) -> pd.DataFrame:
    """Return TESS light curve search results as a plain dataframe (cacheable)."""
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        return pd.DataFrame()
    return search.table.to_pandas()

def fetch_lightcurve(target_name: str):
    """Download the first available TESS light curve (not cached — lightkurve objects are not picklable)."""
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        return None
    return search[0].download()


# ── Main app ──────────────────────────────────────────────────────────────────

if run_search:
    # Resolve coordinates
    with st.spinner("Resolving target…"):
        try:
            coord = resolve_target(target_input.strip(), search_mode)
        except Exception as e:
            st.error(f"Could not resolve target: {e}")
            st.stop()

    st.success(
        f"**{target_input}** → RA {coord.ra.deg:.4f}°, Dec {coord.dec.deg:.4f}°"
    )

    # ── JWST tab ──────────────────────────────────────────────────────────────
    if "JWST" in missions:
        st.subheader("🪐 JWST Observations")
        with st.spinner("Querying MAST for JWST…"):
            df_jwst = query_mast(
                coord.ra.deg, coord.dec.deg,
                radius_arcmin, "JWST", max_results,
            )

        if df_jwst.empty:
            st.info("No JWST observations found for this target / radius.")
        else:
            st.caption(f"{len(df_jwst)} observations returned")

            # Display columns
            display_cols = [c for c in [
                "target_name", "instrument_name", "filters",
                "t_exptime", "obs_date", "proposal_id", "dataproduct_type", "calib_level",
            ] if c in df_jwst.columns]

            st.dataframe(
                df_jwst[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Image previews
            preview_rows = df_jwst[df_jwst["jpegURL"].notna()] if "jpegURL" in df_jwst.columns else pd.DataFrame()
            if not preview_rows.empty:
                st.markdown("#### Image Previews")
                cols = st.columns(4)
                shown = 0
                for _, row in preview_rows.iterrows():
                    if shown >= 8:
                        break
                    img = fetch_preview(row["jpegURL"])
                    if img:
                        label = f"{row.get('target_name','?')} · {row.get('filters','')}"
                        cols[shown % 4].image(img, caption=label, use_container_width=True)
                        shown += 1
                if shown == 0:
                    st.info("Preview images not available for these observations.")

            # Links to MAST portal
            if "obs_id" in df_jwst.columns:
                st.markdown("#### Links to MAST Portal")
                for _, row in df_jwst.head(10).iterrows():
                    oid = row["obs_id"]
                    url = f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={oid}"
                    st.markdown(f"- [{oid}]({url})")

    # ── TESS tab ──────────────────────────────────────────────────────────────
    if "TESS" in missions:
        st.subheader("⭐ TESS Observations")

        with st.spinner("Querying MAST for TESS…"):
            df_tess = query_mast(
                coord.ra.deg, coord.dec.deg,
                radius_arcmin, "TESS", max_results,
            )

        if df_tess.empty:
            st.info("No TESS observations found for this target / radius.")
        else:
            display_cols = [c for c in [
                "target_name", "instrument_name", "filters",
                "t_exptime", "obs_date", "proposal_id", "dataproduct_type", "calib_level",
            ] if c in df_tess.columns]
            st.caption(f"{len(df_tess)} observations returned")
            st.dataframe(df_tess[display_cols], use_container_width=True, hide_index=True)

        # Light curve
        st.markdown("#### Light Curve (TESS via lightkurve)")
        with st.spinner("Searching for light curves…"):
            try:
                lc_meta = search_lightcurve_meta(target_input.strip())
            except Exception as e:
                lc_meta = pd.DataFrame()
                st.warning(f"lightkurve search error: {e}")

        if not lc_meta.empty:
            with st.expander("Light curve search results"):
                st.dataframe(lc_meta, use_container_width=True, hide_index=True)

        lc = None
        with st.spinner("Downloading light curve…"):
            try:
                lc = fetch_lightcurve(target_input.strip())
            except Exception as e:
                st.warning(f"lightkurve download error: {e}")

        if lc is not None:

            fig, ax = plt.subplots(figsize=(10, 3))
            lc.scatter(ax=ax, s=1, c="steelblue")
            ax.set_title(f"TESS light curve — {target_input}")
            ax.set_xlabel("Time (BTJD)")
            ax.set_ylabel("Flux")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "No TESS light curve found via lightkurve for this target name. "
                "Try an exact catalogue name (e.g. 'TRAPPIST-1' or 'TIC 278683025')."
            )

else:
    st.info("Enter a target in the sidebar and press **Search** to begin.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data from [MAST](https://mast.stsci.edu) · "
    "Powered by [astroquery](https://astroquery.readthedocs.io) & "
    "[lightkurve](https://lightkurve.github.io/lightkurve/)"
)
