import os
import io
from datetime import date, timedelta
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# Data / Scoring Helpers (your existing QCMobile BASIC logic)
# ============================================================

def flatten_qcmobile_basics(raw_data: Any) -> List[Dict[str, Any]]:
    """
    Flatten QCMobile /carriers/:dotNumber/basics JSON into a list of simple dicts,
    one per BASIC.

    Handles both:
      1) {"content": [ { "basic": {...} }, ... ], "retrievalDate": "..."}
      2) [ { "basic": {...} }, ... ]
    """
    basics_flat: List[Dict[str, Any]] = []

    if isinstance(raw_data, list):
        content = raw_data
    elif isinstance(raw_data, dict):
        content = raw_data.get("content", [])
    else:
        return basics_flat

    if not isinstance(content, list):
        return basics_flat

    for entry in content:
        basic = (entry or {}).get("basic", {}) or {}
        basics_type = basic.get("basicsType", {}) or {}

        short_desc = basics_type.get("basicsShortDesc")
        code = basics_type.get("basicsCode")

        percentile_raw = basic.get("basicsPercentile")
        measure_raw = basic.get("measureValue")
        threshold_raw = basic.get("basicsViolationThreshold")

        exceeded = (basic.get("exceededFMCSAInterventionThreshold") or "").upper()
        onroad_def = (basic.get("onRoadPerformanceThresholdViolationIndicator") or "").upper()
        serious_def = (basic.get("seriousViolationFromInvestigationPast12MonthIndicator") or "").upper()

        total_insp_viol = basic.get("totalInspectionWithViolation")
        total_viol = basic.get("totalViolation")
        run_date = basic.get("basicsRunDate")

        percentile_num: Optional[float] = None
        if isinstance(percentile_raw, str):
            txt = percentile_raw.strip().lower()
            if txt.endswith("%"):
                try:
                    percentile_num = float(txt.replace("%", ""))
                except ValueError:
                    percentile_num = None

        measure_num: Optional[float] = None
        if measure_raw is not None:
            try:
                measure_num = float(str(measure_raw))
            except ValueError:
                measure_num = None

        threshold_num: Optional[float] = None
        if threshold_raw is not None:
            try:
                threshold_num = float(str(threshold_raw))
            except ValueError:
                threshold_num = None

        basics_flat.append(
            {
                "BASIC": short_desc or code,
                "BASIC Code": code,
                "Percentile": percentile_raw,
                "Percentile (num)": percentile_num,
                "Measure": measure_raw,
                "Measure (num)": measure_num,  # raw FMCSA measure, not our score
                "Violation Threshold": threshold_raw,
                "Violation Threshold (num)": threshold_num,
                "Exceeded FMCSA Threshold": exceeded,
                "On-Road Threshold Violation": onroad_def,
                "Serious Violation (12m)": serious_def,
                "Inspections With Violation (24m)": total_insp_viol,
                "Total Violations (24m)": total_viol,
                "Snapshot Date": run_date,
            }
        )

    return basics_flat


def evaluate_basic_risk(row: Dict[str, Any]) -> Tuple[str, int]:
    """
    Assign a risk band and numeric risk score to a BASIC based on:
    - Percentile vs FMCSA violation threshold (preferred)
    - Fallback: measureValue relative to basicsViolationThreshold
    - Intervention flags (exceeded threshold, on-road, serious violation)
    """
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")
    measure_num = row.get("Measure (num)")

    exceeded = row.get("Exceeded FMCSA Threshold") == "Y"
    onroad_def = row.get("On-Road Threshold Violation") == "Y"
    serious = row.get("Serious Violation (12m)") == "Y"

    band = "Unknown"
    score = 0

    if pct is not None and thresh is not None:
        if pct >= thresh:
            band = "High"
            score += 3
        elif pct >= (thresh - 10):
            band = "Medium"
            score += 2
        elif pct >= 20:
            band = "Low"
            score += 1
        else:
            band = "Very Low"
            score += 0
    elif measure_num is not None and thresh is not None and thresh > 0:
        ratio = measure_num / thresh
        if ratio >= 1.0:
            band = "High"
            score += 3
        elif ratio >= 0.75:
            band = "Medium"
            score += 2
        elif ratio >= 0.3:
            band = "Low"
            score += 1
        else:
            band = "Very Low"
            score += 0
    else:
        band = "Unknown"

    if exceeded:
        score += 2
    if onroad_def:
        score += 1
    if serious:
        score += 2

    return band, score


def compute_carrier_measure(basics_flat: List[Dict[str, Any]]) -> Optional[float]:
    """
    Preferred:
      measure = max(percentile) / 25  → 0–4 (lower is better)
    Fallback:
      ratio = measureValue / violationThreshold → pseudo-percentile
      measure = max(pseudo-percentile) / 25
    """
    percents = [
        b.get("Percentile (num)")
        for b in basics_flat
        if b.get("Percentile (num)") is not None
    ]

    if percents:
        return round(max(percents) / 25.0, 3)

    pseudo_percents: List[float] = []
    for b in basics_flat:
        m = b.get("Measure (num)")
        thresh = b.get("Violation Threshold (num)")
        if m is not None and thresh is not None and thresh > 0:
            pct_est = max(0.0, min(100.0, (m / thresh) * 100.0))
            pseudo_percents.append(pct_est)

    if not pseudo_percents:
        return None

    return round(max(pseudo_percents) / 25.0, 3)


def status_from_measure(measure: Optional[float]) -> Tuple[str, str]:
    if measure is None:
        return "Unknown", "⚪️"
    if measure < 1.5:
        return "Green", "🟢"
    if measure <= 4.0:
        return "Yellow", "🟡"
    return "Red", "🔴"


def summarize_insurance_impact(
    basics_flat: List[Dict[str, Any]],
    measure: Optional[float],
    overall_status: str,
) -> Tuple[str, str]:
    has_high = any(b.get("Risk Level") == "High" for b in basics_flat)
    has_medium = any(b.get("Risk Level") == "Medium" for b in basics_flat)
    has_exceeded = any(b.get("Exceeded FMCSA Threshold") == "Y" for b in basics_flat)

    if measure is None or overall_status == "Unknown":
        return (
            "Limited visibility",
            "Some BASIC percentiles are not public or insufficient. Underwriters will lean more on "
            "loss history, operations, and telematics. Use the Inspection Deep Dive below to show "
            "what’s actually happening on the roadside.",
        )

    if has_high or has_exceeded or overall_status == "Red":
        return (
            "High rate pressure",
            "At least one BASIC is above/near FMCSA intervention thresholds. Plan for roughly +15–30% "
            "rate pressure unless you show a clear corrective story.",
        )
    if has_medium or overall_status == "Yellow":
        return (
            "Moderate rate pressure",
            "Some BASICs are trending toward thresholds. Expect roughly +5–15% rate pressure unless "
            "you can demonstrate improvement.",
        )
    return (
        "Stable / favorable",
        "BASICs appear well below thresholds with no major flags. Expect flat to modest changes "
        "(roughly 0–5%) driven mostly by market conditions.",
    )


def action_items_for_basic(name: str) -> List[str]:
    name_lower = (name or "").lower()

    if "unsafe" in name_lower:
        return [
            "Coach drivers on speeding, following distance, and distractions (the usual ‘big 3’).",
            "Use telematics/cameras to spot harsh braking & speeding before inspections do.",
            "Audit lanes/routes where violations cluster and tighten dispatch expectations.",
        ]
    if "hos" in name_lower or "hours-of-service" in name_lower:
        return [
            "Run weekly ELD audits for form-and-manner + over-hours patterns.",
            "Tighten log edit rules (require documented reasons).",
            "Train drivers on 11/14-hour rules and common roadside traps.",
        ]
    if "driver fitness" in name_lower:
        return [
            "Audit CDLs, endorsements, and medical cards for expirations.",
            "Clean up DQ files and missing docs before renewal.",
            "Set reminders for expiring credentials and annual reviews.",
        ]
    if "drug" in name_lower or "alcohol" in name_lower or "substances" in name_lower:
        return [
            "Verify random testing rates + documentation are correct.",
            "Ensure positives/refusals are handled and documented consistently.",
            "Refresh driver training on the drug & alcohol policy.",
        ]
    if "vehicle maint" in name_lower or "maintenance" in name_lower:
        return [
            "Identify repeat units/shops driving violations (brakes, tires, lights).",
            "Tighten PM schedules and track completion compliance.",
            "Enforce DVIRs and close the loop on repairs.",
        ]
    return [
        "Review repeat patterns and assign an owner to each pattern.",
        "Prioritize fixes that reduce roadside stops/violations quickly.",
        "Document corrective actions for the renewal story.",
    ]


def trend_label_for_basic(row: Dict[str, Any]) -> str:
    band = row.get("Risk Level")
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")

    if band == "High":
        return "🔥 Above threshold"
    if band == "Medium":
        return "⚠️ Near threshold"
    if band == "Low":
        if pct is not None and thresh is not None and pct > (thresh - 15):
            return "● Stable, but watch"
        return "● Stable"
    if band == "Very Low":
        return "✅ Healthy"
    return "● No clear signal"


# ============================================================
# NEW: DOT Open Data “Inspection Deep Dive” (rolling 3 years)
# ============================================================

DOT_VIOLATIONS_DATASETS = [
    # Primary (as discussed)
    ("Vehicle Inspections and Violations", "https://data.transportation.gov/resource/876r-jsdb.json"),
    # Fallback (another related inspection dataset)
    ("Vehicle Inspection File", "https://data.transportation.gov/resource/fx4q-ay7w.json"),
]

# Candidate field names (Socrata schemas vary; we try multiple)
CANDIDATE_FIELDS = {
    "dot": ["dot_number", "usdot", "dot", "carrier_dot_number"],
    "inspection_date": ["inspection_date", "insp_date", "date", "inspection_dt", "inspectiondate"],
    "inspection_id": ["inspection_id", "inspection_number", "insp_id", "report_number", "inspectionreportnumber"],
    "violation_code": ["violation_code", "viol_code", "violation", "code"],
    "oos": ["oos", "out_of_service", "oos_violation", "vehicle_oos", "driver_oos"],
    "state": ["state", "report_state", "inspection_state"],
    "level": ["inspection_level", "level", "insp_level"],
}


def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


@st.cache_data(show_spinner=False, ttl=60 * 30)  # 30 minutes
def fetch_inspection_violations_rolling_3y(
    usdot: str,
    years: int = 3,
    limit: int = 5000,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Pull inspection/violation rows for the last N years from DOT Socrata.

    Returns:
      (df, dataset_name, dataset_url_used)
    """
    usdot = str(usdot).strip()
    if not usdot:
        return pd.DataFrame(), "", ""

    start_date = (date.today() - timedelta(days=365 * years)).isoformat()

    app_token = os.getenv("SOCRATA_APP_TOKEN")
    headers = {"Accept": "application/json"}
    if app_token:
        headers["X-App-Token"] = app_token

    last_error = ""
    for dataset_name, base_url in DOT_VIOLATIONS_DATASETS:
        # Try different schemas: we attempt a few where-clauses with likely date fields.
        where_templates = [
            # Most common guesses
            ("dot_number", "inspection_date"),
            ("dot_number", "insp_date"),
            ("dot_number", "change_date"),  # often exists as update date
            ("usdot", "inspection_date"),
            ("usdot", "change_date"),
        ]

        for dot_field_guess, date_field_guess in where_templates:
            params = {
                "$limit": limit,
                "$order": f"{date_field_guess} DESC",
                "$where": f"{dot_field_guess} = '{usdot}' AND {date_field_guess} >= '{start_date}'",
            }
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    df = pd.DataFrame(data)
                    if df.empty:
                        continue
                    df["__source_dataset__"] = dataset_name
                    df["__source_url__"] = base_url
                    return df, dataset_name, base_url
                else:
                    last_error = f"{dataset_name}: {r.status_code} {r.text[:200]}"
            except Exception as e:
                last_error = f"{dataset_name}: {e}"

    return pd.DataFrame(), "ERROR", last_error


def build_deep_dive_views(df_raw: pd.DataFrame) -> Dict[str, Any]:
    """
    Normalize and compute:
      - violations per month
      - inspections per month
      - top violation codes
      - recent violations table
    """
    out: Dict[str, Any] = {
        "df_norm": pd.DataFrame(),
        "kpis": {},
        "violations_by_month": pd.DataFrame(),
        "inspections_by_month": pd.DataFrame(),
        "top_codes": pd.DataFrame(),
        "recent_rows": pd.DataFrame(),
    }
    if df_raw.empty:
        return out

    df = df_raw.copy()

    dot_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["dot"])
    date_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["inspection_date"])
    insp_id_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["inspection_id"])
    viol_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["violation_code"])
    oos_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["oos"])
    state_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["state"])
    level_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["level"])

    # Normalize dates
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # If we truly can’t find an inspection date, we still return raw
        out["df_norm"] = df
        return out

    df = df.dropna(subset=[date_col])
    df["inspection_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # Violations: if each row is a violation, count rows.
    # Inspections: prefer unique inspection_id; else approximate by unique date+state+level.
    if insp_id_col:
        inspections = df[[insp_id_col, "inspection_month"]].dropna().drop_duplicates()
        inspections_by_month = (
            inspections.groupby("inspection_month")[insp_id_col]
            .nunique()
            .reset_index(name="inspections")
        )
    else:
        approx_cols = ["inspection_month"]
        if state_col:
            approx_cols.append(state_col)
        if level_col:
            approx_cols.append(level_col)

        inspections = df[approx_cols].drop_duplicates()
        inspections_by_month = (
            inspections.groupby("inspection_month")
            .size()
            .reset_index(name="inspections")
        )

    violations_by_month = (
        df.groupby("inspection_month")
        .size()
        .reset_index(name="violations")
    )

    # Top violation codes
    if viol_col:
        top_codes = (
            df[viol_col]
            .astype(str)
            .value_counts()
            .head(15)
            .reset_index()
        )
        top_codes.columns = ["violation_code", "count"]
    else:
        top_codes = pd.DataFrame(columns=["violation_code", "count"])

    # Recent rows view
    recent_cols = []
    for c in [date_col, insp_id_col, viol_col, oos_col, state_col, level_col]:
        if c and c in df.columns and c not in recent_cols:
            recent_cols.append(c)

    recent_rows = df.sort_values(date_col, ascending=False).head(50)[recent_cols] if recent_cols else df.head(50)

    # KPIs
    total_violations = int(len(df))
    total_inspections = int(inspections.shape[0]) if inspections is not None else 0

    oos_rate = None
    if oos_col and oos_col in df.columns:
        # try to interpret Y/1/True as OOS
        oos_series = df[oos_col].astype(str).str.upper()
        oos_rate = float((oos_series.isin(["Y", "YES", "1", "TRUE", "T"])).mean())

    out["df_norm"] = df
    out["kpis"] = {
        "total_violations": total_violations,
        "total_inspections": total_inspections,
        "oos_rate": oos_rate,
        "date_field_used": date_col,
        "inspection_id_field_used": insp_id_col,
        "violation_code_field_used": viol_col,
    }
    out["violations_by_month"] = violations_by_month
    out["inspections_by_month"] = inspections_by_month
    out["top_codes"] = top_codes
    out["recent_rows"] = recent_rows

    return out


# ============================================================
# NEW: Broker-facing export (CSV now, PDF optional)
# ============================================================

def download_csv_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def make_simple_pdf_report_bytes(
    title: str,
    subtitle: str,
    sections: List[Tuple[str, List[str]]],
    table_df: Optional[pd.DataFrame] = None,
) -> Optional[bytes]:
    """
    Minimal PDF generator. Uses reportlab if available.
    If reportlab isn't installed yet, returns None (and UI will hide).
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except Exception:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    x = 0.75 * inch
    y = height - 0.9 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 0.3 * inch

    c.setFont("Helvetica", 11)
    c.drawString(x, y, subtitle)
    y -= 0.4 * inch

    c.setFont("Helvetica", 10)

    def draw_wrapped(lines: List[str], y_pos: float) -> float:
        for line in lines:
            if y_pos < 1.0 * inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                y_pos = height - 0.9 * inch
            c.drawString(x, y_pos, line[:120])
            y_pos -= 0.18 * inch
        return y_pos

    for header, lines in sections:
        if y < 1.2 * inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 0.9 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, header)
        y -= 0.25 * inch
        c.setFont("Helvetica", 10)
        y = draw_wrapped(lines, y)
        y -= 0.15 * inch

    if table_df is not None and not table_df.empty:
        if y < 2.0 * inch:
            c.showPage()
            y = height - 0.9 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Appendix: Recent Inspection / Violation Rows")
        y -= 0.3 * inch

        c.setFont("Helvetica", 8)
        cols = list(table_df.columns)[:6]
        table_small = table_df[cols].head(20).astype(str)

        # Simple fixed-width table
        col_width = (width - 1.5 * inch) / len(cols)
        for col_i, col_name in enumerate(cols):
            c.drawString(x + col_i * col_width, y, col_name[:18])
        y -= 0.2 * inch

        for _, row in table_small.iterrows():
            if y < 1.0 * inch:
                c.showPage()
                y = height - 0.9 * inch
                c.setFont("Helvetica", 8)
            for col_i, col_name in enumerate(cols):
                c.drawString(x + col_i * col_width, y, str(row[col_name])[:18])
            y -= 0.18 * inch

    c.save()
    buffer.seek(0)
    return buffer.read()


# ============================================================
# UI Helpers
# ============================================================

def render_gauge(measure: Optional[float]):
    if measure is None:
        st.info("Score not available – BASIC percentiles are not public or insufficient.")
        return

    max_scale = 4.0
    value = min(max(measure, 0.0), max_scale)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 30}},
            title={"text": "", "font": {"size": 1}},
            gauge={
                "axis": {"range": [0, max_scale]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 1.5], "color": "#22c55e"},
                    {"range": [1.5, 4.0], "color": "#eab308"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def risk_emoji(level: str) -> str:
    return {
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟠",
        "Very Low": "🟢",
    }.get(level, "⚪️")


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛", layout="wide")
st.title("🚛 Credit Karma for Truckers")

st.write(
    "Turn FMCSA safety data into a simple safety score, insurance impact, and a clear action plan. "
    "Lower is better (golf score)."
)

with st.expander("Where is this data coming from? (so you can defend it to clients)", expanded=False):
    st.markdown(
        """
**This app blends two public sources + optional broker-provided data:**

1) **FMCSA QCMobile API (BASIC snapshot)**  
   - What you see: BASIC names, thresholds, intervention flags, counts, and sometimes percentiles.  
   - Why: great for a fast “credit-score style” snapshot.

2) **DOT Open Data (Inspection-level detail)**  
   - What you see: inspection/violation rows over time (the “meat of the matter”).  
   - Why: this is what you use to explain *trends* and *what is actually driving the BASICs*.

3) **Optional Uploads (not public data)**  
   - Claims / loss runs and driver citations are usually private—if you upload them, we overlay them into the story.
        """
    )

col_input, col_info = st.columns([2, 3])

with col_input:
    usdot = st.text_input("USDOT Number", placeholder="e.g. 44110")
    st.caption("Enter a carrier's USDOT number to see their BASIC profile + inspection deep dive.")

with col_info:
    st.info(
        "This is a translation layer between FMCSA roadside safety data and how underwriters talk about risk. "
        "It is **not** an official FMCSA or insurer rating."
    )

tabs = st.tabs(["Snapshot (BASICs)", "Inspection Deep Dive (3y)", "Add Your Data (optional)", "Broker Exports"])

# ---------------------------
# SNAPSHOT (BASICs)
# ---------------------------
with tabs[0]:
    if st.button("Check Carrier", type="primary", key="check_carrier"):
        if not usdot.strip():
            st.error("Please enter a USDOT number.")
        else:
            webkey = os.getenv("QCMOBILE_WEBKEY")
            if not webkey:
                st.error(
                    "QCMOBILE_WEBKEY is not set in Streamlit secrets. "
                    "Go to your app's Settings → Secrets and add it."
                )
            else:
                base_url = "https://mobile.fmcsa.dot.gov/qc/services"
                url = f"{base_url}/carriers/{usdot.strip()}/basics"
                params = {"webKey": webkey}

                try:
                    resp = requests.get(url, params=params, timeout=15)
                    resp.raise_for_status()
                    raw_data = resp.json()

                    basics_flat = flatten_qcmobile_basics(raw_data)

                    if not basics_flat:
                        st.warning(
                            "No BASIC data returned for this USDOT. "
                            "Check that the DOT number is valid and has BASICs."
                        )
                    else:
                        for b in basics_flat:
                            band, score = evaluate_basic_risk(b)
                            b["Risk Level"] = band
                            b["Risk Score"] = score
                            b["Trend Label"] = trend_label_for_basic(b)

                        measure = compute_carrier_measure(basics_flat)
                        status, emoji = status_from_measure(measure)
                        impact_band, impact_text = summarize_insurance_impact(basics_flat, measure, status)

                        df = pd.DataFrame(basics_flat)
                        df_sorted = df.sort_values(
                            by=["Risk Score", "Percentile (num)"],
                            ascending=[False, False],
                        )

                        st.session_state["basics_df"] = df
                        st.session_state["basics_raw"] = raw_data
                        st.session_state["carrier_measure"] = measure
                        st.session_state["carrier_status"] = status
                        st.session_state["impact_band"] = impact_band
                        st.session_state["impact_text"] = impact_text

                        st.markdown("---")
                        top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.6])

                        with top_left:
                            st.subheader("Safety Score")
                            render_gauge(measure)
                            if measure is not None:
                                st.caption("Lower is better. Based on worst BASIC signal (percentile or fallback).")

                        with top_mid:
                            st.subheader("Status")
                            if measure is None:
                                st.info("⚪️ Unknown – score not available.")
                            else:
                                if status == "Green":
                                    st.success(f"{emoji} {status} – {measure}")
                                elif status == "Yellow":
                                    st.warning(f"{emoji} {status} – {measure}")
                                else:
                                    st.error(f"{emoji} {status} – {measure}")
                                st.caption("Green ≈ good credit, Red ≈ substandard.")

                        with top_right:
                            st.subheader("Insurance Impact (underwriting language)")
                            st.write(f"**{impact_band}**")
                            st.write(impact_text)

                        st.markdown("---")
                        st.subheader("Top Factors Affecting Your Insurance (from BASIC snapshot)")
                        top_factors = df_sorted.head(3).to_dict(orient="records")

                        cols = st.columns(len(top_factors)) if top_factors else []
                        for col, b in zip(cols, top_factors):
                            with col:
                                name = b.get("BASIC") or b.get("BASIC Code")
                                risk_level = b.get("Risk Level")
                                pct = b.get("Percentile")
                                thresh = b.get("Violation Threshold")
                                insp = b.get("Inspections With Violation (24m)")
                                viol = b.get("Total Violations (24m)")
                                trend = b.get("Trend Label")

                                st.markdown(f"**{name}**")
                                st.caption(trend)
                                st.metric(label="Risk", value=f"{risk_emoji(risk_level)} {risk_level}")

                                if pct is not None and thresh is not None:
                                    st.write(f"- {pct} vs threshold {thresh}")
                                if insp is not None:
                                    st.write(f"- {insp} inspections w/ violations (24m)")
                                if viol is not None:
                                    st.write(f"- {viol} total violations (24m)")

                        st.markdown("---")
                        st.subheader("Focus Areas & Action Plan (fast, broker-friendly)")
                        for b in top_factors:
                            name = b.get("BASIC") or b.get("BASIC Code")
                            risk_level = b.get("Risk Level")
                            st.markdown(f"### {name} – {risk_emoji(risk_level)} {risk_level}")
                            for item in action_items_for_basic(name)[:3]:
                                st.write(f"- {item}")

                        st.markdown("---")
                        st.subheader("BASIC Details (source: FMCSA QCMobile snapshot)")
                        st.dataframe(df, use_container_width=True)

                        with st.expander("Raw API Response (debug)"):
                            st.json(raw_data)

                except requests.exceptions.HTTPError as e:
                    st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
                except Exception as e:
                    st.error(f"Unexpected error calling QCMobile API: {e}")

    else:
        st.caption("Click **Check Carrier** to load BASIC snapshot.")

# ---------------------------
# INSPECTION DEEP DIVE (3y rolling)
# ---------------------------
with tabs[1]:
    st.subheader("Inspection Deep Dive (rolling last 3 years)")
    st.caption(
        "This section is designed for the broker conversation: trends, repeat violations, and the clearest “point of attack.” "
        "Source: DOT Open Data (Socrata)."
    )

    if not usdot.strip():
        st.info("Enter a USDOT number above, then use the Snapshot tab to check the carrier.")
    else:
        with st.spinner("Pulling inspection/violation rows from DOT Open Data (last 3 years)…"):
            df_raw, ds_name, ds_url = fetch_inspection_violations_rolling_3y(usdot.strip(), years=3, limit=5000)

        if ds_name == "ERROR":
            st.error("Could not query DOT Open Data for inspections/violations.")
            st.code(ds_url)
        elif df_raw.empty:
            st.warning("No inspection/violation rows returned for the last 3 years (or the dataset schema didn’t match expected fields).")
            st.caption("If this carrier is small/new, this can be normal.")
        else:
            st.success(f"Loaded {len(df_raw):,} rows from: **{ds_name}**")

            views = build_deep_dive_views(df_raw)
            df_norm = views["df_norm"]
            kpis = views["kpis"]

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Violations (rows)", f"{kpis.get('total_violations', 0):,}")
            with k2:
                st.metric("Inspections (est.)", f"{kpis.get('total_inspections', 0):,}")
            with k3:
                oos_rate = kpis.get("oos_rate")
                st.metric("OOS Rate (if available)", f"{oos_rate*100:.1f}%" if isinstance(oos_rate, float) else "N/A")
            with k4:
                st.metric("Data window", "Last 3 years")

            st.markdown("---")

            # Trend charts
            vb = views["violations_by_month"]
            ib = views["inspections_by_month"]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Violations per month (3y)**")
                fig = px.bar(vb, x="inspection_month", y="violations")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("**Inspections per month (3y)**")
                fig = px.line(ib, x="inspection_month", y="inspections", markers=True)
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Top codes + recent
            top_codes = views["top_codes"]
            recent_rows = views["recent_rows"]

            left, right = st.columns([1.1, 0.9])
            with left:
                st.markdown("**Top repeat violation codes (what’s driving the story)**")
                if top_codes.empty:
                    st.info("No violation_code field found in this dataset response; showing raw data instead.")
                else:
                    st.dataframe(top_codes, use_container_width=True, height=380)

            with right:
                st.markdown("**Fix-first list (quick wins)**")
                if top_codes.empty:
                    st.write("- Pull the violations CSV and identify repeat codes manually (dataset schema mismatch).")
                else:
                    # simple: top 5 codes
                    for _, row in top_codes.head(5).iterrows():
                        st.write(f"- **{row['violation_code']}** — {int(row['count'])} times (3y)")

                st.markdown("---")
                st.caption("Tip: use DataQs to challenge incorrect inspections/violations (if applicable).")

            st.markdown("---")
            st.markdown("**Most recent inspection/violation rows (for the compliance deep dive)**")
            st.dataframe(recent_rows, use_container_width=True, height=420)

            st.markdown("---")

            # CSV Export (meets the “feed the Violations worksheet” need)
            st.subheader("Export for Spreadsheet (Violations worksheet)")
            st.caption("Download CSV and paste/import into your DOT Compliance spreadsheet “Violations” tab.")
            download_csv_button(
                df_norm,
                filename=f"usdot_{usdot.strip()}_violations_last3y.csv",
                label="⬇️ Download Violations CSV (last 3 years)",
            )

            # Store for broker exports / PDF
            st.session_state["violations_df"] = df_norm
            st.session_state["violations_source"] = f"{ds_name} ({ds_url})"

# ---------------------------
# ADD YOUR DATA (OPTIONAL)
# ---------------------------
with tabs[2]:
    st.subheader("Add Your Data (optional overlays)")
    st.caption("Claims/loss runs and driver citations are typically private. Upload CSVs to overlay them into your report story.")

    claims_file = st.file_uploader("Upload Claims / Loss Runs CSV (optional)", type=["csv"], key="claims_upload")
    cits_file = st.file_uploader("Upload Driver Citations CSV (optional)", type=["csv"], key="cits_upload")

    if claims_file is not None:
        try:
            claims_df = pd.read_csv(claims_file)
            st.session_state["claims_df"] = claims_df
            st.success(f"Loaded claims rows: {len(claims_df):,}")
            st.dataframe(claims_df.head(25), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read claims CSV: {e}")

    if cits_file is not None:
        try:
            cits_df = pd.read_csv(cits_file)
            st.session_state["cits_df"] = cits_df
            st.success(f"Loaded citations rows: {len(cits_df):,}")
            st.dataframe(cits_df.head(25), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read citations CSV: {e}")

    st.info(
        "Recommended CSV columns (not enforced yet):\n"
        "- Claims: loss_date, paid_amount, reserve_amount, cause, driver_id/unit_id\n"
        "- Citations: citation_date, driver_id, violation_type, severity, location\n"
        "These overlays become powerful on the PDF renewal report."
    )

# ---------------------------
# BROKER EXPORTS (PDF / client-ready)
# ---------------------------
with tabs[3]:
    st.subheader("Broker Exports (keep yourself in the picture)")
    st.caption("Export client-ready artifacts so you don’t have to show the site.")

    basics_df = st.session_state.get("basics_df")
    violations_df = st.session_state.get("violations_df")
    impact_band = st.session_state.get("impact_band", "")
    impact_text = st.session_state.get("impact_text", "")
    measure = st.session_state.get("carrier_measure")
    status = st.session_state.get("carrier_status", "Unknown")
    violations_source = st.session_state.get("violations_source", "DOT Open Data (Socrata)")

    if basics_df is None and violations_df is None:
        st.info("Load a carrier snapshot and/or inspection deep dive first.")
    else:
        b1, b2 = st.columns(2)

        with b1:
            if violations_df is not None and not violations_df.empty:
                download_csv_button(
                    violations_df,
                    filename=f"usdot_{usdot.strip()}_violations_last3y.csv",
                    label="⬇️ Download Violations CSV (again)",
                )
            else:
                st.caption("No violations dataset loaded yet.")

        with b2:
            st.caption("PDF export is enabled if `reportlab` is installed (add it to requirements.txt if needed).")

        st.markdown("---")

        pdf_bytes = make_simple_pdf_report_bytes(
            title="Renewal Prep Report (Broker Copy)",
            subtitle=f"USDOT {usdot.strip()} • Status: {status} • Score: {measure if measure is not None else 'N/A'}",
            sections=[
                ("Insurance Impact (summary)", [impact_band, impact_text] if impact_band else ["(Load Snapshot to populate)"]),
                ("Data Sources", [
                    "FMCSA QCMobile API: BASIC snapshot (thresholds, flags, counts, percentiles when available).",
                    f"DOT Open Data: inspection/violation rows (rolling 3 years). Source used: {violations_source}",
                    "Optional uploads: claims/loss runs and driver citations (private, broker/carrier provided).",
                ]),
                ("Broker Action Plan (top-level)", [
                    "1) Use ‘Top repeat violation codes’ to pick 3–5 fix-first items.",
                    "2) Address the newest repeats first (recency drives underwriting narrative).",
                    "3) If any inspections/violations are wrong, file DataQs challenges and document outcomes.",
                ]),
            ],
            table_df=violations_df if isinstance(violations_df, pd.DataFrame) else None,
        )

        if pdf_bytes is None:
            st.warning("PDF export not available yet (missing `reportlab`). Add `reportlab` to requirements.txt.")
        else:
            st.download_button(
                "⬇️ Download Broker PDF Report",
                data=pdf_bytes,
                file_name=f"usdot_{usdot.strip()}_renewal_prep_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

