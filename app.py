import os
import io
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# App Defaults / Config
# ============================================================

APP_TITLE = "🚛 Credit Karma for Truckers"
QC_BASE_URL = "https://mobile.fmcsa.dot.gov/qc/services"

DOT_VIOLATIONS_DATASETS = [
    # Primary (most commonly cited for violations/inspections)
    ("Vehicle Inspections and Violations", "https://data.transportation.gov/resource/876r-jsdb.json"),
    # Fallback (schema varies; used if the first doesn't match)
    ("Vehicle Inspection File", "https://data.transportation.gov/resource/fx4q-ay7w.json"),
]

CANDIDATE_FIELDS = {
    "dot": ["dot_number", "usdot", "dot", "carrier_dot_number"],
    "inspection_date": ["inspection_date", "insp_date", "date", "inspection_dt", "inspectiondate"],
    "inspection_id": ["inspection_id", "inspection_number", "insp_id", "report_number", "inspectionreportnumber"],
    "violation_code": ["violation_code", "viol_code", "violation", "code"],
    "oos": ["oos", "out_of_service", "oos_violation", "vehicle_oos", "driver_oos"],
    "state": ["state", "report_state", "inspection_state"],
    "level": ["inspection_level", "level", "insp_level"],
}

# Rough “maintenance-ish” heuristics (placeholder; you’ll refine with a real mapping later)
MAINTENANCE_HINTS = (
    "BRAKE", "TIRE", "TYRE", "LIGHT", "LAMP", "REFLECT", "COUPL", "HITCH",
    "WHEEL", "RIM", "AXLE", "SUSPEN", "STEER", "AIR", "LEAK", "OIL", "FUEL",
    "EXHAUST", "FRAME", "LOAD SECUR", "CARGO SECUR"
)


# ============================================================
# Utility / Parsing
# ============================================================

def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def pct_str_to_num(pct_raw: Any) -> Optional[float]:
    if not isinstance(pct_raw, str):
        return None
    txt = pct_raw.strip().lower()
    if txt.endswith("%"):
        try:
            return float(txt.replace("%", ""))
        except Exception:
            return None
    return None


def looks_true(x: Any) -> bool:
    s = str(x).strip().upper()
    return s in {"Y", "YES", "TRUE", "T", "1"}


# ============================================================
# QCMobile Fetchers (FMCSA official services; requires webKey)
# ============================================================

def qc_get_json(path: str, webkey: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Generic QCMobile GET helper.
    """
    url = f"{QC_BASE_URL}{path}"
    r = requests.get(url, params={"webKey": webkey}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        return data
    # Some endpoints might return list; wrap to keep things consistent
    return {"content": data}


@st.cache_data(show_spinner=False, ttl=60 * 30)
def qc_fetch_carrier_profile(usdot: str, webkey: str) -> Dict[str, Any]:
    return qc_get_json(f"/carriers/{usdot}", webkey)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def qc_fetch_basics(usdot: str, webkey: str) -> Dict[str, Any]:
    return qc_get_json(f"/carriers/{usdot}/basics", webkey)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def qc_fetch_oos(usdot: str, webkey: str) -> Dict[str, Any]:
    # QCMobile includes an /oos endpoint in their documentation
    return qc_get_json(f"/carriers/{usdot}/oos", webkey)


# ============================================================
# QCMobile BASIC normalization + scoring
# ============================================================

def flatten_qcmobile_basics(raw_data: Any) -> List[Dict[str, Any]]:
    """
    Normalize QCMobile /basics into a list of dicts.
    Handles both:
      1) {"content":[{"basic":{...}}, ...]}
      2) [{"basic":{...}}, ...]
    """
    if isinstance(raw_data, list):
        content = raw_data
    elif isinstance(raw_data, dict):
        content = raw_data.get("content", [])
    else:
        return []

    if not isinstance(content, list):
        return []

    basics_flat: List[Dict[str, Any]] = []
    for entry in content:
        basic = (entry or {}).get("basic", {}) or {}
        basics_type = basic.get("basicsType", {}) or {}

        short_desc = basics_type.get("basicsShortDesc")
        code = basics_type.get("basicsCode")

        percentile_raw = basic.get("basicsPercentile")
        measure_raw = basic.get("measureValue")
        threshold_raw = basic.get("basicsViolationThreshold")

        exceeded_raw = basic.get("exceededFMCSAInterventionThreshold")
        onroad_raw = basic.get("onRoadPerformanceThresholdViolationIndicator")
        serious_raw = basic.get("seriousViolationFromInvestigationPast12MonthIndicator")

        run_date = basic.get("basicsRunDate")

        basics_flat.append(
            {
                "BASIC": short_desc or code,
                "BASIC Code": code,
                "Snapshot Date": run_date,

                "Percentile": percentile_raw,
                "Percentile (num)": pct_str_to_num(percentile_raw),

                "Measure": measure_raw,
                "Measure (num)": safe_float(measure_raw),

                "Violation Threshold": threshold_raw,
                "Violation Threshold (num)": safe_float(threshold_raw),

                # Flags may be "Y/N", "-1", "Not Public", etc.
                "Exceeded FMCSA Threshold (raw)": exceeded_raw,
                "On-Road Threshold Violation (raw)": onroad_raw,
                "Serious Violation (12m) (raw)": serious_raw,

                # counts
                "Inspections With Violation (24m)": basic.get("totalInspectionWithViolation"),
                "Total Violations (24m)": basic.get("totalViolation"),
            }
        )

    return basics_flat


def evaluate_basic_risk(row: Dict[str, Any]) -> Tuple[str, int, str]:
    """
    Risk band and score, with an explanation string.
    Preferred: percentile vs threshold
    Fallback: measure/threshold ratio
    """
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")
    measure_num = row.get("Measure (num)")

    exceeded_raw = row.get("Exceeded FMCSA Threshold (raw)")
    onroad_raw = row.get("On-Road Threshold Violation (raw)")
    serious_raw = row.get("Serious Violation (12m) (raw)")

    exceeded = looks_true(exceeded_raw)
    onroad_def = looks_true(onroad_raw)
    serious = looks_true(serious_raw)

    band = "Unknown"
    score = 0
    why = "No public percentile or comparable measure."

    if pct is not None and thresh is not None:
        if pct >= thresh:
            band = "High"
            score += 3
            why = f"Percentile {pct:.1f}% ≥ threshold {thresh:.1f}%."
        elif pct >= (thresh - 10):
            band = "Medium"
            score += 2
            why = f"Percentile {pct:.1f}% near threshold {thresh:.1f}% (within 10 points)."
        elif pct >= 20:
            band = "Low"
            score += 1
            why = f"Percentile {pct:.1f}% below threshold {thresh:.1f}%."
        else:
            band = "Very Low"
            score += 0
            why = f"Percentile {pct:.1f}% low and well below threshold {thresh:.1f}%."
    elif measure_num is not None and thresh is not None and thresh > 0:
        ratio = measure_num / thresh
        if ratio >= 1.0:
            band = "High"
            score += 3
            why = f"Fallback: measure/threshold ratio {ratio:.2f} (≥ 1.00)."
        elif ratio >= 0.75:
            band = "Medium"
            score += 2
            why = f"Fallback: measure/threshold ratio {ratio:.2f} (≥ 0.75)."
        elif ratio >= 0.3:
            band = "Low"
            score += 1
            why = f"Fallback: measure/threshold ratio {ratio:.2f} (≥ 0.30)."
        else:
            band = "Very Low"
            score += 0
            why = f"Fallback: measure/threshold ratio {ratio:.2f} (< 0.30)."

    # Escalate for flags (if visible)
    flag_notes = []
    if exceeded:
        score += 2
        flag_notes.append("Exceeded intervention threshold")
    if onroad_def:
        score += 1
        flag_notes.append("On-road threshold violation")
    if serious:
        score += 2
        flag_notes.append("Serious violation past 12 months")

    if flag_notes:
        why += " Flags: " + ", ".join(flag_notes) + "."

    return band, score, why


def compute_carrier_measure(basics_flat: List[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    """
    Returns (measure, explanation).
    measure target range 0–4 (lower is better)
    Preferred: max(percentile)/25
    Fallback: max((measure/threshold)*100)/25
    """
    percents = [b.get("Percentile (num)") for b in basics_flat if b.get("Percentile (num)") is not None]
    if percents:
        m = round(max(percents) / 25.0, 3)
        return m, "Computed from worst (highest) public BASIC percentile ÷ 25."

    pseudo = []
    for b in basics_flat:
        measure_num = b.get("Measure (num)")
        thresh = b.get("Violation Threshold (num)")
        if measure_num is not None and thresh is not None and thresh > 0:
            pseudo_pct = max(0.0, min(100.0, (measure_num / thresh) * 100.0))
            pseudo.append(pseudo_pct)

    if pseudo:
        m = round(max(pseudo) / 25.0, 3)
        return m, "Fallback computed from worst measure/threshold ratio (scaled like a percentile) ÷ 25."

    return None, "Insufficient public BASIC percentile or measure/threshold data to compute a score."


def status_from_measure(measure: Optional[float]) -> Tuple[str, str]:
    if measure is None:
        return "Unknown", "⚪️"
    if measure < 1.5:
        return "Green", "🟢"
    if measure <= 4.0:
        return "Yellow", "🟡"
    return "Red", "🔴"


# ============================================================
# OOS Parsing (QCMobile /oos)
# ============================================================

def parse_oos_summary(oos_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    QCMobile /oos schema can vary; we defensively extract anything resembling:
      - vehicleOosRate, driverOosRate (or similar)
      - inspection counts
    """
    # Try a few nested patterns
    content = oos_raw.get("content")
    if isinstance(content, list) and content:
        obj = content[0]
    elif isinstance(content, dict):
        obj = content
    else:
        obj = oos_raw

    # Flatten a level if key exists
    if isinstance(obj, dict) and "oos" in obj and isinstance(obj["oos"], dict):
        obj = obj["oos"]

    # Candidate keys
    candidates = {
        "vehicle_oos_rate": ["vehicleOosRate", "vehicle_oos_rate", "vehicleOOSRate", "vehicle_oos", "vehOosRate"],
        "driver_oos_rate": ["driverOosRate", "driver_oos_rate", "driverOOSRate", "driver_oos", "drvOosRate"],
        "insp_total": ["totalInspections", "inspectionCount", "inspections", "totalInspection"],
        "vehicle_oos_count": ["vehicleOosCount", "vehicleOOSCount", "vehOosCount"],
        "driver_oos_count": ["driverOosCount", "driverOOSCount", "drvOosCount"],
    }

    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out

    for out_key, keys in candidates.items():
        for k in keys:
            if k in obj:
                out[out_key] = obj.get(k)
                break

    # Normalize rates if they are strings with %
    for rate_key in ["vehicle_oos_rate", "driver_oos_rate"]:
        val = out.get(rate_key)
        if isinstance(val, str) and val.strip().endswith("%"):
            out[rate_key] = safe_float(val.strip().replace("%", "")) / 100.0
        else:
            # if numeric but looks like 12.3 (percent), treat as percent; if 0.123 treat as fraction
            num = safe_float(val)
            if num is not None:
                out[rate_key] = num / 100.0 if num > 1.0 else num

    # Normalize counts
    for k in ["insp_total", "vehicle_oos_count", "driver_oos_count"]:
        v = out.get(k)
        try:
            out[k] = int(float(v)) if v is not None else None
        except Exception:
            out[k] = None

    return out


# ============================================================
# DOT Open Data (Socrata) Deep Dive
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_inspection_rows_rolling_3y(usdot: str, years: int = 3, limit: int = 5000) -> Tuple[pd.DataFrame, str, str]:
    """
    Pull inspection/violation rows for last N years from DOT Socrata.
    Returns (df, dataset_name_used, dataset_url_used_or_error).
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
    # Try common where patterns
    where_templates = [
        ("dot_number", "inspection_date"),
        ("dot_number", "insp_date"),
        ("dot_number", "change_date"),
        ("usdot", "inspection_date"),
        ("usdot", "change_date"),
    ]

    for dataset_name, base_url in DOT_VIOLATIONS_DATASETS:
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
                    df["__dot_field_guess__"] = dot_field_guess
                    df["__date_field_guess__"] = date_field_guess
                    return df, dataset_name, base_url
                last_error = f"{dataset_name}: {r.status_code} {r.text[:200]}"
            except Exception as e:
                last_error = f"{dataset_name}: {e}"

    return pd.DataFrame(), "ERROR", last_error


def build_deep_dive_views(df_raw: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "df_norm": pd.DataFrame(),
        "kpis": {},
        "violations_by_month": pd.DataFrame(),
        "inspections_by_month": pd.DataFrame(),
        "top_codes": pd.DataFrame(),
        "recent_rows": pd.DataFrame(),
        "schema_notes": [],
        "maintenance_share_est": None,
    }
    if df_raw.empty:
        return out

    df = df_raw.copy()

    date_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["inspection_date"])
    insp_id_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["inspection_id"])
    viol_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["violation_code"])
    oos_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["oos"])
    state_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["state"])
    level_col = _pick_first_existing_column(df, CANDIDATE_FIELDS["level"])

    schema_notes = []
    schema_notes.append(f"Date field used: {date_col or 'None found'}")
    schema_notes.append(f"Inspection ID field used: {insp_id_col or 'None found'}")
    schema_notes.append(f"Violation code field used: {viol_col or 'None found'}")
    schema_notes.append(f"OOS indicator field used: {oos_col or 'None found'}")

    if not date_col:
        out["df_norm"] = df
        out["schema_notes"] = schema_notes
        return out

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["inspection_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    # Inspections per month
    if insp_id_col:
        inspections = df[[insp_id_col, "inspection_month"]].dropna().drop_duplicates()
        inspections_by_month = inspections.groupby("inspection_month")[insp_id_col].nunique().reset_index(name="inspections")
    else:
        approx_cols = ["inspection_month"]
        if state_col:
            approx_cols.append(state_col)
        if level_col:
            approx_cols.append(level_col)
        inspections = df[approx_cols].drop_duplicates()
        inspections_by_month = inspections.groupby("inspection_month").size().reset_index(name="inspections")

    # Violations per month (row counts)
    violations_by_month = df.groupby("inspection_month").size().reset_index(name="violations")

    # Top codes
    if viol_col:
        top_codes = df[viol_col].astype(str).value_counts().head(15).reset_index()
        top_codes.columns = ["violation_code", "count"]
    else:
        top_codes = pd.DataFrame(columns=["violation_code", "count"])

    # Recent rows
    show_cols = []
    for c in [date_col, insp_id_col, viol_col, oos_col, state_col, level_col]:
        if c and c in df.columns and c not in show_cols:
            show_cols.append(c)
    recent_rows = df.sort_values(date_col, ascending=False).head(50)[show_cols] if show_cols else df.head(50)

    # Maintenance share estimate (only if codes exist)
    maint_share = None
    if viol_col:
        codes = df[viol_col].astype(str).str.upper()
        is_maint = pd.Series(False, index=codes.index)
        # simple heuristic: if code text contains common maintenance hints
        for hint in MAINTENANCE_HINTS:
            is_maint = is_maint | codes.str.contains(hint, na=False)
        maint_share = float(is_maint.mean()) if len(is_maint) else None

    # OOS rate estimate (from deep dive; may not match QCMobile /oos)
    oos_rate = None
    if oos_col and oos_col in df.columns:
        oos_rate = float(df[oos_col].apply(looks_true).mean()) if len(df) else None

    out["df_norm"] = df
    out["kpis"] = {
        "total_rows": int(len(df)),
        "inspections_est": int(inspections.shape[0]) if inspections is not None else 0,
        "oos_rate_est_from_rows": oos_rate,
        "date_field_used": date_col,
        "inspection_id_field_used": insp_id_col,
        "violation_code_field_used": viol_col,
        "oos_field_used": oos_col,
    }
    out["violations_by_month"] = violations_by_month
    out["inspections_by_month"] = inspections_by_month
    out["top_codes"] = top_codes
    out["recent_rows"] = recent_rows
    out["schema_notes"] = schema_notes
    out["maintenance_share_est"] = maint_share
    return out


# ============================================================
# Exports (CSV + optional PDF)
# ============================================================

def download_csv_button(df: pd.DataFrame, filename: str, label: str) -> None:
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
    Add 'reportlab' to requirements.txt to enable.
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

    def draw_lines(lines: List[str], y_pos: float) -> float:
        for line in lines:
            if y_pos < 1.0 * inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                y_pos = height - 0.9 * inch
            c.drawString(x, y_pos, line[:140])
            y_pos -= 0.18 * inch
        return y_pos

    for header, lines in sections:
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.9 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, header)
        y -= 0.25 * inch
        c.setFont("Helvetica", 10)
        y = draw_lines(lines, y)
        y -= 0.12 * inch

    if table_df is not None and not table_df.empty:
        if y < 2.0 * inch:
            c.showPage()
            y = height - 0.9 * inch

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, "Appendix: Recent inspection / violation rows (sample)")
        y -= 0.3 * inch

        c.setFont("Helvetica", 8)
        cols = list(table_df.columns)[:6]
        small = table_df[cols].head(18).astype(str)

        col_width = (width - 1.5 * inch) / len(cols)
        for ci, cn in enumerate(cols):
            c.drawString(x + ci * col_width, y, cn[:18])
        y -= 0.2 * inch

        for _, row in small.iterrows():
            if y < 1.0 * inch:
                c.showPage()
                y = height - 0.9 * inch
                c.setFont("Helvetica", 8)
            for ci, cn in enumerate(cols):
                c.drawString(x + ci * col_width, y, str(row[cn])[:18])
            y -= 0.18 * inch

    c.save()
    buffer.seek(0)
    return buffer.read()


# ============================================================
# Headline risk (combines BASIC + OOS + Deep Dive signals)
# ============================================================

def compute_headline_status(
    basic_status: str,
    basic_measure: Optional[float],
    vehicle_oos_rate: Optional[float],
    maint_share_est: Optional[float],
    yellow_oos_threshold: float,
    red_oos_threshold: float,
) -> Tuple[str, str, List[str]]:
    """
    Headline status = max(BASIC status, OOS risk, maintenance-driven OOS story)
    Returns (status, emoji, reasons)
    """
    reasons: List[str] = []

    # Start from BASIC
    headline = basic_status

    def bump_to(at_least: str) -> None:
        nonlocal headline
        order = {"Unknown": 0, "Green": 1, "Yellow": 2, "Red": 3}
        if order.get(at_least, 0) > order.get(headline, 0):
            headline = at_least

    # OOS override
    if vehicle_oos_rate is not None:
        if vehicle_oos_rate >= red_oos_threshold:
            bump_to("Red")
            reasons.append(f"Vehicle OOS rate {vehicle_oos_rate*100:.1f}% ≥ {red_oos_threshold*100:.1f}% (override).")
        elif vehicle_oos_rate >= yellow_oos_threshold:
            bump_to("Yellow")
            reasons.append(f"Vehicle OOS rate {vehicle_oos_rate*100:.1f}% ≥ {yellow_oos_threshold*100:.1f}% (override).")
        else:
            reasons.append(f"Vehicle OOS rate {vehicle_oos_rate*100:.1f}% below override thresholds.")

    # Maintenance share “story signal” (soft)
    if maint_share_est is not None:
        if maint_share_est >= 0.45:
            # If nearly half+ of violations look maintenance-coded, we escalate narrative risk to Yellow at least.
            bump_to("Yellow")
            reasons.append(f"Maintenance-coded violations ~{maint_share_est*100:.0f}% of rows (soft risk signal).")

    emoji = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴", "Unknown": "⚪️"}.get(headline, "⚪️")

    # If BASIC is unknown but we have strong OOS, explain
    if basic_measure is None and vehicle_oos_rate is not None:
        reasons.append("BASIC score limited/Not Public; using OOS and inspection outcomes to guide headline risk.")

    return headline, emoji, reasons


# ============================================================
# UI helpers
# ============================================================

def render_gauge(measure: Optional[float]) -> None:
    if measure is None:
        st.info("Score not available from public percentiles; we rely more on OOS + inspection outcomes below.")
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
    return {"High": "🔴", "Medium": "🟡", "Low": "🟠", "Very Low": "🟢"}.get(level, "⚪️")


def action_items_for_basic(name: str) -> List[str]:
    n = (name or "").lower()
    if "vehicle" in n or "maint" in n:
        return [
            "Target brakes/tires/lights first (most common OOS drivers).",
            "Identify repeat units/shops producing violations and tighten PM compliance.",
            "Add a pre-trip + DVIR closure loop: defect → repair → verify.",
        ]
    if "unsafe" in n:
        return [
            "Coach high-risk drivers on speeding, following distance, and distractions.",
            "Use telematics/cameras to catch harsh braking + speeding early.",
            "Audit lanes/routes where violations cluster and fix expectations.",
        ]
    if "hos" in n or "hours" in n:
        return [
            "Run weekly ELD audits (form & manner + over-hours patterns).",
            "Tighten log edit rules and require documented reasons.",
            "Train drivers on 11/14-hour rules + common roadside traps.",
        ]
    if "driver fitness" in n:
        return [
            "Audit CDLs/med cards/endorsements for expirations and missing items.",
            "Clean up DQ files before renewal.",
            "Set reminders for expiring credentials.",
        ]
    if "drug" in n or "alcohol" in n:
        return [
            "Verify random testing rates and documentation.",
            "Ensure positives/refusals handled consistently and documented.",
            "Refresh training on drug/alcohol policy.",
        ]
    return [
        "Find repeat patterns (same code, same unit, same terminal) and assign an owner.",
        "Fix the most recent repeats first (recency drives renewal conversations).",
        "Document corrective actions for the renewal story.",
    ]


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛", layout="wide")
st.title(APP_TITLE)

# Sidebar settings (stakeholder-friendly controls)
with st.sidebar:
    st.header("Controls")
    st.caption("These are *explanations and guardrails* so stakeholders trust the output.")
    show_explainers = st.toggle("Show explainers / justifications", value=True)

    st.subheader("OOS override thresholds")
    yellow_oos_threshold = st.slider("Vehicle OOS → Yellow if ≥", 0.00, 0.50, 0.12, 0.01)
    red_oos_threshold = st.slider("Vehicle OOS → Red if ≥", 0.00, 0.50, 0.20, 0.01)

    st.divider()
    st.subheader("Socrata limits")
    st.caption("If you hit timeouts, reduce this.")
    socrata_limit = st.selectbox("Max deep-dive rows", [2000, 5000, 10000], index=1)

st.write(
    "A broker-friendly translation layer: **FMCSA BASIC snapshot + inspection outcomes (OOS) + inspection-level violations**. "
    "Lower is better (golf score)."
)

if show_explainers:
    with st.expander("⬇️ Capture → Input → Output (how this app works)", expanded=False):
        st.markdown(
            """
**Capture (what we fetch):**
- **FMCSA QCMobile API**: BASIC snapshot (`/carriers/{dot}/basics`), carrier profile (`/carriers/{dot}`), and OOS summary (`/carriers/{dot}/oos`) using your `webKey`.
- **DOT Open Data (Socrata)**: inspection/violation rows (rolling last 3 years) used for trends and “what’s driving the story.”

**Input (how we interpret):**
- Score uses public BASIC percentiles when available; if percentiles are not public, we fall back to measure/threshold ratio.
- Headline status can be **overridden** by **Vehicle OOS rate** (because underwriting and renewal conversations care about OOS outcomes).
- Deep Dive uses inspection-level rows to show trends and repeats; it is the best “proof” when BASICs are not public.

**Output (what you show to clients):**
- A simple Green/Yellow/Red story, plus “why” expanders for defensibility.
- A deep-dive export (CSV) to feed your compliance spreadsheet “Violations” tab.
"""
        )

    with st.expander("⬇️ Why we override with Vehicle OOS", expanded=False):
        st.markdown(
            """
BASIC snapshots can be limited (“Not Public”) or not aligned with what’s hurting a renewal right now.
**Vehicle OOS** is a concrete outcome that brokers and underwriters recognize immediately.
So if Vehicle OOS is high, we elevate headline risk even if BASIC score looks green.
"""
        )

# Inputs
col_input, col_info = st.columns([2, 3])
with col_input:
    usdot = st.text_input("USDOT Number", placeholder="e.g. 44110", value=st.session_state.get("usdot", ""))
    st.session_state["usdot"] = usdot.strip()

with col_info:
    st.info(
        "Tip: if you want a human verification point, SAFER has a carrier snapshot page. "
        "It’s useful as a reference link, but it’s not a stable JSON API."
    )

# Tabs
tabs = st.tabs(["Snapshot", "Inspection Deep Dive (3y)", "Exports", "Debug / Raw"])

# Session state containers
if "loaded" not in st.session_state:
    st.session_state["loaded"] = False

# ============================================================
# SNAPSHOT TAB
# ============================================================
with tabs[0]:
    st.subheader("Snapshot")
    st.caption("Sources: FMCSA QCMobile (BASICs + OOS + carrier profile).")

    do_load = st.button("Check Carrier", type="primary")

    if do_load:
        if not usdot.strip():
            st.error("Enter a USDOT number.")
        else:
            webkey = os.getenv("QCMOBILE_WEBKEY")
            if not webkey:
                st.error("Missing `QCMOBILE_WEBKEY` in Streamlit secrets.")
            else:
                try:
                    with st.spinner("Fetching FMCSA QCMobile data…"):
                        carrier_raw = qc_fetch_carrier_profile(usdot.strip(), webkey)
                        basics_raw = qc_fetch_basics(usdot.strip(), webkey)
                        oos_raw = qc_fetch_oos(usdot.strip(), webkey)

                    # Store raw
                    st.session_state["carrier_raw"] = carrier_raw
                    st.session_state["basics_raw"] = basics_raw
                    st.session_state["oos_raw"] = oos_raw

                    # Normalize basics
                    basics_flat = flatten_qcmobile_basics(basics_raw)
                    if not basics_flat:
                        st.warning("No BASICs returned for this carrier (or carrier not found).")
                        st.session_state["loaded"] = False
                    else:
                        # Risk per BASIC
                        for b in basics_flat:
                            band, score, why = evaluate_basic_risk(b)
                            b["Risk Level"] = band
                            b["Risk Score"] = score
                            b["Risk Why"] = why

                        basics_df = pd.DataFrame(basics_flat)
                        basics_df_sorted = basics_df.sort_values(by=["Risk Score", "Percentile (num)"], ascending=[False, False])

                        # Score
                        measure, measure_why = compute_carrier_measure(basics_flat)
                        basic_status, basic_emoji = status_from_measure(measure)

                        # OOS summary
                        oos_summary = parse_oos_summary(oos_raw)

                        # Store computed
                        st.session_state["basics_df"] = basics_df
                        st.session_state["basics_df_sorted"] = basics_df_sorted
                        st.session_state["basic_measure"] = measure
                        st.session_state["basic_measure_why"] = measure_why
                        st.session_state["basic_status"] = basic_status
                        st.session_state["basic_emoji"] = basic_emoji
                        st.session_state["oos_summary"] = oos_summary
                        st.session_state["loaded"] = True

                except requests.exceptions.HTTPError as e:
                    st.error(f"QCMobile HTTP error: {e}")
                    st.session_state["loaded"] = False
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.session_state["loaded"] = False

    if not st.session_state.get("loaded"):
        st.info("Enter a USDOT number and click **Check Carrier**.")
    else:
        # Header profile (best-effort)
        carrier = st.session_state.get("carrier_raw", {})
        # carrier profile can come as {content:[{carrier:{...}}]} or other; defensively find name
        carrier_obj = carrier.get("content")
        if isinstance(carrier_obj, list) and carrier_obj:
            carrier_obj = carrier_obj[0]
        if isinstance(carrier_obj, dict) and "carrier" in carrier_obj and isinstance(carrier_obj["carrier"], dict):
            carrier_obj = carrier_obj["carrier"]
        if not isinstance(carrier_obj, dict):
            carrier_obj = {}

        legal_name = carrier_obj.get("legalName") or carrier_obj.get("legal_name") or ""
        dba_name = carrier_obj.get("dbaName") or carrier_obj.get("dba_name") or ""
        allow_to_operate = carrier_obj.get("allowToOperate")
        out_of_service = carrier_obj.get("outOfService")

        top1, top2, top3 = st.columns([1.4, 1.2, 1.4])
        with top1:
            st.subheader(legal_name if legal_name else f"USDOT {usdot.strip()}")
            if dba_name:
                st.caption(f"DBA: {dba_name}")
            if show_explainers:
                with st.expander("⬇️ Source: carrier profile fields", expanded=False):
                    st.write("Source: FMCSA QCMobile `/carriers/{dot}` (public carrier profile).")

        with top2:
            st.metric("Allowed to operate", allow_to_operate if allow_to_operate is not None else "N/A")
            st.metric("Out of service order", out_of_service if out_of_service is not None else "N/A")

        with top3:
            st.caption("Quick reference link (manual):")
            st.write("SAFER Company Snapshot:")
            st.code("https://safer.fmcsa.dot.gov/query.asp", language="text")
            if show_explainers:
                with st.expander("⬇️ Why SAFER is a reference link only", expanded=False):
                    st.write(
                        "SAFER is helpful for manual verification, but it’s an HTML website (not a stable JSON API). "
                        "For automation, QCMobile + DOT Open Data are more reliable."
                    )

        st.divider()

        basics_df_sorted = st.session_state["basics_df_sorted"]
        basics_df = st.session_state["basics_df"]
        basic_measure = st.session_state["basic_measure"]
        basic_measure_why = st.session_state["basic_measure_why"]
        basic_status = st.session_state["basic_status"]
        basic_emoji = st.session_state["basic_emoji"]
        oos_summary = st.session_state.get("oos_summary", {})

        # Deep dive (if loaded later) can inform headline (maintenance share)
        maint_share_est = st.session_state.get("maintenance_share_est")

        vehicle_oos_rate = oos_summary.get("vehicle_oos_rate")
        headline_status, headline_emoji, headline_reasons = compute_headline_status(
            basic_status=basic_status,
            basic_measure=basic_measure,
            vehicle_oos_rate=vehicle_oos_rate,
            maint_share_est=maint_share_est,
            yellow_oos_threshold=yellow_oos_threshold,
            red_oos_threshold=red_oos_threshold,
        )

        # Top KPI row
        k1, k2, k3 = st.columns([1.2, 1.3, 1.5])
        with k1:
            st.subheader("Safety Score")
            render_gauge(basic_measure)
            if show_explainers:
                with st.expander("⬇️ How the score is computed", expanded=False):
                    st.write(basic_measure_why)

        with k2:
            st.subheader("Headline Status")
            if headline_status == "Green":
                st.success(f"{headline_emoji} {headline_status}")
            elif headline_status == "Yellow":
                st.warning(f"{headline_emoji} {headline_status}")
            elif headline_status == "Red":
                st.error(f"{headline_emoji} {headline_status}")
            else:
                st.info(f"{headline_emoji} {headline_status}")

            if show_explainers:
                with st.expander("⬇️ Why headline status is what it is", expanded=False):
                    for r in headline_reasons:
                        st.write(f"- {r}")

        with k3:
            st.subheader("OOS Summary")
            v_rate = oos_summary.get("vehicle_oos_rate")
            d_rate = oos_summary.get("driver_oos_rate")
            st.metric("Vehicle OOS rate", f"{v_rate*100:.1f}%" if isinstance(v_rate, float) else "N/A")
            st.metric("Driver OOS rate", f"{d_rate*100:.1f}%" if isinstance(d_rate, float) else "N/A")
            if show_explainers:
                with st.expander("⬇️ Source: OOS summary", expanded=False):
                    st.write("Source: FMCSA QCMobile `/carriers/{dot}/oos` (roadside out-of-service summary).")

        st.divider()

        # Top factors
        st.subheader("Top Factors Affecting Your Insurance (from BASIC snapshot)")
        top_factors = basics_df_sorted.head(3).to_dict(orient="records")
        if not top_factors:
            st.write("No BASIC factors found.")
        else:
            cols = st.columns(len(top_factors))
            for col, b in zip(cols, top_factors):
                with col:
                    name = b.get("BASIC") or b.get("BASIC Code")
                    st.markdown(f"**{name}**")
                    st.metric("Risk", f"{risk_emoji(b.get('Risk Level'))} {b.get('Risk Level')}")
                    st.caption(b.get("Risk Why", ""))
                    st.write(f"- Percentile: {b.get('Percentile')}")
                    st.write(f"- Threshold: {b.get('Violation Threshold')}")
                    st.write(f"- Inspections w/ violation (24m): {b.get('Inspections With Violation (24m)')}")
                    st.write(f"- Total violations (24m): {b.get('Total Violations (24m)')}")

        if show_explainers:
            with st.expander("⬇️ Why BASIC snapshot can disagree with OOS reality", expanded=False):
                st.markdown(
                    """
- Percentiles may be **Not Public** for some carriers/BASICs.
- Snapshot dates can be old and not reflect recent operational issues.
- Underwriters often react quickly to **high Vehicle OOS**, especially when it’s maintenance-driven.
That’s why we include an OOS-based override and an inspection-level deep dive.
"""
                )

        st.divider()

        # Action plan (short + broker friendly)
        st.subheader("Action Plan (broker friendly)")
        for b in top_factors:
            name = b.get("BASIC") or b.get("BASIC Code")
            st.markdown(f"### {name}")
            for item in action_items_for_basic(name)[:3]:
                st.write(f"- {item}")

        if vehicle_oos_rate is not None and vehicle_oos_rate >= yellow_oos_threshold:
            st.warning("OOS override triggered: prioritize vehicle maintenance controls and document corrective actions for renewal.")

        st.divider()
        st.subheader("BASIC Details (table)")
        st.dataframe(basics_df, use_container_width=True)

# ============================================================
# DEEP DIVE TAB
# ============================================================
with tabs[1]:
    st.subheader("Inspection Deep Dive (rolling last 3 years)")
    st.caption("Source: DOT Open Data (Socrata). Best for trends, repeats, and explaining what’s driving underwriting conversations.")

    if not usdot.strip():
        st.info("Enter a USDOT above.")
    else:
        with st.spinner("Fetching inspection/violation rows…"):
            df_raw, ds_name, ds_url = fetch_inspection_rows_rolling_3y(usdot.strip(), years=3, limit=int(socrata_limit))

        if ds_name == "ERROR":
            st.error("Could not query DOT Open Data.")
            st.code(ds_url)
        elif df_raw.empty:
            st.warning("No inspection/violation rows returned for the last 3 years (or schema mismatch).")
            if show_explainers:
                with st.expander("⬇️ Why this can happen", expanded=False):
                    st.write(
                        "Some carriers have low inspection volume, or the dataset schema may not match the expected fields. "
                        "We try two datasets and multiple filters, but public datasets can be inconsistent."
                    )
        else:
            views = build_deep_dive_views(df_raw)
            df_norm = views["df_norm"]
            kpis = views["kpis"]
            maint_share_est = views["maintenance_share_est"]

            st.session_state["violations_df"] = df_norm
            st.session_state["violations_source"] = f"{ds_name} ({ds_url})"
            st.session_state["maintenance_share_est"] = maint_share_est

            st.success(f"Loaded {len(df_raw):,} rows from: **{ds_name}**")

            if show_explainers:
                with st.expander("⬇️ Deep dive schema + reliability notes", expanded=False):
                    for note in views["schema_notes"]:
                        st.write(f"- {note}")
                    st.write("Public datasets vary; this view focuses on trends and repeats and may require mapping refinements.")

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Rows (violations/events)", f"{kpis.get('total_rows', 0):,}")
            with c2:
                st.metric("Inspections (est.)", f"{kpis.get('inspections_est', 0):,}")
            with c3:
                oos_est = kpis.get("oos_rate_est_from_rows")
                st.metric("OOS rate (est.)", f"{oos_est*100:.1f}%" if isinstance(oos_est, float) else "N/A")
            with c4:
                st.metric("Maintenance share (est.)", f"{maint_share_est*100:.0f}%" if isinstance(maint_share_est, float) else "N/A")

            st.divider()

            # Trend charts
            vb = views["violations_by_month"]
            ib = views["inspections_by_month"]

            left, right = st.columns(2)
            with left:
                st.markdown("**Violations per month**")
                fig = px.bar(vb, x="inspection_month", y="violations")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown("**Inspections per month**")
                fig = px.line(ib, x="inspection_month", y="inspections", markers=True)
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Top codes + recent
            top_codes = views["top_codes"]
            recent_rows = views["recent_rows"]

            cA, cB = st.columns([1.2, 0.8])
            with cA:
                st.markdown("**Top repeat violation codes (what’s driving the story)**")
                if top_codes.empty:
                    st.info("No violation_code field found; showing raw rows instead.")
                else:
                    st.dataframe(top_codes, use_container_width=True, height=380)

            with cB:
                st.markdown("**Fix-first list**")
                if top_codes.empty:
                    st.write("- Export CSV and identify repeat items manually (schema mismatch).")
                else:
                    for _, row in top_codes.head(7).iterrows():
                        st.write(f"- **{row['violation_code']}** — {int(row['count'])} times (3y)")

                if show_explainers:
                    with st.expander("⬇️ Why fix-first is simple right now", expanded=False):
                        st.write(
                            "This MVP ranks by frequency only. A later iteration will weight by severity, recency, "
                            "and which BASIC category the violation drives."
                        )

            st.divider()

            st.markdown("**Most recent rows (for compliance deep dive)**")
            st.dataframe(recent_rows, use_container_width=True, height=420)

            st.divider()
            st.subheader("Export for Spreadsheet (Violations worksheet)")
            st.caption("Download CSV and paste/import into your compliance spreadsheet’s ‘Violations’ tab.")
            download_csv_button(
                df_norm,
                filename=f"usdot_{usdot.strip()}_violations_last3y.csv",
                label="⬇️ Download Violations CSV (last 3 years)",
            )

# ============================================================
# EXPORTS TAB
# ============================================================
with tabs[2]:
    st.subheader("Exports")
    st.caption("Broker-friendly artifacts so you can deliver the story without showing the app.")

    basics_df = st.session_state.get("basics_df")
    violations_df = st.session_state.get("violations_df")
    violations_source = st.session_state.get("violations_source", "DOT Open Data (Socrata)")
    oos_summary = st.session_state.get("oos_summary", {})
    basic_measure = st.session_state.get("basic_measure")
    basic_status = st.session_state.get("basic_status", "Unknown")

    if basics_df is None and violations_df is None:
        st.info("Load a carrier first (Snapshot or Deep Dive).")
    else:
        col1, col2 = st.columns(2)

        with col1:
            if isinstance(violations_df, pd.DataFrame) and not violations_df.empty:
                download_csv_button(
                    violations_df,
                    filename=f"usdot_{usdot.strip()}_violations_last3y.csv",
                    label="⬇️ Download Violations CSV",
                )
            else:
                st.caption("No deep-dive dataset loaded yet.")

        with col2:
            st.caption("PDF export requires adding `reportlab` to requirements.txt.")
            if show_explainers:
                with st.expander("⬇️ Why PDF is generated instead of 'printing the site'", expanded=False):
                    st.write(
                        "PDF export keeps the broker in the picture and produces a stable deliverable. "
                        "It also avoids relying on a live web app during a client meeting."
                    )

        st.divider()

        v_rate = oos_summary.get("vehicle_oos_rate")
        d_rate = oos_summary.get("driver_oos_rate")

        sections = [
            ("Executive Summary", [
                f"USDOT: {usdot.strip()}",
                f"BASIC score (0–4, lower better): {basic_measure if basic_measure is not None else 'N/A'}",
                f"BASIC status: {basic_status}",
                f"Vehicle OOS rate: {f'{v_rate*100:.1f}%' if isinstance(v_rate, float) else 'N/A'}",
                f"Driver OOS rate: {f'{d_rate*100:.1f}%' if isinstance(d_rate, float) else 'N/A'}",
            ]),
            ("Data Sources (defensible)", [
                "FMCSA QCMobile: carrier profile, BASIC snapshot, and OOS summary (official FMCSA services, requires webKey).",
                f"DOT Open Data (Socrata): inspection/violation rows, rolling last 3 years. Source used: {violations_source}",
                "Note: insurance claims and driver citation datasets are typically private; they can be added as broker/carrier uploads later.",
            ]),
            ("Broker Action Plan (simple)", [
                "1) If Vehicle OOS is high: focus on maintenance controls and document corrective actions for renewal.",
                "2) Use Deep Dive: top repeat codes + most recent rows to define 3–5 fix-first items.",
                "3) Challenge incorrect inspections/violations via DataQs when appropriate and document outcomes.",
            ]),
        ]

        pdf_bytes = make_simple_pdf_report_bytes(
            title="Renewal Prep Report (Broker Copy)",
            subtitle=f"USDOT {usdot.strip()} • Generated {date.today().isoformat()}",
            sections=sections,
            table_df=violations_df if isinstance(violations_df, pd.DataFrame) else None,
        )

        if pdf_bytes is None:
            st.warning("PDF export not available yet. Add `reportlab` to requirements.txt.")
        else:
            st.download_button(
                "⬇️ Download Broker PDF Report",
                data=pdf_bytes,
                file_name=f"usdot_{usdot.strip()}_renewal_prep_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ============================================================
# DEBUG TAB
# ============================================================
with tabs[3]:
    st.subheader("Debug / Raw")
    st.caption("This is for stakeholder trust-building and troubleshooting. Hide later.")

    if not st.session_state.get("loaded") and not st.session_state.get("violations_df") is not None:
        st.info("Load a carrier first.")
    else:
        with st.expander("⬇️ Raw QCMobile carrier profile", expanded=False):
            st.json(st.session_state.get("carrier_raw", {}))
        with st.expander("⬇️ Raw QCMobile basics", expanded=False):
            st.json(st.session_state.get("basics_raw", {}))
        with st.expander("⬇️ Raw QCMobile oos", expanded=False):
            st.json(st.session_state.get("oos_raw", {}))

        vdf = st.session_state.get("violations_df")
        if isinstance(vdf, pd.DataFrame):
            with st.expander("⬇️ Deep dive sample rows (normalized)", expanded=False):
                st.dataframe(vdf.head(50), use_container_width=True)

        with st.expander("⬇️ What additional API calls help (free)", expanded=False):
            st.markdown(
                """
**Recommended additions (still free):**
- **QCMobile `/carriers/{dot}`**: improves the “Credit Karma profile header” (legal name, DBA, allowed to operate, etc.).
- **QCMobile `/carriers/{dot}/oos`**: provides an official OOS summary (vehicle/driver OOS rates/counts).
- **QCMobile `/authority`, `/cargo-carried`, `/operation-classification`**: helpful underwriting context (operations story).

**SAFER (`safer.fmcsa.dot.gov/query.asp`)**:
- Great for manual verification and sharing a reference link.
- Not a stable JSON API (HTML page), so automated scraping is fragile.
"""
            )
