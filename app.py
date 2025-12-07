import os
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ==========================
# Data / Scoring Helpers
# ==========================

def flatten_qcmobile_basics(raw_data: Any) -> List[Dict[str, Any]]:
    """
    Flatten QCMobile /carriers/:dotNumber/basics JSON into a list of simple dicts,
    one per BASIC.

    Handles both:
      1) {"content": [ { "basic": {...} }, ... ], "retrievalDate": "..."}
      2) [ { "basic": {...} }, ... ]
    """
    basics_flat: List[Dict[str, Any]] = []

    # Case 1: response is a list of entries
    if isinstance(raw_data, list):
        content = raw_data
    # Case 2: response is an object with "content"
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

        # Normalize numeric fields
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
                "Measure (num)": measure_num,           # raw FMCSA measure, not our score
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

    # Primary: percentile vs threshold
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

    # Fallback: use measure / threshold ratio if no percentile
    elif measure_num is not None and thresh is not None and thresh > 0:
        ratio = measure_num / thresh  # e.g. 11 / 65 ≈ 0.17
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

    # Escalate for flags
    if exceeded:
        score += 2
    if onroad_def:
        score += 1
    if serious:
        score += 2

    return band, score


def compute_carrier_measure(basics_flat: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute our "golf score" for the carrier.

    Preferred:
        - Use numeric BASIC percentiles:
            measure = max(percentile) / 25  → 0–4 (lower is better).

    Fallback:
        - If no numeric percentiles exist, use measure/threshold:
            ratio = measureValue / basicsViolationThreshold
            pseudo_percentile = ratio * 100 (clipped 0–100)
            measure = max(pseudo_percentile) / 25 → 0–4.

    If neither percentiles nor measure/threshold data exist:
        - Return None (score not available).
    """
    # Preferred: real percentiles
    percents = [
        b.get("Percentile (num)")
        for b in basics_flat
        if b.get("Percentile (num)") is not None
    ]

    if percents:
        measure = max(percents) / 25.0
        return round(measure, 3)

    # Fallback: use measure / threshold ratio
    pseudo_percents: List[float] = []
    for b in basics_flat:
        m = b.get("Measure (num)")
        thresh = b.get("Violation Threshold (num)")
        if m is not None and thresh is not None and thresh > 0:
            ratio = m / thresh        # e.g. 11 / 65 ≈ 0.17
            pct_est = max(0.0, min(100.0, ratio * 100.0))
            pseudo_percents.append(pct_est)

    if not pseudo_percents:
        return None

    measure = max(pseudo_percents) / 25.0
    return round(measure, 3)


def status_from_measure(measure: Optional[float]) -> Tuple[str, str]:
    """
    Map the carrier measure to a traffic light status.
    """
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
    """
    Return a short narrative about likely insurance impact.
    """
    has_high = any(b.get("Risk Level") == "High" for b in basics_flat)
    has_medium = any(b.get("Risk Level") == "Medium" for b in basics_flat)
    has_exceeded = any(b.get("Exceeded FMCSA Threshold") == "Y" for b in basics_flat)

    if measure is None or overall_status == "Unknown":
        band = "Limited visibility"
        text = (
            "BASIC percentiles for this carrier are not public or show insufficient data. "
            "Underwriters will lean more on loss history, operations, and telematics. "
            "Treat this as a neutral to slightly negative signal until more data is available."
        )
        return band, text

    if has_high or has_exceeded or overall_status == "Red":
        band = "High rate pressure"
        text = (
            "At least one BASIC is above or near FMCSA intervention thresholds. "
            "Underwriters will view this as higher risk; plan for roughly +15–30% "
            "rate pressure unless you show a clear fix."
        )
    elif has_medium or overall_status == "Yellow":
        band = "Moderate rate pressure"
        text = (
            "You have BASICs moving toward FMCSA thresholds. Expect moderate rate "
            "pressure (roughly +5–15%) unless you can demonstrate improvement."
        )
    else:
        band = "Stable / favorable"
        text = (
            "Your BASICs are well below thresholds with no major flags. Expect flat "
            "to modest changes (roughly 0–5%), driven mostly by market conditions."
        )

    return band, text


def action_items_for_basic(name: str) -> List[str]:
    """
    Return a short list of recommended actions for a given BASIC name.
    """
    name_lower = (name or "").lower()

    if "unsafe" in name_lower:
        return [
            "Drill into which drivers and lanes are driving Unsafe Driving violations.",
            "Coach high-risk drivers on speeding, following distance, and distractions.",
            "Use telematics/cameras to flag harsh braking and speeding early.",
        ]
    if "hos" in name_lower or "hours-of-service" in name_lower:
        return [
            "Run weekly ELD/HOS audits for form-and-manner and over-hours.",
            "Tighten log edit rules and require documented reasons for changes.",
            "Train drivers on 11/14-hour rules and common roadside traps.",
        ]
    if "driver fitness" in name_lower:
        return [
            "Audit CDLs, endorsements, and medical cards for expirations.",
            "Clean up driver qualification files before renewal.",
            "Set automated reminders for upcoming document expirations.",
        ]
    if "drug" in name_lower or "alcohol" in name_lower or "substances" in name_lower:
        return [
            "Check random testing rates and documentation for accuracy.",
            "Verify all positives/refusals are handled and documented correctly.",
            "Refresh driver training on your drug & alcohol policy.",
        ]
    if "vehicle maint" in name_lower or "maintenance" in name_lower:
        return [
            "Find units/shops generating the most maintenance violations.",
            "Tighten PM schedules and track completion compliance.",
            "Enforce DVIRs and close the loop on repairs for brakes, tires, and lights.",
        ]

    return [
        "Review repeat violation patterns in this BASIC.",
        "Prioritize fixes that directly reduce roadside violations.",
        "Document your corrective plan and share it before renewal.",
    ]


def trend_label_for_basic(row: Dict[str, Any]) -> str:
    """
    Simple pseudo-trend label based on risk level.
    """
    band = row.get("Risk Level")
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")

    if band == "High":
        return "🔥 Critical – above threshold"
    if band == "Medium":
        return "⚠️ Watch – near threshold"
    if band == "Low":
        if pct is not None and thresh is not None and pct > (thresh - 15):
            return "● Stable but rising"
        return "● Stable"
    if band == "Very Low":
        return "✅ Healthy"
    return "● No clear signal"


# ==========================
# UI Helpers
# ==========================

def render_gauge(measure: Optional[float]):
    """
    Render a credit-score-style gauge using Plotly.
    If measure is None, show an info message instead.
    """
    if measure is None:
        st.info("Score not available – BASIC percentiles are not public or insufficient.")
        return

    max_scale = 4.0  # our score range is 0–4 (from percentiles/ratios)
    value = min(max(measure, 0.0), max_scale)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "font": {"size": 30}},
            title={"text": "", "font": {"size": 1}},  # avoid clipping
            gauge={
                "axis": {"range": [0, max_scale]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 1.5], "color": "#22c55e"},   # green
                    {"range": [1.5, 4.0], "color": "#eab308"}, # yellow
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 0},
                    "thickness": 0.0,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def risk_emoji(level: str) -> str:
    mapping = {
        "High": "🔴",
        "Medium": "🟡",
        "Low": "🟠",
        "Very Low": "🟢",
    }
    return mapping.get(level, "⚪️")


# ==========================
# Streamlit App
# ==========================

st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛", layout="wide")
st.title("🚛 Credit Karma for Truckers")

st.write(
    "Turn FMCSA BASIC scores into a simple safety score, insurance impact, and "
    "a clear action plan. Lower is better (golf score)."
)

col_input, col_info = st.columns([2, 3])

with col_input:
    usdot = st.text_input("USDOT Number", placeholder="e.g. 44110")
    st.caption("Enter a carrier's USDOT number to see their BASIC profile.")

with col_info:
    st.info(
        "This is a translation layer between FMCSA BASIC data and how underwriters "
        "think about risk. It is **not** an official FMCSA or insurer rating."
    )

if st.button("Check Carrier", type="primary"):
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
                    # Evaluate risk for each BASIC
                    for b in basics_flat:
                        band, score = evaluate_basic_risk(b)
                        b["Risk Level"] = band
                        b["Risk Score"] = score
                        b["Trend Label"] = trend_label_for_basic(b)

                    # Overall measure + status (may be None)
                    measure = compute_carrier_measure(basics_flat)
                    status, emoji = status_from_measure(measure)

                    # Insurance impact
                    impact_band, impact_text = summarize_insurance_impact(
                        basics_flat, measure, status
                    )

                    df = pd.DataFrame(basics_flat)
                    df_sorted = df.sort_values(
                        by=["Risk Score", "Percentile (num)"],
                        ascending=[False, False],
                    )

                    st.markdown("---")

                    # ===== Top score section: gauge + status + impact =====
                    top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.6])

                    with top_left:
                        st.subheader("Safety Score")
                        render_gauge(measure)
                        if measure is not None:
                            st.caption(
                                "Lower is better. Based on your highest (worst) BASIC percentile."
                            )

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
                            st.caption("Think of Green ≈ good credit, Red ≈ substandard.")

                    with top_right:
                        st.subheader("Insurance Impact")
                        st.write(f"**{impact_band}**")
                        st.write(impact_text)

                    # ===== Top factors =====
                    st.markdown("---")
                    st.subheader("Top Factors Affecting Your Insurance")

                    top_factors = df_sorted.head(3).to_dict(orient="records")
                    if not top_factors:
                        st.write("No significant risk factors identified.")
                    else:
                        cols = st.columns(len(top_factors))
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

                                st.metric(
                                    label="Risk",
                                    value=f"{risk_emoji(risk_level)} {risk_level}",
                                )

                                if pct is not None and thresh is not None:
                                    st.write(f"- {pct} vs threshold {thresh}")
                                if insp is not None:
                                    st.write(f"- {insp} inspections w/ violations (24m)")
                                if viol is not None:
                                    st.write(f"- {viol} total violations (24m)")

                    # ===== Focus areas & action plan =====
                    st.markdown("---")
                    st.subheader("Focus Areas & Action Plan")

                    if not top_factors:
                        st.write("Nothing critical right now. Keep doing what you're doing.")
                    else:
                        for b in top_factors:
                            name = b.get("BASIC") or b.get("BASIC Code")
                            risk_level = b.get("Risk Level")
                            trend = b.get("Trend Label")

                            st.markdown(f"### {name} – {risk_emoji(risk_level)} {risk_level}")
                            st.caption(trend)

                            items = action_items_for_basic(name)[:3]
                            for item in items:
                                st.write(f"- {item}")

                    # ===== BASIC details table =====
                    st.markdown("---")
                    st.subheader("BASIC Details (for deeper dives)")

                    preferred_cols = [
                        "BASIC",
                        "Risk Level",
                        "Risk Score",
                        "Trend Label",
                        "Measure",
                        "Percentile",
                        "Violation Threshold",
                        "Exceeded FMCSA Threshold",
                        "On-Road Threshold Violation",
                        "Serious Violation (12m)",
                        "Inspections With Violation (24m)",
                        "Total Violations (24m)",
                        "Snapshot Date",
                    ]
                    cols_to_show = [c for c in preferred_cols if c in df.columns]
                    for c in df.columns:
                        if c not in cols_to_show:
                            cols_to_show.append(c)

                    st.dataframe(df[cols_to_show], use_container_width=True)

                    # ===== Trend note =====
                    st.markdown("---")
                    st.subheader("Trend & Watch Notes")

                    st.write(
                        "This view uses the latest BASIC snapshot from FMCSA. "
                        "True historical trends (up/down over time) require storing prior "
                        "snapshots, which this MVP does not yet do. For now, treat High "
                        "and Medium BASICs as the main watch items for next renewal."
                    )

                    # Debug
                    with st.expander("Raw API Response (debug)"):
                        st.json(raw_data)

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
            except Exception as e:
                st.error(f"Unexpected error calling QCMobile API: {e}")
