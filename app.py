import os
from typing import List, Dict, Any, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ==========================
# Data / Scoring Helpers
# ==========================

def flatten_qcmobile_basics(raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten QCMobile /carriers/:dotNumber/basics JSON (as in your sample)
    into a list of simple dicts, one per BASIC.
    """
    basics_flat: List[Dict[str, Any]] = []

    if not isinstance(raw_data, dict):
        return basics_flat

    content = raw_data.get("content", [])
    if not isinstance(content, list):
        return basics_flat

    for entry in content:
        basic = entry.get("basic", {}) or {}
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
        percentile_num = None
        if isinstance(percentile_raw, str) and percentile_raw.strip().endswith("%"):
            try:
                percentile_num = float(percentile_raw.strip().replace("%", ""))
            except ValueError:
                percentile_num = None

        measure_num = None
        if measure_raw is not None:
            try:
                measure_num = float(str(measure_raw))
            except ValueError:
                measure_num = None

        threshold_num = None
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
                "Measure (num)": measure_num,
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
    - Percentile vs FMCSA violation threshold
    - Intervention flags (exceeded threshold, on-road, serious violation)
    """
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")
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


def compute_carrier_measure(basics_flat: List[Dict[str, Any]]) -> float:
    """
    MVP carrier "golf score":

        carrier_measure = max(Measure (num) for each BASIC)

    This uses the worst BASIC Measure as the overall score.
    Lower is better.
    """
    measures = [
        row.get("Measure (num)")
        for row in basics_flat
        if row.get("Measure (num)") is not None
    ]

    if not measures:
        return 0.0

    return round(max(measures), 3)


def status_from_measure(measure: float) -> Tuple[str, str]:
    """
    Map the carrier measure to a traffic light status.

    Thresholds from your spec:
        Green:  Measure < 1.5
        Yellow: 1.5 – 4.0
        Red:    > 4.0
    """
    if measure < 1.5:
        return "Green", "🟢"
    if measure <= 4.0:
        return "Yellow", "🟡"
    return "Red", "🔴"


def summarize_insurance_impact(
    basics_flat: List[Dict[str, Any]],
    measure: float,
    overall_status: str,
) -> Tuple[str, str]:
    """
    Return a short narrative about likely insurance impact.
    """
    has_high = any(b.get("Risk Level") == "High" for b in basics_flat)
    has_medium = any(b.get("Risk Level") == "Medium" for b in basics_flat)
    has_exceeded = any(b.get("Exceeded FMCSA Threshold") == "Y" for b in basics_flat)

    if has_high or has_exceeded or overall_status == "Red":
        band = "High rate pressure"
        text = (
            "At least one BASIC is above or near FMCSA intervention thresholds. "
            "Underwriters will treat this as a higher-risk account and you should "
            "plan for roughly +15–30% rate pressure unless you show a clear fix."
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

def render_gauge(measure: float):
    """
    Render a credit-score-style gauge using Plotly.
    Measure: assume 0–6+ range with:
        0–1.5   Green
        1.5–4.0 Yellow
        4.0–6+  Red
    """
    max_scale = 6.0
    value = min(max(measure, 0.0), max_scale)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "font": {"size": 30}},
            # Empty title to avoid clipping at top
            title={"text": "", "font": {"size": 1}},
            gauge={
                "axis": {"range": [0, max_scale]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 1.5], "color": "#22c55e"},   # green
                    {"range": [1.5, 4.0], "color": "#eab308"}, # yellow
                    {"range": [4.0, max_scale], "color": "#ef4444"},  # red
                ],
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

                    # Overall measure + status
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
                        st.caption("Lower is better. Based on your worst BASIC Measure.")

                    with top_mid:
                        st.subheader("Status")
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

                    # ===== Top factors (cleaned up) =====
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

                                # Short bullet stats
                                if pct is not None and thresh is not None:
                                    st.write(f"- {pct} vs threshold {thresh}")
                                if insp is not None:
                                    st.write(f"- {insp} inspections w/ violations (24m)")
                                if viol is not None:
                                    st.write(f"- {viol} total violations (24m)")

                    # ===== Focus areas & action plan (shorter) =====
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

                            items = action_items_for_basic(name)[:3]  # keep it tight
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
