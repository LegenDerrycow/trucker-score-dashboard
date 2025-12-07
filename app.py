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

        exceeded = basic.get("exceededFMCSAInterventionThreshold")
        onroad_def = basic.get("onRoadPerformanceThresholdViolationIndicator")
        serious_def = basic.get("seriousViolationFromInvestigationPast12MonthIndicator")

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
    exceeded = (row.get("Exceeded FMCSA Threshold") or "").upper() == "Y"
    onroad_def = (row.get("On-Road Threshold Violation") or "").upper() == "Y"
    serious = (row.get("Serious Violation (12m)") or "").upper() == "Y"

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
    Return a narrative about likely insurance impact based on:
    - Presence of High/Medium risk BASICs
    - Intervention flags
    - Overall carrier measure
    """
    has_high = any(b.get("Risk Level") == "High" for b in basics_flat)
    has_medium = any(b.get("Risk Level") == "Medium" for b in basics_flat)
    has_exceeded = any(
        (b.get("Exceeded FMCSA Threshold") or "").upper() == "Y" for b in basics_flat
    )

    if has_high or has_exceeded or overall_status == "Red":
        band = "High rate pressure"
        text = (
            "You are above FMCSA intervention thresholds in at least one BASIC or very "
            "close to them. Many insurers will view this as a high-risk account. "
            "Expect meaningful rate pressure (roughly +15–30%+) and fewer carrier options."
        )
    elif has_medium or overall_status == "Yellow":
        band = "Moderate rate pressure"
        text = (
            "You have one or more BASICs approaching FMCSA intervention thresholds. "
            "Carriers will likely price in some deterioration but still view you as "
            "controllable risk. Expect moderate rate pressure (roughly +5–15%) unless "
            "you show clear corrective action."
        )
    else:
        band = "Stable / favorable"
        text = (
            "Your BASICs are well below FMCSA thresholds with no major flags. "
            "Under normal market conditions this looks like a well-managed account. "
            "Expect mostly flat to modest movements (roughly 0–5%), driven more by "
            "market conditions than your own safety performance."
        )

    return band, text


def action_items_for_basic(name: str) -> List[str]:
    """
    Return a list of recommended actions for a given BASIC name.
    Uses simple keyword matching.
    """
    name_lower = (name or "").lower()

    if "unsafe" in name_lower:
        return [
            "Identify top drivers and locations driving Unsafe Driving violations.",
            "Implement targeted coaching and ride-alongs for high-risk drivers.",
            "Tighten policies on speeding, following distance, and mobile phone use.",
            "Consider telematics or cameras for speeding / harsh braking alerts.",
        ]
    if "hos" in name_lower or "hours-of-service" in name_lower:
        return [
            "Audit ELD logs weekly for HOS and form-and-manner violations.",
            "Lock down log edits and require documented reasons for any changes.",
            "Refresh driver training on 11/14-hour rules and split sleeper provisions.",
            "Set alerts for approaching HOS limits to avoid roadside violations.",
        ]
    if "driver fitness" in name_lower:
        return [
            "Audit all CDLs, endorsements, and medical cards for currency.",
            "Set reminders for upcoming expirations and rechecks.",
            "Tighten hiring and qualification file reviews before onboarding.",
            "Document all Q-file cleanup and share with your broker before renewal.",
        ]
    if "drug" in name_lower or "alcohol" in name_lower or "substances" in name_lower:
        return [
            "Review your random testing program for proper rates and follow-up.",
            "Document all positives and refusals with corrective actions.",
            "Refresh driver training on your drug and alcohol policy.",
            "Ensure clearinghouse reporting and return-to-duty processes are followed.",
        ]
    if "vehicle maint" in name_lower or "maintenance" in name_lower:
        return [
            "Map violations back to specific units, routes, and shops.",
            "Tighten your preventive maintenance schedule and track completion.",
            "Enforce DVIRs and close the loop on defect repairs.",
            "Prioritize brakes, tires, and lights—high-impact and visible at roadside.",
        ]

    # Generic fallback
    return [
        "Review recent violations in this BASIC and identify repeat patterns.",
        "Prioritize fixes that reduce future roadside violations.",
        "Document your corrective action plan and share it before renewal.",
    ]


def trend_label_for_basic(row: Dict[str, Any]) -> str:
    """
    Simple pseudo-trend label based on risk level and proximity to threshold.
    (True historical trending would require stored snapshots.)
    """
    band = row.get("Risk Level")
    pct = row.get("Percentile (num)")
    thresh = row.get("Violation Threshold (num)")

    if band == "High":
        return "🔥 Critical – above threshold"
    if band == "Medium":
        return "⚠️ Watch – near threshold"
    if band == "Low":
        # If close to 20% or 10 points below threshold, it’s a light watch
        if pct is not None and thresh is not None and pct > (thresh - 15):
            return "● Stable but rising"
        return "● Stable"
    if band == "Very Low":
        return "✅ Healthy"
    return "● No clear signal"


def simulate_basic_improvement(
    basics_flat: List[Dict[str, Any]],
    target_basic_name: str,
    reduction_pct: float,
) -> float:
    """
    Simple simulator: reduce the Measure of one BASIC by X% and recompute
    the carrier measure (max of BASIC measures).
    """
    reduced_measures = []

    for b in basics_flat:
        m = b.get("Measure (num)")
        if m is None:
            continue
        name = b.get("BASIC") or b.get("BASIC Code")
        if name == target_basic_name:
            m = m * (1.0 - reduction_pct / 100.0)
        reduced_measures.append(m)

    if not reduced_measures:
        return 0.0

    return round(max(reduced_measures), 3)


# ==========================
# UI Helpers
# ==========================

def render_gauge(measure: float, status: str, emoji: str):
    """
    Render a credit-score-style gauge using Plotly.
    Measure: we assume 0–6+ range, with:
        0–1.5   Green
        1.5–4.0 Yellow
        4.0–6+  Red
    """
    # Clamp measure for visualization
    max_scale = 6.0
    value = min(max(measure, 0.0), max_scale)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "font": {"size": 32}},
            title={"text": f"{emoji} {status}", "font": {"size": 18}},
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
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


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
        "This tool is a translation layer between FMCSA BASIC data and how underwriters "
        "actually think about risk. It is **not** an official FMCSA or insurer rating."
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

                    # Compute overall measure and status
                    measure = compute_carrier_measure(basics_flat)
                    status, emoji = status_from_measure(measure)

                    # Insurance impact summary
                    impact_band, impact_text = summarize_insurance_impact(
                        basics_flat, measure, status
                    )

                    df = pd.DataFrame(basics_flat)
                    df_sorted = df.sort_values(
                        by=["Risk Score", "Percentile (num)"],
                        ascending=[False, False],
                    )

                    st.markdown("---")

                    # ========= Top score section: gauge + impact =========
                    top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.6])

                    with top_left:
                        st.subheader("Your Safety Score")
                        render_gauge(measure, status, emoji)
                        st.caption("Lower is better. This uses your worst BASIC Measure.")

                    with top_mid:
                        st.subheader("Status")
                        if status == "Green":
                            st.success(f"{emoji} {status} – Overall Measure: {measure}")
                        elif status == "Yellow":
                            st.warning(f"{emoji} {status} – Overall Measure: {measure}")
                        else:
                            st.error(f"{emoji} {status} – Overall Measure: {measure}")
                        st.caption(
                            "Green ≈ good credit, Yellow ≈ fair, Red ≈ substandard account."
                        )

                    with top_right:
                        st.subheader("Insurance Impact")
                        st.write(f"**{impact_band}**")
                        st.write(impact_text)

                    # ========= Factor cards =========
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
                                st.write(trend)
                                st.metric("Risk Level", risk_level, None)

                                bullet_lines = []
                                if pct is not None and thresh is not None:
                                    bullet_lines.append(
                                        f"Percentile {pct} vs threshold {thresh}"
                                    )
                                if insp is not None:
                                    bullet_lines.append(
                                        f"{insp} inspections with violations (24m)"
                                    )
                                if viol is not None:
                                    bullet_lines.append(
                                        f"{viol} total violations (24m)"
                                    )

                                for line in bullet_lines:
                                    st.write(f"- {line}")

                    # ========= Focus areas & action plan =========
                    st.markdown("---")
                    st.subheader("Focus Areas & Action Plan")

                    if not top_factors:
                        st.write("Nothing critical right now. Keep doing what you're doing.")
                    else:
                        for b in top_factors:
                            name = b.get("BASIC") or b.get("BASIC Code")
                            risk_level = b.get("Risk Level")
                            trend = b.get("Trend Label")

                            st.markdown(f"### {name} – {risk_level} priority")
                            st.caption(trend)

                            # Recommended actions
                            items = action_items_for_basic(name)
                            st.write("**Recommended actions (next 60–90 days):**")
                            for item in items:
                                st.write(f"- {item}")

                    # ========= Simulator =========
                    st.markdown("---")
                    st.subheader("Simulator: What If We Improve a BASIC?")

                    # Choose a BASIC to simulate
                    basic_options = [
                        b.get("BASIC") or b.get("BASIC Code")
                        for b in basics_flat
                        if b.get("Measure (num)") is not None
                    ]
                    basic_options = list(dict.fromkeys(basic_options))  # dedupe

                    if not basic_options:
                        st.info("No BASICs with numeric measures to simulate yet.")
                    else:
                        col_sim1, col_sim2 = st.columns([1.2, 1.8])
                        with col_sim1:
                            target_basic = st.selectbox(
                                "Choose a BASIC to improve",
                                basic_options,
                            )
                            reduction_pct = st.slider(
                                "Reduce this BASIC's Measure by (%)",
                                min_value=0,
                                max_value=50,
                                value=10,
                                step=5,
                            )

                        with col_sim2:
                            if target_basic:
                                new_measure = simulate_basic_improvement(
                                    basics_flat, target_basic, reduction_pct
                                )
                                sim_status, sim_emoji = status_from_measure(new_measure)

                                st.metric(
                                    label="Current Overall Measure",
                                    value=measure,
                                )
                                st.metric(
                                    label="Simulated Overall Measure",
                                    value=new_measure,
                                    delta=round(new_measure - measure, 3),
                                )
                                st.write(
                                    f"If you reduce **{target_basic}** violations enough "
                                    f"to lower its BASIC measure by **{reduction_pct}%**, "
                                    f"your overall status would be approximately "
                                    f"**{sim_emoji} {sim_status}**."
                                )

                    # ========= BASIC details table =========
                    st.markdown("---")
                    st.subheader("BASIC Details")

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

                    # ========= Trend notes =========
                    st.markdown("---")
                    st.subheader("Trend & Watch Notes")

                    st.write(
                        "This view uses the latest BASIC snapshot from FMCSA. "
                        "True historical trends (up/down over time) require storing prior "
                        "snapshots, which this MVP does not yet do. For now, treat "
                        "BASICs marked as High or Medium risk as 'watch items' that could "
                        "drive future rate increases if not corrected."
                    )

                    # Raw JSON for debugging if needed
                    with st.expander("Raw API Response (debug)"):
                        st.json(raw_data)

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
            except Exception as e:
                st.error(f"Unexpected error calling QCMobile API: {e}")
