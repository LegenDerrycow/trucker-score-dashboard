import os
import requests
import pandas as pd
import streamlit as st


# ---------- Flatten QCMobile BASICs ----------

def flatten_qcmobile_basics(raw_data):
    """
    Take the JSON returned by QCMobile /carriers/:dotNumber/basics
    and flatten it into a list of simple dicts, one per BASIC.

    Expected structure (as in your sample):

    {
      "content": [
        {
          "basic": {
            "basicsPercentile": "18%",
            "basicsRunDate": "...",
            "basicsType": {
              "basicsCode": "Unsafe Driving",
              "basicsId": 11,
              "basicsShortDesc": "Unsafe Driving"
            },
            "basicsViolationThreshold": "50",
            "exceededFMCSAInterventionThreshold": "Y/N",
            "measureValue": "1.02",
            "onRoadPerformanceThresholdViolationIndicator": "Y/N",
            "seriousViolationFromInvestigationPast12MonthIndicator": "Y/N",
            "totalInspectionWithViolation": 155,
            "totalViolation": 170
          },
          "_links": {...}
        },
        ...
      ],
      "retrievalDate": "..."
    }
    """
    basics_flat = []

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


# ---------- Risk evaluation & scoring ----------

def evaluate_basic_risk(row):
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


def compute_carrier_measure(basics_flat):
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


def status_from_measure(measure: float):
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


def summarize_insurance_impact(basics_flat, measure, overall_status):
    """
    Return a short narrative about likely insurance impact based on:

    - Presence of High/Medium risk BASICs
    - Intervention flags (exceeded threshold)
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
            "Expect meaningful rate pressure (rough order +15–30%+) at renewal and "
            "reduced market appetite."
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
            "Expect flat to modest movements (roughly 0–5%) driven more by market "
            "conditions than your own safety performance."
        )

    return band, text


def action_items_for_basic(name: str):
    """
    Return a list of recommended actions for a given BASIC name.
    Uses simple keyword matching.
    """
    name_lower = (name or "").lower()

    if "unsafe" in name_lower:
        return [
            "Run a violation drilldown: identify top drivers and locations contributing to Unsafe Driving.",
            "Implement targeted coaching and ride-alongs for high-risk drivers.",
            "Tighten policies on speeding, following distance, and mobile phone use.",
            "Consider telematics or camera-based coaching for hard braking and speeding.",
        ]
    if "hos" in name_lower or "hours-of-service" in name_lower:
        return [
            "Audit ELD logs weekly for form-and-manner and HOS violations.",
            "Lock down log edits and require documented reasons for any changes.",
            "Provide refresher training on 11/14-hour rules and split sleeper provisions.",
            "Use alerts for approaching HOS limits to reduce roadside violations.",
        ]
    if "driver fitness" in name_lower:
        return [
            "Audit all CDLs, endorsements, and medical cards for currency and accuracy.",
            "Implement a pre-trip document check process for newly dispatched drivers.",
            "Tighten hiring and qualification file reviews before onboarding.",
            "Work with your agent to show documented Q-file cleanup before renewal.",
        ]
    if "drug" in name_lower or "alcohol" in name_lower or "substances" in name_lower:
        return [
            "Review your random testing program for proper selection rates and follow-up.",
            "Document all policy violations and corrective actions taken.",
            "Refresh driver training on your drug and alcohol policy.",
            "Ensure all positives and refusals are handled and documented per regulation.",
        ]
    if "vehicle maint" in name_lower or "maintenance" in name_lower:
        return [
            "Map violations back to specific units and shops to find patterns.",
            "Tighten your preventive maintenance schedule and track completion.",
            "Enforce pre- and post-trip DVIRs and close the loop on defects.",
            "Focus first on brakes, tires, and lights—high-impact, visible at roadside.",
        ]

    # Generic fallback
    return [
        "Review recent violations in this BASIC and identify repeat patterns.",
        "Prioritize corrective actions that remove violations from future inspections.",
        "Document your plan and share it with your broker before renewal.",
    ]


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛")
st.title("🚛 Credit Karma for Truckers")

st.write(
    "Enter a USDOT number to translate FMCSA BASICs into an insurance-style score, "
    "risk summary, and action plan. Lower is better (golf score)."
)

usdot = st.text_input("USDOT Number", placeholder="e.g. 44110")

if st.button("Check BASICs"):
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
                    # Evaluate risk per BASIC
                    for b in basics_flat:
                        band, score = evaluate_basic_risk(b)
                        b["Risk Level"] = band
                        b["Risk Score"] = score

                    # Compute carrier measure & status
                    measure = compute_carrier_measure(basics_flat)
                    status, emoji = status_from_measure(measure)

                    # Insurance impact summary
                    impact_band, impact_text = summarize_insurance_impact(
                        basics_flat, measure, status
                    )

                    # Convert to DataFrame for display
                    df = pd.DataFrame(basics_flat)

                    # ---------- Top summary ----------
                    st.markdown("---")
                    st.subheader("Overall Status")

                    bg_color = {
                        "Green": "#dcfce7",   # light green
                        "Yellow": "#fef9c3",  # light yellow
                        "Red": "#fee2e2",     # light red
                    }.get(status, "#e5e7eb")  # gray fallback

                    st.markdown(
                        f"""
                        <div style="
                            background-color:{bg_color};
                            padding:1rem;
                            border-radius:0.75rem;
                            border:1px solid #d4d4d4;
                        ">
                            <h2 style="margin:0; font-size:1.5rem;">
                                {emoji} {status} – Overall Measure: {measure}
                            </h2>
                            <p style="margin:0.25rem 0 0; color:#4b5563;">
                                Lower is better (golf score). This measure reflects your worst BASIC.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ---------- Insurance impact ----------
                    st.subheader("Insurance Impact Summary")

                    st.markdown(
                        f"""
                        <div style="
                            background-color:#eef2ff;
                            padding:1rem;
                            border-radius:0.75rem;
                            border:1px solid #c7d2fe;
                        ">
                            <h3 style="margin:0; font-size:1.2rem;">
                                {impact_band}
                            </h3>
                            <p style="margin:0.25rem 0 0; color:#4b5563;">
                                {impact_text}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ---------- Focus areas / Action plan ----------
                    st.subheader("Focus Areas & Action Plan")

                    # Sort BASICs by Risk Score (descending) and Percentile
                    df_sorted = df.sort_values(
                        by=["Risk Score", "Percentile (num)"],
                        ascending=[False, False],
                    )

                    top_basics = df_sorted.head(3).to_dict(orient="records")

                    if not top_basics:
                        st.write("No significant risk areas identified.")
                    else:
                        for b in top_basics:
                            name = b.get("BASIC") or b.get("BASIC Code")
                            risk_level = b.get("Risk Level")
                            pct = b.get("Percentile")
                            thresh = b.get("Violation Threshold")
                            insp = b.get("Inspections With Violation (24m)")
                            viol = b.get("Total Violations (24m)")

                            st.markdown(f"### {name} – {risk_level} priority")

                            summary_bits = []
                            if pct is not None and thresh is not None:
                                summary_bits.append(f"Percentile {pct} vs threshold {thresh}")
                            if insp is not None:
                                summary_bits.append(f"{insp} inspections with violations")
                            if viol is not None:
                                summary_bits.append(f"{viol} total violations")

                            if summary_bits:
                                st.write(" • " + " | ".join(str(x) for x in summary_bits))

                            # Recommended actions
                            items = action_items_for_basic(name)
                            st.write("**Recommended actions:**")
                            for item in items:
                                st.write(f"- {item}")

                    # ---------- BASIC details table ----------
                    st.subheader("BASIC Details")

                    preferred_cols = [
                        "BASIC",
                        "Risk Level",
                        "Risk Score",
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

                    # ---------- "Trend" / watch explanation ----------
                    st.subheader("Trend & Watch Notes")

                    st.write(
                        "This view uses the latest BASIC snapshot from FMCSA. "
                        "True historical trends (up/down over time) require storing prior "
                        "snapshots, which this MVP does not yet do. "
                        "For now, treat BASICs marked as High or Medium risk as 'watch items' "
                        "that could drive future rate increases if not corrected."
                    )

                    # Raw JSON for debugging
                    with st.expander("Raw API Response (debug)"):
                        st.json(raw_data)

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
            except Exception as e:
                st.error(f"Unexpected error calling QCMobile API: {e}")
