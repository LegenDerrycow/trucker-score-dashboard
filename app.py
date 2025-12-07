import os
import requests
import pandas as pd
import streamlit as st


# ---------- Helper functions ----------

def flatten_qcmobile_basics(raw_data):
    """
    Take the JSON returned by QCMobile /carriers/:dotNumber/basics
    and flatten it into a list of simple dicts, one per BASIC.

    Uses the sample structure you provided:
    {
      "content": [
        {"basic": {...}, "_links": {...}},
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
        basic = entry.get("basic", {})
        basics_type = basic.get("basicsType", {}) or {}

        # Extract & normalize fields we care about
        short_desc = basics_type.get("basicsShortDesc")
        code = basics_type.get("basicsCode")

        percentile_raw = basic.get("basicsPercentile")
        measure_raw = basic.get("measureValue")
        threshold = basic.get("basicsViolationThreshold")

        exceeded = basic.get("exceededFMCSAInterventionThreshold")
        onroad_def = basic.get("onRoadPerformanceThresholdViolationIndicator")
        serious_def = basic.get("seriousViolationFromInvestigationPast12MonthIndicator")

        total_insp_viol = basic.get("totalInspectionWithViolation")
        total_viol = basic.get("totalViolation")

        run_date = basic.get("basicsRunDate")

        # Keep original strings for display; also compute numeric forms
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

        basics_flat.append(
            {
                "BASIC": short_desc or code,
                "BASIC Code": code,
                "Percentile": percentile_raw,
                "Percentile (num)": percentile_num,
                "Measure": measure_raw,
                "Measure (num)": measure_num,
                "Violation Threshold": threshold,
                "Exceeded FMCSA Threshold": exceeded,
                "On-Road Threshold Violation": onroad_def,
                "Serious Violation (12m)": serious_def,
                "Inspections With Violation (24m)": total_insp_viol,
                "Total Violations (24m)": total_viol,
                "Snapshot Date": run_date,
            }
        )

    return basics_flat


def compute_carrier_measure(basics_flat):
    """
    MVP "golf score" for the carrier.

    Use the maximum BASIC Measure (worst BASIC) as the carrier score:

        carrier_measure = max(Measure (num) for each BASIC)

    Lower is better, consistent with FMCSA measure.
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


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛")
st.title("🚛 Credit Karma for Truckers")

st.write(
    "Enter a USDOT number to fetch BASIC safety data from the FMCSA QCMobile API. "
    "We compute an MVP 'golf score' using the worst BASIC Measure (lower is better)."
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
                    # Build dataframe for display
                    df = pd.DataFrame(basics_flat)

                    # Compute carrier measure & status
                    measure = compute_carrier_measure(basics_flat)
                    status, emoji = status_from_measure(measure)

                    # Big traffic light header
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
                                {emoji} {status} – Measure: {measure}
                            </h2>
                            <p style="margin:0.25rem 0 0; color:#4b5563;">
                                Lower is better (golf score). This is an MVP score derived from FMCSA BASIC Measures.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # BASIC table
                    st.subheader("BASIC Details")

                    # Show key columns first if present
                    preferred_cols = [
                        "BASIC",
                        "BASIC Code",
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

                    # Raw JSON for debugging
                    with st.expander("Raw API Response (debug)"):
                        st.json(raw_data)

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
            except Exception as e:
                st.error(f"Unexpected error calling QCMobile API: {e}")
