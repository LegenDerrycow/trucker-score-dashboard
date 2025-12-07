import os
import requests
import streamlit as st

# 1. Page setup
st.set_page_config(page_title="Credit Karma for Truckers", page_icon="🚛")
st.title("🚛 Credit Karma for Truckers (QCMobile Skeleton)")

st.write(
    "Enter a USDOT number to fetch BASIC safety data from the FMCSA QCMobile API."
)

# 2. Input
usdot = st.text_input("USDOT Number", placeholder="e.g. 44110")

if st.button("Check BASICs"):
    if not usdot.strip():
        st.error("Please enter a USDOT number.")
    else:
        webkey = os.getenv("QCMOBILE_WEBKEY")
        if not webkey:
            st.error(
                "QCMOBILE_WEBKEY is not set in Streamlit secrets. "
                "Add it before using the app."
            )
        else:
            base_url = "https://mobile.fmcsa.dot.gov/qc/services"
            # Example: /carriers/44110/basics?webKey=...
            url = f"{base_url}/carriers/{usdot.strip()}/basics"
            params = {"webKey": webkey}

            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                st.success("Successfully fetched BASICs data from QCMobile.")
                st.subheader("Raw API Response (for validation)")
                st.json(data)
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error from QCMobile: {e} (status {resp.status_code})")
            except Exception as e:
                st.error(f"Unexpected error calling QCMobile API: {e}")
