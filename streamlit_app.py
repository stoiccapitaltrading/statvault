import streamlit as st
import base64
import os

# Try to inject logo if it exists
def inject_sidebar_logo(image_path="logo.png"):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_logo = base64.b64encode(image_file.read()).decode()
            st.markdown(
                f"""
                <style>
                [data-testid="stSidebarNav"]::before {{
                    content: "";
                    display: block;
                    margin-top: -60px;
                    margin-bottom: 20px;
                    height: 180px;
                    background-image: url("data:image/png;base64,{encoded_logo}");
                    background-repeat: no-repeat;
                    background-size: contain;
                    background-position: center;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning(f"⚠️ Failed to load logo: {e}")
    else:
        st.info("ℹ️ No logo file found. Skipping logo injection.")

# Inject logo (if present)
inject_sidebar_logo()

# Basic content to confirm it's working
st.set_page_config(page_title="AlphaStats Dashboard", layout="wide")
st.title("✅ AlphaStats is Live")
st.markdown("Welcome! Select a module from the sidebar to begin.")
