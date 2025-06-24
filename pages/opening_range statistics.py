
import base64
import streamlit as st

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

encoded_logo = get_base64_image("logo.png")

st.markdown(
    f"""
    <style>
    [data-testid="stSidebarNav"]::before {{
        content: "";
        display: block;
        margin-top: 0px;
        margin-bottom: 10px;
        height: 120px;
        background-image: url("data:image/png;base64,{encoded_logo}");
        background-repeat: no-repeat;
        background-size: contain;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


import streamlit as st
st.set_page_config(page_title="Opening Range Stats", layout="wide")

st.title("⏱️ Opening Range Stats")
st.write("This page will analyze opening range sweeps. Coming soon!")
