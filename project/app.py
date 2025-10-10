import io
import os
import sys
import streamlit as st
from helper import get_base64, set_background, title_style, result_style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import predict_endpoint

# ----------------- CONFIG -----------------
IMAGE_DIR = "project/images"
BACKGROUND_IMG = os.path.join(IMAGE_DIR, "main.jpg") 

# ------------------- UI -------------------
# Background
set_background(BACKGROUND_IMG)

# Title
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>Traditional Vietnamese Instrument Classification</div>", unsafe_allow_html=True)

# CSS style
st.markdown("""
    <style>
        @media (max-width: 767px){
        .instrument-container {
            flex-direction : column;
        }
        .instrument {
        margin-bottom: 20px;
        }

        .instrument img {
             width: 100%;
        }
        }
        @media (max-width: 1200px) and (min-width:768px){
        .instrument-container {
            display : flex;
            flex-wrap: wrap;
        }
        }
        .instrument-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flew-wrap : wrap;
            gap : 20px; 
        }
        .instrument {
            text-align: center;
            margin: 0 20px;
        }
        .instrument img {
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
            width: 300px;
            height: 300px;
            min-width: 250px;
            transition: transform 0.3s ease;
        }
        .instrument img:hover {
            transform: scale(1.05);
        }
        .instrument-caption {
            margin-top: 10px;
            font-weight: bold;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
    """, unsafe_allow_html=True)

html_images = f"""
    <div class='instrument-container'>
        <div class='instrument'>
            <img src='data:image/png;base64,{get_base64(os.path.join(IMAGE_DIR, "danbau.jpg"))}' alt='Đàn bầu' />
            <div class='instrument-caption'>Đàn bầu</div>
        </div>
        <div class='instrument'>
            <img src='data:image/png;base64,{get_base64(os.path.join(IMAGE_DIR, "dantranh.png"))}' alt='Đàn tranh' />
            <div class='instrument-caption'>Đàn tranh</div>
        </div>
        <div class='instrument'>
            <img src='data:image/png;base64,{get_base64(os.path.join(IMAGE_DIR, "dan nhi.jpg"))}' alt='Đàn nhị' />
            <div class='instrument-caption'>Đàn nhị</div>
        </div>
        <div class='instrument'>
            <img src='data:image/png;base64,{get_base64(os.path.join(IMAGE_DIR, "sao.jpg"))}' alt='Sáo' />
            <div class='instrument-caption'>Sáo</div>
        </div>
    </div>
    """

st.markdown(html_images, unsafe_allow_html=True)

# ----------------- Upload and predict -----------------
st.markdown("### 🎵 Upload file âm thanh (.mp3 hoặc .wav)")
uploaded_file = st.file_uploader("Chọn tệp", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/mp3')
    file_bytes = uploaded_file.read()
    file_like = io.BytesIO(file_bytes)

    st.success("📂 File đã được tải lên!")
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: blue;
            color: red;
            padding: 16px 32px;
            font-size: 40px;
            border-radius: 8px;
            cursor: pointer;
        }
        div.stButton > button:hover {
            opacity : 80%;
        }
        </style>
        """, unsafe_allow_html=True)
    
    if st.button("🎯 Phân loại"):
        result, _, _, _ = predict_endpoint(file_like)
        st.markdown(result_style, unsafe_allow_html=True)
        st.markdown(f"<div class='result'>🎼 Kết quả: {result}</div>", unsafe_allow_html=True)