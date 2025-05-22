import streamlit as st
import streamlit.components.v1 as components
import os
from datetime import datetime
import pandas as pd
from helper import get_base64, set_background, title_style, result_style, create_data_input, save_prediction
import random

# ----------------- Thiết lập cấu hình -----------------
IMAGE_DIR = "project/images"
BACKGROUND_IMG = os.path.join(IMAGE_DIR, "main.jpg") 
EXCEL_LOG = "project/user_input.xlsx"
UPLOAD_DIR = "project/uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Dữ liệu nhạc cụ (hiển thị và phân loại)
instrument_images = {
    "Đàn bầu": "danbau.jpg",
    "Đàn tranh": "dantranh.png",
    "Đàn nhị": "dan nhi.jpg",
    "Sáo": "sao.jpg"
}

# ----------------- Giao diện -----------------
# Nền
set_background(BACKGROUND_IMG)

# Tiêu đề
st.markdown(title_style, unsafe_allow_html=True)
st.markdown("<div class='title'>Vietnamese Traditional Instrument Classifier</div>", unsafe_allow_html=True)

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


# ----------------- Upload và phân loại -----------------
st.markdown("### 🎵 Upload file âm thanh (.mp3 hoặc .wav)")
uploaded_file = st.file_uploader("Chọn tệp", type=["mp3", "wav"])

# Giả lập mô hình phân loại
def classify_instrument(audio_path):
    return random.choice(list(instrument_images.keys()))

if uploaded_file:
    st.audio(uploaded_file, format='audio/mp3')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{uploaded_file.name}")

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("📁 File đã được tải lên!")
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
        result = classify_instrument(save_path)
        st.markdown(result_style, unsafe_allow_html=True)
        st.markdown(f"<div class='result'>🎼 Kết quả: {result}</div>", unsafe_allow_html=True)

        # Tạo file log nếu chưa có
        if not os.path.exists(EXCEL_LOG):
            pd.DataFrame(columns=["Filename", "Predicted Instrument"]).to_excel(EXCEL_LOG, index=False)

        existing_data = pd.read_excel(EXCEL_LOG)
        new_data = create_data_input([uploaded_file.name], [result], existing_data, existing_data.shape[0])
        save_prediction(EXCEL_LOG, existing_data, new_data)

        # Thống kê
        st.markdown("### 📊 Thống kê kết quả phân loại:")
        chart_data = pd.read_excel(EXCEL_LOG)
        stats = chart_data["Predicted Instrument"].value_counts()
        st.bar_chart(stats)
