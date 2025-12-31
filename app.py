import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------- 1. CONFIGURATION & METRICS -----------------
ACCURACY = 0.99  
VAL_ACCURACY = 0.97
LOSS = 0.02
VAL_LOSS = 0.06

# ----------------- 2. ENHANCED MODERN THEME (FIXED VISIBILITY) -----------------
MAIN_BG_COLOR = "#0E1117" 
SIDEBAR_COLOR = "#111827" 
ACCENT_COLOR = "#00FFAA"  # Neon Green for values

def set_modern_theme():
    st.markdown(
        f"""
        <style>
        /* Main Background */
        .stApp {{
            background-color: {MAIN_BG_COLOR};
            color: white;
        }}
        
        /* 1. HEADER REFINEMENT */
        .main-header {{
            background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
            padding: 2.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }}
        .main-header h1 {{
            font-size: 3rem !important;
            margin-bottom: 0.5rem !important;
        }}
        .main-header p {{
            font-size: 1.3rem !important; /* TEXT SIZE ++ */
            color: #E0E7FF !important;
            font-weight: 500;
        }}

        /* 2. SIDEBAR TEXT VISIBILITY FIX */
        [data-testid="stSidebar"] {{
            background-color: {SIDEBAR_COLOR} !important;
        }}
        /* Target all text in sidebar: labels, list items, and paragraphs */
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] li, 
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {{
            color: #FFFFFF !important; 
            font-size: 1.05rem !important;
            opacity: 1 !important;
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {ACCENT_COLOR} !important;
            margin-bottom: 10px !important;
        }}

        /* 3. METRIC LABEL COLORS */
        [data-testid="stMetricLabel"] {{
            color: #BBBBBB !important; /* Makes labels like 'Training Acc' visible */
            font-size: 1.1rem !important;
            font-weight: bold !important;
            text-transform: uppercase;
        }}
        [data-testid="stMetricValue"] {{
            color: {ACCENT_COLOR} !important;
        }}
        div[data-testid="stMetric"] {{
            background-color: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 20px;
            border-radius: 15px;
        }}

        /* 4. FILE UPLOADER VISIBILITY FIX */
        .stFileUploader section {{
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 2px dashed #4F46E5 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }}
        /* "Drag and drop file here" text */
        .stFileUploader label p {{
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
        }}
        /* Browse Files Button */
        button[data-testid="stBaseButton-secondary"] {{
            background-color: #4F46E5 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            font-weight: bold !important;
        }}

        /* Diagnosis Button Styling */
        .stButton>button {{
            background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 1.2rem !important;
            padding: 0.7rem !important;
            border: none !important;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ----------------- 3. MODEL & PREDICTION FUNCTIONS -----------------

@st.cache_resource
def load_model():
    base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.load_weights("brain_tumor.keras")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def is_mri(image_data):
    img_array = np.array(image_data.convert('RGB'))
    b, g, r = cv2.split(img_array)
    return np.std([np.mean(b), np.mean(g), np.mean(r)]) < 12 

def predict_tumor(image_data, model):
    img_array = np.array(image_data.convert('RGB'))
    image_resized = cv2.resize(img_array, (128, 128)) 
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)
    return model.predict(image_input, verbose=0)[0][0]

# ----------------- 4. APP INTERFACE -----------------

st.set_page_config(page_title="🧠 Brain Tumor AI", layout="wide")
set_modern_theme()
model = load_model()

# --- Sidebar ---
st.sidebar.title("📚 Project Documentation")
st.sidebar.markdown("""
### ⚙️ System Specifications
* **Architecture:** VGG16 (CNN)
* **Input Size:** 128x128 Pixels
* **Task:** Binary Classification
""")
st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Team (APEX NEURAL)")
st.sidebar.markdown("""
* Salman Sarwar (Leader)
* Muhammad Maaz Asim
* Muhammad Nadeem
* Muhammad Zohaib Malik
""")

# --- Enhanced Header ---
st.markdown('''
<div class="main-header">
    <h1>🧠 AI-Based Brain Tumor Classifier</h1>
    <p>Advanced Deep Learning for Trusted Medical MRI Analysis</p>
</div>
''', unsafe_allow_html=True)

## Metrics Section
col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Acc", f"{ACCURACY*100:.1f}%")
col2.metric("Validation Acc", f"{VAL_ACCURACY*100:.1f}%")
col3.metric("Training Loss", f"{LOSS:.4f}")
col4.metric("Validation Loss", f"{VAL_LOSS:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

## Upload Section
st.header("🔬 MRI Analysis Portal")
uploaded_file = st.file_uploader("Upload Brain MRI Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_col, res_col = st.columns([1, 1.2], gap="large")
    
    with img_col:
        st.image(image, caption='Uploaded MRI Source', use_container_width=True)
    
    with res_col:
        if not is_mri(image):
            st.warning("⚠️ **Safety Alert:** Non-MRI detected. Results may be unreliable.")

        if st.button('🚀 RUN NEURAL DIAGNOSIS'):
            with st.spinner('Scanning features...'):
                score = predict_tumor(image, model)
                st.markdown("### Diagnosis Result:")

                if 0.47 < score < 0.53:
                    st.warning("## ⚠️ Result: Inconclusive")
                elif score >= 0.5:
                    st.error(f"## 🚨 Tumor Detected (POSITIVE)")
                    st.subheader(f"Confidence Level: {score*100:.2f}%")
                else:
                    st.success(f"## ✅ No Tumor Detected (NEGATIVE)")
                    st.subheader(f"Confidence Level: {(1-score)*100:.2f}%")
                
                st.info(f"Model Confidence Score: {score:.4f}")

# streamlit run app.py
# for run the app


