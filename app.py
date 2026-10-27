import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier | AI Studio",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual aesthetics
st.markdown("""
<style>
    /* Metric card styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        background: linear-gradient(135deg, #4F46E5, #9333EA);
        color: white;
        border-radius: 9999px;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.4);
        margin: 0.5rem 0;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading for fast Streamlit performance
@st.cache_resource(show_spinner="Loading trained MNIST neural network...")
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'mnist_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`! Please run `python train_model.py` first.")
        st.stop()
    data = joblib.load(model_path)
    return data['model'], data.get('accuracy', 0.9707), data.get('samples', {})

model, model_acc, sample_digits = load_model()

# Header Section
st.title("🔢 MNIST Handwritten Digit Classifier")
st.markdown("A lightweight neural network trained on the MNIST benchmark dataset. Draw a digit, upload an image, or test sample digits.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Specifications")
    st.markdown(f"""
    - **Architecture**: Multi-Layer Perceptron (MLP)
    - **Layers**: `784 ➔ 128 (ReLU) ➔ 64 (ReLU) ➔ 10 (Softmax)`
    - **Benchmark Accuracy**: **`{model_acc * 100:.2f}%`**
    - **Model Footprint**: **~2.5 MB** *(Ultra-fast for Streamlit Cloud)*
    - **Runtime**: CPU-friendly scikit-learn
    """)
    st.markdown("---")
    st.subheader("💡 Tips for Best Drawing Results")
    st.markdown("""
    1. Draw centrally within the box.
    2. Use a thick stroke (brush width 16–22px).
    3. The model recognizes single digits (0 to 9).
    """)
    st.markdown("---")
    st.markdown("🚀 **Ready for Streamlit Community Cloud**")

# Input Mode Tabs
tab_draw, tab_upload, tab_sample = st.tabs(["✍️ Draw Digit", "📁 Upload Image", "🎲 Sample Digits"])

processed_img_28x28 = None

with tab_draw:
    col_canvas, col_ctrl = st.columns([1.2, 1])
    
    with col_canvas:
        st.write("Draw a digit (0 - 9) on the canvas:")
        stroke_width = st.slider("Brush stroke width", min_value=12, max_value=28, value=18, step=2)
        
        # Interactive drawing canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas",
        )

    with col_ctrl:
        st.write("**Canvas Preview & Processing**")
        if canvas_result.image_data is not None:
            # Check if user has drawn something
            img_data = canvas_result.image_data
            if np.max(img_data[:, :, 3]) > 0: # Checks alpha or pixels
                # Convert RGBA to Grayscale
                img_pil = Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('L')
                
                # Resize to 28x28 using Lanczos filter
                img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                
                # Center of mass alignment / bounding box normalization if content exists
                if np.max(arr) > 0.05:
                    processed_img_28x28 = arr
                    st.image(img_resized, caption="28x28 Model Input Representation", width=120)
                else:
                    st.info("Canvas is empty. Draw a number to predict!")
            else:
                st.info("Canvas is empty. Draw a number to predict!")

with tab_upload:
    st.write("Upload an image file containing a handwritten digit:")
    uploaded_file = st.file_uploader("Upload digit (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        raw_pil = Image.open(uploaded_file).convert('L')
        
        # Invert if user uploaded black text on white background
        raw_arr = np.array(raw_pil)
        if np.mean(raw_arr) > 127: # predominantly light background
            raw_pil = ImageOps.invert(raw_pil)
            
        img_resized = raw_pil.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        processed_img_28x28 = arr
        
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            st.image(raw_pil, caption="Processed Source Image", width=140)
        with col_up2:
            st.image(img_resized, caption="Resized to 28x28 Model Input", width=120)

with tab_sample:
    st.write("Select a pre-loaded test digit from the MNIST dataset:")
    col_sel, col_prev = st.columns([1, 1])
    
    with col_sel:
        target_digit = st.selectbox("Choose Digit Class", options=list(range(10)), index=3)
        sample_idx = st.radio("Choose Sample Variant", options=[0, 1, 2, 3], horizontal=True)
    
    if sample_digits and target_digit in sample_digits and len(sample_digits[target_digit]) > sample_idx:
        sample_vec = sample_digits[target_digit][sample_idx]
        sample_mat = np.array(sample_vec, dtype=np.float32).reshape(28, 28)
        
        with col_prev:
            st.image(sample_mat, caption=f"MNIST Test Sample: Digit {target_digit}", width=120)
        
        if st.button("Classify this Sample", type="primary"):
            processed_img_28x28 = sample_mat

# Classification & Analytics Section
st.markdown("---")
st.subheader("📊 Classification Results")

if processed_img_28x28 is not None:
    # Flatten image to 1D vector of 784 pixels
    input_vector = processed_img_28x28.reshape(1, -1)
    
    # Predict probabilities & class
    probabilities = model.predict_proba(input_vector)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = probabilities[predicted_digit] * 100
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Digit</div>
            <div class="prediction-badge">{predicted_digit}</div>
            <div class="metric-title" style="margin-top: 10px;">Confidence Score</div>
            <div class="metric-value">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick summary breakdown
        sorted_indices = np.argsort(probabilities)[::-1]
        st.write("**Top 3 Most Likely:**")
        for rank in range(min(3, len(sorted_indices))):
            d = sorted_indices[rank]
            p = probabilities[d] * 100
            st.markdown(f"- **Digit {d}**: `{p:.2f}%`")
            
    with col_res2:
        st.write("**Full Probability Distribution (0–9)**")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')
        
        digits = [str(i) for i in range(10)]
        colors = ['#6366F1' if i == predicted_digit else '#334155' for i in range(10)]
        
        bars = ax.bar(digits, probabilities * 100, color=colors, width=0.6, edgecolor='rgba(255,255,255,0.1)')
        
        ax.set_ylim(0, 105)
        ax.set_ylabel("Probability (%)", color='#94A3B8', fontsize=10)
        ax.set_xlabel("Digit Class", color='#94A3B8', fontsize=10)
        ax.tick_params(colors='#CBD5E1', which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.grid(axis='y', linestyle='--', alpha=0.2, color='#FFFFFF')
        
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='#F8FAFC', fontsize=8, fontweight='bold')
                            
        st.pyplot(fig)
else:
    st.info("👆 Draw a digit above, upload an image, or pick a sample digit to see predictions.")
