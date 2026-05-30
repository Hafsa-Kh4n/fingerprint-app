import io, os
import numpy as np
import streamlit as st
import cv2
from PIL import Image

st.set_page_config(page_title="Fingerprint Matcher", page_icon="🔏", layout="wide")

HF_REPO_ID    = "Hafsa-Hab1b/fingerprint-siamese"
HF_MODEL_FILE = "fingerprint_siamese.onnx"
IMG_SIZE      = 64
THRESHOLD     = 1.0

@st.cache_resource(show_spinner="Loading model... (first time only)")
def load_model():
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    path    = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILE)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session

def preprocess(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Could not read image.")
        st.stop()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(img, -1), 0)  # (1, 64, 64, 1)

def predict(session, a, b, threshold):
    inputs = {
        session.get_inputs()[0].name: a,
        session.get_inputs()[1].name: b,
    }
    dist    = float(session.run(None, inputs)[0][0][0])
    is_same = dist < threshold
    conf    = (1 - dist/threshold)*100 if is_same else ((dist-threshold)/(2-threshold))*100
    return dist, is_same, float(np.clip(conf, 0, 100))

# ── Sidebar ──
with st.sidebar:
    st.title("🔏 Fingerprint Matcher")
    st.markdown("---")
    threshold = st.slider("Decision threshold", 0.1, 2.0, THRESHOLD, 0.05,
        help="Distance below this = same person. Range is 0 to 2.")
    st.markdown("---")
    st.caption("Siamese network · SOCOfing dataset")

# ── Main ──
st.title("🔏 Fingerprint Matching")
st.caption("Upload two fingerprints — model tells you if they belong to the same person")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Fingerprint A")
    fp_a = st.file_uploader("Upload A", type=["bmp","png","jpg","jpeg"], key="a")
    if fp_a:
        st.image(Image.open(io.BytesIO(fp_a.read())).convert("L"), use_column_width=True)
        fp_a.seek(0)

with col2:
    st.subheader("Fingerprint B")
    fp_b = st.file_uploader("Upload B", type=["bmp","png","jpg","jpeg"], key="b")
    if fp_b:
        st.image(Image.open(io.BytesIO(fp_b.read())).convert("L"), use_column_width=True)
        fp_b.seek(0)

st.markdown("---")
btn = st.button("🔍 Compare Fingerprints", use_container_width=True,
                disabled=not (fp_a and fp_b))

if not (fp_a and fp_b):
    st.info("Upload both fingerprint images above to compare.")

if btn and fp_a and fp_b:
    with st.spinner("Comparing..."):
        model            = load_model()
        arr_a            = preprocess(fp_a)
        arr_b            = preprocess(fp_b)
        dist, is_same, conf = predict(model, arr_a, arr_b, threshold)

    st.markdown("## Result")
    r1, r2 = st.columns([1, 2])
    with r1:
        if is_same:
            st.success("## ✅ SAME PERSON")
        else:
            st.error("## ❌ DIFFERENT PERSONS")
        st.metric("Confidence", f"{conf:.1f}%")
        st.metric("Distance",   f"{dist:.4f}")
        st.metric("Threshold",  f"{threshold:.2f}")
    with r2:
        st.markdown("#### Distance gauge (0 = identical → 2 = completely different)")
        st.progress(min(dist / 2.0, 1.0))
        c1, c2, c3 = st.columns(3)
        c1.caption("0 — identical")
        c2.caption(f"▲ threshold {threshold:.2f}")
        c3.caption("2 — max diff")
        st.markdown(f"""
| | Value | Meaning |
|---|---|---|
| Distance | `{dist:.4f}` | How different the fingerprints are |
| Threshold | `{threshold:.2f}` | Below this = same person |
| Confidence | `{conf:.1f}%` | How sure the model is |
""")

st.markdown("---")
st.caption("Siamese network · SOCOfing dataset · threshold default = 1.0")
