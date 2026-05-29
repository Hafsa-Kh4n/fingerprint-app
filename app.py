import io
import numpy as np
import streamlit as st
import cv2
from PIL import Image

st.set_page_config(
    page_title="Fingerprint Matcher",
    page_icon="🔏",
    layout="wide"
)

HF_REPO_ID = "Hafsa-Hab1b/fingerprint-siamese"
HF_MODEL_FILE = "fingerprint_siamese.tflite"

IMG_SIZE = 64
DEFAULT_THRESHOLD = 1.0


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from huggingface_hub import hf_hub_download
    import tflite_runtime.interpreter as tflite

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILE
    )

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=(0, -1))


def predict(interpreter, a, b, threshold):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], a.astype(np.float32))
    interpreter.set_tensor(input_details[1]["index"], b.astype(np.float32))

    interpreter.invoke()

    dist = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    is_same = dist < threshold

    if is_same:
        conf = (1 - dist / threshold) * 100
    else:
        conf = (dist - threshold) / (2 - threshold) * 100

    conf = float(np.clip(conf, 0, 100))

    return dist, is_same, conf


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🔏 Fingerprint Matcher")
    st.markdown("---")

    threshold = st.slider(
        "Decision threshold",
        min_value=0.1,
        max_value=2.0,
        value=DEFAULT_THRESHOLD,
        step=0.05
    )

    st.caption("Siamese Network · TFLite model")


# ---------------- MAIN UI ----------------
st.title("🔏 Fingerprint Matching System")
st.write("Upload two fingerprint images to check if they belong to the same person.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fingerprint A")
    fp_a = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg", "bmp"], key="a")

    if fp_a:
        st.image(Image.open(fp_a).convert("L"), use_container_width=True)

with col2:
    st.subheader("Fingerprint B")
    fp_b = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg", "bmp"], key="b")

    if fp_b:
        st.image(Image.open(fp_b).convert("L"), use_container_width=True)

st.markdown("---")

compare_btn = st.button(
    "🔍 Compare Fingerprints",
    use_container_width=True,
    disabled=not (fp_a and fp_b)
)

if not (fp_a and fp_b):
    st.info("Please upload both fingerprint images to continue.")

if compare_btn and fp_a and fp_b:
    with st.spinner("Running model..."):
        model = load_model()

        a = preprocess(fp_a)
        b = preprocess(fp_b)

        dist, is_same, conf = predict(model, a, b, threshold)

    st.markdown("## Result")

    left, right = st.columns([1, 2])

    with left:
        if is_same:
            st.success("✅ SAME PERSON")
        else:
            st.error("❌ DIFFERENT PERSON")

        st.metric("Confidence", f"{conf:.1f}%")
        st.metric("Distance", f"{dist:.4f}")
        st.metric("Threshold", f"{threshold:.2f}")

    with right:
        st.markdown("### Distance Indicator")
        st.progress(min(dist / 2.0, 1.0))

        st.caption("0 = identical fingerprints, 2 = completely different")
