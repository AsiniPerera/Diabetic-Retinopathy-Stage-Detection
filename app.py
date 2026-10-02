from __future__ import annotations

import base64
import hashlib
import logging

import streamlit as st

from model_inference import TrainedPredictor
from pdf_report import build_pdf_report


MAX_UPLOAD_BYTES = 128 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def get_predictor() -> TrainedPredictor:
    return TrainedPredictor()


def render_analysis() -> None:
    st.markdown(
        "<div class='hero'><div><p class='eyebrow'>RETINAL IMAGE ANALYSIS</p>"
        "<h1>Your retinal image,<br>made easier to understand.</h1>"
        "<p>Upload a retinal image to see the diabetic retinopathy estimate.</p>"
        "</div><div class='hero-art'>◎</div></div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a retinal image",
        type=["jpg", "jpeg", "png"],
        help="JPEG or PNG, up to 128 MB.",
    )
    if uploaded is None:
        st.info("Choose an image to get started.")
        return

    payload = uploaded.getvalue()
    if len(payload) > MAX_UPLOAD_BYTES:
        st.error("The image is larger than 128 MB. Choose a smaller JPEG or PNG.")
        return
    st.image(payload, caption=uploaded.name, width=360)

    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                predictor = get_predictor()
                result = predictor.predict(payload, uploaded.name)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_payload"] = payload
                st.session_state["analysis_upload_hash"] = hashlib.sha256(payload).hexdigest()
        except Exception as exc:
            LOGGER.exception("Retinal image analysis failed")
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("analysis_result")
    if (
        not result
        or st.session_state.get("analysis_upload_hash") != hashlib.sha256(payload).hexdigest()
    ):
        return

    st.divider()
    st.subheader("Result")
    if not result.get("classification_permitted"):
        st.warning("This image was not graded.")
    else:
        st.metric("Estimated grade", result["predicted_class"])
        st.progress(float(result["confidence"]), text=f"Model confidence: {result['confidence']:.1%}")
        st.caption("This is a research estimate, not a diagnosis.")
        st.markdown("**Grade probability**")
        for probability in result["probabilities"]:
            st.progress(
                float(probability["probability"]),
                text=f"{probability['name']}: {probability['probability']:.1%}",
            )

    for reason in result.get("reasons", []):
        st.info(reason)

    report = build_pdf_report(result, st.session_state["analysis_payload"])
    st.download_button(
        "Download PDF report",
        data=report,
        file_name="retinal-analysis-report.pdf",
        mime="application/pdf",
    )

    with st.expander("Image-processing steps"):
        for step in result.get("processing_log", []):
            st.markdown(
                f"**{step['number']:02d}. {step['title']}** · "
                f"{step['status']} · {step['duration_ms']:.0f} ms"
            )
            if step.get("details"):
                st.json(step["details"], expanded=False)

    for visual in result.get("visual_steps", []):
        encoded = visual.get("image", "").partition(",")[2]
        if encoded:
            st.image(
                base64.b64decode(encoded),
                caption=visual.get("title", "Processing preview"),
                width=280,
            )


def main() -> None:
    st.set_page_config(
        page_title="RetiScreen AI",
        page_icon="👁️",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #061311 0%, #101629 55%, #161327 100%); }
        .block-container { max-width: 1180px; padding-top: 2rem; }
        .hero { display:flex; justify-content:space-between; align-items:center; gap:2rem;
                padding:2.3rem; margin:0 0 2rem; border:1px solid #2b5655;
                border-radius:28px; background:linear-gradient(115deg,#103538,#171c32); }
        .hero h1 { font-size:clamp(2rem,5vw,3.4rem); line-height:1.08; }
        .hero p { color:#b4cfcc; }
        .eyebrow { color:#7fe2c4 !important; font-size:.8rem; letter-spacing:.15em; font-weight:700; }
        .hero-art { color:#83e5c8; font-size:8rem; text-shadow:0 0 35px #53cbb977; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    analysis_tab, model_tab = st.tabs(["Image analysis", "About the model"])
    with analysis_tab:
        render_analysis()
    with model_tab:
        st.title("About the model")
        st.markdown(
            """
            - **Architecture:** EfficientNetB0 with ImageNet transfer learning
            - **Dataset:** 3,600 usable APTOS images
            - **Input preparation:** retinal extraction, resize and padding to 224 × 224
            - **Training:** classifier adaptation followed by selective fine-tuning
            - **Grades:** No DR, Mild, Moderate, Severe and Proliferative DR

            This research prototype is not a medical device. Results do not replace
            assessment by a qualified clinician.
            """
        )


if __name__ == "__main__":
    main()
