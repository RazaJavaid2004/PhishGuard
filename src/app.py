"""
PhishGuard: Multi-Source Email Threat Classification
Interactive Prediction Interface (Phase 5)

A Streamlit web application that loads the trained TF-IDF vectorizer
and Logistic Regression model, accepts raw email text from the user,
and returns a threat classification with confidence probability.
"""

from pathlib import Path
import joblib
import streamlit as st
import numpy as np

# ── Resolve project paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"

# ── Benchmark Sample Texts ───────────────────────────────────────────
SAMPLE_PHISHING = (
    "Dear Customer,\n\n"
    "We have detected unusual activity and unauthorized sign-in attempts on your account. "
    "To ensure your security, access has been temporarily restricted. "
    "Please click the secure link below to verify your identity and restore full access:\n\n"
    "http://secure-account-verification-portal.com/login?token=892374\n\n"
    "Failure to verify within 24 hours will result in permanent account suspension.\n\n"
    "Sincerely,\n"
    "Customer Security & Protection Department"
)

SAMPLE_LEGITIMATE = (
    "Hi Team,\n\n"
    "Just a friendly reminder that our quarterly planning session is scheduled for "
    "this Thursday at 2:00 PM in Conference Room B (and via the standard video link).\n\n"
    "Please review the attached project roadmap and have your departmental updates "
    "ready so we can finalize the timeline for next sprint.\n\n"
    "Let me know if anyone has scheduling conflicts.\n\n"
    "Best regards,\n"
    "Sarah Jenkins\n"
    "Senior Project Coordinator"
)


# ── Load artifacts (cached so they persist across reruns) ────────────
@st.cache_resource
def load_artifacts():
    """Load the fitted vectorizer and trained model from disk."""
    if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
        st.error(
            "⚠️ Model artifacts not found. "
            "Please run `python src/train.py` first to train and save the model."
        )
        st.stop()
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model


# ── Prediction logic ─────────────────────────────────────────────────
def predict_email(text: str, vectorizer, model) -> dict:
    """
    Classify a single email and return the prediction label,
    class name, and confidence probability.
    """
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred_label = int(np.argmax(proba))
    confidence = float(proba[pred_label])
    class_name = "Phishing / Threat" if pred_label == 1 else "Legitimate / Safe"
    return {
        "label": pred_label,
        "class_name": class_name,
        "confidence": confidence,
        "prob_legitimate": float(proba[0]),
        "prob_phishing": float(proba[1]),
    }


# ── Session State Callbacks for Sample Loading ───────────────────────
def set_sample_phishing():
    st.session_state["email_text_input"] = SAMPLE_PHISHING

def set_sample_legitimate():
    st.session_state["email_text_input"] = SAMPLE_LEGITIMATE

def clear_email_input():
    st.session_state["email_text_input"] = ""


# ── Streamlit UI ─────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PhishGuard – Email Threat Classifier",
        page_icon="🛡️",
        layout="centered",
    )

    # Initialize session state for text input if not present
    if "email_text_input" not in st.session_state:
        st.session_state["email_text_input"] = ""

    # Header
    st.markdown(
        """
        <div style="text-align:center; padding-bottom:10px;">
            <h1>🛡️ PhishGuard</h1>
            <p style="font-size:1.15em; color:grey;">
                Multi-Source Email Threat Classification System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Load model
    vectorizer, model = load_artifacts()
    st.success("✅ Model and vectorizer loaded successfully.", icon="✅")

    # Sample Buttons
    st.markdown("##### 🧪 Quick Test Samples")
    bcol1, bcol2, bcol3 = st.columns([1, 1, 0.7])
    with bcol1:
        st.button(
            "🔴 Load Phishing Sample",
            on_click=set_sample_phishing,
            use_container_width=True,
        )
    with bcol2:
        st.button(
            "🟢 Load Legitimate Sample",
            on_click=set_sample_legitimate,
            use_container_width=True,
        )
    with bcol3:
        st.button(
            "🧹 Clear",
            on_click=clear_email_input,
            use_container_width=True,
        )

    # Input area bound to session state key
    st.markdown("##### 📧 Email Content")
    email_text = st.text_area(
        label="Email content to classify",
        key="email_text_input",
        height=220,
        placeholder=(
            "Paste the complete email body or headers here to classify..."
        ),
        label_visibility="collapsed",
    )

    # Classify Button
    if st.button("🔍 Classify Email", type="primary", use_container_width=True):
        if not email_text or not email_text.strip():
            st.warning("⚠️ Please enter or load email text before classifying.")
        else:
            with st.spinner("Analyzing email semantics and threat patterns..."):
                result = predict_email(email_text.strip(), vectorizer, model)

            # Result display
            st.divider()
            if result["label"] == 1:
                st.error(f"### 🚨 Prediction: {result['class_name']}", icon="🚨")
            else:
                st.success(f"### ✅ Prediction: {result['class_name']}", icon="✅")

            # Confidence metrics
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Status", result["class_name"].split(" / ")[0])
            mcol2.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
            mcol3.metric("Phishing Probability", f"{result['prob_phishing'] * 100:.2f}%")

            # Probability bars
            st.markdown("##### Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            prob_col1.progress(
                result["prob_legitimate"],
                text=f"Legitimate: {result['prob_legitimate'] * 100:.1f}%",
            )
            prob_col2.progress(
                result["prob_phishing"],
                text=f"Phishing: {result['prob_phishing'] * 100:.1f}%",
            )

    # Footer
    st.divider()
    st.caption(
        "PhishGuard v1.0 — Complex Computing Activity (CCA) · "
        "Logistic Regression (SAGA, L2) on TF-IDF (10,000 features) · "
        "Trained on 163,868 multi-source emails · "
        "Accuracy: 98.60% · Recall: 98.97%"
    )


if __name__ == "__main__":
    main()
