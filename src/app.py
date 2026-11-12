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

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"


# ── Multi-Template Collections ───────────────────────────────────────
PHISHING_TEMPLATES = [
    (
        "Subject: Urgent: Unauthorized Sign-in Attempt Detected\n\n"
        "Dear Customer,\n\n"
        "We detected an unrecognized sign-in attempt to your online banking portal "
        "from an unknown device in Moscow, Russia. For your protection, your access "
        "has been temporarily restricted.\n\n"
        "Please verify your identity within 24 hours to prevent permanent account suspension:\n"
        "http://security-auth-portal-online.net/verify?id=982341\n\n"
        "Thank you,\nAccount Fraud Prevention Team"
    ),
    (
        "Subject: Request for Immediate Wire Transfer - Confidential\n\n"
        "Hi,\n\n"
        "I am currently in an executive meeting and cannot take calls. We have an urgent "
        "vendor payment that must be settled today to avoid contract termination. Please "
        "process an outgoing international wire of $18,450 to the account details attached:\n\n"
        "http://secure-wire-processing-desk.com/invoices/vendor_pay.pdf\n\n"
        "Confirm once the wire transfer is executed.\n\n"
        "Regards,\nChief Executive Officer"
    ),
    (
        "Subject: Delivery Notice: Incomplete Shipping Information (Tracking ID: #US-88291)\n\n"
        "Dear Recipient,\n\n"
        "Your parcel with express tracking #US-88291 could not be dispatched due to an "
        "incorrect shipping address on file. A redelivery fee of $1.85 is required to "
        "release the package.\n\n"
        "Update your shipping preferences and pay the fee here:\n"
        "http://postal-express-redelivery-dispatch.org/update-order\n\n"
        "If unclaimed within 48 hours, the package will be returned to the sender."
    ),
    (
        "Subject: Action Required: Update Your Direct Deposit Information\n\n"
        "Attention Employee,\n\n"
        "Our HR payroll system was upgraded over the weekend. All staff must re-confirm "
        "their direct deposit bank information before Thursday 5:00 PM to ensure this "
        "month's payroll is credited without delay.\n\n"
        "Access the employee self-service portal here:\n"
        "http://internal-hr-benefits-portal.com/payroll/auth\n\n"
        "Human Resources Payroll Department"
    ),
    (
        "Subject: John Doe shared a confidential document with you via OneDrive\n\n"
        "You have received an encrypted financial document via Microsoft SharePoint.\n\n"
        "Document: Q4_Executive_Bonus_Structure.xlsx (3.2 MB)\n"
        "Access Level: Restricted\n\n"
        "Click here to sign in with your corporate credentials to view this document:\n"
        "http://onedrive-live-document-viewer.link/auth-login\n\n"
        "Microsoft 365 Document Sharing Service"
    ),
]

LEGITIMATE_TEMPLATES = [
    (
        "Subject: Weekly Team Standup - Agenda & Video Link\n\n"
        "Hi Team,\n\n"
        "A quick reminder about our weekly engineering standup tomorrow at 10:00 AM. "
        "We will review the sprint burndown, discuss blockers on the API migration, "
        "and introduce our new intern, Alex.\n\n"
        "Agenda:\n"
        "1. Sprint review (15 mins)\n"
        "2. Infrastructure update (10 mins)\n"
        "3. Open Q&A (10 mins)\n\n"
        "See you all tomorrow,\nMark"
    ),
    (
        "Subject: Scheduled IT Maintenance Window: Saturday 02:00 UTC\n\n"
        "Hello all,\n\n"
        "Please be advised that the corporate network team will be applying standard "
        "security patches to our internal file servers this Saturday from 2:00 AM to "
        "5:00 AM UTC. During this period, the internal VPN and shared drives will "
        "experience intermittent downtime.\n\n"
        "No action is required on your part. If you experience lingering issues after "
        "06:00 UTC, please submit an internal ticket at helpdesk.internal.corp.\n\n"
        "Best regards,\nCorporate IT Infrastructure Team"
    ),
    (
        "Subject: Reminder: Quarterly All-Hands Meeting this Thursday\n\n"
        "Hi everyone,\n\n"
        "We look forward to seeing everyone at our Q3 All-Hands meeting this Thursday "
        "at 3:00 PM in the Main Auditorium and streamed live for remote staff. The "
        "executive leadership team will share company highlights, financial milestones, "
        "and celebrate team achievements.\n\n"
        "Submit your questions for the anonymous AMA session via the Slack "
        "#all-hands-qa channel.\n\n"
        "Warmly,\nInternal Communications Team"
    ),
    (
        "Subject: PhishGuard Project Deliverables and Code Review Request\n\n"
        "Hi Professor and Review Committee,\n\n"
        "I have committed the final implementation for our Complex Computing Activity "
        "(PhishGuard: Multi-Source Email Threat Classification). All data pipelines, "
        "exploratory analysis notebooks, trained model artifacts, and evaluation metrics "
        "have been validated.\n\n"
        "The repository contains the complete documentation and technical report "
        "outlining our results.\n\n"
        "Sincerely,\nStudent Research Team"
    ),
    (
        "Subject: Re: Budget Spreadsheet for Q3 Review\n\n"
        "Hi Nadia,\n\n"
        "Thanks for sending over the updated budget spreadsheet. I reviewed the numbers "
        "for the marketing and engineering departments and everything looks good. I made "
        "a few minor adjustments to the travel expense projections based on last quarter's "
        "actuals.\n\n"
        "Could you forward the final version to David and the finance team before Friday "
        "so they can include it in the board presentation?\n\n"
        "Thanks,\nKevin Park\nOperations Analyst"
    ),
]


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
    """Classify a single email and return prediction with confidence."""
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


# ── Session State Callbacks ──────────────────────────────────────────
def cycle_phishing_sample():
    idx = st.session_state.get("phish_idx", -1)
    idx = (idx + 1) % len(PHISHING_TEMPLATES)
    st.session_state["phish_idx"] = idx
    st.session_state["email_text_input"] = PHISHING_TEMPLATES[idx]


def cycle_legitimate_sample():
    idx = st.session_state.get("legit_idx", -1)
    idx = (idx + 1) % len(LEGITIMATE_TEMPLATES)
    st.session_state["legit_idx"] = idx
    st.session_state["email_text_input"] = LEGITIMATE_TEMPLATES[idx]


def clear_email_input():
    st.session_state["email_text_input"] = ""


# ── Streamlit UI ─────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PhishGuard – Email Threat Classifier",
        page_icon="🛡️",
        layout="centered",
    )

    # Initialize session state
    if "email_text_input" not in st.session_state:
        st.session_state["email_text_input"] = ""
    if "phish_idx" not in st.session_state:
        st.session_state["phish_idx"] = -1
    if "legit_idx" not in st.session_state:
        st.session_state["legit_idx"] = -1

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

    # Sample Buttons — show which template is active
    st.markdown("##### 🧪 Quick Test Samples *(click repeatedly to cycle through 5 templates)*")
    bcol1, bcol2, bcol3 = st.columns([1, 1, 0.6])
    with bcol1:
        phish_num = st.session_state["phish_idx"] + 1 if st.session_state["phish_idx"] >= 0 else 0
        st.button(
            f"🔴 Phishing Sample ({phish_num}/{len(PHISHING_TEMPLATES)})",
            on_click=cycle_phishing_sample,
            use_container_width=True,
        )
    with bcol2:
        legit_num = st.session_state["legit_idx"] + 1 if st.session_state["legit_idx"] >= 0 else 0
        st.button(
            f"🟢 Legit Sample ({legit_num}/{len(LEGITIMATE_TEMPLATES)})",
            on_click=cycle_legitimate_sample,
            use_container_width=True,
        )
    with bcol3:
        st.button(
            "🧹 Clear",
            on_click=clear_email_input,
            use_container_width=True,
        )

    # Input area bound to session state
    st.markdown("##### 📧 Email Content")
    email_text = st.text_area(
        label="Email content to classify",
        key="email_text_input",
        height=220,
        placeholder="Paste the complete email body or headers here to classify...",
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
            mcol3.metric("Phishing Prob.", f"{result['prob_phishing'] * 100:.2f}%")

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
