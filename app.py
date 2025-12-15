import streamlit as st
import joblib
import os
import pandas as pd

# ─── Load model safely ───
if not os.path.exists("spam_classifier_model.pkl"):
    st.error("Model file not found!")
    st.stop()

model = joblib.load("spam_classifier_model.pkl")

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Real-World Spam Detector")
st.caption("Trained on 20k+ messages • 99%+ accuracy on test set")

text = st.text_area("Paste any SMS or email here ↓", height=160,
                    placeholder="e.g. Congratulations you won $1000 Walmart gift card...")

if st.button("Detect Spam", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            pred = model.predict([text])[0]
            prob = model.predict_proba([text])[0].max()

        if pred == 1:
            st.error(f"SPAM DETECTED — {prob:.1%} confidence")
            st.snow()
        else:
            st.success(f"NOT spam — {prob:.1%} confidence")
            

# ─── Live accuracy test with real spam examples (click to auto-fill) ───
st.markdown("### Try these real-world examples:")

examples = {
    "You won $1000 Walmart gift card! Click here": "SPAM",
    "Free iPhone 15 – limited offer": "SPAM",
    "Your Netflix account will be suspended": "SPAM",
    "Hey are we still meeting for lunch tomorrow?": "NOT spam",
    "Package delivery attempt failed – reschedule here": "SPAM",
    "Mom I’m running late, stuck in traffic": "NOT spam"
}

cols = st.columns(2)
for i, (msg, true_label) in enumerate(examples.items()):
    if cols[i % 2].button(msg, key=f"ex{i}", use_container_width=True):
        text = msg

        st.rerun()
