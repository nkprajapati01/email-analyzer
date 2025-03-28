import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the pre-trained spam detection models
spam_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")
entailment_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")

def classify_email(email_text):
    # Classify the email text using multiple models
    spam_result = spam_classifier(email_text)
    entailment_result = entailment_classifier(email_text, candidate_labels=["spam", "not spam"])
    
    # Combine results
    spam_score = spam_result[0]['score'] if spam_result[0]['label'] == "spam" else 1 - spam_result[0]['score']
    entailment_score = entailment_result['scores'][entailment_result['labels'].index("spam")]
    
    # Average the scores
    avg_score = (spam_score + entailment_score) / 2
    
    return avg_score

def highlight_suspicious_words(email_text, keywords):
    # Highlight suspicious words in the email text
    highlighted_text = email_text
    for keyword in keywords:
        highlighted_text = highlighted_text.replace(keyword, f"**{keyword}**")
    return highlighted_text

def generate_summary(confidence):
    if confidence > 0.4:
        summary = f"⚠️ This email is likely spam with a confidence score of {confidence:.2f}. It is highly recommended not to reply or click any links. Be cautious of any requests for personal information or urgent actions."
    else:
        summary = f"ℹ️ This email does not appear to be spammy, but always exercise caution. Confidence score: {confidence:.2f}."
    return summary

# Refined list of suspicious keywords
suspicious_keywords = [
    "free", "win", "winner", "congratulations", "claim your prize", "limited time offer",
    "you have been selected", "risk-free", "no credit card required", "act now",
    "click here", "verify your account", "secure your account", "your account is at risk",
    "unusual activity detected", "update your payment details", "login required",
    "suspicious activity", "guaranteed", "lowest price",
    "exclusive deal", "best rates", "double your income",
    "earn money fast", "no investment required", "free trial", "miracle cure",
    "anti-aging", "weight loss", "doctor approved", "100% safe", "instant results",
    "you’re a winner", "casino bonus", "jackpot", "betting tips", "no deposit required",
    "make money online", "financial freedom", "get rich quick"
]

# Streamlit app
st.title("📧 Email Analyzer")
st.write("Paste your email text below to analyze its spam likelihood.")

email_text = st.text_area("Email Text")

if email_text:
    confidence = classify_email(email_text)
    
    st.markdown(f"### Spam Likelihood: **{confidence * 100:.2f}%**")
    
    # Pie chart for spam likelihood
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([confidence, 1 - confidence], labels=[f"Spam", "Not Spam"], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
    
    st.markdown("### Explanation")
    st.write("The pie chart above represents the likelihood that the analyzed email is spam. The percentage indicates the confidence level of the classification model. A higher percentage suggests a higher probability that the email is spam. Always exercise caution and avoid clicking on suspicious links or providing personal information in response to such emails.")
    
    summary = generate_summary(confidence)
    st.markdown(f"### Summary")
    st.write(summary)

    st.markdown("""
    - **Free / Win / Risk-Free**: Common keywords in financial & prize scams.
    - **Click Here / Verify Your Account**: Common phishing keywords.
    - **Guaranteed / Lowest Price**: Marketing & promotional spam.
    - **Miracle Cure / Weight Loss**: Health & medicine spam.
    - **You’re a Winner / Casino Bonus**: Lottery & gambling spam.
    - **Make Money Online / Get Rich Quick**: Work-from-home & easy money scams.
    """)

    if confidence > 0.4:
        st.warning("⚠️ This email appears to be very spammy. It is highly recommended not to reply or click any links.")
    else:
        st.info("ℹ️ This email does not appear to be spammy, but always exercise caution.")
