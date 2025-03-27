import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# Load the pre-trained model for email classification
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def classify_email(email_text):
    # Classify the email text
    result = classifier(email_text)
    return result

def highlight_suspicious_words(email_text, keywords):
    # Highlight suspicious words in the email text
    highlighted_text = email_text
    for keyword in keywords:
        highlighted_text = highlighted_text.replace(keyword, f"**{keyword}**")
    return highlighted_text

# List of suspicious keywords
suspicious_keywords = [
    "free", "win", "winner", "congratulations", "claim your prize", "limited time offer",
    "you have been selected", "risk-free", "no credit card required", "act now", "urgent",
    "click here", "verify your account", "secure your account", "your account is at risk",
    "unusual activity detected", "update your payment details", "login required",
    "suspicious activity", "confidential information", "guaranteed", "lowest price",
    "exclusive deal", "best rates", "special promotion", "double your income",
    "earn money fast", "no investment required", "free trial", "miracle cure",
    "anti-aging", "weight loss", "doctor approved", "100% safe", "instant results",
    "you’re a winner", "casino bonus", "jackpot", "betting tips", "no deposit required",
    "make money online", "work from home", "financial freedom", "get rich quick", "passive income"
]

# Streamlit app
st.title("Email Analyzer")
st.write("Paste your email text below to analyze its category.")

email_text = st.text_area("Email Text")

if email_text:
    result = classify_email(email_text)
    category = result[0]['label']
    confidence = result[0]['score']
    
    st.write(f"Category: {category}")
    st.write(f"Confidence Score: {confidence:.2f}")
    
    highlighted_text = highlight_suspicious_words(email_text, suspicious_keywords)
    
    st.write("Highlighted Email Text:")
    st.write(highlighted_text)
    
    # Pie chart for confidence score
    fig, ax = plt.subplots()
    ax.pie([confidence, 1 - confidence], labels=[f"{category}", "Other"], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)
