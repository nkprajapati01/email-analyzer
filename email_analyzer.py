import streamlit as st
from transformers import pipeline

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
    
    suspicious_keywords = ["urgent", "click", "free", "win", "password"]
    highlighted_text = highlight_suspicious_words(email_text, suspicious_keywords)
    
    st.write("Highlighted Email Text:")
    st.write(highlighted_text)
