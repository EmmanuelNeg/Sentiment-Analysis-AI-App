import streamlit as st # function to create UI components and control the app layout
from transformers import pipeline # pipline is a convenience wrapper from hugging face that loads a model + tokenizer/ pre/post processing for comming tasks (sentiment, summarization, translation)
import os

st.set_page_config(page_title="AI Practise", layout="centered") #sets the browser tab title and lyout style for streamlit

st.title(" AI practise App")
st.write("Type some text and see live sentiment analysis")

#Load sentiment analysis
@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment = load_sentiment() 

st.subheader("1) Sentiment analysis")
user_text = st.text_area("Enter text to analyze", value="Hello I love building AI apps, my name is Emmanuel Atule", height=130)

if st.button("Analyze sentiment"): 
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Running model..."):
             result = sentiment(user_text[:1000]) #limit for safety
        label = result[0]["label"]
        score = result[0]["score"]
        st.markdown(f"**prediction:** '{label}'")
        st.markdown(f"**prediction:** {score:.2f}")
        if label.lower().startswith("pos"):
            st.success("This looks positive")
        else:
            st.error("Oops! this looks negative (or not postive).")

    

