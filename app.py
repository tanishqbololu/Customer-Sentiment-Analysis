import numpy as np
import pandas as pd
import pickle
import streamlit as st
import base64


def add_bg_image(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Add background image
add_bg_image("bg.png")

#Load pickle files
model = pickle.load(open(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\Customer Review NLP and ML\classifier.pkl",'rb'))

tfidf = pickle.load(open(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\Customer Review NLP and ML\vectoriser.pkl",'rb'))

#Title of App
st.title('Customer Sentiment Predictor ')



#Description
st.subheader('📌 Overview:')
st.write("The Customer Sentiment Predictor is an interactive web application designed to analyze customer reviews and determine their sentiment. Whether you're a business owner, a product manager, or just someone interested in understanding customer feedback, this app leverages Natural Language Processing (NLP) and Machine Learning to classify customer reviews as either positive or negative.")

#Input text
InputText = st.text_area("Enter your review 📝")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

if InputText.strip():
    review = re.sub('[^a-zA-Z]',' ',InputText)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    InputVector = tfidf.transform(corpus).toarray() 

#Button 
if st.button('Predict 🪄'):
    if InputText.strip():
        if InputVector is not None:
            # Make the prediction only if review_vector is defined
            prediction = model.predict(InputVector)
            
            # Display the prediction result
            if prediction ==1:
                st.success("Your feedback is **positive** 😊")
            else:
                st.error("Your feedback is **negative** 😞")
    else:
        # Display message if no review is entered
        st.warning("No input recieved, please enter text")   
else:
    st.info("We're eager to hear your thoughts!😀")

st.markdown("""
    ---
    <div style="text-align: center;">
        Created by Tanishq Bololu 😁<br>
        🚀 <a href="https://www.linkedin.com/in/tanishqbololu/" target="_blank">LinkedIn</a>
    </div>
""", unsafe_allow_html=True)