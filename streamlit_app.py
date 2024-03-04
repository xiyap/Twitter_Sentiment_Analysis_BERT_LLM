import streamlit as st
from utils import set_background, preprocess_input, tweet_inference
from TweetClassifierClass import TweetClassifer

set_background('./streamlit_data/background.jpg')

# Streamlit interface
st.title('Twitter Sentiment Analysis')

input_sentence = st.text_input('Enter your sentence here:', '')

# Inference + Result
if st.button('Predict Sentiment'):
    if input_sentence.strip() == '' or len(input_sentence.split()) < 2:
        st.error('Please enter a sentence.')
    else:
        pkl_saved_model = 'TweetClassifer_2class_model.pkl'
        results = tweet_inference(pkl_saved_model, input_sentence)

        st.success(f'Predicted Sentiment: {results[0]}')
        st.info(f'Percentage of Predicted Sentiment: {results[1] * 100:.2f}%')