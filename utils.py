import base64
import streamlit as st
from PIL import ImageOps, Image

import re

import torch
import joblib
from transformers import AutoTokenizer
from torch.nn.functional import softmax

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html = True)

# Preprocess input sentence
def preprocess_input(sentence):
    text_cleaning_re = "@\S+|https?:\S+|[^A-Za-z0-9]+"
    text_clean = re.sub(text_cleaning_re, ' ', str(sentence).lower()).strip()
    return text_clean

def tweet_inference(pkl_model_path, input_sentence):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    # Preprocess and tokenize input sentence
    processed_sentence = preprocess_input(input_sentence)
    tokenized_input = tokenizer.encode_plus(
        processed_sentence,
        max_length = 25,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )

    # Push tensors to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    # Load fine-tuned model
    tweet_classifier_model = joblib.load(pkl_model_path)
    tweet_classifier_model = tweet_classifier_model.to(device)

    # Set model to evaluation mode
    tweet_classifier_model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = tweet_classifier_model(input_ids, attention_mask)
        probabilities = softmax(outputs, dim = 1)
        _, predicted_label = torch.max(outputs, 1)
        predicted_class = predicted_label.item()
        probability_of_predicted_class = probabilities[0][predicted_class].item()

    # Return results
    sentiment_labels = ['Negative', 'Positive']
    predicted_sentiment = sentiment_labels[predicted_class]
    
    return predicted_sentiment, probability_of_predicted_class