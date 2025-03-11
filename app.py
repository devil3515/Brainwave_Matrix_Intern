import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
with open('model/model.sav', 'rb') as file:
    model = pickle.load(file)

##Defining the different apps for the model visualizations and model predictions    
def app1():
    st.title("Fake News Detection")
    st.header("#Dataset Used")
    true_data = pd.read_csv('dataset/True.csv')
    fake_data = pd.read_csv('dataset/Fake.csv')
    st.write("#True News Dataset")
    st.write(true_data.head())
    st.write("#Fake News Dataset")
    st.write(fake_data.head())

    st.header("#Exploratory Data Analysis")
    st.write("###True News Dataset")
    st.write(true_data["subject"].value_counts())
    st.write("###Fake News Dataset")
    st.write(fake_data["subject"].value_counts())

    st.subheader("Count Fake news data types")
    st.image(r'src\fake-news-data-types.png')
    st.subheader("Count True news data types")
    st.image(r'src\true-news-data-types.png')


    st.title("Word Cloud for the Fake and True news")
    st.subheader("Word Cloud for Fake news")
    st.image(r'src\fake-data-wordcloud.png')
    st.subheader("Word Cloud for True news")
    st.image(r'src\true-data-wordcloud.png')


    st.title("Model Performance")
    st.subheader("Model Performance")
    st.image(r'src\model-metrics.png')
    st.subheader("Model Confusion Matrix")
    st.image(r'src\confusion-matrix.png')



def app2():
    input_data = st.text_input("Enter the news to predict: ")

    # Define prediction variable before using it
    prediction = None  

    if st.button("Predict"):
        test_data = []
        for i in range(len(input_data)):
            review = input_data[i].split()
            review = ' '.join(review)
            test_data.append(review)

        voc_size = 100
        onehot_repr = [one_hot(words, voc_size) for words in test_data]
        sent_length = 20
        embedded_docs = pad_sequences(onehot_repr, maxlen=sent_length, padding='pre')
        test_data = np.array(embedded_docs)

        if test_data.size == 0:
            st.error("No data provided for prediction.")
        else:
            prediction = (model.predict(test_data) > 0.5).astype(int)

    # Only check prediction if it exists
    if prediction is not None:
        if prediction[0] == 0:
            st.write("News is Fake")
        else:
            st.write("News is Real")





selected_app = st.sidebar.selectbox("Select an App", ["Fake News Detection Predictions", "Model Visualizations"])

if selected_app == "Fake News Detection Predictions":
    app2()
elif selected_app == "Model Visualizations":
    app1()
