# Fake News Detection

## Overview
This project is a **Fake News Detection** system built using a GRU-based deep learning model. The model is trained on a dataset containing real and fake news articles to classify input text as **Fake** or **Real**.

## Check for the model visualizations in the live links 
[View the Project on Render](https://brainwave-matrix-intern-yzt2.onrender.com)
Note Live Link will take time to load....(Edited: Render unable to handel the tensorflow so live link is useless currently..)
Also the Visualizations Uploaded in Src folder

Make sure to check both Prediction and Model Visualization from the Sidebar
<img src="src\prediction.png" alt="App Preview" width="600">

## Dataset
The dataset used in this project is downloaded from Kaggle:
- `Fake.csv` - Contains fake news articles.
- `True.csv` - Contains real news articles.

## Steps in the Notebook

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
```

### 2. Download and Load the Dataset
```python
!kaggle datasets download emineyetm/fake-news-detection-datasets

with zipfile.ZipFile('fake-news-detection-datasets.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

fake_data = pd.read_csv("News_dataset/Fake.csv")
true_data = pd.read_csv("News_dataset/True.csv")
```

### 3. Exploratory Data Analysis (EDA)
- Count the number of fake vs. real news articles.
- Generate **word clouds** to visualize the most frequent words in both categories.

### 4. Data Preprocessing
- Removing null values and unnecessary columns.
- Checking for unknown publishers.
- Tokenizing and padding the text for neural network input.

### 5. Model Training (GRU-Based Model)
```python
model = Sequential([
    Embedding(vocab_size,embedding_dim,input_shape=(max_len,),trainable=False),
    BatchNormalization(),
    GRU(units=128,activation="tanh",kernel_regularizer=regularizers.l2(0.0001)),
    BatchNormalization(),
    Dropout(0.2),
    BatchNormalization(),
    Dense(1,activation="sigmoid")
])
```
- The model uses an embedding layer, followed by a single GRU layer, multiple batch normalization layers, a dropout layer, and a final dense layer with a sigmoid activation function for binary classification.

### 6. Training the Model
```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=42,callbacks=[callback])
```

### 7. Making Predictions
```python
input_text = "Iran's leaders have portrayed the anti-government protests as  instigated by foreign enemies."
test_data= []
for i in range(0, len(custom_data)):
    review =custom_data[i]
    review = review.split()
    review = ' '.join(review)
    test_data.append(review)
    voc_size = 100
onehot_repr=[one_hot(words,voc_size)for words in test_data]
sent_length=20
embedded_docs=pad_sequences(onehot_repr,maxlen=sent_length,padding='pre')
test_data=np.array(embedded_docs)
prediction=(model.predict(test_data)>0.5).astype(int)

if(prediction[0]==0):
  print("News is Fake")
else:
  print("News is Real")
```
- The model predicts **Fake (0)** or **Real (1)** for the given input news article.

## Running the Project
1. Clone the repository:
```sh
git clone https://github.com/devil3515/Brainwave_Matrix_Intern.git
cd Brainwave_Matrix_Intern
```
2. Install the required libraries:
```sh
pip install -r requirements.txt
```
2. Run the Streamlit App:
```sh
streamlit run app.py
```

## Conclusion
This project successfully classifies news articles as fake or real using a **GRU-based neural network**. Future improvements could include using **transformers (BERT, GPT)** or **hybrid models** to enhance accuracy.
