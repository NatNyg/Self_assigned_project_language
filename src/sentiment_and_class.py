"""
Importing the libraries I'll be using througout the script 
"""
import argparse
# simple text processing tools
import re
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

import numpy as np 
import pandas as pd 
import os
import sys

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# scikit-learn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

#sentiment analysis tools 
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


# Create a function to get the polarity
def getPolarity(text):
    """
function to get the polarity using textblob
    """
    return  TextBlob(text).sentiment.polarity

def getAnalysis(score):
    """
function to getAnalysis (decision boundary for labels)
    """
    if score < 0:
        return 'Negative'
    else:
        return 'Positive'

"""
The three following functions (strip_html_tags, remove_accented_chars and preprocess_corpus) are functions that helps preprocess the corpus we'll be performing deep learning tasks on. Basically they help normalise the docs so the text is clean and ready to work with. 
"""
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
    return norm_docs

def load_data(dataset):
    """
This function loads and processes the data defined by the user - I will be using the "DisneylandReviews.csv" file for my project. The function does the following to prepare the data for textblob and vader sentiment analysis:
- read the data using pandas 
- remove na's from the data (these are marked as "missing" in the dataset, hence the syntacs)
- remove duplicates 
- fetch the reviews column from the data and use the previous defined pre_process_corpus function to prepare the data
    """
    
    filepath=f'in/{dataset}'
    df=pd.read_csv(filepath,encoding="cp1252", na_values=['missing'])
    df=df.dropna().reset_index()
    df.drop_duplicates(subset='Review_Text', inplace=True, keep='first')
    
    Reviews = df['Review_Text'].values
    processed_reviews= pre_process_corpus(Reviews)
    
    return processed_reviews
    
def textblob_sentiment(processed_reviews):
    """
This function performs text_blob_sentiment on the data, trains a deep learning model and predicts labels (positive/negative sentiment). The following steps are being executed:
- the processed reviews is saved in a dataframe, and the getPolarity function using textblob is applied on the reviews
- the getAnalysis function is used to determine whether the sentiment of each review is positive or negative
- the negative and positive sentiment is then split up into two new dataframes, in order to balance the data
- I use the resample function from scikit-learn to downsample the positive sentiment so it matches the negative sentiment size 
- I then concatenate the negative sentiment and the positive downsample into a new dataframe (data_balanced)
- I then fetch the reviews and sentiment so it's ready for further processing

    """
    processed_reviews_df = pd.DataFrame()
    processed_reviews_df["reviews"] = processed_reviews 
    processed_reviews_df['Polarity'] = processed_reviews_df['reviews'].apply(getPolarity)
    processed_reviews_df['Analysis'] = processed_reviews_df['Polarity'].apply(getAnalysis)
    positive_sentiment = processed_reviews_df[processed_reviews_df["Analysis"] == "Positive"]
    negative_sentiment = processed_reviews_df[processed_reviews_df["Analysis"] == "Negative"]
    positive_downsample = resample(positive_sentiment,
                                   replace=True,
                                   n_samples=len(negative_sentiment),
                                   random_state=42)
    data_balanced = pd.concat([negative_sentiment, positive_downsample])
    
    reviews = data_balanced["reviews"].values 
    sentiment = data_balanced["Analysis"].values
    
    return reviews, sentiment

def vader_sentiment(processed_reviews):
    
    analyzer = SentimentIntensityAnalyzer()

    reviews_vader_sentiment = [] 
    for review in processed_reviews:
        reviews_vader_sentiment.append(analyzer.polarity_scores(review))
    
    VADER_sentiment_df = pd.DataFrame() 
    VADER_sentiment_df["reviews"] = processed_reviews
    
    sentiment_df = pd.DataFrame(reviews_vader_sentiment, columns=["neg","neu","pos","compound"])
    VADER_sentiment_df = VADER_sentiment_df.join(sentiment_df)

    VADER_sentiment_df['Analysis'] = VADER_sentiment_df['compound'].apply(getAnalysis)
    
    positive_vader = VADER_sentiment_df[VADER_sentiment_df["Analysis"] == "Positive"]
    negative_vader =  VADER_sentiment_df[ VADER_sentiment_df["Analysis"] == "Negative"]
    positive_vader_downsample = resample(positive_vader,
                                         replace=True,
                                         n_samples=len(negative_vader),
                                         random_state=42)
    
    vader_data_balanced = pd.concat([negative_vader, positive_vader_downsample])
    reviews = vader_data_balanced["reviews"].values 
    sentiment = vader_data_balanced["Analysis"].values
    return reviews, sentiment

def process_data(reviews, sentiment):
    """
This function processes the data from the sentiment and reviews retrieved from either textblob_sentiment or vader_sentiment. It then performs the following:
- Splits the data into train/test 
- Initializing the tokenizer from tensorflow keras including what to do if we encounter a word in the test data, that is not yet in the vocabulary from the training data
- Fit the tokenizer on the documents 
- Set the padding value to 0 (adding 0's to the end of documents that are shorter than others, ensuring that documents have the same length
- Turn the text into sequences 
- Set maximum sequence length to 1000, ensuring a max of 1000 tokens in each document 
- Add padding sequences to test and train data
- Using sci-kit learn's labelbinarizer to binarize the labels 
    """

    X_train, X_test, y_train, y_test = train_test_split(reviews,
                                                        sentiment, 
                                                        test_size = 0.2, 
                                                        random_state = 42)
   
    t = Tokenizer(oov_token = '<UNK>')
    t.fit_on_texts(X_train) 

    t.word_index["<PAD>"] = 0 
    X_train_seqs = t.texts_to_sequences(X_train)
    X_test_seqs = t.texts_to_sequences(X_test)
    MAX_SEQUENCE_LENGTH = 1000
   
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad, y_train, y_test



def define_model(t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad):
    """
This function defines the model I want to use for the text classification through the following steps:
- Clearing session for models and parameters
- Define parameters for model (overall vocabulary size, embedding size, number of epochs and batch size)
- Create the model using the Sequential tensorflow keras model 
- Adding layers (embedding, convolution, pooling and fully connected classification). We'll be using sigmoid and binary crossentropy since we're dealing with a binary classification problem (positve or negative sentiment)
    """
    tf.keras.backend.clear_session()

    VOCAB_SIZE = len(t.word_index)
    EMBED_SIZE = 300
    EPOCH_SIZE = 2 
    BATCH_SIZE = 128
    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                     kernel_size=4, 
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model, EPOCH_SIZE, BATCH_SIZE

def train_and_evaluate_model(X_train_pad, y_train, X_test_pad, y_test, model, EPOCH_SIZE, BATCH_SIZE, sentiment_method):
    """
This function trains and evaluates the model just defined by doing the following:
- fitting the model on the train data and saving the history of the training
- evaluating the model using the evaluate function on the test data and saving the scores 
- making predictions using the predict function and saving the results
- assigning the labels 
- making a classification report using scikit learn's function, and saving the report to the "out" folder. 
    """
    
    history = model.fit(X_train_pad, y_train,
                        epochs = EPOCH_SIZE,
                        batch_size = BATCH_SIZE,
                        validation_split = 0.1, 
                        verbose = True) 
    
    scores = model.evaluate(X_test_pad, y_test, verbose = 1)
    print(f"accuracy = {scores[1]}")
    #0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    with open(f'out/{sentiment_method}_report.txt', 'w') as file:
        file.write(classification_report(y_test, predictions, target_names = ['Negative', 'Positive']))
    

    
def parse_args():
    """
This function intialises the argumentparser and adds the command line parameters 
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-sm","--sentiment_method",required=True, help = "The method used to make sentimentanalysis")
    ap.add_argument("-ds","--dataset",required=True, help = "Dataset to make sentiment analysis and text classification on")
    args = vars(ap.parse_args())
    return args 

def main():
    """
The main function defines which functions to run when the script is executed, and which command line parameters should be passed to what functions. I have added two if statements, allowing the user to input what sentiment analysis method to use (textblob or vader) and also what dataset the sentiment analysis and text classification should be performed on. 
    """
    args = parse_args()
    if args["sentiment_method"] == "textblob":
        processed_reviews = load_data(args["dataset"])
        reviews, sentiment = textblob_sentiment(processed_reviews)
        t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad, y_train, y_test = process_data(reviews, sentiment)
        model, EPOCH_SIZE, BATCH_SIZE = define_model(t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad)
        train_and_evaluate_model(X_train_pad, y_train, X_test_pad, y_test, model, EPOCH_SIZE, BATCH_SIZE, args["sentiment_method"])
        
    elif args["sentiment_method"] == "vader":
        processed_reviews = load_data(args["dataset"])
        reviews, sentiment = vader_sentiment(processed_reviews)
        t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad, y_train, y_test = process_data(reviews, sentiment)
        model, EPOCH_SIZE, BATCH_SIZE = define_model(t, MAX_SEQUENCE_LENGTH, X_train_pad, X_test_pad)
        train_and_evaluate_model(X_train_pad, y_train, X_test_pad, y_test, model, EPOCH_SIZE, BATCH_SIZE, args["sentiment_method"])
        
     
    
if __name__== "__main__":
    main()
    


