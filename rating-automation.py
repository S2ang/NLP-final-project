# Import modules
import os
import re
import math
import random
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

from random import sample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM, Bidirectional, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_word_list = stopwords.words('english')
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
  
# Folder Paths
binary_path = "Drive\\Users\\Location\\aclImdb\\binary_classification\\"
train_path = "Drive\\Users\\Location\\aclImdb\\train"
test_path = "Drive\\Users\\Location\\aclImdb\\test"

# Classes
class Review:
    user_rating = 0 #rating given by user
    description = "" #text body of user review
    objective = False #review objectivity
    sentiment = "" #"positive"/"negative"/"mixed"
    sentiment_polarity = -1 #analysis polarity of sentiment
    sentiment_rating = 0 #rating given by program
    rating_disparity = 0 #percentage disparity
    reliable = False #reliability test

class Analysis_response:
    def __init__(self, name, score, accuracy):
        self.name = name
        self.score = score
        self.accuracy = accuracy
    total_disparity = 100
    total_accuracy = 0
    total_reliability = 0
    disparity_accumulator = 0
    reliability_accumulator = 0

# Global variables
max_rating = 10
responses = []

# Read text File 
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        return f.read()
    
# Iterate through all folders/files in directory
def data_load(path):
    movies = []
    for folder in os.listdir(path):
        # create folder path
        folder_path = path + "/" + folder
        # check folder
        if os.path.isdir(folder_path):
            if folder == "neg":
                print("add negative reviews: start")
                # iterate through files in folder
                for file in os.listdir(folder_path):
                    # Check whether file is in text format or not
                    if file.endswith(".txt"):
                        file_path = f"{folder_path}/{file}"
                        # create review object
                        this_review = Review()
                        # get user given rating from file name
                        if int(file[-5]) == 0:
                            this_review.user_rating = 10
                        else:
                            this_review.user_rating = int(file[-5])
                        # call read text file function and add review to object
                        this_review.description = read_text_file(file_path)
                        this_review.sentiment = "negative"
                        # add review to list of reviews
                        movies.append(this_review)
                print("add negative reviews: complete\n")
            elif folder == "pos":
                print("add positive reviews: start")
                # iterate through files in folder
                for file in os.listdir(folder_path):
                    # Check whether file is in text format or not
                    if file.endswith(".txt"):
                        file_path = f"{folder_path}/{file}"
                        # create review object
                        this_review = Review()
                        # get user given rating from file name
                        if int(file[-5]) == 0:
                            this_review.user_rating = 10
                        else:
                            this_review.user_rating = int(file[-5])
                        # call read text file function and add review to object
                        this_review.description = read_text_file(file_path)
                        this_review.sentiment = "positive"
                        # add review to list of reviews
                        movies.append(this_review)
                print("add positive reviews: complete\n")
            else:
                print("list of movie reviews complete\n")
    return movies

# PRIMARY FUNCTIONS
# Removing stop words
def remove_stopwords(text, is_lower_case=False):
    tokenizer=ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop_word_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def data_parsing(text):
    text = remove_stopwords(text)
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

# Train models & analyse sentiment
def analyse_sentiment(df, name):
    
    tokenizer = Tokenizer(num_words=8000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    
    df['TEXT'] = df['TEXT'].apply(data_parsing)
    
    df['TEXT'] = df['TEXT'].apply(lambda x: re.sub('[^a-zA-Z"]', ' ', x))
    
    tokenizer.fit_on_texts(df['TEXT'].values)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(df['TEXT'].values)
    print('Found %s unique tokens.' % len(word_index))
    
    num_tokens = [len(tokens) for tokens in sequences]
    num_tokens = np.array(num_tokens)
    max_len = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_len = int(max_len)
    
    X = pad_sequences(sequences, maxlen = max_len)
    
    Y = df['SENTIMENT'].map({'negative' : 0, 'positive' : 1}).values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size=0.25, random_state=42)
    
    embed_dim = 64
    lstm_out = 64
    epochs = 5
    batch_size = 32

    print((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    
    model = Sequential()
    model.add(Embedding(8000, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=3, min_delta=0.0001)])
    
    validation_size = len(df)//10

    X_validate = X_test[-validation_size:]
    Y_validate = Y_test[-validation_size:]
    X_test = X_test[:-validation_size]
    Y_test = Y_test[:-validation_size]

    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    
    response = Analysis_response(name, score*100, acc*100)
    responses.append(response)

    print("score: %.3f" % (score))
    print("acc: %.3f" % (acc))
    
    return model, tokenizer, max_len

def calculate_polarity(description, model, token, max_len):
    token_gen = [description]
    seq_gen = token.texts_to_sequences(token_gen)
    pad_gen = pad_sequences(seq_gen, maxlen=max_len)
    pol_gen = model.predict(pad_gen)
    return pol_gen[0][0]

def predict_rating(pol):
    rating = math.ceil(pol*10)
    return rating

def calculate_disparity(r): 
    disparity = r.user_rating - r.sentiment_rating
    if disparity < 0:
        disparity *= -1
    r.rating_disparity = (disparity/max_rating) * 100 #percentage disparity

def test_reliability(r): 
    test_rating = ""

    if r.sentiment == "positive":
        if 6 <= r.sentiment_rating <= 10:
            test_rating = "pass"
        else:
            test_rating = "fail"
    elif r.sentiment == "negative":
        if 1 <= r.sentiment_rating <= 5:
            test_rating = "pass"
        else:
            test_rating = "fail"
    else:
        print("review sentiment tagged incorrectly")

    if test_rating == "pass":
        r.reliable = True
    else:
        r.reliable = False

# Secondary functions
def calculate_distribution(list):
    pos = 0
    neg = 0
    list_pos = []
    list_neg = []
    for i in list:
        if i.sentiment == 'positive':
            pos += 1
            list_pos.append(i)
        else:
            neg += 1
            list_neg.append(i)
    return pos, neg, list_pos, list_neg

# Define sets
print("LOADING TRAIN SET...")
train_set = data_load(train_path)
print("LOADING TEST SET...")
test_set = data_load(test_path)

# Open classification model
classify_model = pickle.load(open('bayes_classifier', 'rb'))

# Define new arrays with only text descriptions
train_set_text = []
test_set_text = []

for i in train_set:
    train_set_text.append(i.description)
for i in test_set:
    test_set_text.append(i.description)

# Define word features
def word_feats(words):
    data = words.split()
    return dict([(word.lower(), True) for word in data])

# Define feature sets
train_featuresets = [(word_feats(text)) for text in train_set_text]
test_featuresets = [(word_feats(text)) for text in test_set_text]

# Create new sanitized sets
for i in range(0, len(train_featuresets)):
    classified = classify_model.classify(train_featuresets[i])
    if classified == 0:
        train_set[i].objective = True    
for i in range(0, len(test_featuresets)):
    classified = classify_model.classify(test_featuresets[i])
    if classified == 0:
        test_set[i].objective = True 

train_sanitized_movies = []
train_sub_movies = []
test_sanitized_movies = []
test_sub_movies = []

for movie in train_set:
    if movie.objective:
        train_sanitized_movies.append(movie)
    else:
        train_sub_movies.append(movie)
for movie in test_set:
    if movie.objective:
        test_sanitized_movies.append(movie)
    else:
        test_sub_movies.append(movie)

# Calculate review distribution across sets [TRAIN]
pos_san, neg_san, san_pos_list, san_neg_list = calculate_distribution(train_sanitized_movies)
pos_sub, neg_sub, sub_pos_list, sub_neg_list = calculate_distribution(train_sub_movies)

# Create equally sized datasets
sizes = [pos_san, neg_san, pos_sub, neg_sub]
half_size = min(sizes)
full_size = half_size*2

sanitized_movies = sample(san_pos_list, half_size) + sample(san_neg_list, half_size)
subjective_movies = sample(sub_pos_list, half_size) + sample(sub_neg_list, half_size)
original_movies = sample(train_set[:12500], half_size) + sample(train_set[12500:], half_size)

# Format datasets
train_text = [i.description for i in train_set]
train_pol = [i.sentiment for i in train_set]
org_text = [i.description for i in original_movies]
org_pol = [i.sentiment for i in original_movies]
san_text = [i.description for i in sanitized_movies]
san_pol = [i.sentiment for i in sanitized_movies]
sub_text = [i.description for i in subjective_movies]
sub_pol = [i.sentiment for i in subjective_movies]

# Define dataframes from new datasets
train_df = pd.DataFrame(np.column_stack([train_text, train_pol]), columns=['TEXT', 'SENTIMENT'])
org_df = pd.DataFrame(np.column_stack([org_text, org_pol]), columns=['TEXT', 'SENTIMENT'])
san_df = pd.DataFrame(np.column_stack([san_text, san_pol]), columns=['TEXT', 'SENTIMENT'])
sub_df = pd.DataFrame(np.column_stack([sub_text, sub_pol]), columns=['TEXT', 'SENTIMENT'])

# Train models
train_model, token_tr, t_max = analyse_sentiment(train_df, "Original")
org_model, org_tr, o_max = analyse_sentiment(org_df, "Un-santized sample [of original]")
san_model, token_san, sa_max = analyse_sentiment(san_df, "Sanitized")
sub_model, token_sub, su_max = analyse_sentiment(sub_df, "Subjective-only")

# calculate review distribution across sets [TEST]
T_pos_san, T_neg_san, T_san_pos_list, T_san_neg_list = calculate_distribution(test_sanitized_movies)
T_pos_sub, T_neg_sub, T_sub_pos_list, T_sub_neg_list = calculate_distribution(test_sub_movies)

# main program - using trained model on test set data
print("analyses: start")

i=0
for r in test_set:
    r.sentiment_polarity = calculate_polarity(r.description, train_model, token_tr, t_max)
    r.sentiment_rating = predict_rating(r.sentiment_polarity)
    calculate_disparity(r)
    responses[0].disparity_accumulator += r.rating_disparity
    test_reliability(r)
    if r.reliable == True:
        responses[0].reliability_accumulator +=1
    if i<2:
        print("COMPLETE ORIGINAL TRAINED MODEL\n", str(r.__dict__), "\n")
        i += 1

responses[0].total_disparity = responses[0].disparity_accumulator / len(test_set)
responses[0].total_accuracy = 100 - responses[0].total_disparity
responses[0].total_reliability = (responses[0].reliability_accumulator / len(test_set)) * 100

for r in test_set:
    r.sentiment_polarity = calculate_polarity(r.description, org_model, org_tr, o_max)
    r.sentiment_rating = predict_rating(r.sentiment_polarity)
    calculate_disparity(r)
    responses[1].disparity_accumulator += r.rating_disparity
    test_reliability(r)
    if r.reliable == True:
        responses[1].reliability_accumulator +=1
    if i<4:
        print("SAMPLE of ORIGINAL TRAINED MODEL\n", str(r.__dict__), "\n")
        i += 1

responses[1].total_disparity = responses[1].disparity_accumulator / len(test_set)
responses[1].total_accuracy = 100 - responses[1].total_disparity
responses[1].total_reliability = (responses[1].reliability_accumulator / len(test_set)) * 100

for r in test_set:
    r.sentiment_polarity = calculate_polarity(r.description, san_model, token_san, sa_max)
    r.sentiment_rating = predict_rating(r.sentiment_polarity)
    calculate_disparity(r)
    responses[2].disparity_accumulator += r.rating_disparity
    test_reliability(r)
    if r.reliable == True:
        responses[2].reliability_accumulator +=1
    if i<6:
        print("OBJECTIVE TRAINED MODEL\n", str(r.__dict__), "\n")
        i += 1

responses[2].total_disparity = responses[2].disparity_accumulator / len(test_set)
responses[2].total_accuracy = 100 - responses[2].total_disparity
responses[2].total_reliability = (responses[2].reliability_accumulator / len(test_set) * 100)

for r in test_set:
    r.sentiment_polarity = calculate_polarity(r.description, sub_model, token_sub, su_max)
    r.sentiment_rating = predict_rating(r.sentiment_polarity)
    calculate_disparity(r)
    responses[3].disparity_accumulator += r.rating_disparity
    test_reliability(r)
    if r.reliable == True:
        responses[3].reliability_accumulator +=1
    if i<8:
        print("SUBJECTIVE TRAINED MODEL\n", str(r.__dict__), "\n")
        i += 1

responses[3].total_disparity = responses[3].disparity_accumulator / len(test_set)
responses[3].total_accuracy = 100 - responses[3].total_disparity
responses[3].total_reliability = (responses[3].reliability_accumulator / len(test_set) * 100)

print("analyses: completed.\n")

#output
print("IN TRAIN\nThere are", len(train_sanitized_movies), "sanitized movies,", pos_san, "positive reviews and", neg_san, "negative reviews.")
print("There are", len(train_sub_movies), "subjective movies", pos_sub, "positive reviews and", neg_sub, "negative reviews.")
print("There were", len(train_set), "original movies, 12,500 positive reviews and 12,500 negative reviews.")
print("\nThere are 4 training sets total. The complete original [25,000 reviews] and 3 training sets of equal size and distribution: original, sanitized and subjective. Each set has been randomly sampled to contain:", full_size, "reviews, 1/2 positive, 1/2 negative.")

print("\nIN TEST\nThere are", len(test_sanitized_movies), "sanitized movies,", T_pos_san, "positive reviews and", T_neg_san, "negative reviews.")
print("There are", len(test_sub_movies), "subjective movies", T_pos_sub, "positive reviews and", T_neg_sub, "negative reviews.")
print("There were", len(test_set), "original movies 12,500 positive reviews and 12,500 negative reviews.")
print("\nAll testing will be performed on the complete test set [25,000 reviews] for fair comparison.")

for res in responses:
    print("\nTotal reviews tested: ", len(test_set))
    print(res.name, "- model training loss-score: %.3f" %res.score, "%")
    print(res.name, "- model training accuracy: %.3f" %res.accuracy, "%")
    print(res.name, "- model testing rating disparity [mean avg.]: %.3f" %res.total_disparity, "%")
    print(res.name, "- model testing rating accuracy [mean avg.]: %.3f" %res.total_accuracy, "%")
    print(res.name, "- model testing reliability: %.3f" %res.total_reliability, "%")
