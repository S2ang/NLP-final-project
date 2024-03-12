# import modules
import pandas as pd
import numpy as np
import pickle
import nltk.classify.util

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier

# read file(s)
def file_read(path):
    with open(path, 'r', encoding = "ISO-8859-1") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
    return data

# classify objects
sub = file_read("rotten_imdb/quote.tok.gt9.5000")
obj = file_read("rotten_imdb/plot.tok.gt9.5000")
sub_0 = ["subjective" for i in sub]
obj_1 = ["objective" for i in obj]

# create dataset
df1 = pd.DataFrame(np.column_stack([sub, sub_0]), columns=['text', 'subjectivity'])
df2 = pd.DataFrame(np.column_stack([obj, obj_1]), columns=['text', 'subjectivity'])

# concatanate objective/subjective data
frames = [df1, df2]
df = pd.concat(frames)

# create labels for dataframe
le = LabelEncoder()
df["label"] = le.fit_transform(df["subjectivity"])

# define dataframe objects
df = df[["text", "label"]]

# define dataframe sample
df = df.sample(frac = 1)

# define word feature method
def word_feats(words):
    data = words.split()
    return dict([(word.lower(), True) for word in data])

# define feature sets
featuresets = [(word_feats(text), label) for index, (text, label) in df.iterrows()]

# define size of feature sets
num = int(len(featuresets)*(0.75))

# define test and train sets
train_x = featuresets[:num]
test_x = featuresets[num:]

# train classification model
classifier = nltk.NaiveBayesClassifier.train(train_x)

# display accuracy of tested model
accuracy = nltk.classify.util.accuracy(classifier, test_x)
print("\nAccuracy:", accuracy, "\n-------------------------")

classifier.show_most_informative_features()

# open classification model
with open('bayes_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
