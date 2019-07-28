import re
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().magic('matplotlib inline')

train = pd.read_csv("train_file.csv")
train = train.drop(['IDLink', 'Source', 'PublishDate', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn'], axis=1)
train.head()

test = pd.read_csv("test_file.csv")
test = test.drop(['Source', 'PublishDate', 'Topic', 'Facebook', 'GooglePlus', 'LinkedIn'], axis=1)
test.head()

train.shape, test.shape

length_train = train['Title'].str.len()
length_test = test['Title'].str.len()

plt.hist(length_train, bins=20, label="train_title")
plt.hist(length_test, bins=20, label="test_title")
plt.legend()
plt.show()

length_train = train['Headline'].str.len()
length_test = test['Headline'].str.len()

plt.hist(length_train, bins=20, label="train_headline")
plt.hist(length_test, bins=20, label="test_headline")
plt.legend()
plt.show()

stop_words = set(stopwords.words('english'))
result = []
title = pd.DataFrame()
for s in train['Title'].values:
    tokens = word_tokenize(s)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    result.append([i for i in stemmed if not i in stop_words])
title['result'] = result
title = pd.DataFrame(title)
title.head()

stop_words = set(stopwords.words('english'))
result1 = []
headline = pd.DataFrame()
for s in train['Headline'].values:
    tokens = word_tokenize(s)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    result1.append([i for i in stemmed if not i in stop_words])
headline['result'] = result1
headline = pd.DataFrame(headline)
headline.head()

for i in range(len(title['result'])):
    title['result'][i] = TreebankWordDetokenizer().detokenize(title['result'][i])
title['result'] = title
title.head()

for i in range(len(headline['result'])):
    headline['result'][i] = TreebankWordDetokenizer().detokenize(headline['result'][i])
headline['result'] = headline
headline.head()


all_words = ' '.join([text for text in title['result']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

#Bag-of-Words Features
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(title['result'])
bow.shape

#Bag-of-Words Features
bow_vectorizer1 = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow1 = bow_vectorizer1.fit_transform(headline['result'])
bow1.shape

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#Bagofwords features
train_bow = bow[:55932,:]
test_bow = bow[55932:,:]
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['SentimentTitle'],
                                                          random_state=42,
                                                          test_size=0.3)

#Bagofwords features
train_bow1 = bow1[:55932,:]
test_bow1 = bow1[55932:,:]
# splitting data into training and validation set
xtrain_bow1, xvalid_bow1, ytrain1, yvalid1 = train_test_split(train_bow1, train['SentimentHeadline'],
                                                          random_state=42,
                                                          test_size=0.3)

model = xgb.XGBRegressor()
xgb_model = model.fit(xtrain_bow, ytrain)
prediction = xgb_model.predict(xvalid_bow)

xgb_model1 = model.fit(xtrain_bow1, ytrain1)
prediction1 = xgb_model1.predict(xvalid_bow1)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(yvalid, prediction))
print('Mean Squared Error:', metrics.mean_squared_error(yvalid, prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yvalid, prediction)))

print('Mean Absolute Error:', metrics.mean_absolute_error(yvalid1, prediction1))
print('Mean Squared Error:', metrics.mean_squared_error(yvalid1, prediction1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yvalid1, prediction1)))

stop_words = set(stopwords.words('english'))
result = []
title = pd.DataFrame()
for s in test['Title'].values:
    tokens = word_tokenize(s)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    result.append([i for i in stemmed if not i in stop_words])
title['result'] = result
title = pd.DataFrame(title)
title.head()

stop_words = set(stopwords.words('english'))
result = []
headline = pd.DataFrame()
for s in test['Headline'].values:
    tokens = word_tokenize(s)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    result.append([i for i in stemmed if not i in stop_words])
headline['result'] = result
headline = pd.DataFrame(headline)
headline.head()

for i in range(len(title['result'])):
    title['result'][i] = TreebankWordDetokenizer().detokenize(title['result'][i])
title['result'] = title
title.head()

for i in range(len(headline['result'])):
    headline['result'][i] = TreebankWordDetokenizer().detokenize(headline['result'][i])
headline['result'] = headline
headline.head()

#Bag-of-Words Features
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(title['result'])
bow.shape
test_bow = bow[:37288:,:]

#Bag-of-Words Features
bow_vectorizer1 = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow1 = bow_vectorizer1.fit_transform(headline['result'])
bow1.shape
test_bow1 = bow1[:37288:,:]

test_pred = xgb_model.predict(test_bow)
test_pred1 = xgb_model1.predict(test_bow1)
print(test_pred)
final = pd.DataFrame()
final['IDLink'] = test['IDLink']
final['SentimentTitle'] = test_pred
final['SentimentHeadline'] = test_pred1
final.to_csv('bow.csv', index=False)
