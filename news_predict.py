import os
import string
import numpy as np
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import xgboost as xgb
from sklearn import metrics
from sklearn.externals import joblib
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from textblob import TextBlob

train = pd.read_csv("train_file.csv")
train = train.drop(['IDLink', 'Source', 'PublishDate', 'Facebook', 'GooglePlus', 'LinkedIn'], axis=1)
train.head()

sia = SIA()
results = []

for line in train['Title']:
    pol_score = sia.polarity_scores(line)
    pol_score['Title'] = line
    pol_score['subjectivity'] = TextBlob(line).subjectivity
    pol_score['polarity'] = TextBlob(line).polarity
    results.append(pol_score)

print(results[:5])

sia = SIA()
results1 = []

for line in train['Headline']:
    pol_score = sia.polarity_scores(line)
    pol_score['Headline'] = line
    pol_score['subjectivity'] = TextBlob(line).subjectivity
    pol_score['polarity'] = TextBlob(line).polarity
    results1.append(pol_score)

print(results1[:5])

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
train["topic"] = lb_make.fit_transform(train["Topic"])
train[["topic", "Topic"]].head(11)

title = pd.DataFrame.from_records(results)
title = title[['Title', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']]
title['SentimentTitle'] = train['SentimentTitle']
title['topic'] = train['topic']
title = pd.DataFrame(title)
title.head()
corr = title.corr()
corr.style.background_gradient(cmap='coolwarm')

headline = pd.DataFrame.from_records(results1)
headline = headline[['Headline', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']]
headline['SentimentHeadline'] = train['SentimentHeadline']
headline['topic'] = train['topic']
headline = pd.DataFrame(headline)
headline.head()
corr = headline.corr()
corr.style.background_gradient(cmap='coolwarm')

X = title.drop(['Title', 'SentimentTitle', 'neg', 'subjectivity'], axis=1)
Y = title['SentimentTitle']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

X1 = headline.drop(['Headline', 'SentimentHeadline', 'neg', 'subjectivity'], axis=1)
Y1 = headline['SentimentHeadline']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 400, random_state = 0)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

regressor1 = xgb.XGBRegressor()
regressor1.fit(X1_train,y1_train)

y1_pred = regressor1.predict(X1_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y1_test, y1_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y1_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y1_pred)))

filename = 'model.sav'
joblib.dump(regressor, filename)

filename1 = 'model1.sav'
joblib.dump(regressor1, filename1)

test = pd.read_csv("test_file.csv")
test = test.drop(['IDLink', 'Source', 'PublishDate', 'Facebook', 'GooglePlus', 'LinkedIn'], axis=1)
test.head()

lb_make = LabelEncoder()
test["topic"] = lb_make.fit_transform(test["Topic"])
test[["topic", "Topic"]].head(11)

sia = SIA()
results = []

for line in test['Title']:
    pol_score = sia.polarity_scores(line)
    pol_score['Title'] = line
    pol_score['subjectivity'] = TextBlob(line).subjectivity
    pol_score['polarity'] = TextBlob(line).polarity
    results.append(pol_score)

print(results[:5])

sia = SIA()
results1 = []

for line in test['Headline']:
    pol_score = sia.polarity_scores(line)
    pol_score['Headline'] = line
    pol_score['subjectivity'] = TextBlob(line).subjectivity
    pol_score['polarity'] = TextBlob(line).polarity
    results1.append(pol_score)

print(results1[:5])

title = pd.DataFrame.from_records(results)
title = title[['Title', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']]
title['topic'] = test['topic']
title = pd.DataFrame(title)
title.head()

headline = pd.DataFrame.from_records(results1)
headline = headline[['Headline', 'neg', 'neu', 'pos', 'compound', 'polarity', 'subjectivity']]
headline['topic'] = test['topic']
headline = pd.DataFrame(headline)
headline.head()

X = title.drop(['Title','neg', 'subjectivity'], axis=1)
loaded_model = joblib.load(filename)
result = loaded_model.predict(X)
print(result)

X1 = headline.drop(['Headline','neg', 'subjectivity'], axis=1)
loaded_model1 = joblib.load(filename1)
result1 = loaded_model1.predict(X1)
print(result1)

test = pd.read_csv("test_file.csv")
final = pd.DataFrame()
final['IDLink'] = test['IDLink']
final['SentimentTitle'] = result
final['SentimentHeadline'] = result1
final.to_csv('output.csv',index=False)

feature_importances = pd.DataFrame(regressor.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)

feature_importances1 = pd.DataFrame(regressor1.feature_importances_,index = X1_train.columns,columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)

print(feature_importances1)
