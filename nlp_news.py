import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import jieba
import jieba.posseg as pseg
from jieba import analyse


df_train = pd.read_csv("train_data.csv")

jieba.analyse.set_stop_words('stopwords.txt')

for i in range(len(df_train)):
     df_train.title[i]= str(jieba.analyse.extract_tags(df_train.title[i]))


X_train, X_val, y_train, y_val = train_test_split(
    df_train['title'], 
    df_train['label'], 
    random_state = 44
)


pipe = Pipeline([('vect', CountVectorizer(encoding ='UTF-8')),
                 ('tfidf', TfidfTransformer()),
                 ('model', SGDClassifier())])

model = pipe.fit(X_train, y_train)
prediction = model.predict(X_val)
print("accuracy: {}%".format(round(accuracy_score(y_val, prediction)*100,2)))



df_test = pd.read_csv("test_data.csv")
df_test.head()

df_test.head()

for i in range(len(df_test)):
     df_test.title[i]= str( jieba.analyse.extract_tags(df_test.title[i]))

predictions = model.predict(df_test['title'])

import csv

with open('sample.csv', 'w', newline='') as csvFile:
  
  writer = csv.writer(csvFile)

  writer.writerow(['id','label'])
  
  for i in range(len(predictions)):
      writer.writerow([i,predictions[i]])
  


