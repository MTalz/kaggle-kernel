# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import json
data  = []
limit = 100000
with open("/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json", 'r') as file:
    for index, line in enumerate(file): 
        data.append(json.loads(line))
        if index == limit:
            break

# Load into Pandas dataframe
            
arxiv = pd.DataFrame([])
for entry in data:
        arxiv = arxiv.append(pd.DataFrame([entry]))
arxiv = arxiv.reset_index(drop=True)
arxiv[['title','categories']] = arxiv[['title','categories']].astype('str')

# Filter categories

arxiv['categories'] = arxiv["categories"].str.replace(r'[^\w\s]', '')
cat_type = ['hepph','hepex','hepth','grqc','astroph','quantph']
arxiv = arxiv[arxiv['categories'].isin(cat_type)]

# Remove punctuation

arxiv['title'] = arxiv["title"].str.replace(r'[^\w\s]', '')

# Remove Stopwords

from nltk.corpus import stopwords
stop = stopwords.words('english')
arxiv['title']= arxiv['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Stemming

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
arxiv['title']= arxiv['title'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Model Training

X_train,X_test,y_train,y_test = train_test_split(arxiv.title,arxiv.categories,test_size=0.25,random_state=42)

from sklearn.naive_bayes import MultinomialNB
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)

sgd = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf',SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
              ])
sgd.fit(X_train,y_train)
y_pred = sgd.predict(X_test)
print(accuracy_score(y_test, y_pred)*100)