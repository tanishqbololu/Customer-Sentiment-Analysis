# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:37:36 2024

@author: TANISHQ
"""

# Importing Libraries
import pandas as pd
import numpy as np


#Importing Dataset

dataset=pd.read_csv(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\Customer Review NLP and ML\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)


#Cleaning the text

import re 
#Regular expressions provide a powerful way to manipulate and search through text data

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #re.sub(pattern, replacement, string)
    
    review = review.lower()
    
    review = review.split()
    
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    
    corpus.append(review)
    
    
#Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer()

X = tv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Classification
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train,y_train)


#Preicting the test result
y_pred = classifier.predict(X_test)


#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


from sklearn.metrics import accuracy_score

ac = accuracy_score(y_test, y_pred)

print(ac)


bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance

#===============================================
'''
CASE STUDY --> model is underfitted  & we got less accuracy 

1> Implementation of tfidf vectorization , lets check bias, variance, ac, auc, roc 
2> Impletemation of all classification algorihtm (logistic, knn, randomforest, decission tree, svm, xgboost,lgbm,nb) with bow & tfidf 
4> You can also reduce or increase test sample 
5> xgboost & lgbm as well
6> you can also try the model with stopword 


6> then please add more recores to train the data more records 
7> ac ,bias, varian - need to equal scale ( no overfit & not underfitt)

'''


import pickle 

filename1 = 'vectoriser.pkl'

with open(filename1,'wb') as file:pickle.dump(tv,file)

filename2 = 'classifier.pkl'

with open(filename2,'wb') as file:pickle.dump(classifier,file)


