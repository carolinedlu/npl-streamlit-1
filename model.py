from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import numpy as np


df=pd.read_csv("base_reviews.csv")

#SENTIMENT ANALYSIS

xtrain=df.loc[:35000,"Review"].values
ytrain=df.loc[:35000,"Sentiment"].values

xtest=df.loc[35000:,"Review"].values
ytest=df.loc[35000:,"Sentiment"].values



tfidf=TfidfVectorizer()
#transform text into numbers 
lr_tfidf=Pipeline([("vect",tfidf),
                   ("clf",LogisticRegression(solver="liblinear",C=10,penalty="l2",random_state=42))])
#we do the LR model

lr_tfidf.fit(xtrain,ytrain)

#we create the pickle
model=open("model_lr_tfidf.pickle","wb")
pickle.dump(lr_tfidf,model)
model.close()

