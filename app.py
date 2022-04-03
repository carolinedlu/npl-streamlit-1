import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re,string,unicodedata
from string import punctuation
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")
porter=PorterStemmer()
stop=set(stopwords.words("english"))
punct=list(string.punctuation)
stop.update(punct)

st.title("REVIEWS SENTIMENT ANALYSIS")
st.write("This model uses a supervised model to predict if a movie review that someone writes is positive or negative")

model=open("model_lr_tfidf.pickle","rb")
lr_tfidf=pickle.load(model)
model.close()

def clean_html(text):
  soup=BeautifulSoup(text,"html.parser") 
  return soup.get_text()

def clean_url(text):
  return re.sub(r"http\S+","",text)

def clean_stopwords(text):
  final_text=[]  
  for i in text.split():
    if i.strip().lower() not in stop and i.strip().lower().isalpha():
      final_text.append(i.strip().lower())
  return " ".join(final_text)

def stemmer(text):
  final_text=[porter.stem(word)for word in text.split()]
  return " ".join(final_text)

def clean_text(text):
  text=clean_html(text)
  text=clean_url(text)
  text=clean_stopwords(text)
  text=stemmer(text)
  return text

def prueba(review):
  review=clean_text(review)
  review=np.array([review])
  sent=lr_tfidf.predict(review)[0]
  if sent==0:
    st.write("Negative")
  else:
    st.write("Positive")

review=st.text_input("Write here your review:")

prueba(review)

