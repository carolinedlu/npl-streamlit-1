import tarfile 
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

with tarfile.open ("./aclImdb_v1.tar.gz","r") as tar:
  tar.extractall()


df=pd.DataFrame()

basepath="aclImdb"
labels={"pos":1,"neg":0}

for s in ("test","train"):
  for l in ("pos","neg"):
    path=os.path.join(basepath,s,l)
    for file in sorted(os.listdir(path)):
      with open(os.path.join(path,file),"r",encoding="utf-8") as infile:
        txt=infile.read()
      df=df.append([[txt,labels[l]]],ignore_index=True)

df.columns=["Review","Sentiment"]


np.random.seed(0)

df=df.reindex(np.random.permutation(df.index))

df=df.reset_index(drop=True)


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

  #Stemmer return the root of the word. example houses->hous

def clean_text(text):
  text=clean_html(text)
  text=clean_url(text)
  text=clean_stopwords(text)
  text=stemmer(text)
  return text

df["Review"]=df["Review"].apply(clean_text)

df.to_csv("base_reviews.csv",index=False)