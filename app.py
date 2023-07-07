import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv("netflixData.csv")

data=data[["Title","Description","Content Type","Genres"]]

data= data.dropna()

import nltk
import re
import string

#nltk.download('stopwords')
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))

def clean(text):
  text=str(text).lower()
  text=re.sub('\[.*?\]','',text)
  text=re.sub('https?://\S+|www\.\S+', '', text)
  text=re.sub('<.*?>', '', text)
  text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
  text=re.sub('\n','',text)
  text=re.sub('\w*\d\w*','',text)
  text=[word for word in text.split(' ') if word not in stopword]
  text=" ".join(text)
  text=[stemmer.stem(word) for word in text.split(' ')]
  text=" ".join(text)
  return text
data["Title"]=data["Title"].apply(clean)


feature=data["Genres"].tolist()
tfidf = text.TfidfVectorizer(input=feature,stop_words="english")
tfidf_matrix=tfidf.fit_transform(feature)
similarity= cosine_similarity(tfidf_matrix)


indices= pd.Series(data.index, index=data['Title']).drop_duplicates()


def NetFlix_Recommendation(title,similarity=similarity):
  index=indices[title]
  similarity_scores=list(enumerate(similarity[index]))
  similarity_scores=sorted(similarity_scores,key=lambda x:x[1],reverse=True)
  similarity_scores=similarity_scores[0:10]
  movie_indices=[i[0] for i in similarity_scores]
  return data['Title'].iloc[movie_indices]

import streamlit as st

st.title("Netflix Movies/TV Shows Recommendation")

def rec():
  name=st.text_area("Enter Movie / Tv Show Name: ")
  if(st.button(label='Recommend')):
    if len(name)<1:
      st.write(" ")
    else:
      st.write(NetFlix_Recommendation(name))

rec()
