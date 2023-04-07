import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn

model=pickle.load(open('model.pkl','rb'))
cv=pickle.load(open('cv.pkl','rb'))

st.header('Duplicate Questions Pairs')

q1=st.text_input('Enter Question 1')
q2=st.text_input('Enter Question 2')

def common_words(q1,q2):
  w1=set(map(lambda x:x.lower().strip(),q1.split()))
  w2=set(map(lambda x:x.lower().strip(),q2.split()))
  return len(w1&w2)

def total_words(q1,q2):
  w1=set(map(lambda x:x.lower().strip(),q1.split()))
  w2=set(map(lambda x:x.lower().strip(),q2.split()))
  return (len(w1)+len(w2))

from fuzzywuzzy import fuzz

def fetch_fuzzy_features(q1,q2):

  fuzzy_features=[0.0]*4
  fuzzy_features[0]=fuzz.QRatio(q1,q2)
  fuzzy_features[1]=fuzz.partial_ratio(q1,q2)
  fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)
  fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)
  return fuzzy_features

def query_point_creator(q1,q2):
  input_query=[]
  #preprocess
  input_query.append(len(q1))
  input_query.append(len(q2))

  input_query.append(len(q1.split()))
  input_query.append(len(q2.split()))

  input_query.append(common_words(q1,q2))
  input_query.append(total_words(q1,q2))
  input_query.append(round(common_words(q1,q2)/(total_words(q1,q2)),2))

  fuzzy_features=fetch_fuzzy_features(q1,q2)
  input_query.extend(fuzzy_features)

  q1_bow=cv.transform([q1]).toarray()
  q2_bow=cv.transform([q2]).toarray()

  return np.hstack((q1_bow,q2_bow,np.array(input_query).reshape(1,11)))

if st.button('Find'):
    result=query_point_creator(q1,q1)[0]
    if result.any():
        st.header('Duplicate')
    else:
       st.header('Not Duplicate')



