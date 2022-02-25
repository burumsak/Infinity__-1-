#loading necessary libraries
import streamlit as st
import pandas as pd
import pickle
from io import StringIO
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import re
import nltk
from bs4 import BeautifulSoup
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import html.parser
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

st.set_page_config(
    page_title="Infinity",
    page_icon="♾",
    layout="wide"
)

okc = pd.read_excel("Copy of User Details_Faf.xlsx")

#st.title("Infinity")
st.image("image.png",width=250)

st.markdown('##')  

with st.expander("Show data"):
    st.write(okc)
st.write("") 
st.markdown('##')
  
'''
*Answer these questions to find new people to interact with!*
'''   
st.markdown('#')

okc1 = pd.DataFrame()

#form for recommendations

with st.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age",18,79)
        status = st.selectbox("Relationship Status", ["Single","In a relationship","Unknown"])
        gender = st.selectbox("Gender", ["Female","Male"])
        orientation = st.selectbox("Sexual Orientation", ["Straight","Bisexual","Gay"])
        body_type = st.selectbox("How would you describe your body type?", ["Fit","Average","Curvy","Thin","Overweight","Rather not say"])
        education = st.selectbox("What level of education have you completed?", ["college/university","Masters and above","other","Two-year college","High school","Med / Law school"])
        ethnicity= st.selectbox("Which ethnicity do you belong to?", ["White","Asian","Hispanic","African American","Mixed","Unknown","others"])
        religion = st.selectbox("Which religion do you practice?", ["Agnosticism","Atheism","Christianity","Catholicism","Judaism","Buddhism","Islam","Hinduism","Unknown","others"])
        essay3 = st.text_input("The first thing people usually notice about me")
        essay4 = st.text_input("Favourite books, Movies, Show,Music, Food")
        essay5 = st.text_input("The 6 things that I could never do without")
        essay6 = st.text_input("I spend a lot of time thinking about")
        essay7 = st.text_input("On a typical Friday night I am")
        essay8 = st.text_input("The most private thing I am willing to admit")
    with col2:
        smokes = st.selectbox("Do you smoke?", ["Yes","No"])
        drink = st.selectbox("Do you drink alcohol?", ["Yes","No"])
        diet = st.selectbox("What kind of diet do you follow?", ["Anything","Vegan","Vegetarian","Halal","Kosher","other"])
        speaks = st.text_input("Which language do you speak other than English?")
        sign = st.selectbox("What is your astrological sign?", ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpion","Sagittarius","Capricorn","Aquarius","Pisces"])
        offspring = st.selectbox("What is your opinion on having children?", ["Wants Kids","Does not want kids","Has kid","Unknown"])
        drugs = st.selectbox("Do you consume drugs?", ["Yes","No"])
        height = st.number_input("What is your height? \n (In inches)",30,100)
        income = st.number_input("What is your annual income?",0,100000)
        pets = st.selectbox("What is your opinion on having pets?", ["Likes Cats and Dogs","Dislikes Cats and Dogs","Likes only cats","Likes only Dogs","Unknown"])
        job = st.selectbox("What is your current profession?", ["Office/Professional","Science/Tech","Business Management","Creative"])
        essay0 = st.text_input("My self summary")
        essay1 = st.text_input("What I am doing with my life")
        essay2 = st.text_input("I am really good at")
        essay9 = st.text_input("You should message me if")
    submit_button= st.form_submit_button(label='Submit')


data = {"Name": name, "age": age, 
     "gender": gender, "orientation":orientation,
     "status":status,"education":education, "ethnicity":ethnicity, 
                 "religion" : religion,"smokes":smokes, "drink":drink, "body_type":body_type,"diet":diet,"job":job,
                 "speaks":speaks,"sign":sign,"offspring":offspring,"drugs":drugs,"height":height,
                 "income":income,"pets":pets,"essay0":essay0,"essay1":essay1,"essay2":essay2,
                 "essay3":essay3,"essay4":essay4,"essay5":essay5,"essay6":essay6,"essay7":essay7,
                 "essay8":essay8,"essay9":essay9}

data = pd.DataFrame(data,index=[0])
data.columns = data.columns.str.lower()
okc.columns = okc.columns.str.lower()
features = pd.DataFrame(data) 
         

okc1 = pd.concat([features,okc],axis=0,join="inner",ignore_index=True) 


ok= okc1.copy(deep=True)


#Data cleaning and pre-processing
ok['essay']=ok[['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']].apply(lambda x: ' '.join(x), axis=1)
ok.drop(['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9'],axis=1,inplace=True)

corpus_df = ok.copy(deep=True)

corpus_df['corpus'] = ok[['age', 'status', 'gender', 'orientation', 'body_type', 'diet', 'drink',
       'drugs', 'education', 'ethnicity', 'height', 'income', 'job',
       'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks', 'essay']].astype(str).agg(' '.join, axis=1)
corpus_df = corpus_df.astype(str)

corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('\n', ' '))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('nan', ' '))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("\'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("-'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("--'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("='", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("/", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(".", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(":", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(",", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("(", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(")", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("?", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("!", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(";", " "))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('"', " "))
corpus_df['corpus'] = corpus_df['corpus'].str.replace('\d+', '')
corpus_list = corpus_df['corpus']


# vectorization
stemmer = SnowballStemmer("english")

tfidf = TfidfVectorizer(stop_words = "english", ngram_range = (1,3), max_df=0.8, min_df=0.2) 
corpus_tfidf = tfidf.fit(corpus_list)
corpus_2d = pd.DataFrame(tfidf.transform(corpus_list).todense(),
                   columns = tfidf.get_feature_names(),)
tfidf_vec = tfidf.fit_transform(corpus_list)

corpus_2d.head()
corpus_mat_sparse = csr_matrix(corpus_2d.values)


#Model specification
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute') # cosine takes 1-cosine( as cosine distance)
model_knn.fit(corpus_mat_sparse)

#recommendation algorithm (cosine)
@st.cache
def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
  result= pd.DataFrame()
  for i in indices:
    result= result.append(okc1.iloc[i,:])
  result['similarity distance']= distances.flatten()
  return result[['name',"similarity distance","age","status","orientation","ethnicity","religion","diet","essay0"]] 

st.markdown('##')
st.markdown('##')   

if submit_button:
    st.write("Your Matches are:")
    st.write(rec(0))
