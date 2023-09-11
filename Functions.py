 # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import re
import joblib
from config import stop_words,df,df_clean


products_gem=[[text for text in x.split()] for x in df_clean.description_ws]
# Obtain the number of features based on dictionary: Use corpora.Dictionary
dictionary = corpora.Dictionary(products_gem)
# List of features in dictionary
dictionary.token2id
# Numbers of features (word) in dictionary
feature_cnt = len(dictionary.token2id)
# Obtain corpus based on dictionary (dense matrix)
corpus = [dictionary.doc2bow(text) for text in products_gem]
print(corpus[0]) # id, so lan xuat hien cua token trong van ban/ san pham
# Use TF-IDF Model to process corpus, obtaining index
tfidf = models.TfidfModel(corpus)
# tính toán sự tương tự trong ma trận thưa thớt
index = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                            num_features = feature_cnt)

# 3. Modeling - Gensim
# tìm kiếm theo ID

def recommender_id(input_id, dictionary, tfidf_model, index):
      # Convert search words into sparse vector
    selected_id = df_clean.loc[df_clean['item_id'] == input_id, 'description_ws']
    combined_text = ' '.join(selected_id)
    view_product = combined_text.split()
    kw_vector = kw_vector = dictionary.doc2bow(view_product)
    print("view product's vector",kw_vector)

  # Similarity calculation
    sim = index[tfidf[kw_vector]]

  # print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])

    df_result=pd.DataFrame({'id': list_id,'score': list_score})

  # Find five highest scores
    five_highest_score = df_result.sort_values(by='score',ascending=False).head(11)
    print('Five highest scores: ')
    print(five_highest_score)
  # Ids to list
    idToList=list(five_highest_score['id'])
    print('idToList',idToList)

    products_find=df_clean[df_clean.index.isin(idToList)]
    results=products_find[['item_id','name','price','description','image']]
    print('Recommender: ')
    results=pd.concat([results,five_highest_score],axis=1).sort_values(by='score',ascending=False)
    results=results.tail(10)
    return results



#TH2: Khi khách hàng nhập chữ trên thanh công cụ tìm kiếm
def text_clean_2(text):
    # Convert text to lowercase
    cleaned_text = text.lower()
    # Remove special characters, punctuation, and symbols
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    #tokenzie
    cleaned_text = word_tokenize(cleaned_text)
    # remove stop_words
    def remove_stopwords(cleaned_text, stop_words):
        cleaned_text_split = cleaned_text.split()
        cleaned_text = [word for word in cleaned_text_split if word not in stop_words]
        return ' '.join(cleaned_text)

    return cleaned_text

def recommender_text(input_text, dictionary, tfidf_model, index, stop_words):
    # Preprocess the input text
    #processed_input = input_text.split()

    # Clean the input text
    cleaned_text = text_clean_2_model(input_text)

    # Convert the processed input into a sparse vector
    kw_vector = dictionary.doc2bow(cleaned_text)

    # Similarity calculation
    sim = index[tfidf_model[kw_vector]]

    # Create a DataFrame with scores
    df_result = pd.DataFrame({'id': range(len(sim)), 'score': sim})
    # Find five highest scores
    five_highest_score = df_result.sort_values(by='score', ascending=False).head(11)

    # Get the corresponding product IDs
    idToList = list(five_highest_score['id'])

    # Filter products based on IDs
    products_find = df_clean[df_clean.index.isin(idToList)]

    # Create the results DataFrame
    results = products_find[['item_id', 'name','description','price','image']].copy()
    results['score'] = five_highest_score['score'].values
    results = results.sort_values(by='score', ascending=False)
    results=results.head(10)
    return results

#SAVE FUNCTIONs
#Save the function to a file
joblib.dump(recommender_id, 'recommender_id.joblib')
joblib.dump(recommender_text, 'recommender_text.joblib')
joblib.dump(text_clean_2, 'text_clean_2.joblib') # dùng để làm sạch text
#Load the function from the file

recommender_id_model = joblib.load('recommender_id.joblib')
recommender_text_model = joblib.load('recommender_text.joblib')
text_clean_2_model=joblib.load('text_clean_2.joblib')

input_id=48102821
result_id=recommender_id_model(input_id, dictionary, tfidf, index)

input_id_2=916784
result_id_2=recommender_id_model(input_id_2, dictionary, tfidf, index)

input_id_3=2860621
result_id_3=recommender_id_model(input_id_3, dictionary, tfidf, index)



input_text = "Tai nghe Bluetooth"
result_text = recommender_text_model(input_text, dictionary, tfidf, index, stop_words)

input_text_2 = "LOA"
result_text_2 = recommender_text_model(input_text_2, dictionary, tfidf, index, stop_words)

input_text_3 = "PIN SẠC"
result_text_3 = recommender_text_model(input_text_3, dictionary, tfidf, index, stop_words)