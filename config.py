 # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json



# with open('result2_final.json', 'r') as json_file:
#     # Load the JSON data into a Python data structure
#     data = json.load(json_file)

# data.head(5)




STOP_WORD_FILE = "vietnamese-stopwords.txt"
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')

# Content based

df=pd.read_parquet('df.parquet')
df_clean=pd.read_parquet('product_ws.parquet')
df_sub=df_clean


# Collab 

df_review = pd.read_parquet('df_review.parquet')
df_collab = pd.read_csv('df_collab.csv',header=0)

# save file as parquet


# # Load data
# df_review = pd.read_csv('ReviewRaw.csv', header=0)
# df_review_sub = pd.read_parquet('df_sub.parquet')
# df_review_result = pd.read_csv('result_df.csv', header=0)
# df_review_result=df_review_result[["customer_id","product_id","rating"]]



# # Rename columns
# column_name_mapping = {
#     "item_id": "product_id",
#     "name": "product_name",
# }
# df_product = df.rename(columns=column_name_mapping)

# # Convert data types
# df_review['product_id'] = pd.to_numeric(df_review['product_id'], errors='coerce')

# # Merge dataframes
# #df_review_full = df_review.merge(df_product, on="product_id", how="left")
# df_collab=df_review_result.merge(df_product, on="product_id", how="left")

# print(df_collab.head(5))

# df_collab.to_csv('df_collab.csv', index=False)
