
import streamlit as st
from config  import stop_words, df,df_clean,df_collab,df_review,df_sub
from Functions import text_clean_2_model,dictionary, tfidf, index,recommender_id_model,recommender_text_model,result_id,result_text,result_id_3,result_id_2,result_text_2,result_text_3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from gensim import corpora, models, similarities
import re
import warnings
import os, sys
import joblib
import random





def display_product_info(product):
    # Customize this function to display the product information as you need
    st.write(f"Name: {product['product_name']}")
    st.write(f"Rating: {product['rating_x']}")
    
    # Create an expander for the description
    with st.expander("Read More"):
        st.write(product['description'])



# Create a Streamlit app

def remove_diacritics(input_str):
    # Dictionary mapping characters with diacritics to their corresponding characters without diacritics
    diacritics_map = {
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        # Add more mappings as needed
    }
    
    # Replace characters with diacritics using the mapping
    output_str = ''.join([diacritics_map[char] if char in diacritics_map else char for char in input_str])
    
    return output_str


def get_recommendations(customer_id, selected_option):
    # Replace this with your recommendation logic
    recommendations = df_collab[df_collab['customer_id'] == customer_id].head(selected_option)
    return recommendations

def run_recommender_app_collab():
    st.title("Collaborative Filtering Recommendation")
    
    # Allow the user to enter a customer name or ID
    customer_input = st.text_input("Enter a Customer Name or ID")
    
    # Define a list of available options for the number of products
    num_products_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Allow the user to select the number of products to display
    selected_option = st.selectbox("Select the number of products to display:", num_products_options, index=num_products_options.index(10))
    
    if st.button("Find Recommendations"):
        if customer_input:
            customer_input_clean = remove_diacritics(customer_input)  # Bỏ dấu trong tên khách hàng

            # Check if the input is a numeric customer ID
            if customer_input_clean.isdigit():
                customer_id = int(customer_input_clean)
                if customer_id > 0:
                    st.subheader(f"Top {selected_option} Recommended Products for Customer ID {customer_id}")

                    # Retrieve recommendations using the cached function
                    recommendations = get_recommendations(customer_id, selected_option)

                    if not recommendations.empty:
                        # Create a container for the product images
                        col1, col2, col3 = st.columns(3)

                        # Add some CSS to control the spacing between images
                        st.write(
                            """
                            <style>
                            .stImage {
                                margin-right: 20px; /* Adjust the right margin as needed */
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        for i, (_, product) in enumerate(recommendations.iterrows(), start=1):
                            # Display product information in a column
                            with col1:
                                st.image(product['image'], caption=product['product_name'], use_column_width=True, output_format='JPEG')
                                display_product_info(product)

                            # Start a new row every three products
                            if i % 3 == 0:
                                col1, col2, col3 = st.columns(3)
                else:
                    st.warning("Please enter a valid numeric customer ID.")
            else:
                customer_name = customer_input_clean
                # Filter out rows where the 'name' column is not a string
                df_collab_filtered = df_collab[df_collab['name'].apply(lambda x: isinstance(x, str))]
                # Retrieve the collaborative filtering recommendations for the customer name
                recommendations = df_collab_filtered[df_collab_filtered['name'].str.lower() == customer_name].head(selected_option)

                if not recommendations.empty:
                    # Create a container for the product images
                    col1, col2, col3 = st.columns(3)

                    # Add some CSS to control the spacing between images
                    st.write(
                        """
                        <style>
                        .stImage {
                            margin-right: 20px; /* Adjust the right margin as needed */
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    for i, (_, product) in enumerate(recommendations.iterrows(), start=1):
                        # Display product information in a column
                        with col1:
                            st.image(product['image'], caption=product['product_name'], use_column_width=True, output_format='JPEG')
                            display_product_info(product)

                        # Start a new row every three products
                        if i % 3 == 0:
                            col1, col2, col3 = st.columns(3)
                else:
                    st.info("No recommendations available for the selected customer name.")
        else:
            st.warning("Please enter a customer name or ID.")
    
    # Add a "Find Another Customer" button to reset the page
    if st.button("Find Another Customer"):
        st.experimental_rerun()

    # Suggest 10 random customer IDs
    random_customer_ids = random.sample(df_collab['customer_id'].tolist(), 10)
    st.write("Randomly suggested customer IDs:")
    st.write(random_customer_ids)





def display_product_info_2(product):
    # Customize this function to display the product information as you need
    st.write(f"Name: {product['name']}")
    st.write(f"Price: {product['price']}")
    
    # Create an expander for the description
    with st.expander("Read More"):
        st.write(product['description'])



def run_contend_based_recommender_app(choice):
    if choice == 'Content-Based Recommendation':
        st.write("#### Content-Based Recommendation")
    
        # Allow the user to enter a product ID or name
        product_input = st.text_input("Enter a Product ID or Name")

        # Define a list of available options for the number of products
        num_products_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Allow the user to select the number of products to display
        selected_option = st.selectbox("Select the number of products to display:", num_products_options)

        if st.button("Find Recommendations"):
            if product_input:
                if product_input.isdigit():
                    product_id = int(product_input)  # Convert to integer
                    
                    # Retrieve product information for the searched product
                    info_id_search = df[df['item_id'] == product_id].iloc[0]  # Get the first row as it contains the searched product

                    # Replace this with your actual recommendation logic
                    result_id_search = recommender_id_model(product_id, dictionary, tfidf, index)

                    if not info_id_search.empty:
                        # Display product information for the searched product
                        st.subheader("Your searched product:")
                        st.image(info_id_search['image'], caption=info_id_search['name'], use_column_width=100, output_format='JPEG')
                        display_product_info_2(info_id_search)

                        if not result_id_search.empty:
                            # Display recommended products
                            st.subheader("Recommended Products:")
                            col1, col2, col3 = st.columns(3)

                            # Add some CSS to control the spacing between images
                            st.write(
                                """
                                <style>
                                .stImage {
                                    margin-right: 20px; /* Adjust the right margin as needed */
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )

                            for i, (_, product) in enumerate(result_id_search.iterrows(), start=1):
                                # Display product information in a column
                                with col1:
                                    st.image(product['image'], caption=product['name'], use_column_width=True, output_format='JPEG')
                                    display_product_info_2(product)

                                # Start a new row every three products
                                if i % 3 == 0:
                                    col1, col2, col3 = st.columns(3)
                        else:
                            st.info("No recommendations available for the selected product.")
                    else:
                        st.info("No information available for the selected product ID.")
                else:
                    product_name = product_input
                    # Implement content-based recommendation
                    content_based_results = recommender_text_model(product_name, dictionary, tfidf, index, stop_words)

                    if not content_based_results.empty:
                        st.subheader(f"Top {selected_option} Recommended Products for Product Name: {product_name}")
                        
                        # Create a container for the product images
                        col1, col2, col3 = st.columns(3)
                        
                        # Add some CSS to control the spacing between images
                        st.write(
                            """
                            <style>
                            .stImage {
                                margin-right: 20px; /* Adjust the right margin as needed */
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        for i, (_, product) in enumerate(content_based_results.iterrows(), start=1):
                            # Display product information in a column
                            with col1:
                                st.image(product['image'], caption=product['name'], use_column_width=True, output_format='JPEG')
                                display_product_info_2(product)

                            # Start a new row every three products
                            if i % 3 == 0:
                                col1, col2, col3 = st.columns(3)
                    else:
                        st.info("No content-based recommendations available for the entered product name.")

                # Add a button to enter another product
                if st.button("Enter Another Product"):
                    st.text_input("Enter a Product ID or Name", key='product_input')  # Reset the input field
            else:
                st.warning("Please enter a product ID or Name.")
        else:
            # Suggest 10 random customer IDs
            random_product_ids = random.sample(df['item_id'].tolist(), 10)
            st.write("Randomly suggested Product IDs:")
            st.write(random_product_ids)



# Main Streamlit app
if __name__ == "__main__":
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 2.5rem;">Data Science Project 2</h1>
        <p style="font-size: 1.5rem;">Recommender System</p>
    </div>
    """,
    unsafe_allow_html=True,
    )
   
    
    menu = ["Business Objective", "Content-based filtering overview", "Content-Based Recommendation","Collaborative Filtering overview","Collaborative Filtering Recommendation"]
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Business Objective':
        st.subheader("Business Objective")
        st.write("""
        A content-based recommender system is a sophisticated technology used to enhance user experiences. It suggests items by analyzing user preferences and matching them with item attributes. This personalized approach ensures that users receive recommendations tailored to their unique tastes and interests.
        """)  
        st.write("""
        Problem/Requirement: In this project, data is collected from an e-commerce website, assuming that the website does not have a recommender system. The objective is to build a Recommendation System to suggest and recommend products to users/customers. The goal is to create recommendation models, including Content-based filtering and Collaborative filtering, for one or multiple product categories on A.vn, providing personalized choices for users.
        """)
        st.image("Content_filtering.png")
        st.image("Collaborative_filtering.png")
    elif choice == 'Content-based filtering overview':
        st.write("#### Build Project Content-based filtering")
        st.write("##### 1. Product Overview")
        st.dataframe(df[["item_id", "name","price","rating","description","brand","group"]].head(3))
        st.write("###### Data before cleaning includes the combination of name, description, brand, and group")
        st.dataframe(df[["item_id", "description"]].head(3))
        st.write("###### Data after cleaning and tokenization with 'Underthesea' ")
        st.dataframe(df_clean[["item_id", "description_ws"]].head(3))  
        st.write("##### 2. Some Pictures of  Products")
        image_urls = [
        'https://salt.tikicdn.com/cache/280x280/ts/product/9e/af/79/39855aad21aaa6ed4459909c7c0aea4e.jpg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/0e/03/6e/1e82e11419bd4aae424b10a5457eb932.jpeg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/28/12/73/3f373c0bd557df40f6c8c404622d16a2.jpg',
        'https://salt.tikicdn.com/cache/280x280/ts/product/0a/07/39/b9050cc9f02a8d01cd45466a9e21b9d4.jpg'
        ]
        st.write("<div style='display: flex; flex-wrap: wrap;'>", unsafe_allow_html=True)

        # Display each image side by side with reduced width
        for image_url in image_urls:
            st.write(f"<img src='{image_url}' style='width: 45%; margin-right: 5px;'>", unsafe_allow_html=True)
        # Close the container
        st.write("</div>", unsafe_allow_html=True)

        st.write("##### 3. Build Content-based filtering with gensim")
        st.write("This is an explanation of how our product recommendation system works.")

        # Step 1: Cleaning and Tokenizing Text
        st.write("Step 1: Cleaning and Tokenizing Text with underthesea")
        st.write("In the first crucial step of our recommendation system, we meticulously clean and tokenize the product descriptions using the advanced 'underthesea' natural language processing technology. This intricate process ensures that each product description is transformed into a collection of individual words, ready to undergo further analysis and scrutiny")

        # Step 2: Creating a Dictionary and Corpus
        st.write("Step 2: Creating a Dictionary and Corpus")
        st.write("Following the initial cleaning process, we embark on the journey of creating a comprehensive dictionary and corpus. This step involves meticulously cataloging all the unique words encountered in the product descriptions. Furthermore, we painstakingly count the occurrences of each word in every product description. This meticulous record-keeping enables us to gain valuable insights into the linguistic richness of our product library.")

        # Step 3: TF-IDF Transformation
        st.write("Step 3: TF-IDF Transformation")
        st.write("As we delve deeper into the heart of our recommendation system, we harness the power of TF-IDF (Term Frequency-Inverse Document Frequency) transformation. This transformation is instrumental in identifying the most salient and impactful words within each product description. By assigning weights to each word based on its significance, we can pinpoint the words that define a product's uniqueness and appeal.")

        # Step 4: Calculating Similarities
        st.write("Step 4: Calculating Similarities")
        st.write("In this pivotal phase, we embark on the mathematical journey of calculating similarities between products. Using the well-established cosine similarity metric, we unravel the hidden relationships between products. By measuring the cosine of the angle between product vectors, we gain insights into how similar or dissimilar each product is to others in our extensive library.")

        # Step 5: Finding Recommendations
        st.write("Step 5: Finding Recommendations")
        st.write(" The core of our recommendation system lies in finding the most relevant product recommendations for each item in our collection. To achieve this, we employ a sophisticated algorithm that meticulously analyzes the product similarities we've uncovered. For each product, we painstakingly identify the top 5 most similar products as recommendations, ensuring that our suggestions are highly tailored to your preferences.")

        # Step 6: Displaying Recommendations
        st.write("Step 6: Displaying Recommendations")
        st.write("As a culmination of our extensive efforts, we present you with the cream of the crop—our meticulously curated recommendations. These handpicked products have earned their place as the top recommended items in our library. With a keen eye for quality and relevance, we ensure that your shopping experience is nothing short of exceptional.")

        # Conclusion
        st.write("Our recommendation system acts as your trusted shopping companion, resembling a helpful librarian in the digital realm. By suggesting products based on their descriptions and inherent characteristics, we aim to enhance your shopping journey, making it more enjoyable and tailored to your unique preferences.")
        st.write("##### 4. Show result example")
        result_id_1 = f"The customer clicks on the product with ID 916784, then suggests similar products: {result_id}"
        st.code(result_id_1)
        result_id_2 = f"The customer clicks on the product with ID 48102821, then suggests similar products: {result_id_2}"
        st.code(result_id_2)
        result_id_3 = f"The customer clicks on the product with ID 2860621, then suggests similar products: {result_id_3}"
        st.code(result_id_3)
        result_text_1 = f"When customers search for Bluetooth headphones, recommend products such as: {result_text}"
        st.code(result_text_1)
        result_text_2 = f"When customers search for lOA , recommend products such as: {result_text_2}"
        st.code(result_text_2)
        result_text_3 = f"When customers search for PIN SAC, recommend products such as: {result_text_3}"
        st.code(result_text_3)
    elif choice=='Collaborative Filtering overview':
        st.write("#### Over Collaborative Filtering project")
        st.write("##### 1. Data Overview")
        st.dataframe(df_review.head(3))
        st.write("Selecting necessary attributes for analysis")
        st.dataframe(df_review[["customer_id","product_id","rating"]].head(3))
        st.write("##### 2. Some important steps of project")
        st.write("Step 1: Data Preparation")         
        st.write("In this project, the dataset was collected and cleaned to ensure it contained user interactions with products, such as ratings and purchase histories. Missing values and outliers were handled to ensure data quality.")
        st.write("Step 2: Feature Engineering")
        st.write("To make the data suitable for modeling, categorical user and product identifiers were converted into numerical indices using techniques like StringIndexer. This transformation allowed for the mathematical operations required for collaborative filtering.")
        st.write("Step 3: Train-Test Split")
        st.write("The dataset was divided into a training set and a test set to accurately assess the model's performance. This division enabled the model to be trained on one portion of the data while reserving the other for evaluating how well recommendations were made on unseen interactions.")
        st.write("Step 4: ALS Model")
        st.write("The ALS (Alternating Least Squares) algorithm was chosen for building the collaborative filtering recommendation system. Hyperparameters, such as the rank of latent factors and regularization strength, were tuned to optimize the model's performance.")
        st.write("Step 5: Model Training")
        st.write("With the ALS algorithm and optimal hyperparameters in place, the recommendation model was trained on the training dataset. During training, the ALS model learned latent factors representing users and products, aiming to minimize the error in predicting user-item interactions.")
        st.write("Step 6: Evaluation and Deployment")
        st.write("After training, the model's performance was evaluated using the test dataset. Metrics  RMSE (Root Mean Squared Error)was used to measure how well the model predicted user preferences. Once the model's accuracy was deemed satisfactory, it was deployed to provide personalized recommendations to users, enhancing their experience on the platform.")
        st.write("Following these steps, a collaborative filtering recommendation system with ALS was successfully implemented, providing users with relevant product suggestions based on their interactions and preferences.")
    elif choice == 'Content-Based Recommendation':
        run_contend_based_recommender_app(choice)
        
    elif choice == 'Collaborative Filtering Recommendation':
        run_recommender_app_collab()





