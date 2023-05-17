import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import streamlit as st
# Load the recipe dataset
recipes_df = pd.read_csv('preprocessed.csv')

# Select relevant features for content-based filtering
features = ['name', 'process', 'ingredient']
recipes_df = recipes_df[features]

# Clean the data (if necessary)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Apply the TF-IDF vectorizer on the process and ingredient columns
recipe_matrix = vectorizer.fit_transform(recipes_df['process'] + ' ' + recipes_df['ingredient'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(recipe_matrix, recipe_matrix)

# Function to get recipe recommendations based on content similarity
def recommend_recipes_content(recipe_name, top_n=5):
    try:
        # Get the index of the recipe with the given name
        recipe_index = recipes_df[recipes_df['name'] == recipe_name].index[0]
    except IndexError:
        print("Recipe not found in the dataset.")
        return None

    # Get the similarity scores of all recipes with the given recipe
    similarity_scores = list(enumerate(cosine_sim[recipe_index]))

    # Sort the recipes based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n similar recipes (excluding the recipe itself)
    top_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    # Return the top-n similar recipes
    return recipes_df.iloc[top_indices]


# Collaborative filtering

# Load the dataset
df = pd.read_csv('preprocessed.csv')

# Feature selection
features = ['name', 'rating', 'process']
df = df[features]

# Create binary user-item matrix
user_item_matrix = pd.get_dummies(df, columns=['rating'], prefix='', prefix_sep='')
user_item_matrix = user_item_matrix.groupby('name').max()

# Identify and exclude columns with object data type
object_columns = user_item_matrix.select_dtypes(include='object').columns
user_item_matrix = user_item_matrix.drop(object_columns, axis=1)

# Convert to sparse matrix
sparse_user_item_matrix = csr_matrix(user_item_matrix.values)

# Matrix factorization
SVD = TruncatedSVD(n_components=9)
matrix = SVD.fit_transform(sparse_user_item_matrix)

# Similarity matrix
corr = cosine_similarity(matrix)

# Function to get recipe recommendations based on collaborative filtering
def recommend_recipes_collaborative(recipe_name, top_n=5):
    try:
        # Get the index of the recipe with the given name
        recipe_index = user_item_matrix.index.get_loc(recipe_name)
    except KeyError:
        print("Recipe not found in the dataset.")
        return None

    # Get the similarity scores of all recipes with the given recipe
    similarity_scores = list(enumerate(corr[recipe_index]))

    # Sort the recipes based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n similar recipes (excluding the recipe itself)
    top_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    # Return the top-n similar recipes
    return user_item_matrix.iloc[top_indices]

# Knowledge based
# Function to filter recipes based on dietary restrictions
def filter_recipes_by_dietary_restrictions(dietary_restrictions, recipes_df):
    filtered_recipes = recipes_df.copy()

    for restriction in dietary_restrictions:
        filtered_recipes = filtered_recipes[~filtered_recipes['ingredient'].str.contains(restriction, case=False)]

    return filtered_recipes

# Function to recommend recipes based on a combination of content, collaborative, and knowledge-based filtering
def recommend_recipes(recipe_name, dietary_restrictions=None, top_n=5):
    # Content-based filtering
    recommended_content = recommend_recipes_content(recipe_name, top_n=top_n)

    # Collaborative filtering
    recommended_collaborative = recommend_recipes_collaborative(recipe_name, top_n=top_n)

    # Knowledge-based filtering
    filtered_recipes = recipes_df.copy()
    if dietary_restrictions:
        filtered_recipes = filter_recipes_by_dietary_restrictions(dietary_restrictions, filtered_recipes)

    # Combine recommendations from different approaches
    recommended_recipes = pd.concat([recommended_content, recommended_collaborative, filtered_recipes], ignore_index=True)
    recommended_recipes = recommended_recipes.drop_duplicates(subset=['name']).head(top_n)

    return recommended_recipes
def main():
    st.title("Recipe Recommender")

    # Get user input
    recipe_name = st.text_input("Enter the recipe name:")
    dietary_restrictions = []
    if st.checkbox("Gluten-Free"):
        dietary_restrictions.append("gluten")
    if st.checkbox("Dairy-Free"):
        dietary_restrictions.append("dairy")

    # Recommend recipes based on user input
    recommended_recipes = recommend_recipes(recipe_name, dietary_restrictions)

    # Display recommended recipes
    if recommended_recipes is None:
        st.warning("Recipe not found in the dataset.")
    else:
        st.subheader("Recommended Recipes:")
        st.write(recommended_recipes)

if __name__ == "__main__":
    main()
