import streamlit as st
import pickle

# Load pickled data
with open("recommendation_system_data.pickle", "rb") as f:
    pickle_data = pickle.load(f)

tfidf = pickle_data["tfidf"]
cosine_sim = pickle_data["cosine_sim"]
cosine_sim2 = pickle_data["cosine_sim2"]
indices = pickle_data["indices"]
movies_df = pickle_data["movies_df"]

C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)

# Function to get content-based recommendations
def get_content_based_recommendations(title, cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first entry (itself), and select top 10 similar movies
    movie_indices = [ind[0] for ind in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Function to get demographic recommendations
def get_demographic_recommendations(new_movies_df, C, m):
    new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
    new_movies_df = new_movies_df.sort_values('score', ascending=False)
    recommendations = new_movies_df["title"].head(10)
    return recommendations



# Function to get collaborative recommendations
def get_collaborative_recommendations():
    return "Under construction..."

# Main Streamlit app
st.title('Movie Recommendation System')

option = st.sidebar.selectbox('Select Recommendation Type', ['Content-Based', 'Demographic', 'Collaborative'])

if option == 'Content-Based':
    st.subheader('Content-Based Filtering Recommendations')
    selected_movie = st.selectbox('Select a movie:', movies_df['title'].values)
    if st.button('Get Recommendations'):
        recommendations = get_content_based_recommendations(selected_movie, cosine_sim)
        st.write(recommendations)

elif option == 'Demographic':
    st.subheader('Demographic Filtering Recommendations')
    new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
    recommendations = get_demographic_recommendations(new_movies_df, C, m)
    st.write(recommendations)

else:
    st.subheader('Collaborative Filtering Recommendations')
    st.write(get_collaborative_recommendations())
    

