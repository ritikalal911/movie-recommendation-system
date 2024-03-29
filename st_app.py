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

# Function to get metadata-based recommendations
def get_metadata_based_recommendations(title, cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first entry (itself), and select top 10 similar movies
    movie_indices = [ind[0] for ind in sim_scores]
    return movies_df['title'].iloc[movie_indices]

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
    st.subheader('Content-Based and Metadata Recommendations')

    selected_movie = st.selectbox('Select a movie:', movies_df['title'].values, key='content_based_selectbox', placeholder="Choose a Movie", index=None)

    if st.button('Get Recommendations'):
        if selected_movie !=None:
            st.write('Title : ', selected_movie)
            st.write(f"Genres: {', '.join(movies_df.loc[movies_df['title'] == selected_movie, 'genres'].values[0])}")

            st.write("Director:", movies_df.loc[movies_df['title'] == selected_movie, 'director'].values[0])
            st.write("Cast:", ' '.join(movies_df.loc[movies_df['title'] == selected_movie, 'cast'].values[0]))
            
            content_based_recommendations = get_content_based_recommendations(selected_movie, cosine_sim)
            metadata_recommendations = get_metadata_based_recommendations(selected_movie, cosine_sim2)

            # Display recommendations side by side
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Recommendations for {selected_movie} based on content:")
                st.write(content_based_recommendations)
            with col2:
                st.write(f"Recommendations for {selected_movie} based on metadata:")
                st.write(metadata_recommendations)
        else:
            st.error("Please Select a Movie")

elif option == 'Demographic':
    st.subheader('Demographic Filtering Recommendations')
    new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
    recommendations = get_demographic_recommendations(new_movies_df, C, m)
    st.write("Top 10 Popular and Well-Rated Movies :")
    st.write(recommendations)

else:
    st.subheader('Collaborative Filtering Recommendations')
    st.write(get_collaborative_recommendations())
