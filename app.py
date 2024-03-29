import pickle
with open("recommendation_system_data.pickle", "rb") as f:
    pickle_data = pickle.load(f)

tfidf = pickle_data["tfidf"]
cosine_sim = pickle_data["cosine_sim"]
cosine_sim2 = pickle_data["cosine_sim2"]
indices = pickle_data["indices"]
movies_df = pickle_data["movies_df"]

def get_content_based_recommendations(title, cosine_sim_matrix):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first entry (itself), and select top 10 similar movies
    movie_indices = [ind[0] for ind in sim_scores]
    return movies_df['title'].iloc[movie_indices]

def get_demographic_recommendations(new_movies_df, C, m):
    new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
    new_movies_df = new_movies_df.sort_values('score', ascending=False)
    print("\n-----------Both Popular and Well-rated--------------\n")
    print(new_movies_df["title"].head(10))
    popularity = movies_df.sort_values("popularity", ascending = False)
    print("\n-----------Top 10 Popular Movies-------------------\n")
    print(popularity["title"].head(10))

def get_collaborative_recommendations():
    return "Under construction..."



# Example of using content-based filtering
print("\nContent-Based Filtering Recommendations:")
print("\n----------------Recommendations for The Dark Knight Rises--------------\n")
print(get_content_based_recommendations("The Dark Knight Rises", cosine_sim))
print()

# Example of using demographic filtering
print("Demographic Filtering Recommendations:")
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]

def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)

print(get_demographic_recommendations(new_movies_df, C, m))
print()

# Example of using collaborative filtering
print("Collaborative Filtering Recommendations:")
print(get_collaborative_recommendations())
