import argparse
import json
import numpy as np
from compute_scores import pearson_score
from collaborative_filtering import find_similar_users


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser


def get_recommendations(dataset, input_user):
    """Get movie recommendations for the input user"""
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')
    overal_scores = {}
    similarity_scores = {}

    # Compute a similarity score between the input user and all the other users in the dataset
    for user in [x for x in dataset if x != input_user]:
        similarity_core = pearson_score(dataset, input_user, user)
        if similarity_core <= 0:
            continue
        # Extract a list of movies that have been rated by the current user
        # but haven't been rated by the input user
        filtered_list = [x for x in dataset[user] if x not in dataset[input_user]
                         or dataset[input_user][x] == 0]

        # For each item in the filtered list, keep a track of the weighted rating based on the similarity
        # score. Also keep a track of the similarity scores
        for item in filtered_list:
            overal_scores.update({item: dataset[user][item] * similarity_core})
            similarity_scores.update(({item: similarity_core}))

    if len(overal_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalization
    movie_scores = np.array([[score/similarity_scores[item], item]
                             for item, score in overal_scores.items()])
    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    rating_file = 'knn/ratings.json'
    with open(rating_file) as f:
        data = json.loads(f.read())

    print("\nMovie recommendations for " + user + ":")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)
