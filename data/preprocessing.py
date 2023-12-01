import pandas as pd
import torch

def preprocessing(movie_path, rating_path): 
  '''
    Parameters:
         movie_path (str): A string representing the file path to the movies dataset.
         rating_path (str): A string representing the file path to the ratings dataset.
    
    Returns:
         edge_index (torch.Tensor): the indices of edges in the adjacency matrix for the ratings dataset.
         num_users (int): number of unique users in the ratings dataset.
         num_movies (int): number of unique movies in the ratings dataset.
         user_mapping (pd.DataFrame): the list that map user id to continguous new ids
         movie_df (pd.DataFrame): the movie dataset
         rating_df (pd.DataFrame): the rating dataset
  '''
  # load movies and ratings dataset
  movie_df = pd.read_csv(movie_path, index_col = 'movieId')
  rating_df = pd.read_csv(rating_path, index_col = 'userId') 

  # create mapping to continous range
  movie_mapping = {idx: i for i, idx in enumerate(movie_df.index.unique())}
  user_mapping = {idx: i for i, idx in enumerate(rating_df.index.unique())}
  num_users, num_movies = len(rating_df.index.unique()), len(movie_df.index.unique())

  rating_df = pd.read_csv(rating_path)
  edge_index = None
  users = [user_mapping[idx] for idx in rating_df['userId']]
  movies = [movie_mapping[idx] for idx in rating_df['movieId']]

  # filter for edges with a high rating
  ratings = rating_df['rating'].values
  recommend_bool = torch.from_numpy(ratings).view(-1, 1).to(torch.long) >= 4

  edge_index = [[],[]]
  for i in range(recommend_bool.shape[0]):
    if recommend_bool[i]:
      edge_index[0].append(users[i])
      edge_index[1].append(movies[i])
    
  edge_index = torch.tensor(edge_index)
  return edge_index, num_users, num_movies, movie_mapping, user_mapping, movie_df, rating_df