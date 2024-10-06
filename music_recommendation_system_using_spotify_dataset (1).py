#!/usr/bin/env python
# coding: utf-8

# # **Building Music Recommendation System using Spotify Dataset**
# 
# 
# 
# Hello and welcome , I have created Music Recommendation System using Spotify Dataset. To do this, I presented some of the visualization processes to understand data and done some EDA(Exploratory Data Analysis) so we can select features that are relevant to create a Recommendation System.

# # **Importing Libraries**

# In[1]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


# # **Reading Data**

# In[2]:


data=pd.read_csv("/content/data.csv")
artist_data=pd.read_csv("/content/data_by_artist.csv")
genres_data=pd.read_csv("/content/data_by_genres.csv")
year_data=pd.read_csv("/content/data_by_year.csv")
w_genres_data=pd.read_csv("/content/data_w_genres.csv")


# # Analysing all Dataset using df.head() and info()

# In[3]:


data.head()


# In[4]:


print(data.info())


# In[5]:


artist_data.head()


# In[6]:


print(artist_data.info())


# In[7]:


genres_data.head()


# In[8]:


print(genres_data.info())


# In[9]:


year_data.head()


# In[10]:


print(year_data.info())


# In[11]:


w_genres_data.head()


# In[12]:


print(w_genres_data.info())


# # CHECKING NUMBER OF UNIQUE VALUES IN EACH DATASET

# In[13]:


for x in data.columns:
  print(x,':',len(data[x].unique()))
print(data.shape)


# In[14]:


for x in artist_data.columns:
  print(x,':',len(artist_data[x].unique()))
print(artist_data.shape)


# In[15]:


for x in genres_data.columns:
  print(x,':',len(genres_data[x].unique()))
print(genres_data.shape)


# In[16]:


for x in year_data.columns:
  print(x,':',len(year_data[x].unique()))
print(year_data.shape)


# In[17]:


for x in w_genres_data.columns:
  print(x,':',len(w_genres_data[x].unique()))
print(w_genres_data.shape)


# # **Data Understanding by Visualization and EDA**

# In[18]:


top10_popular_artists = artist_data.nlargest(10, 'popularity')
top10_most_song_produced_artists = artist_data.nlargest(10, 'count')
print('Top 10 Artists that produced most songs:')
top10_most_song_produced_artists[['count','artists']].sort_values('count',ascending=False)


# In[19]:


print('Top 10 Artists that had most popularity score:')
top10_popular_artists[['popularity','artists']].sort_values('popularity',ascending=False)


# **Conclusions from EDA**
# 
# * Most of the songs range between 1950s-2010s.
# * Energy in songs have increased over the time.
# * Acousticness in songs have reduced greately over the decades.
# * We can clearly see that loudness has dominantly increased over the decades and is at it's peak in 2020.
# * In top 10 genres we can see that energy and dancebility are most noticable features.

# # **Music Over Time**
# 
# Using the data grouped by year, we can understand how the overall sound of music has changed from 1921 to 2020.

# In[20]:


data['decade'] = data['year'].apply(lambda year : f'{(year//10)*10}s')
data['decade'] = pd.Categorical(data['decade'])
sns.set(rc={'figure.figsize':(8 ,5)})
sns.countplot(data=data, x='decade',color="blue")


# In[21]:


sound_features = ['valence','acousticness','instrumentalness', 'danceability', 'energy',  'liveness','speechiness']
fig = px.line(year_data, x='year', y=sound_features,title=' Various sound features over decades', line_shape='hv')
fig.show()


# # **Characteristics of Different Genres**
# 
# This dataset contains the audio features for different songs along with the audio features for different genres. We can use this information to compare different genres and understand their unique differences in sound.

# In[22]:


top10_genres = genres_data.nlargest(10, 'popularity')

fig = px.bar(top10_genres, x='genres', y=['acousticness','liveness','valence', 'energy', 'danceability' ], barmode='group')
fig.show()


# In[23]:


fig = px.line(year_data, x='year', y='loudness',title='Loudness over decades')
fig.show()


# **Below is a word cloud visualization based on the text data in the 'genres' column of the DataFrame genres_data. Words that appear more frequently in the 'genres' column will be displayed larger in the word cloud.**

# In[24]:


from wordcloud import WordCloud,STOPWORDS

stopwords = set(STOPWORDS)
comment_words = " ".join(genres_data['genres'])+" "
wordcloud = WordCloud(width =
                      800, height = 800,
                background_color ='green',
                stopwords = stopwords,
                max_words=40,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title("Genres Wordcloud")
plt.show()


# **Below is a word cloud visualization for the 'artists' column in the DataFrame artist_data, where the size of each artist's name in the word cloud is determined by its frequency in the dataset. The word cloud provides a visual representation of the most commonly occurring artists in the dataset.**

# In[25]:


comment_words = " ".join(artist_data['artists'])+" "
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='green',
                stopwords = stopwords,
                min_word_length=3,
                max_words=40,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Artists Wordcloud")
plt.tight_layout(pad = 0)
plt.show()


# # **Clustering Genres with K-Means**
# 
# Using the K-means clustering algorithm, we organize genres in the dataset into 12 clusters based on the numerical audio features specific to each genre.

# In[26]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genres_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genres_data['cluster'] = cluster_pipeline.predict(X)


# **Employing t-SNE for Cluster Visualization: t-distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised machine learning algorithm that has gained widespread popularity in bioinformatics and data science. Its primary utility lies in visualizing the structure of high-dimensional data in two or three dimensions. While t-SNE serves as a dimensionality reduction technique, its predominant use is for visualization rather than data preprocessing. Consequently, dimensionality is typically reduced to two when utilizing t-SNE, facilitating the visualization of data in a two-dimensional plot**
# 
# 
# 
# 
# 
# 

# In[27]:


# Visualizing the Clusters with t-SNE
tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genres_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genres_embedding)
projection['genres'] = genres_data['genres']
projection['cluster'] = genres_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'],color_continuous_scale='magenta')
fig.show()


# # **Clustering Songs with K-Means**
# 
# Principal Component Analysis (PCA) is an unsupervised learning algorithm employed for dimensionality reduction in machine learning. A key distinction between PCA and t-SNE lies in their preservation approaches: while t-SNE maintains local similarities, PCA focuses on preserving large pairwise distances to maximize variance. PCA takes a set of points in high-dimensional data and transforms it into low-dimensional data."

# In[28]:


# Create a pipeline with scaler, imputer, and kmeans
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', SimpleImputer(strategy='mean')),  # You can change the strategy as needed
    ('kmeans', KMeans(n_clusters=20, verbose=False))
], verbose=False)

# Select numeric columns from the data
X = data.select_dtypes(np.number)

# Fit the pipeline
song_cluster_pipeline.fit(X)

# Access the labels_ attribute
song_cluster_labels = song_cluster_pipeline.named_steps['kmeans'].labels_

# Assign the cluster labels to the data
data['cluster_label'] = song_cluster_labels


# In[29]:


# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'],color_continuous_scale='bupu')
fig.show()


# # **Build Recommender System**
# 
# * Based on the analysis and visualizations, it’s clear that similar genres tend to have data points that are located close to each other while similar types of songs are also clustered together.
# * This observation makes perfect sense. Similar genres will sound similar and will come from similar time periods while the same can be said for songs within those genres. We can use this idea to build a recommendation system by taking the data points of the songs a user has listened to and recommending songs corresponding to nearby data points.
# * [Spotipy](https://spotipy.readthedocs.io/en/2.16.1/) is a Python client for the Spotify Web API that makes it easy for developers to fetch data and query Spotify’s catalog for songs. You have to install using `pip install spotipy`
# * After installing Spotipy, you will need to create an app on the [Spotify Developer’s page](https://developer.spotify.com/) and save your Client ID and secret key.

# In[30]:


# installing spotify
get_ipython().system('pip install spotipy')


# In[31]:


#importing necessary libraries and creating find_song function
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

get_ipython().run_line_magic('env', 'SPOTIPY_CLIENT_ID=c265a743fb0b4ca0b3eb0f6c5225ed1f')
get_ipython().run_line_magic('env', 'SPOTIPY_CLIENT_SECRET=6d6dfd102ec44c86926dd81548e70b88')

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=os.environ["SPOTIPY_CLIENT_ID"],
        client_secret=os.environ["SPOTIPY_CLIENT_SECRET"]
    )
)


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
       song_data[key] = value

    return pd.DataFrame(song_data)


# # Define find_song Function:
# A function named find_song takes two parameters: name (song name) and year (release year).
# Inside the function:
# * A defaultdict named song_data is created to store song-related information.
# * A Spotify API search is performed using sp.search with the provided song name and year.
# * If no matching tracks are found, the function returns None.
# * If a matching track is found, its details are retrieved, including track ID and audio features.
# * The song information is collected in the song_data dictionary.
# * The function returns a Pandas DataFrame containing the collected information.

# In[32]:


from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):

    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


# # get_song_data Function:
# * Returns information about the song from spotify_data if found.
# * If the song is not found, it calls a function find_song with the song's name and year.
# 
# # get_mean_vector Function:
# * Computes the mean vector of the song features for the given list of songs.
# 
# # flatten_dict_list Function:
# Flattens the list of dictionaries into a defaultdict where keys are column names and values are lists of corresponding values.
# 
# # recommend_songs Function:
# * Recommends songs based on similarity to the input songs.
# * Uses cosine distance to measure similarity.
# * Returns a list of dictionaries containing recommended song metadata.

# # **Song Recommender User Interface**
# 
# This code provides a simple user interface for recommending songs based on user input. It utilizes the `ipywidgets` library for creating interactive widgets within Google colab environment.

# In[33]:


import ipywidgets as widgets
from IPython.display import display, HTML

def recommend_songs_ui(b):
    song_name = entry_name.value
    song_year = entry_year.value

    song_list = [{'name': song_name, 'year': int(song_year)}]

    recommended_songs = recommend_songs(song_list, data)

    result_html = "<b>Recommended Songs:</b><br>"
    for song in recommended_songs:
        artists = ", ".join(eval(song['artists'])) if 'artists' in song else 'Unknown Artists'
        result_html += f"{song['name']} ({song['year']}) by {artists}<br>"

    result_text.value = result_html

# Create widgets
entry_name = widgets.Text(description="Enter the name of the song:")
entry_year = widgets.Text(description="Enter the year of the song:")
button_recommend = widgets.Button(description="Recommend Songs")
result_text = widgets.HTML(value="")

# Set up event handlers
button_recommend.on_click(recommend_songs_ui)

# Display widgets
display(entry_name, entry_year, button_recommend, result_text)


# # Conclusion
# We are able to recommend top 10 similar songs to user based on the input. The recommendation is based on similarity of numerical features of the songs. We have calculated the cosine distance and identified the songs with highest similarity.
# 
# 
# THANK YOU !!!
