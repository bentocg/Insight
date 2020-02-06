import geopandas as gpd
import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from shapely.geometry import Point
from utils.encoder.encode_user import UserEncoder, MinMaxScaler
from utils.encoder.match_nn import MatchNN
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch

'# Birds of a Feather'
' *bringing birders together*'

st.sidebar.text('User inputs:')
lat = st.sidebar.number_input(label='latitude', value=40.7128)
lon = st.sidebar.number_input(label="longitude", value=-74.0060)

st.sidebar.text('Filters:')
max_distance = st.sidebar.number_input(label='maximun distance (mi)', min_value=1, max_value=10000, value=5)
active_since = st.sidebar.slider(label='active since (year)', min_value=2005, max_value=2020, value=2019)


# helper function to preprocess dataframes
def drop_unused(dataframe, unused=['geometry', 'latitude', 'longitude', 'sample_che']):
    dataframe = dataframe.drop(unused, axis=1)
    dataframe = dataframe.fillna(0)
    return dataframe


@st.cache(persist=True)
def load_data(path='Datasets/Shapefiles/users_NA_2005-2019.shp'):
    """

    :param path:
    :return:
    """
    # read full dataset
    data = gpd.read_file(path)

    # read training users
    with open('utils/users_train.txt', 'r') as src:
        check_train = src.readline().strip().split(' ')

    # get preprocessing scaling from training users
    data_train = data.loc[data.sample_che.isin(check_train)]
    data_train = drop_unused(data_train)

    # start minmax scaler
    preprocessing = MinMaxScaler()
    preprocessing.fit(data_train)

    # return user data and preprocessing function
    return data, preprocessing


@st.cache(persist=True)
def load_encodings(data, preprocessing):
    # read model
    model = MatchNN()
    model.load_state_dict(torch.load('Saved_models/MatchNN_200_200_2_0.pth'))
    model.eval()

    # get encodings
    encoder = UserEncoder(preprocessing, model)

    # drop unused columns and fill missing values
    data = drop_unused(data)
    data = data.astype(np.float32).values

    # generate encodings of existing users
    encodings = np.zeros([data.shape[0], 200])
    for i in range(data.shape[0]):
        encodings[i, :] = encoder.encode_user(data[i, :].reshape([1, -1]))

    # return encodings and encoder
    return encodings


# helper function to find matches
def find_match(user, encodings):
    best_similarity = -50000
    best_idx = 0
    for i in range(len(encodings)):
        curr = cosine_similarity(user, encodings[i].reshape([1, -1]))
        if curr > best_similarity:
            best_similarity = curr
            best_idx = best_idx
    return best_idx


# load data
users, preprocessing = load_data()

# get encodings and encoder
encodings = load_encodings(users, preprocessing)

# add user location
user_loc = Point(lon, lat)

# filter by period
older = users.since >= (2019 - active_since)
filtered_users = users.loc[older]
filtered_enc = encodings[[older]]

# filter results by distance
user_buffer = user_loc.buffer(max_distance / 69)
within = filtered_users.within(user_buffer)
filtered_users = filtered_users.loc[within]
filtered_enc = filtered_enc[[within]]

# get best match
model = MatchNN()
model.load_state_dict(torch.load('Saved_models/MatchNN_200_200_2_0.pth'))
model.eval()
encoder = UserEncoder(preprocessing, model)
user = users.iloc[1500]
user = user.drop(['sample_che', 'geometry', 'latitude', 'longitude'])
user = user.fillna(0)
user = user.astype(np.float32).values
encoded_user = encoder.encode_user(user.reshape([1, -1]))
best_idx = find_match(encoded_user, filtered_enc)

st.map(filtered_users)
