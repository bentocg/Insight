import geopandas as gpd
import numpy as np
import streamlit as st
import torch
from shapely.geometry import Point
from sklearn.metrics.pairwise import cosine_similarity

from utils.encoder.encode_user import UserEncoder, MinMaxScaler
from utils.encoder.match_nn import MatchNN

'# Birds of a Feather'
'$$\hspace{0.5cm}$$ *Bringing birders together*'

st.subheader('Preferences:')
st.sidebar.text('User inputs:')

st.sidebar.text('Filters:')
max_distance = st.sidebar.number_input(label='maximun distance (mi)', min_value=1, max_value=10000, value=5)
active_since = st.sidebar.slider(label='active since (year)', min_value=2005, max_value=2020, value=2019)


# helper function to preprocess dataframes
def drop_unused(dataframe, unused=['geometry', 'latitude', 'longitude', 'sample_che', 'user_name', 'state']):
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


# helper function to get k-nearest neighbors in cosine distance space
def find_k_nearest(user, encodings, k=10):
    similarity = []
    for idx in range(len(encodings)):
        similarity.append([idx, cosine_similarity(user, encodings[idx].reshape([1, -1]))])

    return [ele[0] for ele in sorted(similarity, key=lambda x: x[1])[:k]]


# load data
users, preprocessing = load_data()

# get encodings and encoder
encodings = load_encodings(users, preprocessing)

# filter by period
older = users.since >= (2019 - active_since)
filtered_users = users.loc[older]
filtered_enc = encodings[[older]]

# get user information
user_exists = st.radio("Existing eBird user? (with a minimum of 3 checklists)", ['Yes', 'No'], 0)
if user_exists:
    user_name = st.sidebar.selectbox("User name", filtered_users.user_name, 0)
    user = users.loc[users.user_name == user_name]

# type of location
loc_type = st.radio("Choose location", ['User location', 'Other location'], 0)
if loc_type == "Other location":
    lat = st.sidebar.number_input(label='latitude', value=40.7128)
    lon = st.sidebar.number_input(label="longitude", value=-74.0060)
else:
    lat = user['latitude']
    lon = user['longitude']

# user loc
user_loc = Point(lon, lat)

# filter results by distance
user_buffer = user_loc.buffer(max_distance / 69)
within = filtered_users.within(user_buffer)
filtered_users = filtered_users.loc[tuple(within)]
filtered_enc = filtered_enc[tuple([within])]

# filter by state
filter_by_state = st.sidebar.radio("Filter by state?", ['Yes', 'No'], 1)
if filter_by_state == "Yes":
    state = st.sidebar.selectbox("State", sorted(set(filtered_users.state)), 0)
    filtered_users = filtered_users.loc[filtered_users.state == state]

# start encoder
model = MatchNN()
model.load_state_dict(torch.load('Saved_models/MatchNN_200_200_2_0.pth'))
model.eval()
encoder = UserEncoder(preprocessing, model)

# select number of matches
n_matches = st.sidebar.number_input("Top N matches", min_value=1, max_value=len(filtered_users), value=10)

# get match for user
user = user.drop(['sample_che', 'geometry', 'latitude', 'longitude'])
user = user.fillna(0)
user = user.astype(np.float32).values
encoded_user = encoder.encode_user(user.reshape([1, -1]))
matches = find_k_nearest(encoded_user, encodings, n_matches)

st.text(' ')
st.map(filtered_users)
