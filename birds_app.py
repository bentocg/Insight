"""
Birds App
==========================================================

Streamlit app for the front end of Birds of a Feather. Spawns a map interface to visualize matches.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

import geopandas as gpd
import numpy as np
import streamlit as st
import torch
from shapely.geometry import Point
from sklearn.metrics.pairwise import cosine_similarity

from utils.encoder.encode_user import UserEncoder, MinMaxScaler
from utils.encoder.match_nn import MatchNN


# helper function to preprocess dataframes
def drop_unused(dataframe, unused=['geometry', 'latitude', 'longitude', 'sample_che', 'user_name', 'state',
                                   'profile']):
    dataframe = dataframe.drop(unused, axis=1)
    dataframe = dataframe.fillna(0)
    return dataframe


@st.cache(persist=True)
def load_data(path='Datasets/Shapefiles/users_US_2005-2019.shp'):
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
    model.load_state_dict(torch.load('Saved_models/MatchNN_100_500_1_0.pth'))
    model.eval()

    # get encodings
    encoder = UserEncoder(preprocessing, model)

    # drop unused columns and fill missing values
    data = drop_unused(data)
    data = data.astype(np.float32).values

    # generate encodings of existing users
    encodings = np.zeros([data.shape[0], 500])
    for i in range(data.shape[0]):
        encodings[i, :] = encoder.encode_user(data[i, :].reshape([1, -1]))

    # return encodings and encoder
    return encodings


# helper function to get k-nearest neighbors in cosine distance space
def find_k_nearest(user, encodings, k=10):
    similarity = []

    # get cosine similarity between user and other users
    for idx in range(len(encodings)):
        try:
            similarity.append([idx, cosine_similarity(user, encodings[idx].reshape([1, -1]))])
        except ValueError:
            print(f'Could not get similarity between user and row {idx}')
            similarity.append([idx, 0])

    # sort and extract k top similarity values
    matches = [ele[0] for ele in sorted(similarity, key=lambda x: x[1], reverse=True)[:k]]
    return matches


st.markdown('# <span style="color:green"> **Birds of a Feather** </span>:duck:',
            unsafe_allow_html=True)

'$$\hspace{0.5cm}$$ *Bringing birders together* '
'Please input your eBird user name and preferences for matches on the sideboard'
'Press **generate matches** to get a map of potential birding partners'

st.sidebar.markdown('## <span style="color:green"> **User inputs:**  </span>',
                    unsafe_allow_html=True)

# load data
users, preprocessing = load_data()

# get encodings and encoder
encodings = load_encodings(users, preprocessing)

# get user information
states = sorted(set(users.state))
user_state = st.sidebar.selectbox("User state", states, 34)
input_users = users.loc[users.state == user_state]
user_name = st.sidebar.selectbox("User name", list(input_users['user_name']))
user = input_users.loc[input_users.user_name == user_name]
lat = user['latitude']
lon = user['longitude']

# remove user from dataframe
filtered_users = users.loc[users.user_name != user_name]
filtered_enc = encodings[users.user_name != user_name]

# filters
st.sidebar.markdown('## <span style="color:green"> **Filters:**  </span>',
                    unsafe_allow_html=True)

# by state
target_state = st.sidebar.selectbox("Target state", states, 34)
state_users = filtered_users.state == target_state
filtered_users = filtered_users.loc[state_users]
filtered_enc = filtered_enc[state_users]

# by distance
if target_state == user_state:
    # user loc
    user_loc = Point(lon, lat)
    max_distance = st.sidebar.number_input(label='maximun distance (mi)', min_value=1, max_value=10000, value=5)

    # filter results by distance
    user_buffer = user_loc.buffer(max_distance / 69)
    within = np.array(filtered_users.within(user_buffer))
    filtered_users = filtered_users.loc[within]
    filtered_enc = filtered_enc[within]

# filter results by period
active_since = st.sidebar.slider(label='active since (year)', min_value=2005, max_value=2020, value=2019)
older = np.array(filtered_users.since >= (2019 - active_since))
filtered_users = filtered_users.loc[older]
filtered_enc = filtered_enc[older]

# get number of matches
n_matches = st.sidebar.number_input("Top N matches", min_value=1, max_value=len(filtered_users),
                                    value=min(len(filtered_users),
                                              3))

generate = st.button('generate matches')

if generate:

    # start encoder
    model = MatchNN()
    model.load_state_dict(torch.load('Saved_models/MatchNN_100_500_1_0.pth'))
    model.eval()
    encoder = UserEncoder(preprocessing, model)

    # select number of matches
    if len(filtered_users) > 0:

        # get match for user
        user_num = user.drop(['geometry', 'latitude', 'longitude', 'sample_che', 'user_name', 'state', 'profile'],
                             axis=1)
        user_num = user_num.astype(np.float32).values
        encoded_user = encoder.encode_user(user_num.reshape([1, -1]))
        matches = find_k_nearest(encoded_user, filtered_enc, n_matches)

        # plot map
        plot_users = filtered_users[['latitude', 'longitude']]
        match_users = plot_users.iloc[matches]
        lat = match_users.iloc[0]['latitude']
        lon = match_users.iloc[0]['longitude']
        match_users.loc[:, 'pos'] = [300 + (1 / (ele + 1) * 350) for ele in range(len(match_users))]
        st.deck_gl_chart(
            viewport={
                'latitude': lat,
                'longitude': lon,
                'zoom': 11,
                'pitch': 50,
            },
            layers=[
                {'type': 'ScatterplotLayer',
                 'data': plot_users,
                 'getFillColor': [90, 0, 0]
                 },
                {'type': 'ScatterplotLayer',
                 'data': match_users,
                 'getRadius': "pos",
                 'getFillColor': [0, 100, 0],

                 },
            ])

        # display top results
        st.markdown('## Best matches :trophy:')
        for idx, match in enumerate(matches):
            st.markdown(f'{idx + 1}. {filtered_users.iloc[match].user_name} -> {filtered_users.iloc[match].profile}')

    else:
        st.text(
            f'No available matches for {user.user_name} in {target_state} with current filters')
    st.text(' ')

st.sidebar.markdown('## <span style="color:green"> **How it works:**  </span>',
                    unsafe_allow_html=True)
st.sidebar.markdown('>Birds of a Feather is a web app to recommend potential birding partners from a list of '
                    'over 100.000 active eBird users in the US. It finds good partners by matching the users birding '
                    'preferences with encodings for other eBird users processed by a siamese neural network trained to '
                    'distinguish suitable matches from unsuitable ones. With only a few clicks, birders can be pointed '
                    'to ideal partners which they might otherwise never meet.')

st.sidebar.markdown('#### Source code: https://github.com/bentocg/Insight')
