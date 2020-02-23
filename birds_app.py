"""
Birds App
==========================================================

Streamlit app for the front end of Birds of a Feather. Spawns a map interface to visualize matches.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

import base64

import folium
import geopandas as gpd
import geopy
import numpy as np
import streamlit as st
import torch
from PIL import Image
from shapely.geometry import Point
from sklearn.metrics.pairwise import cosine_similarity

from utils.encoder.encode_user import UserEncoder, MinMaxScaler
from utils.encoder.match_nn import MatchNN


# map wrapper to user folio with streamlit
class MapWrapper:
    def __init__(self, m):
        self.html = m.get_root().render()

    def add_html(self, html_string):
        self.html += html_string

    def _repr_html_(self):
        # Copied from folium.element.Figure
        html = "data:text/html;charset=utf-8;base64," + base64.b64encode(self.html.encode('utf8')).decode('utf8')
        iframe = (
            '<div style="width:100%;">'
            '<div style="position:relative;width:100%;height:0;padding-bottom:60%;">'
            '<iframe src="{html}" style="position:absolute;width:100%;height:100%;left:0;top:0;'
            'border:none !important;" '
            'allowfullscreen webkitallowfullscreen mozallowfullscreen>'
            '</iframe>'
            '</div></div>').format
        return iframe(html=html)


# helper function to preprocess dataframes
def drop_unused(dataframe, unused=['geometry', 'latitude', 'longitude', 'sample_che', 'user_name', 'state',
                                   'profile', 'county']):
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


# display home image
@st.cache
def display_home():
    home_image = Image.open('homePage.jpg')
    return home_image


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

# load data
users, preprocessing = load_data()

# get encodings and encoder
encodings = load_encodings(users, preprocessing)

# get user information
user_name = st.text_input("Please input your eBird profile name to get matches")
if user_name not in list(users.user_name):
    if len(user_name) > 0:
        st.markdown('User not found, please try again.')
    st.image(display_home(), use_column_width=True)

elif user_name != '':
    user = users.loc[users.user_name == user_name]
    if len(user) > 1:
        county = st.selectbox('Select county', options=list(user.county))
        user = user.loc[user.county == county]
    lat = user['latitude'].values[0]
    lon = user['longitude'].values[0]

    # remove user from dataframe
    filtered_users = users.loc[users.user_name != user_name]
    filtered_enc = encodings[users.user_name != user_name]

    # filters
    st.sidebar.markdown('## <span style="color:green"> **Filters:**  </span>',
                        unsafe_allow_html=True)

    # public profiles
    public = st.sidebar.checkbox('only show matches with public profiles', value=True)
    if public:
        has_profile = -filtered_users.profile.isna()
        filtered_enc = filtered_enc[has_profile]
        filtered_users = filtered_users.loc[has_profile]

    # by location
    locator = geopy.Nominatim(user_agent="myGeocoder")
    input_loc = st.sidebar.text_input(label="target location")
    if input_loc:
        location = locator.geocode(input_loc)
        try:
            lat, lon = location.latitude, location.longitude
        except:
            st.sidebar.text('invalid location, check for typos!')

    # user loc
    user_loc = Point(lon, lat)
    max_distance = st.sidebar.number_input(label='maximum distance (mi)', min_value=1, max_value=10000, value=5)

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

    # start encoder
    model = MatchNN()
    model.load_state_dict(torch.load('Saved_models/MatchNN_100_500_1_0.pth'))
    model.eval()
    encoder = UserEncoder(preprocessing, model)

    # select number of matches
    if len(filtered_users) > 0:
        # n matches
        n_matches = st.sidebar.number_input("Top N matches", min_value=1, max_value=len(filtered_users),
                                            value=min(len(filtered_users),
                                                      3))

        # get match for user
        user_num = drop_unused(user)
        user_num = user_num.astype(np.float32).values
        encoded_user = encoder.encode_user(user_num.reshape([1, -1]))
        matches = find_k_nearest(encoded_user, filtered_enc, n_matches)

        # prepare users for plotting map
        plot_users = filtered_users[['latitude', 'longitude']]
        match_users = plot_users.iloc[matches]
        match_users.loc[:, 'pos'] = [(500 + (1 / (ele + 1) * 400)) * (1 + max_distance / 20) for ele in
                                     range(len(match_users))]

        # start folium map
        matches_map = folium.Map(location=[lat, lon], width=660, height=410,
                                 tiles='Stamen Terrain')

        matches_map.fit_bounds([[lat - max_distance / 100, lon - max_distance / 100],
                                [lat + max_distance / 100, lon + max_distance / 100]])

        # add marker for user
        folium.Marker(tuple([lat, lon]), color='green').add_to(matches_map)

        # add distance radius
        folium.Circle(tuple([lat, lon]), color='black', radius=max_distance * 1600,
                      dash_array='20, 20', weight=1.8).add_to(matches_map)

        # add markers for matches
        n = 0
        for idx, match in match_users.iterrows():
            n += 1
            folium.Circle(tuple([match.latitude, match.longitude]), color='black',
                          weight=1.3, fill_color='red', tooltip=f"{filtered_users.loc[idx]['user_name']}, "
                                                                f"eBirder since: "
                                                                f" {int(2019 - filtered_users.loc[idx]['since'])}, "
                                                                f"Species seen: "
                                                                f"{int(filtered_users.loc[idx]['n_species'])}",
                          popup=filtered_users.loc[idx]['profile'],
                          fill=True, fill_opacity=0.4, radius=match.pos).add_to(matches_map)
            folium.Marker(tuple([match.latitude, match.longitude]),
                          icon=folium.DivIcon(
                              html=f"""<div style="font-family: courier new; color: black;">{n}</div>""")).add_to(
                matches_map)

        # plot map
        matches_map = MapWrapper(matches_map)
        st.write(matches_map._repr_html_(), unsafe_allow_html=True)

        # display top results
        st.markdown('## Best matches :trophy:')
        for idx, match in enumerate(matches):
            if filtered_users.iloc[match].profile:
                st.markdown(
                    f'{idx + 1}. {filtered_users.iloc[match].user_name} -> {filtered_users.iloc[match].profile}')
            else:
                st.markdown(
                    f'{idx + 1}. {filtered_users.iloc[match].user_name} -> {filtered_users.iloc[match].sample_che}')


    else:
        st.markdown(
            f'>No matches found with current filters, please increase maximum distance or '
            f'change target location.')
    st.text(' ')

st.sidebar.markdown('## <span style="color:green"> **How it works:**  </span>',
                    unsafe_allow_html=True)
st.sidebar.markdown('>Birds of a Feather is a web app to recommend potential birding partners from a list of '
                    'over 100.000 active eBird users in the US. It finds good partners by matching the users birding '
                    'preferences with encodings for other eBird users processed by a siamese neural network trained to '
                    'distinguish suitable matches from unsuitable ones. With only a few clicks, birders can be pointed '
                    'to ideal partners which they might otherwise never meet.')

st.sidebar.markdown('#### Source code: https://github.com/bentocg/Insight')
