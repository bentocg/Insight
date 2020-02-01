import geopandas as gpd
import streamlit as st
import pydeck as pdk
from shapely.geometry import Point

'# Birds of a Feather'
' *bringing birders together*'

st.sidebar.text('User inputs:')
lat = st.sidebar.number_input(label='latitude', value=40.7128)
lon = st.sidebar.number_input(label="longitude", value=-74.0060)

st.sidebar.text('Filters:')
max_distance = st.sidebar.number_input(label='maximun distance (mi)', min_value=1, max_value=10000, value=5)
active_since = st.sidebar.slider(label='active since (year)', min_value=2005, max_value=2020, value=2019)


@st.cache(persist=True)
def load_encoded(path='Datasets/Shapefiles/users_data.shp'):
    data = gpd.read_file(path)
    return data


# read encoded users
encoded_users = load_encoded()

# add user location
user_loc = Point(lon, lat)

# filter by period
filtered_users = encoded_users.loc[encoded_users.since <= active_since]

# filter results by distance
user_buffer = user_loc.buffer(max_distance / 69)
filtered_users = filtered_users.loc[filtered_users.within(user_buffer)]

print(filtered_users)

st.map(filtered_users)
