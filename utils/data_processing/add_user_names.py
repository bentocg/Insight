from argparse import ArgumentParser
from collections import Counter

import geopandas as gpd
import numpy as np
from ebird.api import Client


def parse_args():
    parser = ArgumentParser('Adds users ID column to users shapefile')
    parser.add_argument('--users_shp', '-u', type=str, default='Datasets/Shapefiles/users_NA_2005-2019.shp',
                        help='path to users shapefile')
    parser.add_argument('--counties_shp', '-c', type=str, default='Datasets/Shapefiles/counties.shp',
                        help='path to counties shapefile')
    return parser.parse_args()


def main():
    # read arguments
    args = parse_args()
    users_shp = args.users_shp
    counties_shp = args.counties_shp

    # read user data
    users = gpd.read_file(users_shp)

    # load ebird client
    client = Client('6qmfvb8pg9dk', 'en_US')

    # find eBird display name for every user
    display_names = []
    for checklist in users.sample_che:
        try:
            display_name = client.get_checklist(checklist.split('/')[-1])['userDisplayName']
        except:
            display_name = 'None'
        display_names.append(display_name)

    # get state/county shapefile
    counties = gpd.read_file(counties_shp)

    # find state/county for every user
    state = []
    for idx, row in users.iterrows():
        point = row.geometry
        closest = np.argmin(point.distance(polygon) for polygon in counties.geometry)
        state_code = counties.iloc[closest]['HASC_2'].split('.')[1]
        display_names[idx] += f" / {state_code}, {counties.iloc[closest]['NAME_2']}"
        state.append(state_code)

    # add user display name and state
    users['user_name'] = display_names
    users['state'] = state

    # write update shapefile
    users.to_file(users_shp)


if __name__ == "__main__":
    main()
