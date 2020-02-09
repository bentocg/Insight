"""
Add user names
==========================================================
Script to add user names to users shapefile. Uses ebird-api to get display names from user checklists.
Display names are combined with the county where the user birded most within the column 'user_name'. User name
and state are appended as columns to the original shapefile.

Author: Bento Gonçalves
License: MIT
Copyright: 2020-2021
"""

from argparse import ArgumentParser

import geopandas as gpd
import numpy as np
from ebird.api import Client


def parse_args():
    parser = ArgumentParser('Adds users ID column to users shapefile')
    parser.add_argument('--users_shp', '-u', type=str, default='Datasets/Shapefiles/users_US_2005-2019.shp',
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

    # get state/county shapefile
    counties = gpd.read_file(counties_shp)
    counties = counties.dropna(subset=['HASC_2'])

    # get only users in US

    # load ebird client
    client = Client('6qmfvb8pg9dk', 'en_US')

    # find eBird display name for every user
    display_names = []
    for checklist in users.sample_che:
        try:
            display_name = client.get_checklist(checklist.split('/')[-1])['userDisplayName']
        except ValueError:
            print(f'No valid username for checklist {checklist}')
            display_name = 'None'
        display_names.append(display_name)

    # find state/county for every user
    states = []
    for idx, row in users.iterrows():
        point = row.geometry
        closest = np.argmin([point.distance(polygon) for polygon in counties.geometry])
        row = counties.iloc[closest]
        state_code = row['HASC_2'].split('.')[1]
        display_names[idx] += f" / {counties.iloc[closest]['NAME_2']}"
        states.append(state_code)

    # add user display name and state
    users['user_name'] = display_names
    users['state'] = states

    # write update shapefile
    users.to_file(users_shp)


if __name__ == "__main__":
    main()
