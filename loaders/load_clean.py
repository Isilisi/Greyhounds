# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:14:15 2021

@author: CharlesBray
"""

# Set the current working directory to the project's root
# and make all module imports relative to root
import os
import decouple
config = decouple.AutoConfig(' ')
os.chdir(config('ROOT_DIRECTORY'))
import sys
sys.path.insert(0, '')

# Import libraries
import itertools
import math
import numpy as np
import pandas as pd
import tutorials.fast_track_tutorial.fasttrack as ft
import json

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import tz
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler

'''
Load AU race data from earliest month available to present in batches using 
the FastTrack API wrapper and save to the folder ./data/raw/ and append to dataframes.
If the raw data has been previously loaded then skip and append to dataframes.
'''

# Load environment variables
FASTTRACK_KEY = config('FASTTRACK_KEY')

# Validate FastTrack API connection
client = ft.Fasttrack(FASTTRACK_KEY)
track_codes = client.listTracks()

# Import race data excluding NZ races
au_tracks_filter = list(track_codes[track_codes['state'] != 'NZ']['track_code'])

# create a data directory to store raw and clean data
if not os.path.exists('./data/'):
    os.makedirs('./data/raw/')
    os.makedirs('./data/clean/')

# Time window to import data
# First day of month that we have access to data
date_from = '2018-07-01'

# First day of previous month
date_to = (datetime.today() - relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')

# Dataframes to populate data with
race_details = pd.DataFrame()
dog_results = pd.DataFrame()

# For each month, either fetch data from API or use local CSV file if we already have downloaded it
for start in pd.date_range(date_from, date_to, freq='MS'):
    start_date = start.strftime("%Y-%m-%d")
    end_date = (start + MonthEnd(1)).strftime("%Y-%m-%d")
    try:
        filename_races = f'FT_AU_RACES_{start_date}.csv'
        filename_dogs = f'FT_AU_DOGS_{start_date}.csv'

        filepath_races = f'./data/raw/{filename_races}'
        filepath_dogs = f'./data/raw/{filename_dogs}'

        print(f'Loading data from {start_date} to {end_date}')
        if os.path.isfile(filepath_races):
            # Load local CSV file
            month_race_details = pd.read_csv(filepath_races) 
            month_dog_results = pd.read_csv(filepath_dogs) 
        else:
            # Fetch data from API
            month_race_details, month_dog_results = client.getRaceResults(start_date, end_date, au_tracks_filter)
            month_race_details.to_csv(filepath_races, index=False)
            month_dog_results.to_csv(filepath_dogs, index=False)

        # Combine monthly data
        race_details = race_details.append(month_race_details, ignore_index=True)
        dog_results = dog_results.append(month_dog_results, ignore_index=True)
    except:
        print(f'Could not load data from {start_date} to {end_date}')
        
'''
Clean raw data and store as a single .csv in ./data/clean
'''

# Clean up the race dataset
race_details = race_details.rename(columns = {'@id': 'FastTrack_RaceId'})
race_details['Distance'] = race_details['Distance'].apply(lambda x: int(x.replace("m", "")))
race_details['date_dt'] = pd.to_datetime(race_details['date'], format = '%d %b %y')
race_details['TrackDist'] = race_details['Track'] + race_details['Distance'].astype(str)

# Clean up the dogs results dataset
dog_results = dog_results.rename(columns = {'@id': 'FastTrack_DogId', 'RaceId': 'FastTrack_RaceId'})

# Combine dogs results with race attributes
dog_results = dog_results.merge(
    race_details[['FastTrack_RaceId', 'Distance', 'RaceGrade', 'Track', 'RaceNum', 'TrackDist', 'date_dt']], 
    how = 'left',
    on = 'FastTrack_RaceId'
)

# Clean StartPrice column
dog_results['StartPrice'] = dog_results['StartPrice'].apply(lambda x: None if x is None else float(x.replace('$', '').replace('F', '')) if isinstance(x, str) else x)

# Discard entries without results (scratched or did not finish)
dog_results = dog_results[~dog_results['Box'].isnull()]
dog_results['Box'] = dog_results['Box'].astype(int)

# Clean up other attributes
dog_results['RunTime'] = dog_results['RunTime'].astype(float)
dog_results['SplitMargin'] = dog_results['SplitMargin'].astype(float)
dog_results['Prizemoney'] = dog_results['Prizemoney'].astype(float).fillna(0)
dog_results['Place'] = pd.to_numeric(dog_results['Place'].apply(lambda x: x.replace("=", "") if isinstance(x, str) else x), errors='coerce')
dog_results = dog_results[~dog_results['Place'].isna()]
dog_results['win'] = dog_results['Place'].apply(lambda x: 1 if x == 1 else 0)
dog_results['DogName'] = dog_results['DogName'].apply(lambda x: x.upper() if type(x) == str else x)
dog_results['TrainerName'] = dog_results['TrainerName'].apply(lambda x: x.upper() if type(x) == str else x)

# Remove columns that are empty
dog_results = dog_results.drop(['Handicap', 'Comments'], axis=1)

# Remove races with no track/distance data (1 race)
dog_results = dog_results[~dog_results['TrackDist'].isna()]

# Save to ./data/clean/ directory
dog_results.to_csv('./data/clean/dog_results.csv', index=False)