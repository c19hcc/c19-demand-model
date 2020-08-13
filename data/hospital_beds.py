import json
import pandas as pd


# Clean Beds Data
data_beds = []

with open('./usa-hospital-beds.geojson') as f:
    for line in f:
        data_beds.append(json.loads(line))
        
data_beds = pd.DataFrame(data_beds)
data_beds_county = data_beds[['FIPS', 'NUM_LICENSED_BEDS', 'NUM_ICU_BEDS']].groupby('FIPS').agg(sum)
data_beds_county = data_beds_county.fillna(0)

# Clean Cases Data
df = pd.read_csv('covid_confirmed_usafacts.csv', converters={'countyFIPS': str})
df['countyFIPS'] = df['countyFIPS'].apply(lambda x: x.zfill(5))
df = df[df['countyFIPS'] != '00000']
df = df[df['countyFIPS'] != '00001']

# Merge Data
df = pd.merge(df, data_beds_county, left_on='countyFIPS', right_on='FIPS', how='left')
df = df.fillna(0)

df = df[['countyFIPS', 'County Name', 'State', 'NUM_LICENSED_BEDS', 'NUM_ICU_BEDS']]
df.to_csv('hospital_ICU_beds_by_county.csv', index=False)