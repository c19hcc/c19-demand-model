import pandas as pd
import numpy as np

data = pd.read_csv('county_info.csv')
data['countyFIPS'] = data['countyFIPS'].apply(str).apply(lambda x: x.zfill(5))
data.to_csv('county_info.csv', index=False)