import pandas as pd

df = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv')
df.to_csv('./data/covid_confirmed_usafacts.csv', index=False)

df = pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv')
df.to_csv('./data/covid_deaths_usafacts.csv', index=False)