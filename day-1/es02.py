import pandas as pd

# Read the csv
df = pd.read_csv('auto.csv')

# Remove all rows with 'mpg' lower than 16
filtered = df.query('mpg >= 16')
# Another solution
df[df.mpg >= 16]
print(filtered.head())

# Get the first 7 rows of the columns 'weights' and 'acceleration'
first7 = df[['weight', 'acceleration']][:7]
# Or
first7 = df[['weight', 'acceleration']].head(7)

# Remove the rows in the 'horsepower' column that has the value '?' and convert the column
# to an int type instead of a string
cols = df[df.horsepower != '?']
cols['horsepower'] = cols['horsepower'].astype(int)

# Average of every column but name
df.loc[:, df.columns != 'name'].mean()
