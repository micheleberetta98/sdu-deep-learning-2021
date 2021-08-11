import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------ STEP 1
# Download the auto.csv from the course website and load it into python.
# Use the pandas.read_csv function for importing the dataset.
# Be aware, there are some missing values in the dataset, indicated by '?'.
# You have to remove those lines and then make sure the corresponding columns are casted to a numerical type.

# Reading the CSV
df = pd.read_csv('auto.csv')
print(df.head())

# Removing the missing values (they are only in the horsepower column)
df = df[df.horsepower != '?']
df['horsepower'] = df['horsepower'].astype(int)

# ------------------------------ STEP 2
# Inspect the data. Plot the relationships between the different variables and mpg.
# Use for example the matplotlib.pyplot scatter plot.
# Do you already suspect what features might be helpful to regress the consumption? Save the graph.

# All the different variables (excluding "name")
variables = ['cylinders', 'displacement', 'horsepower',
             'weight', 'acceleration', 'model_year', 'origin']

# Plotting everything
for v in variables:
    plt.figure()
    plt.scatter(df[v], df['mpg'])
    plt.xlabel(v)
    plt.ylabel('MPG')
    plt.title(f'MPG against {v}')
    plt.savefig(f'images/mpg-{v}')
    plt.show()

# Displacement seems to have some kind of correlation
# As do horsepower, weight
# Acceleration seems pretty random, but we can try
# So we'll try displacement, horsepower, weight and acceleration

# ------------------------------ STEP 3
# Perform a linear regression using the OLS function from the statsmodels package.
# Use 'horsepower' as feature and regress the value 'mpg'.
# It is a good idea to look up the statsmodels documentation on OLS, to understand how to use it.
# Further, plot the results including your regression line.

X = sm.add_constant(df['horsepower'])  # Add the column of 1s
y = df['mpg']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# R-squared: 0.6 = 60% of data explained
# beta0 ~= 40
# beta1 ~= -0.16
# p-value is very low for both, so this means they are actually significant

y_hat = results.predict(X)  # Predictions of the model (just for graphing)

plt.figure()
plt.scatter(df['horsepower'], df['mpg'], label='Data')
plt.plot(df['horsepower'], y_hat, 'r', label='OLS prediction')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.legend()
plt.show()

# ------------------------------ STEP 4
# Now extend the model using all features.
# How would you determine which features are important and which aren't?
# Try to find a good selection of features for your model.

X = df.loc[:, df.columns != 'mpg']
X = X.loc[:, X.columns != 'name']
X = sm.add_constant(X)
y = df['mpg']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# R-squared is 82% - better than before
# Significant params seems to be (p-value < 0.05)
# - The constant
# - Displacement
# - Weight
# - Model year
# - Origin

# Let's see if omitting model year and origin gives something better
X = df[['displacement', 'weight']]
X = sm.add_constant(X)
y = df['mpg']
model = sm.OLS(y, X)
results = model.fit()
print(results.rsquared)  # R2 = 0.70 <- worse than before

# Maybe adding horsepower back?
X = df[['displacement', 'weight', 'horsepower']]
X = sm.add_constant(X)
y = df['mpg']
model = sm.OLS(y, X)
results = model.fit()
print(results.rsquared)  # R2 = 0.70 <- still worse

# So the features we car about are
# - Displacement
# - Weight
# - Model year
# - Origin

# ------------------------------ STEP 5
# Can you improve your regression performance by trying different transformations of the variables,
# such as log(X), sqrt(X), 1/X, X^2 and so on.
# Why are some transformations better?

labels = ['Log', 'Square root', 'Inverse', 'Square']
functions = [np.log, np.sqrt, lambda x: 1 / x, lambda x: x ** 2]

X = df.loc[:, df.columns != 'mpg']
X = X.loc[:, X.columns != 'name']

for name, f in zip(labels, functions):
    X1 = sm.add_constant(f(X))
    model = sm.OLS(df['mpg'], X1)
    results = model.fit()
    print(f'R-squared with {name}: {results.rsquared}')

# Some transformations are better because they tend to linearize the data
# So the linear regression performs better
