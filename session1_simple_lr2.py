import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression

n = 100
df = pd.DataFrame()
np.random.seed(1)
df['x1'] = np.random.randn(n)
df['x2'] = np.random.randn(n)
df['x3'] = np.random.randn(n)
df['x4'] = np.random.randn(n)
df['y'] = 10 + -100 * df['x1'] +  75*df['x3'] + np.random.randn(n)

X_test=df[['x1','x2','x3','x4']]
y_test=df['y']
print(y_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_test, y_test)

# The coefficients
print('Coefficients: \n', regr.coef_)


# Plot outputs
#plt.scatter(X_test, y_test,  color='black')
#plt.plot(X_test, y_test, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
