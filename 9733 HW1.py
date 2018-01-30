# 9733 HW1 Qustion 1

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================================
# =========================        Question 1        =========================
# ============================================================================

n = 500
np.random.seed(6996)
Epsilon_vector = np.random.normal(0, 1, n)
X_matrix = np.random.normal(0, 2, n * n)
X_matrix = np.reshape(X_matrix, (n, n))
slopeSet_vector = np.random.uniform(1, 5, 500)
sapply = lambda x: 1 + X_matrix[:, :x].dot(slopeSet_vector[:x]) + Epsilon_vector
z = range(2, 501)
Y = list(map(sapply, z))
Y = np.array(Y).T


# ============================================================================
# =========================        Question 2        =========================
# ============================================================================

# Number of predictors is 490
n = 490
# Define training sets and targets
X_test = sm.add_constant(X_matrix[:, 0:n])
Y_test = Y[:, n - 3]
# Fit regression model
results = sm.OLS(Y_test, X_test).fit()
pval = results.pvalues


plt.figure()
plt.scatter(range(1, n + 1), pval[1:n + 1], marker='o', c='', edgecolors='k')
plt.title('Coefficients P-Values for 490 Predictors')
plt.xlim(0, 500)
plt.xlabel('Coefficient')
plt.ylim(0, 0.01)
plt.ylabel('P-Value')
plt.show()


# ============================================================================
# =======================        Question 3, 4        ========================
# ============================================================================
r_sqaured = []
low = []
upp = []

for i in range(2, 501):
    X_test = sm.add_constant(X_matrix[:, :i])
    Y_test = Y[:, i - 2]
    # Fit regression model
    results = sm.OLS(Y_test, X_test).fit()
    r2 = results.rsquared
    l, u = results.conf_int()[1]
    r_sqaured.append(r2)
    low.append(l)
    upp.append(u)


plt.figure()
plt.plot(range(2, 501), r_sqaured, c='b')
plt.title('Improvement of Fit with Number of Predictors')
plt.xlabel('Number of Predictors')
plt.ylabel('Determination Coefficient')
plt.show()


plt.figure()
plt.plot(range(2, 501), low, c='r')
plt.plot(range(2, 501), upp, c='b')
plt.title('Confidence Intervals for Beta_1')
plt.xlabel('Number of Predictors')
plt.ylabel('95% Confidence Intervals')
plt.ylim(0, 3)
plt.show()

