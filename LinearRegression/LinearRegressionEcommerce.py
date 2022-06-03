import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

customers = pd.read_csv("Ecommerce Customers")


# == EXPLORATION
# print(customers.info())
# print(customers.describe())
# print(customers.head())

# sns.jointplot(x=customers["Time on Website"], y=customers["Yearly Amount Spent"], data=customers)  # There doesn't seem to be much correlation via the joint plot
# sns.jointplot(x=customers["Time on App"], y=customers["Yearly Amount Spent"], data=customers)  # Seems to be some sort of positive correlation
# sns.jointplot(x=customers["Time on App"], y=customers["Length of Membership"], data=customers, kind="hex")  # Correlation of around 12 for time and 3 for length
# sns.pairplot(data=customers)  # In general, seems to be a good correlation between length of membership and amount spent

# x = sns.lmplot(x="Yearly Amount Spent", y="Length of Membership", data=customers)  # This shows a strong correlation
# print(stats.pearsonr(customers["Yearly Amount Spent"], customers["Length of Membership"]))  # Shows a 0.81 r value, so pretty strong
# plt.show(block=True)

# ====  TRAINING AND TESTING
# == LINEAR REGRESSION
# Clean
clean_customers = customers.drop("Email", axis=1).drop("Address", axis=1).drop("Avatar", axis=1)
X = clean_customers.drop("Yearly Amount Spent", axis=1)
y = clean_customers["Yearly Amount Spent"]
# Split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)
# Fit
ecomm_lm = LinearRegression()
ecomm_lm.fit(X_train, y_train)
coef_df = pd.DataFrame(ecomm_lm.coef_, X_test.columns, columns=["Coefficients"])  # Should definitely focus more on app and retaining customers
print(coef_df)
pred = ecomm_lm.predict(X_test)
# sns.scatterplot(x=y_test, y=pred)  # Shows a good prediction
# Metrics
mae = metrics.mean_absolute_error(y_test, pred)
mse = metrics.mean_squared_error(y_test, pred)
rmse = mse ** 0.5

print(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")
residuals = pred - y_test
# plt.hist(residuals, bins=30, rwidth=0.9)  # Shows most residuals around 0
# plt.show()
pass
