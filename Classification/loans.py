import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

loan_data = pd.read_csv("loan_data.csv")
# == EXPLORATION
loan_data.info()
# print(loan_data.head())
# print(loan_data.describe())

# sns.histplot(data=loan_data, x="fico", hue="credit.policy", alpha=0.3, binwidth=7)  # Very few below 660 have a policy 1
# sns.histplot(data=loan_data, x="fico", hue="not.fully.paid", alpha=0.3, binwidth=7)  # Majority of scores around 650, most not fully paid
# sns.histplot(data=loan_data, x="purpose", hue="not.fully.paid", alpha=0.3, binwidth=7)  # Most loans are debt consolidation
# sns.jointplot(x="fico", y="int.rate", data=loan_data)  # Seems to be a slight negative correlation between fico score and interest rate
# sns.lmplot(x="fico", y="int.rate", data=loan_data, hue="not.fully.paid", col="credit.policy")  # Correlations seem the same
# plt.show(block=True)

# === TRAINING AND TESTING
# CLEANING
# loan_data.drop(["purpose"], inplace=True, axis=1)
dummied_loan_data = pd.get_dummies(loan_data, columns=["purpose"], drop_first=True)
# SPLIT
X = dummied_loan_data.drop("not.fully.paid", axis=1)
y = dummied_loan_data["not.fully.paid"]

# SCALING
scaler = StandardScaler(with_mean=True, with_std=True)
scaled_X = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)
scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

# == LOGISTIC REGRESSION
# FIT
logm = LogisticRegression(max_iter=1000)
logm.fit(X_train, y_train)
logm_coef_df = pd.DataFrame(data=logm.coef_.transpose(), index=X_train.columns, columns=["Coefficients"])

# PREDICT
log_preds = logm.predict(X_test)

# METRICS
print("Logistic Model")
print(f"Coefficients:\n{logm_coef_df}")
print(metrics.classification_report(y_test, log_preds))


# == RANDOM FOREST
# FIT
# rand_forest_search = GridSearchCV(RandomForestClassifier(), {"n_estimators": list(range(100, 1000, 100))}, verbose=2)
# rand_forest_search.fit(X_train, y_train)
# rand_forest = rand_forest_search.best_estimator_
rand_forest = RandomForestClassifier(n_estimators=500)  # The recall is dreadful. Need more data really
rand_forest.fit(X_train, y_train)

# PREDICT
rand_forest_preds = rand_forest.predict(X_test)

# METRICS
print("Random Forest")
print(metrics.classification_report(y_test, rand_forest_preds))
# print(f"Best random forest params: {rand_forest_search.best_params_}")

# == DECISION TREE
# FIT
dec_tree = DecisionTreeClassifier()
# PREDICT
dec_tree.fit(X_train, y_train)
dec_tree_preds = dec_tree.predict(X_test)

# METRICS
print("Decision Tree")
print(metrics.classification_report(y_test, dec_tree_preds))

# == KNN
knn_search = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": list(range(1, 31, 2))}, verbose=2)
knn_search.fit(scaled_X_train, scaled_y_train)
knn = knn_search.best_estimator_  # 0 precision for class 1

# PREDICT
knn_preds = knn.predict(X_test)

# METRICS
print("KNN")
print(metrics.classification_report(y_test, knn_preds))

# Overall notes, the logistic model seems to perform best, but it's still not good. It's close to just guessing
