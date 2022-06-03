import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

ad_data = pd.read_csv("advertising.csv")
# == EXPLORATION
print(ad_data.info())
# print(ad_data.head())
# print(ad_data.describe())

# sns.displot(data=ad_data, x="Age")  # Seems to be concentrated in 30s-40s
# sns.jointplot(y="Area Income", x="Age", data=ad_data)  # Doesn"t seem to be a super strong correlation here
# sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde", cmap="OrRd", fill=True)  # A concentration of around age 30 / 70-80 mins
# sns.jointplot(x="Daily Time Spent on Site", y="Daily Internet Usage", data=ad_data)
# sns.pairplot(ad_data, hue="Clicked on Ad")
# The above shows some clear correlations; Higher area income -> more likely to click on ad,
# more internet usage -> more likely to click on ad
# Younger -> more likely to click on ad
# More time spent on site -> more likely to click on ad
# plt.show(block=True)

# === TRAINING AND TESTING
# CLEAN
cleaned_ad_data = ad_data.drop(["Ad Topic Line", "City", "Country", "Timestamp"], axis=1)  # Feel like adding some way to include at least the Ad Topic Line would improve scoring

# SPLIT
X = cleaned_ad_data.drop("Clicked on Ad", axis=1)
y = cleaned_ad_data["Clicked on Ad"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

# == LOGISTIC REGRESSION
# FIT
logm = LogisticRegression()
logm.fit(X_train, y_train)

# PREDICT
log_preds = logm.predict(X_test)

# METRICS
print("Logistic Model")
print(metrics.classification_report(y_test, log_preds))  # Precision is more likely to be a good indicator than recall. And macro avg of precision is 0.93, which seems pretty good

# == RANDOM FOREST
# FIT
rand_forest = RandomForestClassifier()
rand_forest.fit(X_train, y_train)

# PREDICT
rand_forest_preds = rand_forest.predict(X_test)

# METRICS
print("Random Forest")
print(metrics.classification_report(y_test, rand_forest_preds))  # Precision is more likely to be a good indicator than recall. And macro avg of precision is 0.93, which seems pretty good
# Performs better than the logistic model (actually performs best)

# == DECISION TREE
# FIT
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# PREDICT
dec_tree_preds = dec_tree.predict(X_test)

# METRICS
print("Decision Tree")
print(metrics.classification_report(y_test, dec_tree_preds))  # Precision is more likely to be a good indicator than recall. And macro avg of precision is 0.93, which seems pretty good
# Also performs better than the logistic model

# == KNN
# FIT
# All of the KNNs seem terrible
recalls = []
for k in range(1, len(y_train)):
    print(f"{k} Nearest Neighbours")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # PREDICT
    knn_preds = knn.predict(X_test)

    # METRICS
    recalls.append(metrics.recall_score(y_test, knn_preds))
plt.plot(list(range(1, len(y_train))), recalls)
plt.show()
