import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

classified_data = pd.read_csv("Classified Data", index_col=0)
# == EXPLORATION
print(classified_data.info())
print(classified_data.head())
print(classified_data.describe())

# sns.pairplot(classified_data, hue="TARGET CLASS")  # Some pretty good correlations with most vars
# plt.show(block=True)

# === TRAINING AND TESTING
# SPLIT
X = classified_data.drop("TARGET CLASS", axis=1)
y = classified_data["TARGET CLASS"]

# SCALING
scaler = StandardScaler(with_mean=True, with_std=True)
scaled_X = scaler.fit_transform(X)
X = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

# == LOGISTIC REGRESSION
# FIT
logm = LogisticRegression(max_iter=1000)
logm.fit(X_train, y_train)
logm_coef_df = pd.DataFrame(data=logm.coef_.transpose(), index=X_train.columns, columns=["Coefficients"])

# PREDICT
log_preds = logm.predict(X_test)

# METRICS
print("Logistic Model")
print(f"Coefficients:\n{logm_coef_df}")  # This actually shows that age is the strongest positive correlation
print(metrics.classification_report(y_test, log_preds))  # Since it's classified data, f1 score is probably good
# f1 score for logistic model seems to be better than random forest and decision tree

# == RANDOM FOREST
# FIT
rand_forest = RandomForestClassifier()
rand_forest.fit(X_train, y_train)

# PREDICT
rand_forest_preds = rand_forest.predict(X_test)

# METRICS
print("Random Forest")
print(metrics.classification_report(y_test, rand_forest_preds))  # Precision is more likely to be a good indicator than recall. And macro avg of precision is 0.93, which seems pretty good
# Actually pretty good, close to logistic model

# == DECISION TREE
# FIT
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)

# PREDICT
dec_tree_preds = dec_tree.predict(X_test)

# METRICS
print("Decision Tree")
print(metrics.classification_report(y_test, dec_tree_preds))  # Precision is more likely to be a good indicator than recall. And macro avg of precision is 0.93, which seems pretty good
# Worst of the bunch

# == KNN

# KNN seems good at around 30-46 neighbours. As good as logistic model at least
# f1_scores = []
# for k in range(1, len(y_test)):
#     print(f"{k} Nearest Neighbours")
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train, y_train)
#
#     # PREDICT
#     knn_preds = knn.predict(X_test)
#
#     # METRICS
#     f1_scores.append(metrics.f1_score(y_test, knn_preds))
# plt.plot(list(range(len(f1_scores))), f1_scores)
# plt.show(block=True)
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

# PREDICT
knn_preds = knn.predict(X_test)

# METRICS
print("KNN")
print(metrics.classification_report(y_test, knn_preds))