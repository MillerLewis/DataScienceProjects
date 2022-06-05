import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

iris = sns.load_dataset('iris')
iris.info()
print(iris.head())
# sns.pairplot(iris, hue='species')  # setosa species seems the most separable. The other 2 appear to be very similar
# sns.kdeplot(data=iris, x="sepal_width", y="sepal_length", fill=True, palette="crest")  # Concentrated around (3, 6)
# sns.kdeplot(data=iris, x="sepal_width", y="sepal_length", fill=True, palette="crest", hue="species", alpha=0.4)  # Again, shows that setosa is quite separable
plt.show(block=True)

# === TRAINING AND TESTING
# CLEANING
X = iris.drop("species", axis=1)
y = iris["species"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

svc = SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print(metrics.classification_report(y_test, pred))  # Actually pretty good without any hyper param tuning. Perfect guess on setosa and virginica

# Let's try to tune
svc_search = GridSearchCV(SVC(), {"C": [0.1, 1, 100], "gamma": [1, 0.1, 0.01, 0.001]}, verbose=2)
svc_search.fit(X, y)
print(f"Best params: {svc_search.best_params_}")
search_pred = svc_search.best_estimator_.predict(X_test)
print(metrics.classification_report(y_test, search_pred))  # Doesn't really improve anything
