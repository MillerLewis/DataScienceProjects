import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

data = pd.read_csv("College_Data", index_col=0)
data.info()
# print(data.describe())
print(data.head())

# sns.scatterplot(x="Room.Board", y="Grad.Rate", data=data, hue="Private")  # Seems to be a minor correlation with Private having a higher graduation
# sns.scatterplot(x="Outstate", y="F.Undergrad", data=data, hue="Private")  # More non-private on lower end of Outstate but wider spread of F.Undergrad
# sns.histplot(x="Outstate", hue="Private", data=data, alpha=0.4)  # This shows a similar note to above
# sns.histplot(x="Grad.Rate", hue="Private", data=data, alpha=0.4)  # Appears that more private have a higher grad rate
# plt.show(block=True)

# === TRAINING AND TESTING
# CLEANING
data["Grad.Rate"].clip(upper=100, inplace=True)
data["Private"] = data["Private"] == "Yes"
X = data.drop("Private", axis=1)

# FITTING
kmc = KMeans(n_clusters=2)
kmc.fit(X)
print(kmc.cluster_centers_)

# Since we have the actual label, we can evaluate. So this is purely academic
data["Cluster"] = kmc.predict(data.drop("Private", axis=1))
print(metrics.classification_report(data["Private"], data["Cluster"]))  # A really dreadful prediction really, guessing would be better
