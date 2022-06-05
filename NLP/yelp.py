import pandas as pd
import sklearn.model_selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

data = pd.read_csv("yelp.csv")
data.info()
data["text length"] = data["text"].apply(len)
print(data.head())

# DATA EXPLORATION
# fg = sns.FacetGrid(data, col="stars")
# fg.map(sns.histplot, "text length")  # In all cases, shorter text lengths

# sns.boxplot(data=data, x="stars", y="text length")

sns.countplot(data=data, x="stars")  # Majority 4 stars

meaned_data = data.groupby(by="stars").mean()  # Shows that most reviews are useful, lower star reviews are more funny, higher star reviews are more cool
# print(meaned_data)
# print(meaned_data.corr())  # Cool reviews have a pretty strong negative correlation with useful, funny, and text length useful strongly positive correlates with funny and text length, funny strongly positive correlates with text length
# sns.heatmap(meaned_data.corr(), cmap="plasma")
# plt.show(block=True)

yelp_class = data[(data["stars"] == 1) | (data["stars"] == 5)]
X = yelp_class["text"]
y = yelp_class["stars"]

cv = CountVectorizer()
X = pd.DataFrame(cv.fit_transform(X).toarray(), columns=cv.get_feature_names_out())
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

mnnb_model = MultinomialNB()
mnnb_model.fit(X_train, y_train)
preds = mnnb_model.predict(X_test)
print(metrics.classification_report(y_test, preds))

class CustomTfidfVectorizer(TfidfVectorizer):
    def transform(self, raw_documents):
        sparse = super().transform(raw_documents)
        return pd.DataFrame(sparse.toarray(), columns=self.get_feature_names_out())

    def fit(self, raw_documents, y=None):
        super().fit(raw_documents, y=y)

pipeline = Pipeline([
    ("tfid", CustomTfidfVectorizer()),
    ("mnnb", MultinomialNB())
], verbose=2)

X = yelp_class["text"]
y = yelp_class["stars"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=101)

pipeline.fit(X_train, y=y_train)
tf_preds = pipeline.predict(X_test)
print(metrics.classification_report(y_test, tf_preds))  # A lot worse
