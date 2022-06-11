import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.losses import BinaryCrossentropy


bank_note_df = pd.read_csv("bank_note_data.csv")
bank_note_df.info()
# print(bank_note_df.head(5))
# bank_note_df.describe()

# sns.countplot(x="Class", data=bank_note_df)  # It's almost an even split ~750 class 0, ~600 class 1
# sns.heatmap(bank_note_df.isnull())  # No missing values, sort of as expected
# sns.heatmap(bank_note_df.corr(), cmap="RdYlGn")
"""
Image variance seems to have the most strong negative correlation with class
Image curtosis seems to have NO correlation with class directly
"""

sns.pairplot(data=bank_note_df, hue="Class")
"""
As above, Image curtosis plot seems the hardest to distinguish from, but there is still something with Image Curtosis 
against Image Variance
"""

plt.show(block=True)

### SPLITTING
X = bank_note_df.drop("Class", axis=1).values
y = bank_note_df["Class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

### STANDARDIZE
scaler = StandardScaler()
scaler.fit(X_train)
X_train, X_test = map(scaler.transform, (X_train, X_test))

### MODEL
saved_model_path = "test_model"
overwrite = True
model = Sequential()
model.add(Dense(10, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.compile(optimizer="adam", loss=BinaryCrossentropy())
if not os.path.exists(saved_model_path) or overwrite:
    log_dir = "logs\\fit\\"
    stopper = EarlyStopping(patience=10)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(X_train, y_train, epochs=600, validation_split=0.1, callbacks=[stopper, tb])
    model.save(saved_model_path)
    plt.plot(model.history.history["loss"])
    plt.plot(model.history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.show(block=True)
else:
    model.load_weights(saved_model_path)
preds = (model.predict(x=X_test) > 0.5).astype("int32")
print(classification_report(y_test, preds))
