import pathlib

import idx2numpy
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import losses
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from MNIST.data_prep import DATA_DIR

# Will just assume there's one train/test so we don't need to match together the train and labels in some way
train_images = np.expand_dims(idx2numpy.convert_from_file(str(next(pathlib.Path(DATA_DIR).glob("train*images*")))), -1).astype("float32") / 255
train_labels = idx2numpy.convert_from_file(str(next(pathlib.Path(DATA_DIR).glob("train*labels*"))))

test_images = np.expand_dims(idx2numpy.convert_from_file(str(next(pathlib.Path(DATA_DIR).glob("t10k*images*")))), -1).astype("float32") / 255
test_labels = idx2numpy.convert_from_file(str(next(pathlib.Path(DATA_DIR).glob("t10k*labels*"))))


### MODEL CREATION
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), activation="relu"))
model.add(Conv2D(64, kernel_size=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(250, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss=losses.sparse_categorical_crossentropy)
stopper = EarlyStopping(patience=5)
model.fit(train_images, train_labels, validation_split=0.1, epochs=64, batch_size=128, callbacks=[stopper])
model.save("model_attempt")
model.summary()

predictions = np.argmax(model.predict(test_images), axis=-1)
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
