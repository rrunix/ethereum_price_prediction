from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import data_loader
import numpy as np

WINDOW = 10

X, Y, options = data_loader.load_data("etherium_data_pretty.json", window_size=WINDOW, remove_features=["date"],

                                      preprocess_args={'normaliser': StandardScaler()})

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = Sequential()

model.add(LSTM(
    input_shape=(None, 7),
    units=100,
    dropout=0.2,
    return_sequences=True))

model.add(LSTM(
    200,
    dropout=0.2,
    return_sequences=False))

model.add(Dense(units=1))

model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')

model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05)


predictions = model.predict(X[:1000], batch_size=512)
real_values = Y.flatten()[:1000]

indexes = np.arange(0, len(real_values))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


plt.plot(indexes, real_values, color="red")
plt.plot(indexes, predictions, color="blue")

plt.show()
