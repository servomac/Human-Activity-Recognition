from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

from data import load_data

X_train, X_test, Y_train, Y_test = load_data()

#(batch_size, timesteps, input_dim)
timesteps = 128
input_dim = 9
output_dim = 6

model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(output_dim, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train,
          Y_train,
          batch_size=16,
          validation_data=(X_test, Y_test),
          epochs=30)

score = model.evaluate(X_test, Y_test, batch_size=16)
print(score)