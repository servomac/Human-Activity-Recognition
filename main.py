import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)

# for reproducibility
# https://github.com/fchollet/keras/issues/2280
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

from data import load_data
from utils import confusion_matrix

epochs = 30
batch_size = 16
n_hidden = 32

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

X_train, X_test, Y_train, Y_test = load_data()

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

model = Sequential()
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          validation_data=(X_test, Y_test),
          epochs=epochs)

# Evaluate
print(confusion_matrix(Y_test, model.predict(X_test)))
