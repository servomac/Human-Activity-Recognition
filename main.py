from hyperopt import Trials, STATUS_OK, tpe

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from data import _read_csv, load_y, load_signals, load_data

def _count_classes(y):
    return len(set([tuple(category) for category in y]))

def model(X_train, X_test, Y_train, Y_testt):
    timesteps = len(X_train[0])
    input_dim = len(X_train[0][0])
    n_classes = _count_classes(Y_train)

    model = Sequential()
    model.add(LSTM(32, input_shape=(timesteps, input_dim)))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(n_classes, activation={{choice(['sigmoid', 'softmax'])}}))

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit(X_train,
              Y_train,
              batch_size={{choice([16, 32, 64, 128])}},
              validation_data=(X_test, Y_test),
              verbose=2,
              epochs=30)

    score, acc = model.evaluate(X_test, Y_test, batch_size=16)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


X_train, X_test, Y_train, Y_test = load_data()
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=load_data,
                                          algo=tpe.suggest,
                                          functions=[_read_csv, load_y, _count_classes, load_signals],
                                          max_evals=5,
                                          trials=Trials())
    X_train, X_test, Y_train, Y_test = load_data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)