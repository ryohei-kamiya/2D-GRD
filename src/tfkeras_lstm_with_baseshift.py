from logzero import logger
import math
import argparse
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM as Lstm
from tensorflow.python.keras import optimizers

class LSTM:
    def __init__(self, config):
        self._process = config.process
        self._cols_size = config.columns_size
        self._x_input_length = config.x_input_length
        self._x_split_step = config.x_split_step
        self._batch_size = config.batch_size
        self._epochs = config.epochs
        self._learning_rate = config.learning_rate
        self._validation_split = config.validation_split
        self._model_params_path = config.model_params_path
        self._training_dataset_path = config.training_dataset_path
        self._evaluation_dataset_path = config.evaluation_dataset_path
        self._lstm_units = config.lstm_units
        self._model = None

    def __del__(self):
        tf.keras.backend.clear_session()

    def create_model(self, xin, xout, n_hidden=32):
        return Sequential([
            Lstm(n_hidden, input_shape=(xin.shape[1], xin.shape[2]), return_sequences=False),
            Dense(xout.shape[1], activation='linear')
        ])

    def load_dataset(self, dataset_path, x_col_name='x', x_input_length=16, x_split_step=1):
        logger.info("Load a dataset from {}.".format(dataset_path))
        dataset_dirpath = os.path.dirname(dataset_path)
        xinlist = []
        xoutlist = []
        indexcsv = pd.read_csv(dataset_path)
        for cell in indexcsv[x_col_name]:
            df = pd.read_csv(os.path.join(dataset_dirpath, cell), header=None)
            series = df.as_matrix()
            i_last = series.shape[0] - x_input_length - 1
            for i in range(0, i_last, x_split_step):
                past = series[i:i+x_input_length,:]
                current = series[i+x_input_length-1,:]
                future = series[i+x_input_length:i+x_input_length+1,:]
                xinlist.append(np.subtract(past, current))
                xoutlist.append(np.subtract(future, current))
        Xin = np.asarray(xinlist)
        Xout = np.asarray(xoutlist)
        Xout = Xout.reshape((Xout.shape[0], Xout.shape[2]))
        return (Xin, Xout)

    # Load the model parameters
    def _load_model(self, model_params_path='model.h5'):
        return load_model(model_params_path)

    # Training
    def train(self):
        Xin_train, Xout_train = self.load_dataset(self._training_dataset_path)
        model = self.create_model(Xin_train, Xout_train, self._lstm_units)
        optimizer = optimizers.Adam(lr=self._learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        hist = model.fit(Xin_train, Xout_train, batch_size=self._batch_size, verbose=1, epochs=self._epochs, validation_split=self._validation_split)
        model.save(self._model_params_path)

    # Evaluation
    def evaluate(self):
        Xin_test, Xout_test = self.load_dataset(self._evaluation_dataset_path)
        model = self._load_model(self._model_params_path)
        score = model.evaluate(Xin_test, Xout_test, verbose=1)
        print("test accuracy : ", score[1])

    # Initializer for inference
    def init_for_infer(self):
        self._model = self._load_model(self._model_params_path)

    # Inference
    def infer(self, series):
        if len(series.shape) == 2:
            series = series.reshape(1, series.shape[0], series.shape[1])
        xin = series[:1,series.shape[1]-self._x_input_length:,:]
        result = self._model.predict(xin, batch_size=1, verbose=0)
        return result

def get_args(model_params_path='model.h5', training_dataset_path="trining.csv",
        evaluation_dataset_path="evaluation.csv", epochs=100, learning_rate=0.001,
        validation_split=0.1, batch_size=100, process="train", columns_size=2,
        x_input_length=128, x_split_step=16, lstm_units=32, description=None):
    if description is None:
        description = "LSTM"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-r",
                        type=float, default=learning_rate)
    parser.add_argument("--validation-split", "-vs",
                        type=float, default=validation_split)
    parser.add_argument("--epochs", "-e", type=int, default=epochs,
                        help='Epochs of training.')
    parser.add_argument("--model-params-path", "-m",
                        type=str, default=model_params_path,
                        help='Path of the model parameters file.')
    parser.add_argument("--training-dataset-path", "-dt",
                        type=str, default=training_dataset_path,
                        help='Path of the training dataset.')
    parser.add_argument("--evaluation-dataset-path", "-de",
                        type=str, default=evaluation_dataset_path,
                        help='Path of the evaluation dataset.')
    parser.add_argument('--process', '-p', type=str,
                        default='train', help="(train|evaluate|infer).")
    parser.add_argument("--x-input-length", "-xil", type=int, default=x_input_length,
                        help='Length of time-series into the network.')
    parser.add_argument("--x-split-step", "-xss", type=int, default=x_split_step,
                        help='Step size to split time-series.')
    parser.add_argument("--columns-size", "-cs", type=int, default=columns_size,
                        help='Columns size of time-series matrix.')
    parser.add_argument("--lstm-units", "-lstmu", type=int, default=lstm_units,
                        help='The number of LSTM units.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_args()

    model = LSTM(config)
    if config.process == 'train':
        model.train()
    elif config.process == 'evaluate':
        model.evaluate()
    elif config.process == 'infer':
        logger.info("Load a dataset from {}.".format(config.evaluation_dataset_path))
        model.init_for_infer()
        Xin, Xout = model.load_dataset(config.evaluation_dataset_path)
        for i in range(Xin.shape[0]):
            x = Xin[i,:]
            result = model.infer(x)
            print("inference result = {}, true value = {}".format(result.reshape(result.shape[1]), Xout[i]))
