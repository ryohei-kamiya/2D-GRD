from logzero import logger
import math
import argparse
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras import optimizers

class MLP:
    def __init__(self, config):
        self._process = config.process
        self._batch_size = config.batch_size
        self._epochs = config.epochs
        self._learning_rate = config.learning_rate
        self._validation_split = config.validation_split
        self._model_params_path = config.model_params_path
        self._training_dataset_path = config.training_dataset_path
        self._evaluation_dataset_path = config.evaluation_dataset_path
        self._model = None

    def __del__(self):
        tf.keras.backend.clear_session()

    def create_model(self, x, y):
        return Sequential([
            Dense(100, activation='tanh', input_shape=(x.shape[1],x.shape[2])),
            Flatten(),
            Dense(y.shape[1], activation='softmax')
        ])

    def load_dataset(self, dataset_path, x_col_name='x', y_col_name='y'):
        logger.info("Load a dataset from {}.".format(dataset_path))
        dataset_dirpath = os.path.dirname(dataset_path)
        xlist = []
        ylist = []
        indexcsv = pd.read_csv(dataset_path)
        for cell in indexcsv[x_col_name]:
            df = pd.read_csv(os.path.join(dataset_dirpath, cell), header=None)
            xlist.append(df.as_matrix())
        for cell in indexcsv[y_col_name]:
            ylist.append(cell)
        X = np.asarray(xlist)
        y = np.asarray(ylist)
        y = tf.keras.utils.to_categorical(y)
        return (X, y)

    # Load the model parameters
    def _load_model(self, model_params_path='model.h5'):
        return load_model(model_params_path)

    # Training
    def train(self):
        X_train, y_train = self.load_dataset(self._training_dataset_path)
        model = self.create_model(X_train, y_train)
        optimizer = optimizers.Adam(lr=self._learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        hist = model.fit(X_train, y_train, batch_size=self._batch_size, verbose=1, epochs=self._epochs, validation_split=self._validation_split)
        model.save(self._model_params_path)

    # Evaluation
    def evaluate(self):
        X_test, y_test = self.load_dataset(self._evaluation_dataset_path)
        model = self._load_model(self._model_params_path)
        score = model.evaluate(X_test, y_test, verbose=1)
        print("test accuracy : ", score[1])

    # Initializer for inference
    def init_for_infer(self):
        self._model = self._load_model(self._model_params_path)

    # Inference
    def infer(self, series):
        if len(series.shape) == 2:
            series = series.reshape(1, series.shape[0], series.shape[1])
        result = self._model.predict(series, batch_size=1, verbose=0)
        return result.argmax(1)[0]

def get_args(model_params_path='model.h5', training_dataset_path="trining.csv",
        evaluation_dataset_path="evaluation.csv", epochs=100, learning_rate=0.001,
        validation_split=0.1, batch_size=100, process="train", description=None):
    if description is None:
        description = "MLP"
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_args()

    model = MLP(config)
    if config.process == 'train':
        model.train()
    elif config.process == 'evaluate':
        model.evaluate()
    elif config.process == 'infer':
        logger.info("Load a dataset from {}.".format(config.evaluation_dataset_path))
        model.init_for_infer()
        X_test, y_test = model.load_dataset(config.evaluation_dataset_path)
        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            result = model.infer(x)
            print("inference result = {}, true label = {}".format(result, y_test[i].argmax(0)))
