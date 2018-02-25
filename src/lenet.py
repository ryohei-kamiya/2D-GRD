# Copyright 2018 Ryohei Kamiya <ryohei.kamiya@lab2biz.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import sys

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save
import nnabla.logger as logger
from nnabla.utils.data_iterator import data_iterator_csv_dataset
from nnabla.contrib.context import extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

import numpy as np
from numpy.random import seed

from sklearn.metrics import confusion_matrix, accuracy_score

class LeNet:
    def __init__(self, config):
        self._context = config.context
        if self._context is None:
            self._context = 'cpu'
        self._device_id = config.device_id
        self._process = config.process
        self._width = config.width
        self._height = config.height
        self._batch_size = config.batch_size
        self._max_iter = config.max_iter
        self._learning_rate = config.learning_rate
        self._weight_decay = config.weight_decay
        self._val_interval = config.val_interval
        self._val_iter = config.val_iter
        self._monitor_path = config.monitor_path
        self._model_params_path = config.model_params_path
        self._training_dataset_path = config.training_dataset_path
        self._validation_dataset_path = config.validation_dataset_path
        self._evaluation_dataset_path = config.evaluation_dataset_path
        self._x = None
        self._pred = None

    def network(self, x, test=False):
        # Input -> 1,28,28
        # Convolution -> 16,22,22
        with nn.parameter_scope('Convolution'):
            h = PF.convolution(x, 16, (7,7), (0,0))

        # ReLU
        h = F.relu(h, True)

        # BatchNormalization
        with nn.parameter_scope('BatchNormalization'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)
    
        # MaxPooling -> 16,11,11
        h = F.max_pooling(h, (2,2), (2,2), True)

        # Convolution_2 -> 30,9,9
        with nn.parameter_scope('Convolution_2'):
            h = PF.convolution(h, 30, (3,3), (0,0))

        # ReLU_2
        h = F.relu(h, True)

        # BatchNormalization_2
        with nn.parameter_scope('BatchNormalization_2'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)

        # MaxPooling_2 -> 30,4,4
        h = F.max_pooling(h, (2,2), (2,2), True)
        # Affine -> 160
        with nn.parameter_scope('Affine'):
            h = PF.affine(h, (160,))

        # ReLU_3
        h = F.relu(h, True)

        # BatchNormalization_3
        with nn.parameter_scope('BatchNormalization_3'):
            h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test)

        # Affine_2 -> 26
        with nn.parameter_scope('Affine_2'):
            h = PF.affine(h, (26,))

        return h

    def _load_dataset(self, dataset_path, batch_size=100, shuffle=False):
        if os.path.isfile(dataset_path):
            logger.info("Load a dataset from {}.".format(dataset_path))
            return data_iterator_csv_dataset(dataset_path, batch_size, shuffle=shuffle)
        return None

    def _categorical_error(self, pred, y):
        pred_label = pred.argmax(1)
        return (pred_label != y.flat).mean()

    # Load the model parameters
    def _load_model(self, model_params_path='parameters.h5'):
        nn.load_parameters(model_params_path)

    # Evaluation core
    def _evaluate(self, pred, x, y, data_iterator):
        preds = []
        corrects = []
        for i in range(data_iterator.size):
            data = data_iterator.next()
            x.d = data[0]
            y.d = data[1]
            pred.forward(clear_buffer=True)
            preds.extend(pred.d.argmax(1).flatten().tolist())
            corrects.extend(y.d.flatten().tolist())
        return (confusion_matrix(corrects, preds), accuracy_score(corrects, preds))

    # Validation
    def _validate(self, pred, x, y, data_iterator, val_iter=1):
        e = 0.0
        for i in range(val_iter):
            data = data_iterator.next()
            x.d = data[0]
            y.d = data[1]
            pred.forward(clear_buffer=True)
            e += self._categorical_error(pred.d, y.d)
        return e

    # Training core
    def _train(self, pred, solver, loss, x, y, data, weight_decay=0.0):
        x.d = data[0]
        y.d = data[1]
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(weight_decay)
        solver.update()
        return self._categorical_error(pred.d, y.d)

    # Training
    def train(self):
        # variables for training
        tx = nn.Variable([self._batch_size, 1, self._height, self._width])
        ty = nn.Variable([self._batch_size, 1])
        tpred = self.network(tx)
        tpred.persistent = True
        loss = F.mean(F.softmax_cross_entropy(tpred, ty))
        solver = S.Adam(self._learning_rate)
        solver.set_parameters(nn.get_parameters())

        # variables for validation
        vx = nn.Variable([self._batch_size, 1, self._height, self._width])
        vy = nn.Variable([self._batch_size, 1])
        vpred = self.network(vx, True)

        # data iterators
        tdata = self._load_dataset(self._training_dataset_path, self._batch_size, shuffle=True)
        vdata = self._load_dataset(self._validation_dataset_path, self._batch_size, shuffle=True)

        # monitors
        from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
        monitor = Monitor(self._monitor_path)
        monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
        monitor_err = MonitorSeries("Training error", monitor, interval=10)
        monitor_time = MonitorTimeElapsed("Training time", monitor, interval=100)
        monitor_verr = MonitorSeries("Validation error", monitor, interval=10)

        # Training loop
        for i in range(self._max_iter):
            if i % self._val_interval == 0:
                ve = self._validate(vpred, vx, vy, vdata, self._val_iter)
                monitor_verr.add(i, ve / self._val_iter)
            te = self._train(tpred, solver, loss, tx, ty, tdata.next(), self._weight_decay)
            monitor_loss.add(i, loss.d.copy())
            monitor_err.add(i, te)
            monitor_time.add(i)
        ve = self._validate(vpred, vx, vy, vdata, self._val_iter)
        monitor_verr.add(i, ve / self._val_iter)

        # Save a best model parameters
        nn.save_parameters(self._model_params_path)

    # Evaluation
    def evaluate(self):
        # variables for evaluation
        x = nn.Variable([self._batch_size, 1, self._height, self._width])
        y = nn.Variable([self._batch_size, 1])
        pred = self.network(x)
        self._load_model(self._model_params_path)

        # data iterator
        edata = self._load_dataset(self._evaluation_dataset_path, self._batch_size)
        result = self._evaluate(pred, x, y, edata)
        print("\n[confusion matrix]")
        np.savetxt(sys.stdout, result[0], fmt="%.0f", delimiter=",")
        print("\n[accuracy]\n{}".format(result[1]))

    # Initializer for inference
    def init_for_infer(self):
        if self._pred is None:
            # variables for inference
            self._x = nn.Variable([1, 1, self._height, self._width])
            self._pred = self.network(self._x, True)

            self._load_model(self._model_params_path)

    # Inference
    def infer(self, image):
        if self._pred is None:
            self.init_for_infer()
        if image.ndim == 2:
            image = image[np.newaxis,np.newaxis,:,:]
        elif image.ndim == 3:
            image = image[np.newaxis,:,:,:]
        if image.ndim != 4:
            return None
        if image.shape[2] < self._height:
            return None
        self._x.d = image[:1,:,:,:]
        self._pred.forward(clear_buffer=True)
        return self._pred.d.argmax(1)[0]


def get_args(model_params_path='parameters.h5', training_dataset_path="trining.csv",
        validation_dataset_path="validation.csv", evaluation_dataset_path="evaluation.csv",
        monitor_path='.', max_iter=100, learning_rate=0.001, batch_size=100, weight_decay=0,
        width=28, height=28, process="train", description=None):
    if description is None:
        description = "MLP"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size)
    parser.add_argument("--learning-rate", "-r",
                        type=float, default=learning_rate)
    parser.add_argument("--max-iter", "-i", type=int, default=max_iter,
                        help='Max iteration of training.')
    parser.add_argument("--val-interval", "-v", type=int, default=10,
                        help='Validation interval.')
    parser.add_argument("--val-iter", "-j", type=int, default=1,
                        help='Each validation runs `val_iter mini-batch iteration.')
    parser.add_argument("--weight-decay", "-w",
                        type=float, default=weight_decay,
                        help='Weight decay factor of SGD update.')
    parser.add_argument('--context', '-c', type=str,
                        default='cpu', help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--monitor-path", "-mon",
                        type=str, default=monitor_path,
                        help='Path monitoring logs saved.')
    parser.add_argument("--model-params-path", "-m",
                        type=str, default=model_params_path,
                        help='Path of the model parameters file.')
    parser.add_argument("--training-dataset-path", "-dt",
                        type=str, default=training_dataset_path,
                        help='Path of the training dataset.')
    parser.add_argument("--validation-dataset-path", "-dv",
                        type=str, default=validation_dataset_path,
                        help='Path of the validation dataset.')
    parser.add_argument("--evaluation-dataset-path", "-de",
                        type=str, default=evaluation_dataset_path,
                        help='Path of the evaluation dataset.')
    parser.add_argument('--process', '-p', type=str,
                        default='train', help="(train|evaluate|infer).")
    parser.add_argument("--width", "-wt", type=int, default=width,
                        help='Image width.')
    parser.add_argument("--height", "-ht", type=int, default=height,
                        help='Image height.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_args()
    logger.info("Running in %s" % config.context)

    seed(0)
    ctx = extension_context(config.context, device_id = config.device_id)
    nn.set_default_context(ctx)
    nn.clear_parameters()
    net = LeNet(config)

    if config.process == 'train':
        net.train()
    elif config.process == 'evaluate':
        net.evaluate()
    elif config.process == 'infer':
        net.init_for_infer()
        if os.path.isfile(config.evaluation_dataset_path):
            logger.info("Load a dataset from {}.".format(config.evaluation_dataset_path))
            edata = data_iterator_csv_dataset(config.evaluation_dataset_path, 1, shuffle=False)
            for i in range(edata.size):
                data = edata.next()
                x_d = data[0]
                result = net.infer(x_d)
                print("inference result = {}".format(result))
