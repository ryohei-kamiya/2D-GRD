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
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed

from timeseries_data_with_baseshift import data_iterator_timeseries

import numpy as np
from numpy.random import seed


class LSTM:
    def __init__(self, config, h=None, c=None):
        self._context = config.context
        if self._context is None:
            self._context = 'cpu'
        self._device_id = config.device_id
        self._process = 'train'
        self._cols_size = config.columns_size
        self._x_input_length = config.x_input_length
        self._x_output_length = config.x_output_length
        self._x_split_step = config.x_split_step
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
        self._lstm_unit_name = config.lstm_unit_name
        self._lstm_units = config.lstm_units

        self._h = h
        self._c = c
        self._x_in = None
        self._pred = None

    def _load_dataset(self, dataset_path, batch_size=100, shuffle=False):
        if os.path.isfile(dataset_path):
            logger.info("Load a dataset from {}.".format(dataset_path))
            return data_iterator_timeseries(dataset_path, batch_size,
                        x_input_length=self._x_input_length,
                        x_output_length=self._x_output_length,
                        x_split_step=self._x_split_step,
                        rng=None,
                        shuffle=shuffle,
                        with_memory_cache=True,
                        with_parallel=False,
                        with_file_cache=True)
        return None

    def _lstm_cell(self, name, n_hidden, x_in, h=None, c=None):
        if h is None:
            h = nn.Variable.from_numpy_array(np.zeros((self._batch_size, self._cols_size)))
        if c is None:
            c = nn.Variable.from_numpy_array(np.zeros((self._batch_size, n_hidden)))
        h = F.concatenate(h, x_in, axis=1) # LSTM_Concatenate -> cols_size * 2
        with nn.parameter_scope(name + '_Affine'): # LSTM_Affine -> n_hidden
            h1 = PF.affine(h, (n_hidden,), base_axis=1)
        with nn.parameter_scope(name + '_IGate'): # LSTM_IGate -> n_hidden
            h2 = PF.affine(h, (n_hidden,), base_axis=1)
        with nn.parameter_scope(name + '_FGate'): # LSTM_FGate -> n_hidden
            h3 = PF.affine(h, (n_hidden,), base_axis=1)
        with nn.parameter_scope(name + '_OGate'): # LSTM_OGate -> n_hidden
            h4 = PF.affine(h, (n_hidden,), base_axis=1)
        h1 = F.tanh(h1)     # LSTM_Tanh
        h2 = F.sigmoid(h2)  # LSTM_Sigmoid
        h3 = F.sigmoid(h3)  # LSTM_Sigmoid_2
        h4 = F.sigmoid(h4)  # LSTM_Sigmoid_3
        h5 = F.mul2(h2, h1) # LSTM_Mul2 -> n_hidden
        h6 = F.mul2(h3, c)  # LSTM_Mul2_2 -> n_hidden
        h7 = F.add2(h5, h6 ,inplace=True) # LSTM_Add2 -> n_hidden
        h8 = F.tanh(h7)     # LSTM_Tanh_2 -> n_hidden
        h9 = F.mul2(h4, h8) # LSTM_Mul2_3 -> n_hidden
        c = h7              # LSTM_C
        h = h9              # LSTM_H
        return (h, c)

    def network(self, x_in, name='LSTM', n_hidden=32):
        hlist = []
        for x_i in F.split(x_in, axis=1):
            self._h, self._c = self._lstm_cell(name, n_hidden, x_i, self._h, self._c)
            with nn.parameter_scope(name + '_Affine_2'):
                self._h = PF.affine(self._h, (self._cols_size,))
            hlist.append(self._h)
        h = F.stack(*hlist, axis=1)
        h = F.slice(h, start=[0, h.shape[1]-self._x_output_length, 0],
                stop=[self._batch_size, h.shape[1], self._cols_size],
                step=[1, 1, 1])
        return h

    def _regression_error(self, pred, x_out):
        return np.mean((pred - x_out)**2)

    # Validation
    def _validate(self, pred, x_in, x_out, data_iterator, val_iter=1):
        e = 0.0
        for i in range(val_iter):
            data = data_iterator.next()
            x_in.d = data[0]
            x_out.d = data[1]
            pred.forward(clear_buffer=True)
            e += self._regression_error(pred.d, x_out.d)
        return e

    # Training core
    def _train(self, pred, solver, loss, x_in, x_out, data, weight_decay=0.0):
        x_in.d = data[0]
        x_out.d = data[1]
        solver.zero_grad()
        loss.forward(clear_no_need_grad=True)
        loss.backward(clear_buffer=True)
        solver.weight_decay(weight_decay)
        solver.update()
        return self._regression_error(pred.d, x_out.d)

    # Training
    def train(self):
        # variables for training
        tx_in = nn.Variable([self._batch_size, self._x_input_length, self._cols_size])
        tx_out = nn.Variable([self._batch_size, self._x_output_length, self._cols_size])
        tpred = self.network(tx_in, self._lstm_unit_name, self._lstm_units)
        tpred.persistent = True
        loss = F.mean(F.squared_error(tpred, tx_out))
        solver = S.Adam(self._learning_rate)
        solver.set_parameters(nn.get_parameters())

        # variables for validation
        vx_in = nn.Variable([self._batch_size, self._x_input_length, self._cols_size])
        vx_out = nn.Variable([self._batch_size, self._x_output_length, self._cols_size])
        vpred = self.network(vx_in, self._lstm_unit_name, self._lstm_units)

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
                ve = self._validate(vpred, vx_in, vx_out, vdata, self._val_iter)
                monitor_verr.add(i, ve / self._val_iter)
            te = self._train(tpred, solver, loss, tx_in, tx_out, tdata.next(), self._weight_decay)
            monitor_loss.add(i, loss.d.copy())
            monitor_err.add(i, te)
            monitor_time.add(i)
        ve = self._validate(vpred, vx_in, vx_out, vdata, self._val_iter)
        monitor_verr.add(i, ve / self._val_iter)

        # Save a best model parameters
        nn.save_parameters(self._model_params_path)


def get_args(model_params_path='parameters.h5', training_dataset_path="trining.csv",
        validation_dataset_path="validation.csv", monitor_path='.',
        max_iter=100, learning_rate=0.001, batch_size=100, weight_decay=0,
        x_input_length=128, x_output_length=64, x_split_step=16, columns_size=2,
        lstm_unit_name="LSTM", lstm_units=32, description=None):
    if description is None:
        description = "LSTM"
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
    parser.add_argument("--x-input-length", "-xil", type=int, default=x_input_length,
                        help='Length of time-series into the network.')
    parser.add_argument("--x-output-length", "-xol", type=int, default=x_output_length,
                        help='Length of time-series from the network.')
    parser.add_argument("--x-split-step", "-xss", type=int, default=x_split_step,
                        help='Step size to split time-series.')
    parser.add_argument("--columns-size", "-cs", type=int, default=columns_size,
                        help='Columns size of time-series matrix.')
    parser.add_argument("--lstm-unit-name", "-lstmn", type=str, default=lstm_unit_name,
                        help='LSTM unit name.')
    parser.add_argument("--lstm-units", "-lstmu", type=int, default=lstm_units,
                        help='The number of LSTM units.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = get_args()
    logger.info("Running in %s" % config.context)

    seed(0)
    ctx = get_extension_context(config.context, device_id = config.device_id)
    nn.set_default_context(ctx)
    nn.clear_parameters()
    net = LSTM(config)
    net.train()
