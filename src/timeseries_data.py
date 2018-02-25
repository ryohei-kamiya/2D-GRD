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


from collections import OrderedDict
import csv
import numpy as np
import os

from nnabla.utils.data_source_implements import CsvDataSource
from nnabla.utils.data_source_loader import FileReader, load
from nnabla.utils.data_iterator import data_iterator
from nnabla.logger import logger


class TimeseriesDataSource(CsvDataSource):

    def __init__(self, dataset_path, x_input_length=48, x_output_length=16, x_split_step=16, shuffle=False, rng=None):
        if rng is None:
            rng = np.random.RandomState(313)
        self.rng = rng
        super(TimeseriesDataSource, self).__init__(filename=dataset_path, shuffle=shuffle, rng=rng, normalize=False)
        self._x_input_length = x_input_length
        self._x_output_length = x_output_length
        self._x_split_step = x_split_step
        self._original_size = self._size
        self._size = 0
        self._xdata_in = []
        self._xdata_out = []
        self._ydata = []
        self._load_timeseries()
        self.reset()

    def _get_data(self, position):
        xdata_in = self._xdata_in[self._indexes[position]]
        xdata_out = self._xdata_out[self._indexes[position]]
        ydata = self._ydata[self._indexes[position]]
        return (xdata_in, xdata_out, ydata)

    def _load_timeseries(self):
        for i in range(self._original_size):
            series = super(TimeseriesDataSource, self)._get_data(i)
            for j in range(0,len(series[0])-self._x_input_length-self._x_output_length,self._x_split_step):
                self._xdata_in.append(series[0][j:j+self._x_input_length])
                self._xdata_out.append(series[0][j+self._x_input_length:j+self._x_input_length+self._x_output_length])
                self._ydata.append(series[1])
                self._size += 1

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(TimeseriesDataSource, self).reset()

    @property
    def xdata_in(self):
        return self._xdata_in.copy()

    @property
    def xdata_out(self):
        return self._xdata_out.copy()

    @property
    def ydata(self):
        return self._ydata.copy()


def data_iterator_timeseries(dataset_path, batch_size,
                        x_input_length=32,
                        x_output_length=16,
                        x_split_step=32,
                        rng=None,
                        shuffle=True,
                        with_memory_cache=False,
                        with_parallel=False,
                        with_file_cache=False):
    return data_iterator(TimeseriesDataSource(dataset_path=dataset_path,
                                x_input_length=x_input_length, x_output_length=x_output_length,
                                x_split_step=x_split_step, shuffle=shuffle, rng=rng),
                         batch_size,
                         with_memory_cache,
                         with_parallel,
                         with_file_cache)
