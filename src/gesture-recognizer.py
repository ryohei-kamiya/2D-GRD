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


import sys
import argparse
import os
import os.path
import tkinter
import datetime
import pointsbuffer
from PIL import Image, ImageDraw

import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
from nnabla.contrib.context import extension_context

import numpy as np
from numpy.random import seed

from mlp import MLP
from lenet import LeNet
from lstm import LSTM

def get_args(mlp_model_params_path="mlp-parameters.h5", lenet_model_params_path="lenet-parameters.h5",
            lstm_model_params_path="lstm-parameters.h5", labels_path="labels.txt",  description=None):
    if description is None:
        description = "Gesture recognizer"
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--context', '-c', type=str,
                        default=None, help="Extension modules. ex) 'cpu', 'cuda.cudnn'.")
    parser.add_argument("--device-id", "-d", type=int, default=0,
                        help='Device ID the training run on. This is only valid if you specify `-c cuda.cudnn`.')
    parser.add_argument("--net", "-n", type=str,
                        default='mlp',
                        help="Neural network architecure type : ('mlp'|'lenet'|'mlp-with-lstm')")
    parser.add_argument("--mlp-model-params-path", "-mlp",
                        type=str, default=mlp_model_params_path,
                        help='Path of the mlp model parameters file.')
    parser.add_argument("--lenet-model-params-path", "-lenet",
                        type=str, default=lenet_model_params_path,
                        help='Path of the lenet model parameters file.')
    parser.add_argument("--lstm-model-params-path", "-lstm",
                        type=str, default=lstm_model_params_path,
                        help='Path of the lstm model parameters file.')
    parser.add_argument("--labels-path", "-l",
                        type=str, default=labels_path,
                        help='Path of the labels file.')
    args = parser.parse_args()
    return args

class GestureRecognizer:
    def __init__(self, args):
        class Config:
            pass
        config = Config()
        config.context = args.context
        config.device_id = args.device_id
        config.process = 'predict'
        config.columns_size = 2
        config.x_input_length = 32
        config.x_output_length = 16
        config.x_split_step = 16
        config.batch_size = 1
        config.max_iter = 0
        config.learning_rate = 0.0
        config.weight_decay = 0.0
        config.val_interval = 0
        config.val_iter = 0
        config.monitor_path = '.'
        config.training_dataset_path = None
        config.validation_dataset_path = None
        config.evaluation_dataset_path = None

        seed(0)
        if config.context is None:
            config.context = 'cpu'
        logger.info("Running in %s" % config.context)
        self._ctx = extension_context(config.context, device_id = config.device_id)
        nn.set_default_context(self._ctx)

        self._net_type = args.net
        logger.info("Network type is {}.".format(self._net_type))
        self._mlp = None
        self._lenet = None
        self._lstm = None

        nn.clear_parameters()
        if self._net_type == 'mlp':
            config.model_params_path = args.mlp_model_params_path
            if not os.path.isfile(config.model_params_path):
                logger.error("Model params path {} is not found.".format(config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(config.model_params_path))
            config.x_input_length = 64
            self._mlp = MLP(config)
            self._mlp.init_for_predict()
        elif self._net_type == 'lenet':
            config.model_params_path = args.lenet_model_params_path
            if not os.path.isfile(config.model_params_path):
                logger.error("Model params path {} is not found.".format(config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(config.model_params_path))
            config.width = 28
            config.height = 28
            self._lenet = LeNet(config)
            self._lenet.init_for_predict()
        elif self._net_type == 'mlp-with-lstm':
            config.model_params_path = args.lstm_model_params_path
            if not os.path.isfile(config.model_params_path):
                logger.error("Model params path {} is not found.".format(config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(config.model_params_path))
            self._lstm = LSTM(config)
            self._lstm.init_for_predict()
            config.model_params_path = args.mlp_model_params_path
            if not os.path.isfile(config.model_params_path):
                logger.error("Model params path {} is not found.".format(config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(config.model_params_path))
            config.x_input_length = 64
            self._mlp = MLP(config)
            self._mlp.init_for_predict()
        else:
            raise ValueError("Unknown network type {}".format(self._net_type))
        self._labels = None
        self._labels_path = args.labels_path
        if not os.path.isfile(self._labels_path):
            logger.error("Labels path {} is not found.".format(self._labels_path))
        else:
            logger.info("Path of the labels file is {}.".format(self._labels_path))
            with open(self._labels_path) as f:
                self._labels = f.readlines()
        self._points_buf = pointsbuffer.PointsBuffer()

    def _standardize(self, points):
        result = []
        for point in points:
            result.append((2.0 * point[0] / 256.0 - 1.0, 2.0 * point[1] / 256.0 - 1.0))
        return result

    def _normalize(self, points):
        result = []
        minx = float("inf")
        miny = float("inf")
        maxx = float("-inf")
        maxy = float("-inf")
        for point in points:
            minx = float(point[0]) if minx > point[0] else minx
            miny = float(point[1]) if miny > point[1] else miny
            maxx = float(point[0]) if maxx < point[0] else maxx
            maxy = float(point[1]) if maxy < point[1] else maxy
        width = maxx - minx
        height = maxy - miny
        midx = (maxx + minx) * 0.5
        midy = (maxy + miny) * 0.5
        scale = width if width > height else height
        for point in points:
            result.append(((point[0] - midx) / scale, (point[1] - midy) / scale))
        return result

    def _get_image(self, points):
        image = Image.new('L', (28, 28), (255))
        draw = ImageDraw.Draw(image)
        for i in range(len(points) - 1):
            x0 = int((points[i][0]+1.0) * 14.0)
            y0 = int((points[i][1]+1.0) * 14.0)
            x1 = int((points[i+1][0]+1.0) * 14.0)
            y1 = int((points[i+1][1]+1.0) * 14.0)
            x0 = 27 if x0 > 27 else (0 if x0 < 0 else x0)
            y0 = 27 if y0 > 27 else (0 if y0 < 0 else y0)
            x1 = 27 if x1 > 27 else (0 if x1 < 0 else x1)
            y1 = 27 if y1 > 27 else (0 if y1 < 0 else y1)
            draw.line((x0, y0, x1, y1), fill=0)
        return np.asarray(image)

    def get_network_type(self):
        return self._net_type

    # recognize a gesture
    def predict(self, points, stroke_terminal=True):
        result = None
        if self._net_type == 'mlp':
            self._points_buf.set_points(points)
            self._points_buf.adjust()
            points = self._points_buf.get_points()
            result = self._mlp.predict(np.asarray(points))
        elif self._net_type == 'lenet':
            points = self._normalize(points)
            image = self._get_image(points)
            result = self._lenet.predict(image/255.0)
        else:
            points = self._standardize(points)
            if not stroke_terminal :
                pred = self._lstm.predict(np.asarray(points[-32:])).tolist()
                points.extend(pred)
            self._points_buf.set_points(points)
            self._points_buf.adjust()
            points = self._points_buf.get_points()
            result = self._mlp.predict(np.asarray(points))
        if self._labels is not None:
            result = self._labels[result]
        return result

class GesturePainter:
    def __init__(self):
        self._minx = 0
        self._miny = 0
        self._maxx = 255
        self._maxy = 255
        self._output_dir = None
        self._x = 0
        self._y = 0
        self._points = []
        self._recognizer = None
        self._window = None
        self._canvas = None
        self._result_area = None
        self._result_txt = None
        self._initWindow()

    def _initWindow(self):
        self._window = tkinter.Tk()

        self._canvas = tkinter.Canvas(self._window, bg = "white", width = self._maxx + 1 - self._minx, height = self._maxy + 1 - self._miny)
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_pressed)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_released)
        self._canvas.bind("<B1-Motion>", self._on_canvas_dragged)

        quitBtn = tkinter.Button(self._window, text = "QUIT", command = self._window.quit)
        quitBtn.pack(side = tkinter.RIGHT)

        self._result_txt = tkinter.StringVar()
        self._result_txt.set("")

        self._result_area = tkinter.Label(self._window, textvariable = self._result_txt, anchor = tkinter.N, height = 1, font=2)
        self._result_area.pack(side = tkinter.LEFT)

    def _on_canvas_pressed(self, event):
        self._canvas.delete("all")
        self._points = []
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                self._points.append((event.x, event.y))
                self._x = event.x
                self._y = event.y

    def _on_canvas_released(self, event):
        result = self._recognizer.predict(self._points)
        self._result_txt.set("You drew " + result)

    def _on_canvas_dragged(self, event):
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                self._canvas.create_line(self._x, self._y, event.x, event.y, fill = "black", width=1)
                self._points.append((event.x, event.y))
                self._x = event.x
                self._y = event.y
                if self._recognizer.get_network_type() == 'mlp-with-lstm':
                    if len(self._points) % 32 == 0 :
                        result = self._recognizer.predict(self._points, False)
                        self._result_txt.set("You will draw " + result)

    def _on_clear_btn_pressed(self):
        self._canvas.delete("all")
        self._points = []

    def run(self, args):
        self._recognizer = GestureRecognizer(args)
        self._window.mainloop()

if __name__ == '__main__':
    args = get_args()
    GesturePainter().run(args)
    sys.exit(0)
