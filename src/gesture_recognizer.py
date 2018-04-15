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
from nnabla.ext_utils import get_extension_context

import numpy as np
from numpy.random import seed

from mlp import MLP
from lenet import LeNet
from lstm_with_baseshift import LSTM

def get_args(mlp_model_params_path="mlp-parameters.h5", lenet_model_params_path="lenet-parameters.h5",
            lstm_model_params_path="lstm-parameters.h5", labels_path="labels.txt",
            x_length=64, x_input_length=16, x_output_length=1, x_split_step=1, width=28, height=28,
            lstm_unit_name="LSTM", lstm_units=32, description=None):
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
    parser.add_argument("--x-length", "-xl", type=int, default=x_length,
                        help='Length of time-series into the mlp network.')
    parser.add_argument("--x-input-length", "-xil", type=int, default=x_input_length,
                        help='Length of time-series into the lstm network.')
    parser.add_argument("--x-output-length", "-xol", type=int, default=x_output_length,
                        help='Length of time-series from the lstm network.')
    parser.add_argument("--x-split-step", "-xss", type=int, default=x_split_step,
                        help='Step size to split time-series.')
    parser.add_argument("--width", "-wt", type=int, default=width,
                        help='Image width.')
    parser.add_argument("--height", "-ht", type=int, default=height,
                        help='Image height.')
    parser.add_argument("--lstm-unit-name", "-lstmun", type=str, default=lstm_unit_name,
                        help='LSTM unit name.')
    parser.add_argument("--lstm-units", "-lstmu", type=int, default=lstm_units,
                        help='The number of LSTM units.')
    args = parser.parse_args()
    return args

class GestureRecognizer:
    def __init__(self, args):
        class Config:
            pass
        self._config = Config()
        self._config.context = args.context
        self._config.device_id = args.device_id
        self._config.process = 'infer'
        self._config.columns_size = 2
        self._config.x_length = args.x_length
        self._config.x_input_length = args.x_input_length
        self._config.x_output_length = args.x_output_length
        self._config.x_split_step = args.x_split_step
        self._config.width = args.width
        self._config.height = args.height
        self._config.lstm_unit_name = args.lstm_unit_name
        self._config.lstm_units = args.lstm_units
        self._config.batch_size = 1
        self._config.max_iter = 0
        self._config.learning_rate = 0.0
        self._config.weight_decay = 0.0
        self._config.val_interval = 0
        self._config.val_iter = 0
        self._config.monitor_path = '.'
        self._config.training_dataset_path = None
        self._config.validation_dataset_path = None
        self._config.evaluation_dataset_path = None

        seed(0)
        if self._config.context is None:
            self._config.context = 'cpu'
        logger.info("Running in %s" % self._config.context)
        self._ctx = get_extension_context(self._config.context, device_id = self._config.device_id)
        nn.set_default_context(self._ctx)

        self._net_type = args.net
        logger.info("Network type is {}.".format(self._net_type))
        self._mlp = None
        self._lenet = None
        self._lstm = None

        nn.clear_parameters()
        if self._net_type == 'mlp':
            self._config.model_params_path = args.mlp_model_params_path
            if not os.path.isfile(self._config.model_params_path):
                logger.error("Model params path {} is not found.".format(self._config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(self._config.model_params_path))
            self._mlp = MLP(self._config)
            self._mlp.init_for_infer()
        elif self._net_type == 'lenet':
            self._config.model_params_path = args.lenet_model_params_path
            if not os.path.isfile(self._config.model_params_path):
                logger.error("Model params path {} is not found.".format(self._config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(self._config.model_params_path))
            self._lenet = LeNet(self._config)
            self._lenet.init_for_infer()
        elif self._net_type == 'mlp-with-lstm':
            self._config.model_params_path = args.lstm_model_params_path
            if not os.path.isfile(self._config.model_params_path):
                logger.error("Model params path {} is not found.".format(self._config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(self._config.model_params_path))
            self._lstm = LSTM(self._config)
            self._lstm.init_for_infer()
            self._config.model_params_path = args.mlp_model_params_path
            if not os.path.isfile(self._config.model_params_path):
                logger.error("Model params path {} is not found.".format(self._config.model_params_path))
            else:
                logger.info("Path of the model parameters file is {}.".format(self._config.model_params_path))
            self._mlp = MLP(self._config)
            self._mlp.init_for_infer()
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

    def _diff_points(self, points):
        result = []
        if len(points) > 1:
            p0 = points[0]
            for p1 in points[1:]:
                result.append([p1[0] - p0[0], p1[1] - p0[1]])
                p0 = p1
        return result

    def _undiff_points(self, diff_points, start_point=[0, 0]):
        result = []
        if len(diff_points) > 0:
            p0 = start_point
            for dp in diff_points:
                p1 = [p0[0]+dp[0], p0[1]+dp[1]]
                result.append(p1)
                p0 = p1
        return result

    def _subtract_point_from_points(self, points, point):
        return [[p[0]-point[0], p[1]-point[1]] for p in points]

    def _add_point_to_points(self, points, point):
        return [[p[0]+point[0], p[1]+point[1]] for p in points]

    def _standardize(self, points):
        result = []
        for point in points:
            result.append((point[0] / 128.0 - 1.0, point[1] / 128.0 - 1.0))
        return result

    def _unstandardize(self, points):
        result = []
        for point in points:
            result.append(((point[0] + 1.0) * 128.0, (point[1] + 1.0) * 128.0))
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
            result.append(((point[0] - midx) * 0.8 / scale, (point[1] - midy) * 0.8 / scale))
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
    def infer(self, points, stroke_terminal=True):
        result_label = None
        result_points = []
        if self._net_type == 'mlp':
            self._points_buf.set_points(points)
            self._points_buf.adjust()
            points = self._points_buf.get_points()
            result_label = self._mlp.infer(np.asarray(points))
        elif self._net_type == 'lenet':
            points = self._normalize(points)
            image = self._get_image(points)
            result_label = self._lenet.infer(image/255.0)
        else:
            if not stroke_terminal :
                tmp_points = self._standardize(points)
                for i in range(0, self._config.x_input_length, self._config.x_split_step):
                    xin = self._subtract_point_from_points(
                            tmp_points[-self._config.x_input_length:],
                            tmp_points[-1])
                    xout = self._lstm.infer(np.asarray(xin)).tolist()
                    pred = self._add_point_to_points(
                            xout, tmp_points[-1])
                    result_points.extend(pred)
                    tmp_points.extend(pred)
                result_points = self._unstandardize(result_points)
                points = self._unstandardize(tmp_points)
            self._points_buf.set_points(points)
            self._points_buf.adjust()
            points = self._points_buf.get_points()
            result_label = self._mlp.infer(np.asarray(points))
        if self._labels is not None and result_label is not None:
            result_label = self._labels[result_label]
        return (result_label, result_points)

class GesturePainter:
    def __init__(self, args):
        self._config = args

        self._minx = 0
        self._miny = 0
        self._maxx = 255
        self._maxy = 255
        self._output_dir = None
        self._x = 0
        self._y = 0
        self._points = []
        self._pred_points = []
        self._recognizer = None
        self._window = None
        self._canvas = None
        self._result_area = None
        self._result_txt = None
        self._points_buf = pointsbuffer.PointsBuffer()
        self._initWindow()

    def _initWindow(self):
        self._window = tkinter.Tk()
        self._window.title('Gesture Recognizer')

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

    def _fulfill_interpolation(self, points):
        self._points_buf.set_points(points)
        self._points_buf.fulfill_linear_interpolation()
        return self._points_buf.get_points()

    def _on_canvas_pressed(self, event):
        self._canvas.delete("all")
        self._points = []
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                self._x = event.x
                self._y = event.y

    def _on_canvas_released(self, event):
        result_label, _ = self._recognizer.infer(self._points)
        if result_label is not None:
            self._result_txt.set("You drew " + result_label)

    def _draw_line(self, p0, p1, color, width=1):
        self._canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill = color, width=width)

    def _on_canvas_dragged(self, event):
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                p0 = [self._x, self._y]
                p1 = [event.x, event.y]
                points = self._fulfill_interpolation([p0, p1])[:-1]
                self._points.extend(points)
                self._x = event.x
                self._y = event.y
                if self._recognizer.get_network_type() == 'mlp-with-lstm':
                    if len(self._points) % self._config.x_split_step == 0 and \
                            len(self._points) >= self._config.x_input_length:
                        result_label, result_points = self._recognizer.infer(self._points, False)
                        if len(self._pred_points) > 1:
                            p2 = self._pred_points[0]
                            for p3 in self._pred_points[1:]:
                                self._draw_line(p2, p3, color="white", width=2)
                                p2 = p3
                        if len(result_points) > 1:
                            p2 = result_points[0]
                            for p3 in result_points[1:]:
                                self._draw_line(p2, p3, color="red", width=2)
                                p2 = p3
                        self._pred_points = result_points
                        if result_label is not None:
                            self._result_txt.set("You will draw " + result_label)
                self._draw_line(p0, p1, color="black", width=2)

    def _on_clear_btn_pressed(self):
        self._canvas.delete("all")
        self._points = []

    def run(self):
        self._recognizer = GestureRecognizer(self._config)
        self._window.mainloop()

if __name__ == '__main__':
    args = get_args()
    GesturePainter(args).run()
    sys.exit(0)
