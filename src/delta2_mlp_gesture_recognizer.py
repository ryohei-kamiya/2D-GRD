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
import os.path
import tkinter
import pointsbuffer

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.logger as logger
from nnabla.ext_utils import get_extension_context

import numpy as np
from numpy.random import seed


def get_args(mlp_model_params_path="mlp-parameters.h5", labels_path="labels.txt",
            x_length=64, description=None):
    if description is None:
        description = "Gesture recognizer"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--mlp-model-params-path", "-mlp",
                        type=str, default=mlp_model_params_path,
                        help='Path of the mlp model parameters file.')
    parser.add_argument("--labels-path", "-l",
                        type=str, default=labels_path,
                        help='Path of the labels file.')
    parser.add_argument("--x-length", "-xl", type=int, default=x_length,
                        help='Length of points into the mlp network.')
    args = parser.parse_args()
    return args


class MLP:
    def __init__(self, config):
        self._cols_size = config.columns_size
        self._x_length = config.x_length
        self._model_params_path = config.mlp_model_params_path
        self._x = nn.Variable([1, self._x_length, self._cols_size])
        self._pred = self.network(self._x)
        nn.load_parameters(self._model_params_path)

    def network(self, x):
        # Input -> 64,2
        # Affine -> 100
        with nn.parameter_scope('Affine'):
            h = PF.affine(x, (100,))
        # Tanh
        h = F.tanh(h)
        # Affine_2 -> 26
        with nn.parameter_scope('Affine_2'):
            h = PF.affine(h, (26,))
        return h

    # Inference
    def infer(self, points):
        if points.ndim == 2:
            points = np.reshape(points, (1, points.shape[0], -1))
        if points.ndim != 3:
            return None
        if points.shape[1] < self._x_length:
            return None
        self._x.d = points[:1,:,:]
        self._pred.forward(clear_buffer=True)
        return self._pred.d.argmax(1)[0]


class GestureRecognizer:
    def __init__(self, args):
        class Config:
            pass
        self._config = Config()
        self._config.context = 'cpu'
        self._config.device_id = 0
        self._config.columns_size = 2
        self._config.x_length = args.x_length
        self._config.mlp_model_params_path = args.mlp_model_params_path
        self._config.labels_path = args.labels_path

        if not os.path.isfile(self._config.mlp_model_params_path):
            logger.error("Model params path {} is not found.".format(self._config.mlp_model_params_path))
            sys.exit(-1)
        else:
            logger.info("Path of the model parameters file is {}.".format(self._config.mlp_model_params_path))
        if not os.path.isfile(self._config.labels_path):
            logger.error("Labels path {} is not found.".format(self._config.labels_path))
            sys.exit(-1)
        else:
            logger.info("Path of the labels file is {}.".format(self._config.labels_path))

        seed(0)
        logger.info("Running in %s" % self._config.context)
        self._ctx = get_extension_context(self._config.context, device_id = self._config.device_id)
        nn.set_default_context(self._ctx)
        nn.clear_parameters()
        self._mlp = MLP(self._config)
        self._labels = None
        with open(self._config.labels_path) as f:
            self._labels = f.readlines()
        self._points_buf = pointsbuffer.PointsBuffer()

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

    # recognize a gesture
    def infer(self, points, stroke_terminal=True):
        result_label = None
        self._points_buf.set_points(points)
        self._points_buf.adjust()
        points = self._points_buf.get_points()
        result_label = self._mlp.infer(np.asarray(points))
        if self._labels is not None:
            result_label = self._labels[result_label]
        return result_label

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
        self._window.title('MLP Gesture Recognizer')

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

    def _draw_line(self, p0, p1, color, width=1):
        self._canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill = color, width=width)

    def _on_canvas_pressed(self, event):
        self._canvas.delete("all")
        self._points = []
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                self._x = event.x
                self._y = event.y

    def _on_canvas_dragged(self, event):
        if self._minx <= event.x and event.x <= self._maxx:
            if self._miny <= event.y and event.y <= self._maxy:
                p0 = [self._x, self._y]
                p1 = [event.x, event.y]
                points = self._fulfill_interpolation([p0, p1])[:-1]
                self._points.extend(points)
                self._x = event.x
                self._y = event.y
                self._draw_line(p0, p1, color="black", width=2)

    def _on_canvas_released(self, event):
        result_label = self._recognizer.infer(self._points)
        self._result_txt.set("You drew " + result_label)

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
