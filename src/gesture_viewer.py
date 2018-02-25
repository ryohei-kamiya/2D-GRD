# -*- coding:utf-8 -*-
import sys
import argparse
import os.path
import glob
import tkinter
import numpy as np
import pointsbuffer

def get_args(scale=1.0, shift=0.0, description=None):
    if description is None:
        description = "Gesture viewer"
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--scale", "-s", type=float, default=scale,
                        help='Scale value.')
    parser.add_argument("--shift", "-t", type=float, default=shift,
                        help='Shift value.')
    parser.add_argument("inputpath", type=str,
                        help='Path of an input file or a directory.')
    args = parser.parse_args()
    if not os.path.isfile(args.inputpath) and not os.path.isdir(args.inputpath):
        parser.print_help()
        sys.exit(1)
    return args

class GestureViewer:
    def __init__(self, config):
        self._minx = 0
        self._miny = 0
        self._maxx = 255
        self._maxy = 255
        self._window = None
        self._canvas = None
        self._clearBtn = None
        self._prevBtn = None
        self._nextBtn = None
        self._files = []
        self._fileno = 0
        self._points_buf = pointsbuffer.PointsBuffer()
        self._config = config
        self._initWindow()

    def _initWindow(self):
        self._window = tkinter.Tk()
        self._window.title("Gesture Viewer")

        self._canvas = tkinter.Canvas(self._window, bg = "white", width = self._maxx + 1 - self._minx, height = self._maxy + 1 - self._miny)
        self._canvas.pack()

        self._clearBtn = tkinter.Button(self._window, text = "CLEAR", command = self._on_clear_btn_pressed)
        self._clearBtn.pack(side = tkinter.LEFT)

        self._prevBtn = tkinter.Button(self._window, text = "PREV", command = self._on_prev_btn_pressed, state = tkinter.DISABLED)
        self._prevBtn.pack(side = tkinter.LEFT)

        self._nextBtn = tkinter.Button(self._window, text = "NEXT", command = self._on_next_btn_pressed, state = tkinter.DISABLED)
        self._nextBtn.pack(side = tkinter.LEFT)

        quitBtn = tkinter.Button(self._window, text = "QUIT", command = self._window.quit)
        quitBtn.pack(side = tkinter.RIGHT)

    def _load_points(self, filepath):
        data = np.loadtxt(filepath, delimiter=',')
        data = (data + self._config.shift) * self._config.scale
        return data.tolist()

    def _paint_points(self, points):
        if len(points) < 1:
            return None
        p0 = points[0]
        for i, p1 in enumerate(points):
            self._canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill = "black", width=1)
            p0 = p1

    def _on_clear_btn_pressed(self):
        self._canvas.delete("all")

    def _on_prev_btn_pressed(self):
        self._fileno -= 1
        if self._fileno < 0:
            self._fileno = 0
        points = self._load_points(self._files[self._fileno])
        self._paint_points(points)
        self._update_btn_state()

    def _on_next_btn_pressed(self):
        self._fileno += 1
        if self._fileno < 0:
            self._fileno = 0
        points = self._load_points(self._files[self._fileno])
        self._paint_points(points)
        self._update_btn_state()

    def _update_btn_state(self):
        if self._fileno < len(self._files) - 1:
            self._nextBtn.configure(state = tkinter.NORMAL)
        else:
            self._nextBtn.configure(state = tkinter.DISABLED)
        if self._fileno > 0:
            self._prevBtn.configure(state = tkinter.NORMAL)
        else:
            self._prevBtn.configure(state = tkinter.DISABLED)

    def run(self):
        if os.path.isdir(self._config.inputpath):
            self._files = glob.glob(os.path.join(self._config.inputpath, "*.csv"))
        elif os.path.isfile(self._config.inputpath):
            self._files.append(self._config.inputpath)
        self._update_btn_state()
        for file in self._files:
            points = self._load_points(file)
            self._paint_points(points)
        self._window.mainloop()

if __name__ == '__main__':
    args = get_args()
    GestureViewer(args).run()
    sys.exit(0)
