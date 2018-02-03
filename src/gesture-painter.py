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
import os.path
import tkinter
import datetime

def print_usage():
    print("python {} /path/to/output/directory".format(argv[0]), file=sys.stderr)

class GesturePainter:
    def __init__(self):
        self._minx = 0
        self._miny = 0
        self._maxx = 255
        self._maxy = 255
        self._output_dir = None
        self._x = 0
        self._y = 0
        self._window = None
        self._canvas = None
        self._points = None
        self._fileno = 0
        self._initWindow()

    def _initWindow(self):
        self._window = tkinter.Tk()

        self._canvas = tkinter.Canvas(self._window, bg = "white", width = self._maxx + 1 - self._minx, height = self._maxy + 1 - self._miny)
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_pressed)
        self._canvas.bind("<B1-Motion>", self._on_canvas_dragged)

        quitBtn = tkinter.Button(self._window, text = "QUIT", command = self._window.quit)
        quitBtn.pack(side = tkinter.RIGHT)

        clearBtn = tkinter.Button(self._window, text = "CLEAR", command = self._on_clear_btn_pressed)
        clearBtn.pack(side = tkinter.LEFT)

        saveBtn = tkinter.Button(self._window, text = "SAVE", command = self._on_save_btn_pressed)
        saveBtn.pack(side = tkinter.LEFT)

    def _on_canvas_pressed(self, event):
        self._canvas.delete("all")
        self._points = []
        if self._minx > event.x:
            self._x = self._minx
        elif self._maxx < event.x:
            self._x = self._maxx
        else:
            self._x = event.x
        if self._miny > event.y:
            self._y = self._miny
        elif self._maxy < event.y:
            self._y = self._maxy
        else:
            self._y = event.y
        self._points.append((self._x, self._y))

    def _on_canvas_dragged(self, event):
        x = None
        y = None
        if self._minx > event.x:
            x = self._minx
        elif self._maxx < event.x:
            x = self._maxx
        else:
            x = event.x
        if self._miny > event.y:
            y = self._miny
        elif self._maxy < event.y:
            y = self._maxy
        else:
            y = event.y
        self._canvas.create_line(self._x, self._y, x, y, fill = "black", width=1)
        self._x = x
        self._y = y
        self._points.append((self._x, self._y))

    def _on_clear_btn_pressed(self):
        self._canvas.delete("all")
        self._points = []

    def _on_save_btn_pressed(self):
        now = datetime.datetime.now()
        unixtime = now.timestamp()
        filepath = os.path.join(self._output_dir, "{}-{}.csv".format(self._fileno, int(unixtime)))
        self._fileno += 1
        self._output(filepath)
        self._on_clear_btn_pressed()

    def _output(self, filepath):
        with open(filepath, "w") as f:
            for point in self._points:
                print("{},{}".format(point[0], point[1]), file=f)

    def run(self, output_dir):
        self._output_dir = output_dir
        self._window.mainloop()

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    output_dir = "."
    if argc >= 2:
        output_dir = argv[1]
    if not os.path.isdir(output_dir):
        print_usage()
        sys.exit(1)
    GesturePainter().run(output_dir)
    sys.exit(0)
