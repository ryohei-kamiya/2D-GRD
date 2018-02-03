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
import os
import os.path
import shutil
import glob
import csv
import random
import math
import numpy as np
from PIL import Image, ImageDraw

import pointsbuffer

def print_usage():
    print("python {} /path/to/input/directory /path/to/output/directory".format(argv[0]), file=sys.stderr)

def make_dir(dirpath):
    try:
        os.mkdir(dirpath)
    except OSError:
        pass

class DatasetMaker:
    def __init__(self):
        self._minx = 0.0
        self._miny = 0.0
        self._maxx = 255.0
        self._maxy = 255.0
        self._sigma = 32.0
        self._factor = 1.0
        self._homography_distort_range = 32.0
        self._distortion_level0_repeat = 32
        self._distortion_level1_repeat = 32
        self._points_buf = pointsbuffer.PointsBuffer()
        random.seed()

    def _load_points(self, filepath):
        points = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                points.append(tuple(map(float, row)))
        return points

    def _save_points(self, points, filepath):
        if len(points) < 1:
            return None
        with open(filepath, "w") as f:
            for point in points:
                print("{},{}".format(point[0], point[1]), file=f)

    def _save_image(self, points, filepath):
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
        image.save(filepath)

    def _homography_distort(self, points):
        x1 = float(self._minx) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        x2 = float(self._maxx) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        x3 = float(self._maxx) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        x4 = float(self._minx) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        y1 = float(self._miny) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        y2 = float(self._miny) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        y3 = float(self._maxy) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        y4 = float(self._maxy) + (random.random() - 0.5) * self._homography_distort_range * 0.5
        X1 = float(self._minx)
        X2 = float(self._maxx)
        X3 = float(self._maxx)
        X4 = float(self._minx)
        Y1 = float(self._miny)
        Y2 = float(self._miny)
        Y3 = float(self._maxy)
        Y4 = float(self._maxy)
        width = float(self._maxx - self._minx + 1)
        height = float(self._maxy - self._miny + 1)

        sx = x1 - x2 + x3 - x4
        sy = y1 - y2 + y3 - y4
        dx1 = x2 - x3
        dy1 = y2 - y3
        dx2 = x4 - x3
        dy2 = y4 - y3

        z = dx1 * dy2 - dy1 * dx2
        g = (sx * dy2 - sy * dx2) / z
        h = (sy * dx1 - sx * dy1) / z

        m0 = x2 - x1 + g * x2
        m1 = x4 - x1 + h * x4
        m2 = x1
        m3 = y2 - y1 + g * y2
        m4 = y4 - y1 + h * y4
        m5 = y1
        m6 = g
        m7 = h

        minx = float("inf")
        miny = float("inf")
        maxx = float("-inf")
        maxy = float("-inf")
        distorted_points = []
        for point in points:
            x = point[0]
            y = point[1]
            u = x / width
            v = y / height
            t = m6 * u * m7 * v + 1.0
            X = (m0 * u + m1 * v + m2) / t
            Y = (m3 * u + m4 * v + m5) / t
            if minx > X:
                minx = X
            if maxx < X:
                maxx = X
            if miny > Y:
                miny = Y
            if maxy < Y:
                maxy = Y
            distorted_points.append((X, Y))
        if minx < self._minx or self._maxx < maxx or miny < self._miny or self._maxy < maxy:
            center = ((self._maxx - self._minx) * 0.5, (self._maxy - self._miny) * 0.5)
            overs = [center[0] - minx, maxx - center[0], center[1] - miny, maxy - center[1]]
            limits = [center[0], center[0], center[1], center[1]]
            max_index = overs.index(max(overs))
            scale = limits[max_index] / overs[max_index]
            tmp_points = []
            for point in distorted_points:
                tmp_points.append((scale * (point[0] - center[0]) + center[0], scale * (point[1] - center[1]) + center[1]))
            distorted_points = tmp_points
        results = []
        for point in distorted_points:
            X = point[0]
            Y = point[1]
            if self._minx > X :
                X = self._minx
            if self._maxx < X :
                X = self._maxx
            if self._miny > Y :
                Y = self._miny
            if self._maxy < Y :
                Y = self._maxy
            results.append((X, Y))
        return results

    def _gauss_distort(self, points):
        results = []
        cx = random.randrange(self._minx, self._maxx + 1)
        cy = random.randrange(self._miny, self._maxy + 1)
        for point in points:
            x = point[0] - float(cx)
            y = point[1] - float(cy)
            d = - self._factor * math.exp(-(x**2 + y**2)/(2.0 * self._sigma**2))
            dx = d * x
            dy = d * y
            results.append((point[0] + dx, point[1] + dy))
        return results

    def _multi_distortion(self, points, num):
        results = []
        for i in range(num):
            distorted_points1 = self._gauss_distort(points)
            distorted_points2 = self._homography_distort(distorted_points1)
            results.append(distorted_points2)
        return results

    def _standardize(self, points):
        result = []
        for point in points:
            result.append((2.0 * point[0] / (self._maxx - self._minx + 1) - 1.0, 2.0 * point[1] / (self._maxy - self._miny + 1) - 1.0))
        return result

    def _normalize(self, points):
        result = []
        minx = float("inf")
        miny = float("inf")
        maxx = float("-inf")
        maxy = float("-inf")
        for point in points:
            minx = point[0] if minx > point[0] else minx
            miny = point[1] if miny > point[1] else miny
            maxx = point[0] if maxx < point[0] else maxx
            maxy = point[1] if maxy < point[1] else maxy
        width = maxx - minx
        height = maxy - miny
        scale = width if width > height else height
        for point in points:
            result.append((2.0 * point[0] / scale - 1.0, 2.0 * point[1] / scale - 1.0))
        return result

    def run(self, input_dir, output_dir):
        points_dir = os.path.join(output_dir, 'points')
        image_dir = os.path.join(output_dir, 'image')
        adjusted_points_dir = os.path.join(output_dir, 'adjusted_points')
        adjusted_image_dir = os.path.join(output_dir, 'adjusted_image')
        make_dir(points_dir)
        make_dir(image_dir)
        make_dir(adjusted_points_dir)
        make_dir(adjusted_image_dir)
        for dname in os.listdir(input_dir):
            i_dname = os.path.join(input_dir, dname)
            if os.path.isdir(i_dname):
                print("processing {}".format(dname))
                o_points_dname = os.path.join(points_dir, dname)
                o_image_dname = os.path.join(image_dir, dname)
                o_adjusted_points_dname = os.path.join(adjusted_points_dir, dname)
                o_adjusted_image_dname = os.path.join(adjusted_image_dir, dname)
                make_dir(o_points_dname)
                make_dir(o_image_dname)
                make_dir(o_adjusted_points_dname)
                make_dir(o_adjusted_image_dname)
                i_files = glob.glob(os.path.join(i_dname, '*.*'))
                for i_file in i_files:
                    points = self._load_points(i_file)
                    points_list = [points]
                    points_list0 = self._multi_distortion(points_list[0], self._distortion_level0_repeat)
                    points_list.extend(points_list0)
                    points_list1 = []
                    for points in points_list0:
                        points_list1.extend(self._multi_distortion(points, self._distortion_level1_repeat))
                    points_list.extend(points_list1)
                    fname, ext = os.path.splitext(os.path.basename(i_file))
                    fnum = 0
                    for points in points_list:
                        standardized_points = self._standardize(points)
                        o_points_fname = os.path.join(o_points_dname, "{0}-{1:04d}.csv".format(fname, fnum))
                        o_image_fname = os.path.join(o_image_dname, "{0}-{1:04d}.png".format(fname, fnum))
                        self._save_points(standardized_points, o_points_fname)
                        self._save_image(standardized_points, o_image_fname)
                        o_adjusted_points_fname = os.path.join(o_adjusted_points_dname, "{0}-{1:04d}.csv".format(fname, fnum))
                        o_adjusted_image_fname = os.path.join(o_adjusted_image_dname, "{0}-{1:04d}.png".format(fname, fnum))
                        self._points_buf.set_points(standardized_points)
                        self._points_buf.adjust()
                        adjusted_points = self._points_buf.get_points()
                        self._save_points(adjusted_points, o_adjusted_points_fname)
                        self._save_image(adjusted_points, o_adjusted_image_fname)
                        fnum += 1

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    if argc < 3:
        print_usage()
        sys.exit(1)
    if not os.path.isdir(argv[1]):
        print_usage()
        sys.exit(1)
    if os.path.isdir(argv[2]):
        shutil.rmtree(argv[2])
    make_dir(argv[2])
    DatasetMaker().run(argv[1], argv[2])
    sys.exit(0)
