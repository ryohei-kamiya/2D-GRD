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
import enum
import math
import numpy as np

class PointSearchCondition(enum.Enum):
    MOST_FAR_POINT = 1
    MOST_NEAR_POINT = 2
    SAME_POINT = 3
    MIN_ENTROPY_POINT = 4

class LineSearchCondition(enum.Enum):
    MAX_LENGTH_LINE = 1
    MIN_LENGTH_LINE = 2
    ZERO_LENGTH_LINE = 3

class TriangleSearchCondition(enum.Enum):
    ZERO_SIZE_TRIANGLE = 1

class PointsBuffer:
    def __init__(self):
        self._adjusted_scale = 1.0
        self._adjusted_length = 64
        self._center = [0.0, 0.0]
        self._points = []

    def set_adjusted_scale(self, scale):
        self._adjusted_scale = scale

    def set_adjusted_length(self, length):
        self._adjusted_length = length

    def set_points(self, points):
        self._points = points

    def append(self, x, y):
        self._points.append([x, y])

    def extend(self, points):
        self._points.extend(points)

    def get_point(self, index):
        if index >= self.length():
            return None
        return self._points[index]

    def get_points(self):
        return self._points

    def remove_point(self, index):
        if index >= self.length():
            return None
        return self._points.pop(index)

    def length(self):
        return len(self._points)

    def clear(self):
        self._points = []

    def output(self, filepath):
        with open(filepath, "w") as f:
            for point in self._points:
                print("{},{}".format(point[0], point[1]), file=f)

    def centroid(self):
        points_length = self.length()
        if points_length < 1 :
            return None
        centroid_x = 0.0
        centroid_y = 0.0
        for point in self._points:
            centroid_x += point[0]
            centroid_y += point[1]
        centroid_x /= points_length
        centroid_y /= points_length
        return [centroid_x, centroid_y]

    def distance2(self, point1, point2):
        return (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2

    def triangle_area(self, point1, point2, point3):
        v0 = [point2[0] - point1[0], point2[1] - point1[1]]
        v1 = [point3[0] - point1[0], point3[1] - point1[1]]
        return math.fabs(v0[0] * v1[1] - v0[1] * v1[0]) * 0.5

    def _search_most_far_point_index(self, point):
        index = None
        max_distance = 0.0
        for i, p in enumerate(self._points):
            distance = self.distance2(point, p)
            if max_distance < distance:
                index = i
                max_distance = distance
        return index

    def _search_most_near_point_index(self, point):
        index = None
        min_distance = float("inf")
        for i, p in enumerate(self._points):
            distance = self.distance2(point, p)
            if distance != 0.0 and min_distance > distance:
                index = i
                min_distance = distance
        return index

    def _search_same_point_index(self, point):
        index = None
        for i, p in enumerate(self._points):
            if p[0] == point[0] and p[1] == point[1]:
                index = i
                break
        return index

    def _search_min_entropy_point_index(self):
        points_length = self.length()
        if points_length < 3:
            return None
        index = None
        min_area = float("inf")
        for i in range(1, points_length - 1):
            area = self.triangle_area(self._points[i - 1], self._points[i], self._points[i + 1])
            if area < min_area:
                index = i
                min_area = area
            elif area == min_area:
                d1 = self.distance2(self._points[index - 1], self._points[index])
                d1 += self.distance2(self._points[index], self._points[index + 1])
                d1 += self.distance2(self._points[index - 1], self._points[index + 1])
                d2 = self.distance2(self._points[i - 1], self._points[i])
                d2 += self.distance2(self._points[i], self._points[i + 1])
                d2 += self.distance2(self._points[i - 1], self._points[i + 1])
                if d1 > d2:
                    index = i
        return index

    def search_point_index(self, condition, point = None):
        if not isinstance(condition, PointSearchCondition):
            return None

        if condition is PointSearchCondition.MIN_ENTROPY_POINT:
            return self._search_min_entropy_point_index()

        if not isinstance(point, tuple):
            point = (0, 0)
        if condition is PointSearchCondition.MOST_FAR_POINT:
            return self._search_most_far_point_index(point)
        elif condition is PointSearchCondition.MOST_NEAR_POINT:
            return self._search_most_near_point_index(point)
        elif condition is PointSearchCondition.SAME_POINT:
            return self._search_same_point_index(point)
        return None

    def _search_max_length_line_index(self):
        index = None
        max_distance = 0.0
        p0 = self._points[0]
        for i in range(1, self.length()):
            p1 = self._points[i]
            distance = self.distance2(p0, p1)
            if max_distance < distance:
                index = i
                max_distance = distance
            p0 = p1
        if index is not None:
            return index - 1
        return None

    def _search_min_length_line_index(self):
        index = None
        min_distance = float("inf")
        p0 = self._points[0]
        for i in range(1, self.length()):
            p1 = self._points[i]
            distance = self.distance2(p0, p1)
            if distance != 0.0 and min_distance > distance:
                index = i
                min_distance = distance
            p0 = p1
        if index is not None:
            return index - 1
        return None

    def _search_zero_length_line_index(self):
        index = None
        p0 = self._points[0]
        for i in range(1, self.length()):
            p1 = self._points[i]
            if p0[0] == p1[0] and p0[1] == p1[1]:
                index = i
                break
            p0 = p1
        if index is not None:
            return index - 1
        return None

    def search_line_index(self, condition):
        if not isinstance(condition, LineSearchCondition):
            return None
        if condition is LineSearchCondition.MAX_LENGTH_LINE:
            return self._search_max_length_line_index()
        elif condition is LineSearchCondition.MIN_LENGTH_LINE:
            return self._search_min_length_line_index()
        elif condition is LineSearchCondition.ZERO_LENGTH_LINE:
            return self._search_zero_length_line_index()
        return None

    def _search_zero_size_triangle_index(self):
        index = None
        p0 = self._points[0]
        p1 = self._points[1]
        for i in range(2, self.length()):
            p2 = self._points[i]
            area = self.triangle_area(p0, p1, p2)
            if area == 0.0:
                index = i
                break
            p0 = p1
            p1 = p2
        if index is not None:
            return index - 1
        return None

    def search_triangle_index(self, condition):
        if not isinstance(condition, TriangleSearchCondition):
            return None
        if condition is TriangleSearchCondition.ZERO_SIZE_TRIANGLE:
            return self._search_zero_size_triangle_index()
        return None

    def interpolate_with_line_index(self, index):
        points_length = self.length()
        if points_length < 2:
            return -2
        elif points_length < 4:
            x = (self._points[index + 1][0] + self._points[index][0]) * 0.5
            y = (self._points[index + 1][1] + self._points[index][1]) * 0.5
            self._points.insert(index, [x, y])
            return 0
        p0 = None
        p1 = None
        p2 = None
        p3 = None
        if index == 0:
            p0 = self._points[index]
            p1 = self._points[index]
            p2 = self._points[index + 1]
            p3 = self._points[index + 2]
        elif index == points_length - 2:
            p0 = self._points[index - 1]
            p1 = self._points[index]
            p2 = self._points[index + 1]
            p3 = self._points[index + 1]
        else:
            p0 = self._points[index - 1]
            p1 = self._points[index]
            p2 = self._points[index + 1]
            p3 = self._points[index + 2]
        v0 = np.array([0.0, 0.0])
        d0 = math.sqrt(self.distance2(p0, p2))
        if d0 == 0.0:
            return -1
        v0[0] = (p2[0] - p0[0])
        v0[1] = (p2[1] - p0[1])
        v0[0] /= d0
        v0[1] /= d0
        v1 = np.array([0.0, 0.0])
        d1 = math.sqrt(self.distance2(p1, p3))
        if d1 == 0.0:
            return -1
        v1[0] = (p3[0] - p1[0])
        v1[1] = (p3[1] - p1[1])
        v1[0] /= d1
        v1[1] /= d1
        line_length = self.distance2(p1, p2)
        line_length = math.sqrt(line_length)
        dts = np.array([0.5**3, 0.5**2, 0.5, 1.0])
        H   = np.array([
            [ 2.0, -2.0,  1.0,  1.0],
            [-3.0,  3.0, -2.0, -1.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 1.0,  0.0,  0.0,  0.0]
        ])
        G   = np.array([
            p1,
            p2,
            line_length * v0,
            line_length * v1
        ])
        result = np.dot(np.dot(dts, H), G)
        self._points.insert(index + 1, [result[0], result[1]])
        return 0

    def change_scale(self, scale):
        points_length = self.length()
        if points_length < 2 :
            return -1
        centroid = self.centroid()
        if centroid is not None:
            most_far_point_index = self.search_point_index(PointSearchCondition.MOST_FAR_POINT, centroid)
            if most_far_point_index is not None:
                d = self.distance2(centroid, self._points[most_far_point_index])
                if d is not None and d > 0.0:
                    d = math.sqrt(d)
                    points = []
                    for (i, p) in enumerate(self._points):
                        x = (p[0] - centroid[0]) * scale / d + self._center[0]
                        y = (p[1] - centroid[1]) * scale / d + self._center[1]
                        points.append([x, y])
                    self._points = points
                    return 0
        return -1

    def change_length(self, length):
        points_length = self.length()
        if points_length < 2:
            return -1
        index = self.search_line_index(LineSearchCondition.ZERO_LENGTH_LINE)
        while index is not None:
            self.remove_point(index)
            index = self.search_line_index(LineSearchCondition.ZERO_LENGTH_LINE)
        index = self.search_triangle_index(TriangleSearchCondition.ZERO_SIZE_TRIANGLE)
        while index is not None:
            self.remove_point(index)
            index = self.search_triangle_index(TriangleSearchCondition.ZERO_SIZE_TRIANGLE)
        points_length = self.length()
        if points_length < 2:
            return -1
        elif points_length == length:
            return 0
        elif points_length < length:
            while points_length < length:
                index = self.search_line_index(LineSearchCondition.MAX_LENGTH_LINE)
                if index is None:
                    return -1
                ret = self.interpolate_with_line_index(index)
                if ret == -1:
                    index = self.search_triangle_index(TriangleSearchCondition.ZERO_SIZE_TRIANGLE)
                    while index is not None:
                        self.remove_point(index)
                        index = self.search_triangle_index(TriangleSearchCondition.ZERO_SIZE_TRIANGLE)
                if ret == -2:
                    return -1
                points_length = self.length()
            return 0
        else:
            while points_length > length:
                index = self.search_point_index(PointSearchCondition.MIN_ENTROPY_POINT)
                if index is None:
                    return -1
                self.remove_point(index)
                points_length = self.length()
            return 0
        return -1

    def adjust_scale(self):
        return self.change_scale(self._adjusted_scale)

    def adjust_length(self):
        return self.change_length(self._adjusted_length)

    def adjust(self):
        ret = self.adjust_scale()
        if ret != 0:
            return ret
        ret = self.adjust_length()
        if ret != 0:
            return ret
