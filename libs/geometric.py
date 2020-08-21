"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from shapely.geometry import LineString, Point, Polygon
from collections import deque


def get_polygon(point_list):
    return Polygon(point_list)


def get_line(data):
    return LineString(data)


def get_point(data):
    return Point(data)


def get_distance(l, p):
    return p.distance(l)


def get_x(y , a, k):
    return (y - k) / a


def get_y(x, a, k):
    return a * x + k


