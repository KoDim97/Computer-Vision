from math import sin, cos, atan
import config
import cv2


def point2line(p1, p2):
    a = atan(- (p1[0] - p2[0]) / (p1[1] - p2[1]))
    d = p1[1] * sin(a) + p1[0] * cos(a)
    return a, d


def intersect_line(a1, d1, a2, d2):
    x = (d1 * sin(a2) - d2 * sin(a1)) / (sin(a2)*cos(a1) - sin(a1)*cos(a2))
    y = (d1 - x * cos(a1)) / sin(a1)
    return x, y


def distance_from_focus_to_line(x, y, a, d):
    return abs(y * sin(a) + x * cos(a) -d)


def create_table_template():
    table_template = cv2.imread(config.PATH_TO_TEMPLATE_TABLE_IMG)
    table_template = cv2.cvtColor(table_template, cv2.COLOR_BGR2RGB)
    table_key_point = [[409, 1793], [1653, 1801], [1609, 3041],[397, 3457]]
    return table_template, table_key_point
