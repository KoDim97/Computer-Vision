import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny
from skimage.morphology import binary_closing
from skimage.transform import hough_line, hough_line_peaks

import config
import helper


def use_brown_mask(img):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    mask = (r - g > 20) * (g - b > 20) * (b < 80) * (r > 70)
    mask = binary_closing(mask, selem=np.ones((50, 50)))
    if config.SHOW_STEPS:
        plt.subplots(figsize=(15, 6))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()
    return mask


# this function does not using in current implemntation
def hide_table_on_mask(mask, ap, bp, cp, dp):
    i2 = int(max(ap[0], bp[0], cp[0], dp[0]))
    j2 = int(max(ap[1], bp[1], cp[1], dp[1]))
    for j in range(0, j2):
        for i in range(0, i2):
            mask[j][i] = False
    if config.SHOW_STEPS:
        plt.subplots(figsize=(15, 6))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()
    return mask


# show key points on table
def show_transform(img1, img2, points, transform):
    cv2.polylines(img1, [np.int32(points)], True, 255, 3, cv2.LINE_AA)
    cv2.polylines(img2, [np.int32(transform)], True, 255, 3, cv2.LINE_AA)

    points = np.reshape(points, (-1, 2))
    transform = np.reshape(transform, (-1, 2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    color = ['or', 'ob', 'og', 'oy']

    ax[0].set_title('Table template')
    ax[0].imshow(img1)
    for i in range(4):
        ax[0].plot(points[i][0], points[i][1], color[i], markersize=5)

    ax[1].imshow(img2)
    i = 0
    for i in range(4):
        ax[1].plot(transform[i][0], transform[i][1], color[i], markersize=5)
    plt.show()


# match key points
def transform_points(img1, img2, points):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    mymatches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            mymatches.append(m)

    src = np.float32([kp1[m.queryIdx].pt for m in mymatches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in mymatches]).reshape(-1, 1, 2)

    M, status = cv2.findHomography(src, dst, cv2.RANSAC, 2)
    p = points
    points = np.float32(points).reshape(-1, 1, 2)
    transform = cv2.perspectiveTransform(points, M)

    if config.SHOW_STEPS:
        show_transform(img1, img2, points, transform)

    return np.reshape(transform, (-1, 2))


# show all obtained lines
def show_line(image, angles, dists, ap, bp, cp, dp):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(image, cmap='gray')
    for angle, dist in zip(angles, dists):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, image.shape[1]), (y0, y1), '-r')

    ax[1].plot((ap[0], bp[0], cp[0], dp[0]), (ap[1], bp[1], cp[1], dp[1]), 'ob')
    ax[1].plot((ap[0], bp[0]), (ap[1], bp[1]), 'b')
    ax[1].plot((cp[0], dp[0]), (cp[1], dp[1]), 'b')
    ax[1].set_xlim((0, image.shape[1]))
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title('All detected lines')
    plt.show()


# show only filtered lines
def show_appropriate_line(image, angles, dists, ap, dp, x, y):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    ax[1].imshow(image, cmap='gray')
    for angle, dist in zip(angles, dists):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - x * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, x), (y0, y1), '-r')
    ax[1].plot((ap[0], x), (ap[1], y), 'b')
    ax[1].plot((x, dp[0]), (y, dp[1]), 'b')
    ax[1].plot(x, y, 'ob')
    ax[1].set_axis_off()
    ax[1].set_title('Filtered lines')
    plt.show()


def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if config.SHOW_STEPS:
        plt.imshow(img)
    table_template, table_key_point = helper.create_table_template()
    table_transform = transform_points(table_template, img, table_key_point)
    ap, bp, cp, dp = table_transform[0], table_transform[1], table_transform[2], table_transform[3]
    mask = use_brown_mask(img)

    # Using hough detector
    _, angles, dists = hough_line_peaks(*hough_line(canny(mask)), threshold=config.HOUGH_THRESHOLD)
    if config.SHOW_STEPS:
        show_line(mask, angles, dists, ap, bp, cp, dp)

    a1, d1 = helper.point2line(bp, ap)
    a2, d2 = helper.point2line(cp, dp)

    x, y = helper.intersect_line(a1, d1, a2, d2)

    appropr_angle = []
    appropr_dist = []
    if config.SHOW_STEPS:
        show_line(mask, angles, dists, ap, [x, y], [x, y], dp)
    for a, d in zip(angles, dists):
        po = helper.distance_from_focus_to_line(x, y, a, d)
        if po < config.FILTER_LINE_THRESHOLD:
            appropr_angle.append(a)
            appropr_dist.append(d)

    if config.SHOW_STEPS:
        show_appropriate_line(mask, appropr_angle, appropr_dist, ap, dp, x, y)

    return len(appropr_angle) > config.RESOLVE_THRESHOLD


def estimate_model(image_name_list, correct_ans_list, silent_mode=False):
    size = len(image_name_list)
    i = 0
    count = 0
    for name in image_name_list:
        if silent_mode:
            print('Progress:  ', i + 1, '/', size, sep='')
        prediction = predict(name)
        if silent_mode:
            print(prediction, 'for', name)
        if prediction == correct_ans_list[i]:
            count += 1
        i += 1
    return count / size
