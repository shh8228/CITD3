# Referring to https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

import cv2
import numpy as np
from math import pi, atan2, asin


def headpose_estimation(image_points, im):

    size = im.shape

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    rod = cv2.Rodrigues(rotation_vector)[0]
    pitch = asin(rod[2][0])
    yaw = atan2(-rod[1][0], rod[0][0])
    headpose = np.array([-pitch, yaw])

    # Display Head Pose Estimated Image
    #
    # (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
    #                                                  translation_vector,
    #                                                  camera_matrix, dist_coeffs)
    #
    # for p in image_points:
    #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
    #
    # p1 = (int(image_points[0][0]), int(image_points[0][1]))
    # p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    #
    # cv2.line(im, p1, p2, (255, 0, 0), 2)
    # Display image
    # cv2.imshow("Output", im)

    return headpose
