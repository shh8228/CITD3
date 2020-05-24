# Referring to
# https://github.com/Tobias-Fischer/rt_gene/blob/master/rt_gene/src/rt_gene/tracker_face_encoding.py

import numpy as np
import cv2

desired_right_eye = (0.3, 0.3)
desired_face_width = 150
desired_face_height = 150


def face_alignment(image, eye_landmarks):
    right_eye_pts = np.array([eye_landmarks[0], eye_landmarks[1]])
    left_eye_pts = np.array([eye_landmarks[2], eye_landmarks[3]])

    # compute the center of mass for each eye
    left_eye_centre = left_eye_pts.mean(axis=1).astype("int")
    right_eye_centre = right_eye_pts.mean(axis=1).astype("int")
    # compute the angle between the eye centroids
    d_y = left_eye_centre[1] - right_eye_centre[1]
    d_x = left_eye_centre[0] - right_eye_centre[0]
    angle = np.degrees(np.arctan2(d_y, d_x))

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyes_center = ((left_eye_centre[0] + right_eye_centre[0]) // 2,
                   (left_eye_centre[1] + right_eye_centre[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1)

    # update the translation component of the matrix
    rotation_matrix[0, 2] += (image.shape[1] * 0.5 - eyes_center[0])
    rotation_matrix[1, 2] += (image.shape[0] * 0.5 - eyes_center[1])

    # apply the affine transformation
    output = cv2.warpAffine(image, rotation_matrix,
                            (image.shape[1], image.shape[0]),
                            flags=cv2.INTER_CUBIC)
    return output, angle, eyes_center
