# Referring to https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from face_alignment import face_alignment
from headpose_estimation import headpose_estimation

cap = cv2.VideoCapture(0)  # 0: default camera

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while cap.isOpened():
    # 카메라 프레임 읽기
    success, frame = cap.read()
    if success:
        # 프레임 출력
        # resize it, and convert it to grayscale
        image = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        # loop over the face detections
        for (k, rect) in enumerate(rects):

            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            eye_landmarks = [[], [], [], []]
            # loop over the face parts individually
            for (name, (i, j)) in [('right eye', (36, 42)), ('left eye', (42, 48))]:
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                # loop over the subset of facial landmarks, drawing the
                # specific face part
                for (x, y) in shape[i:j]:
                    if name == 'right eye':
                        eye_landmarks[0].append(x)
                        eye_landmarks[1].append(y)
                    elif name == 'left eye':
                        eye_landmarks[2].append(x)
                        eye_landmarks[3].append(y)

                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

                cv2.imshow("Image", clone)

            # FACE ALIGNMENT & GET ANGLE, EYES CENTER
            output, angle, eyes_center = face_alignment(image=image, eye_landmarks=eye_landmarks)
            cv2.imshow("output", output)
            gray2 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

            # EYE PATCH EXTRACTION
            # detect faces in the grayscale image
            rect2 = detector(gray2, 1)

            if len(rect2) == 0:
                break

            rect2 = rect2[0]

            shape2 = predictor(gray2, rect2)
            shape2 = face_utils.shape_to_np(shape2)
            roi = []
            roi2 = []
            for (m, (name, (i, j))) in enumerate([('right eye', (36, 42)), ('left eye', (42, 48))]):
                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape2[i:j]]))
                y = y + int(h / 2)
                h = int(w / 2)
                temp = imutils.resize(output[y - h:y + h, x:x + w], width=60, inter=cv2.INTER_CUBIC)
                y = int(temp.shape[0]/2)
                roi.append(temp[y - 17:y + 19, :])
                roi2.append(output[y - h:y + h, x:x + w])

            # show the face parts
            cv2.imshow("Left", roi[1])
            cv2.imshow("Right", roi[0])
            cv2.imshow("Left2", roi2[1])
            cv2.imshow("Right2", roi2[0])
            print(roi[0].shape, roi2[0].shape)

            # HEAD POSE EXTRACTION
            headpose = headpose_estimation(np.array([
                shape2[30],  # Nose tip 30
                shape2[8],  # Chin 8
                shape2[36],  # Left eye left corner 36
                shape2[45],  # Right eye right corner 45
                shape2[48],  # Left Mouth corner 48
                shape2[54]  # Right mouth corner 54
            ], dtype="double"), output)

            # cv2.imshow("image", image)
            cv2.waitKey(0)

    # ESC를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
