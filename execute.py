# Referring to https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
from imutils import face_utils
import torch
import numpy as np
import imutils
import dlib
import cv2
from face_alignment import face_alignment
from headpose_estimation import headpose_estimation
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from network import GazeEstimationModelVGG
from collections import OrderedDict
from math import pi
from torchvision.transforms import transforms
import pygame

# Demo Settings
bgm = ['The_Fur_Purrrade.mp3', 'The_High_Line.mp3', 'Divine_Life_Society.mp3']
scene = ['city.jpg', 'hip-hop.jpg', 'mountains.png']
ch=0
vol=0.5
onoff = 0
black = np.zeros((1082, 1920, 3), np.uint8)
pygame.init()
pygame.mixer.music.load(bgm[ch])
cv2.imshow('TV', black)
list_lr = []
list_ud = []


# Not used (f : degree -> x,y,z)
def gaze_xyz(y_pred):
    pred_x = -1 * np.cos(y_pred[0]) * np.sin(y_pred[1])
    pred_y = -1 * np.sin(y_pred[0])
    pred_z = -1 * np.cos(y_pred[0]) * np.cos(y_pred[1])
    pred = np.array([pred_x, pred_y, pred_z])
    pred = pred / np.linalg.norm(pred)
    # pred = np.rad2deg(np.arccos(np.dot(pred, gt)))

    return pred


cap = cv2.VideoCapture(0)  # 0: default camera

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = GazeEstimationModelVGG()
model.float()
state_dict = torch.load('model.pt', map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

_transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


while cap.isOpened():
    success, frame = cap.read()
    if success:
        # resize it, and convert it to grayscale
        image = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

        # not detected
        if len(rects) == 0:
            onoff += 1
        # detected
        if len(rects) != 0:
            onoff = 0
            imgch = cv2.imread(scene[ch], cv2.IMREAD_COLOR)
            cv2.imshow('TV', imgch)
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.load(bgm[ch])
                pygame.mixer.music.play(-1)
                pygame.mixer.music.set_volume(vol)

        # turn off
        elif onoff >= 100:
            onoff = 0
            cv2.imshow('TV', black)
            pygame.mixer.music.stop()

        for (k, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            eye_landmarks = [[], [], [], []]
            # loop over the face parts individually
            for (name, (i, j)) in [('right eye', (36, 42)), ('left eye', (42, 48))]:
                # clone = image.copy()
                # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.7, (0, 0, 255), 2)

                # loop over the subset of facial landmarks, drawing the specific face part
                for (x, y) in shape[i:j]:
                    if name == 'right eye':
                        eye_landmarks[0].append(x)
                        eye_landmarks[1].append(y)
                    elif name == 'left eye':
                        eye_landmarks[2].append(x)
                        eye_landmarks[3].append(y)

                #     cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                #
                # cv2.imshow("Image", clone)

            # FACE ALIGNMENT & GET ANGLE, EYES CENTER
            output, angle, eyes_center = face_alignment(image=image, eye_landmarks=eye_landmarks)
            # cv2.imshow("output", output)

            # EYE PATCH EXTRACTION
            gray2 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rect2 = detector(gray2, 1)
            if len(rect2) == 0:
                break
            rect2 = rect2[0]
            shape2 = predictor(gray2, rect2)
            shape2 = face_utils.shape_to_np(shape2)
            roi = []
            for (m, (name, (i, j))) in enumerate([('right eye', (36, 42)), ('left eye', (42, 48))]):
                # extract the ROI of the face region as a separate image
                (x, y, w, h) = cv2.boundingRect(np.array([shape2[i:j]]))
                y = y + int(h / 2)
                h = int(w / 2)
                temp = imutils.resize(output[y - h:y + h, x:x + w], width=60, inter=cv2.INTER_CUBIC)
                y = int(temp.shape[0] / 2)
                roi.append(temp[y - 17:y + 19, :])

            # show the face parts
            # cv2.imshow("Right", roi[0])
            # cv2.imshow("Left", roi[1])

            # BGR to RGB
            roi[0] = cv2.cvtColor(roi[0], cv2.COLOR_BGR2RGB)
            roi[1] = cv2.cvtColor(roi[1], cv2.COLOR_BGR2RGB)

            # HEAD POSE EXTRACTION
            headpose = headpose_estimation(np.array([
                shape2[30],  # Nose tip 30
                shape2[8],  # Chin 8
                shape2[36],  # Left eye left corner 36
                shape2[45],  # Right eye right corner 45
                shape2[48],  # Left Mouth corner 48
                shape2[54]  # Right mouth corner 54
            ], dtype="double"), output)

            # FEED INTO NN
            roi[0] = _transform(roi[0])
            roi[1] = _transform(roi[1])
            with torch.no_grad():
                roi[0] = roi[0].unsqueeze(0)
                roi[1] = roi[1].unsqueeze(0)
                headpose = torch.from_numpy(headpose).unsqueeze(0)
                roi[1], roi[0], headpose = Variable(roi[1]), Variable(roi[0]), Variable(headpose)

                gaze_out = model(roi[1].float(), roi[0].float(), headpose.float())
                gaze_out = gaze_out*180/pi
                out_list.append(gaze_out)
                # print(gaze_out)

            # FOR DEMO
            if len(rects) != 0:
                list_ud.append(gaze_out[0][1])
                list_lr.append(gaze_out[0][0])
                if len(list_lr) > 5:
                    del list_lr[0]
                    del list_ud[0]
                if len(list_lr) == 5:
                    if abs(sum(list_lr)) > abs(sum(list_ud)):
                        if sum(list_lr)/5 < -13:
                            # left
                            print('l')
                            if vol<=0.3:
                                vol = 0
                            else:
                                vol -= 0.3
                            pygame.mixer.music.set_volume(vol)
                            list_lr = []
                            list_ud = []
                        elif sum(list_lr)/5 > 15:
                            # right
                            print('r')
                            if vol >= 0.7:
                                vol = 1
                            else:
                                vol += 0.3
                            pygame.mixer.music.set_volume(vol)
                            list_lr = []
                            list_ud = []
                    else:
                        if sum(list_ud)/5 < -15:
                            # down
                            print('d')
                            if ch == 0:
                                ch = 0
                            else:
                                ch -= 1
                                pygame.mixer.music.load(bgm[ch])
                                pygame.mixer.music.play(-1)
                                pygame.mixer.music.set_volume(vol)
                            list_lr=[]
                            list_ud=[]
                        elif sum(list_ud)/5 > 15:
                            # up
                            print('u')
                            if ch == 2:
                                ch = 2
                            else:
                                ch += 1
                                pygame.mixer.music.load(bgm[ch])
                                pygame.mixer.music.play(-1)
                                pygame.mixer.music.set_volume(vol)
                            list_lr = []
                            list_ud = []

            # cv2.waitKey(0)

    # ESC to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
