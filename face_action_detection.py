import config as cfg
import dlib
import utils
import numpy as np
from imutils import face_utils
import cv2
class face_action_detector():
    def paint_shape(self, im):
        if self.shape is not None:
            paint_idx = [0,1,2,14,15,16,27,29,30,8,3,13]
            for i in paint_idx:
                cv2.circle(im, self.shape[i], 3, (0, 255, 0), 2)
        return im

    def reset(self):
        self.eye_total = 0
        self.mouth_total = 0
        self.left_total = 0
        self.right_total = 0
        self.up_total = 0
        self.down_total = 0

    def __init__(self, ):
        # cargar modelo para deteccion de puntos de ojos
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(cfg.face_68_landmarks)
        self.shape = None

        #eye blink
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.eye_counter = 0

        #mouth open
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.mouth_counter = 0

        #left
        self.left_counter, self.right_counter, self.up_counter,  self.down_counter= 0, 0, 0, 0
        self.reset()

    def turn_face(self):
        l = self.avg_dist([0, 1, 2], [27, 29, 30], 2)
        r = self.avg_dist([16, 15, 14], [27, 29, 30], 2)
        self.left_counter, self.left_total = self.stat(-l + r, abs(l + r) * cfg.TURN_H_THRESH, cfg.DEFAULT_CONT_FRAMES,
                                                       self.left_counter,
                                                       self.left_total, 'left')
        self.right_counter, self.right_total = self.stat(l - r, abs(l + r) * cfg.TURN_H_THRESH,
                                                         cfg.DEFAULT_CONT_FRAMES, self.right_counter,
                                                         self.right_total, 'right')

        u = self.avg_dist([1, 15], [30, 30], 1)
        d = self.avg_dist([2, 14, 3, 13], [29, 29, 29, 29], 1)
        self.up_counter, self.up_total = self.stat(u, cfg.TURN_V_THRESH, cfg.DEFAULT_CONT_FRAMES,
                                                       self.up_counter,
                                                       self.up_total, 'up')
        self.down_counter, self.down_total = self.stat(d,  cfg.TURN_V_THRESH,
                                                         cfg.DEFAULT_CONT_FRAMES, self.down_counter,
                                                         self.down_total, 'down')


    # average distance between shape[list1] and shape[list2],
    # xy: 0 x1-x2, 1 y1-y2, 3 |x1-x2|, 4 |y1-y2|, 5 ((x1-x2)^2+(y1-y2)^2)^0.5
    def avg_dist(self, list1, list2, xy):
        l = len(list1)
        assert(l==len(list2))

        if xy<2:
            d = sum([self.shape[list1[i]][xy] - self.shape[list2[i]][xy] for i in range(l)])
        elif xy<4:
            d = sum([abs(self.shape[list1[i]][xy-2] - self.shape[list2[i]][xy-2]) for i in range(l)])
        else:
            d = sum([np.linalg.norm(self.shape[list1[i]] - self.shape[list2[i]]) for i in range(l)])

        return d/l

    def det(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # face detection
        rectangles = self.frontal_face_detector(gray, 0)
        boxes_face = utils.convert_rectangles2array(rectangles, im)
        if len(boxes_face) != 0:
            areas = utils.get_areas(boxes_face)
            index = np.argmax(areas)
            rectangles = rectangles[index]

            self.shape = face_utils.shape_to_np(self.predictor(gray, rectangles))
            self.eye_blink()
            self.mouth_open()
            self.turn_face()
        else:
            self.shape = None

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def eye_blink(self):
        # 提取左眼和右眼坐标，然后使用该坐标计算两只眼睛的眼睛纵横比
        leftEye = self.shape[self.lStart: self.lEnd]
        rightEye = self.shape[self.rStart: self.rEnd]
        ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2.0
        self.eye_counter, self.eye_total = self.stat(-ear, -cfg.EYE_AR_THRESH, cfg.DEFAULT_CONT_FRAMES, self.eye_counter, self.eye_total, 'eye blink')

    # when v <= th, total add 1
    def stat0(self, v, th, frameth, counter, total, desc=''):
        #超过阈值，计数加一
        if v > th:
            counter += 1
        else:
            # 不满足条件，判断是否满足连续帧数量要求
            if counter >= frameth:
                total += 1
                counter = 0
                print(desc)
            if counter>0:
                counter -= 1
        return counter, total

    # when v > th, total add 1
    def stat(self, v, th, frameth, counter, total, desc=''):
        #超过阈值，计数加一
        if v > th:
            counter += 1
            # 判断是否满足连续帧数量要求
            if counter == frameth:
                total += 1
                print(desc)
        else:
            counter = 0
        return counter, total

    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59（人脸68个关键点）
        B = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        return (A + B) / (2.0 * C)

    def mouth_open(self, ):
        mouth = self.shape[self.mStart:self.mEnd]
        mar = self.mouth_aspect_ratio(mouth)
        self.mouth_counter, self.mouth_total = self.stat(mar, cfg.MOUTH_AR_THRESH, cfg.DEFAULT_CONT_FRAMES, self.mouth_counter,
                                                     self.mouth_total, 'mouth open')
