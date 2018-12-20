# Ref: https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py

# set_image(self: dlib.image_window, image: object)
# add_overlay(self: dlib.image_window, rectangles: dlib.rectangles, color: dlib.rgb_pixel=rgb_pixel(255,0,0))
# add_overlay(self: dlib.image_window, detection: dlib.full_object_detection, color: dlib.rgb_pixel=rgb_pixel(0,0,255))

import cv2
import dlib
import time
import numpy as np

time.clock()
ts=time.clock()

if(__name__=='__main__'): pass
else:
    global face, face_proc
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("FACE Init time:",time.clock()-ts,"s")

    def face(img):
        if type(img) is str:
            img = cv2.imread(img)
            if img is None: return None
        d_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector(d_img, 1) # dlib.dlib.rectangles
        faces = []
        for n in range(len(dets)):
            for k, d in enumerate([dets[n]]):   # dlib.dlib.rectangle
                shape = predictor(img, d)       # dlib.dlib.full_object_detection
                if(shape.num_parts != 68): break
            pt = []
            for i in range(68):
                pt.append((shape.part(i).x,shape.part(i).y))
            faces += [[(dets[n].left(), dets[n].top()), (dets[n].right(), dets[n].bottom()), pt]]
        return faces

    def face_draw(src):
        rects = face(src)
        pix = src.copy()
        for rect in rects:
            if(not rect is None):
                cv2.rectangle(pix, *(rect[0:2]), (0,0,255))
                for i in range(0,16):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (255,0,0))
                for i in range(17,21):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (255,255,0))
                for i in range(22,26):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (255,255,0))
                for i in range(27,30):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (255,0,0))
                for i in range(31,35):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (255,0,0))
                for i in range(36,41):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (128,255,0))
                for i in range(42,47):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (128,255,0))
                for i in range(48,67):
                    cv2.line(pix, rect[2][i], rect[2][i+1], (0,255,0))
        return pix

    def gen_lines(pix):
        rects = face(pix)
        faces = []
        for rect in rects:
            lines = []
            if(not rect is None):
                for i in range(0,16):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(17,21):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(22,26):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(27,30):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(31,35):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(36,41):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(42,47):
                    lines += [[rect[2][i],rect[2][i+1]]]
                for i in range(48,67):
                    lines += [[rect[2][i],rect[2][i+1]]]
                #lines += [[[0,0],[0,pix.shape[1]-1]],[[0,pix.shape[1]-1],[pix.shape[0]-1,pix.shape[1]-1]],[[pix.shape[0]-1,pix.shape[1]-1],[pix.shape[0]-1,0]],[[pix.shape[0]-1,0],[0,0]]]
            faces += [lines]
        if(not len(faces)): return None
        return np.array(faces[0])