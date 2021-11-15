'''
读取视频、检测前景目标，调用DiameseRPN进行跟踪
'''
import cv2
import torch
import numpy as np
from os.path import realpath, dirname, join
from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()


def videoTrack():
    video_path = " "
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    fgbg = cv2.createBackgroundSubtractorMOG2()

    startTrack = False
    restartTrack = False
    stopTrack = False
    isTracking = False

    while (ret):
        # 前景检测
        fgmask = fgbg.apply(frame)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # 前景处理
        fgmask = cv2.erode(fgmask, element)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)


        # 轮廓查找
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame,contours,-1,(0,0,255),2)

        if (startTrack):
            # 找到最大轮廓
            if (len(contours) > 0):
                maxContour = contours[0]
                for contour in contours:
                    if contour.size > maxContour.size:
                        maxContour = contour
                x, y, w, h = cv2.boundingRect(maxContour)
                if (w * h > 400):
                    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    target_pos, target_sz = np.array([x, y]), np.array([w, h])
                    state = SiamRPN_init(frame, target_pos, target_sz, net)
                    isTracking = True
                    startTrack = False

        if (isTracking):
            state = SiamRPN_track(state, frame)
            res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            res = [int(l) for l in res]
            cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)

        # 重新跟踪
        if (restartTrack):
            if (isTracking):
                isTracking = False
            else:
                startTrack = True
                restartTrack = False

        if (stopTrack):
            isTracking = False
            stopTrack = False

        cv2.imshow("track", frame)
        # cv2.imshow("fgmask", fgmask)
        # cv2.imshow("masked", masked)

        key = cv2.waitKey(10)
        if (key == 83):  # S键开始跟踪
            print("------------开始跟踪-----------------")
            startTrack = True
        elif (key == 82):  # R键重新跟踪
            print("------------重新跟踪-----------------")
            restartTrack = True
        elif (key == 80):  # P键停止跟踪
            print("------------停止跟踪-----------------")
            stopTrack = True

        ret, frame = cap.read()


if __name__ == '__main__':
    videoTrack()