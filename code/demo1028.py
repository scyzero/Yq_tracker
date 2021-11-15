# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
net.eval().cuda()

# image and init box
# image_files = sorted(glob.glob('./bag/*.jpg'))
# init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
# [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

video_path = "/home/scy/PycharmProjects/DaSiamRPN/code/data/kuaisu005.mp4"
cap = cv2.VideoCapture(video_path)
_, frame = cap.read()
init_rbox = [1236,328,1326,316,1220,646,1348,646]
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
# frame=cv2.resize(frame,(960,540))
# cv2.imshow('SiamRPN', frame)
# cv2.waitKey(0)


# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
#im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(frame, target_pos, target_sz, net)

# tracking and visualization
toc = 0
video_index = 0

while(1):
    ret, im = cap.read()
    if not ret:
        break
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    video_index +=1
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)





# for f, image_file in enumerate(image_files):
#     im = cv2.imread(image_file)
#     tic = cv2.getTickCount()
#     state = SiamRPN_track(state, im)  # track
#     toc += cv2.getTickCount()-tic
#     res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
#     res = [int(l) for l in res]
#     cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
#     cv2.imshow('SiamRPN', im)
#     cv2.waitKey(1)

print('Tracking Speed {:.1f}fps'.format((video_index-1)/(toc/cv2.getTickFrequency())))
