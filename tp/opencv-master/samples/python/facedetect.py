#!/usr/bin/env python

'''
face detection using haar cascades
USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str

# perform the detection
def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

# draw the rectangle
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "data/haarcascades/haarcascade_eye.xml")
# detect objects in a video stream : casacade - face, nested - eyes 
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('samples/data/lena.jpg')))
    # initialization of the coordinates of the four vertices of the rectangle
    [xp1,yp1,xp2,yp2] = [0,0,0,0]
    while True:
        # get the image from the video stream, if it succeeds, ret = true
        ret, img = cam.read()
        # get the greyscale image of "img" 
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        # the rectangle detected
        rects = detect(gray, cascade)    
        # if last time the face is detected, 
        # detection should be performed in the neighborhood of the previous rectangle
        if xp1 :
            gray2 = gray[yp1:yp2,xp1:xp2]
            rects = detect(gray2.copy(), cascade)
            
        t = clock()
        rects = detect(gray, cascade)
        # the output image
        vis = img.copy()
        # draw the rectangle of face with color green
        draw_rects(vis, rects, (0, 255, 0))
        # the neighborhood of the previous rectangle : a bigger rectangle
        rect_p = np.array([[x-60, y-60, w+60, h+60] for (x, y, w, h) in rects])
        # the neighborhood exists, get the coordinates of its four vertices
        if  len(rect_p) :
            [xp1,yp1,xp2,yp2] = rect_p[0,:][:]
        # draw the neighborhood with color red
        if xp1 :
            cv.rectangle(vis, (xp1, yp1), (xp2, yp2), (0, 0, 255), 3)
            # change the output image
            vis = vis[yp1:yp2,xp1:xp2]
        # detection of eyes in the frame of face(roi)
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                # draw the rectangle of eyes with color bleu
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t
        
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break
    cv.destroyAllWindows()