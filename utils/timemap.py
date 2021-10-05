##########################################################################################
#
# This script aims to create a time map of the BZ reaction, like the one that can be seen
# in many BZ papers. The idea is to take a column of pixels every frame (always the same 
# one) and display an image with these columns adjacent to one another as the video goes.
#
# The core idea is to convert a video into a 2D image, where frames are flattened and
# concatenated one after the other.
#
##########################################################################################



import cv2
import numpy as np
import sys
import multiprocessing
import glob

BZWIDTH = 5


def add_column(frame, timemap, xcol, reps=10):

    x1, y1, x2, y2 = 0,0,500,500
    step_w = int( (x2 - x1) / BZWIDTH )
    step_h = int( (y2 - y1) / BZWIDTH )
    height = y2 - y1
    width = x2 - x1

    for i in range(BZWIDTH):
        col = frame[y1:y2, x1 + step_w*i]
        #row = frame[y1+20-(i*2) + step_h*i, x1:x2]
        for j in range(reps):
            timemap[i*height:i*height+height, xcol*reps+j] = col
            #timemap[i*width:i*width+width, xcol*reps+j] = row



def TimemapSinglevideo(path, processLimiter=multiprocessing.Lock()):

    with processLimiter:
        print("Processing video "+path)
        video = cv2.VideoCapture(path)
        frame_counter = 0
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        speed = 1
        reps = 20 # number of times row will be repeated

        start_frame = 0 # 0 from beggining, 1800 half,...
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        timemap = np.zeros((500*BZWIDTH, (total_frames - start_frame)*reps, 3), np.uint8)
         
        while(True):

            ret, frame = video.read()

            if frame_counter % speed != 0:
                continue

            if ret is False:
                break

            add_column(frame, timemap, frame_counter, reps)

            #cv2.imshow('Time map', timemap)
            #key = cv2.waitKey(1) & 0xFF
            frame_counter += 1


        outname = path.split(".")[0] + ".png"
        cv2.imwrite(outname, timemap)
        video.release()



def TimeMapfolder(pathtofolder):
    ''' This function will execute the previous single TimeMap video function
    in all the files of a folder.'''

    s = multiprocessing.Semaphore(4)
    # we only want to process the videos with "fast5 svm" in the name
    allvideos = glob.glob(pathtofolder+'*.avi')
    for video in allvideos:
        print(video)
        p = multiprocessing.Process(target=TimemapSinglevideo, args=(video,s))
        p.start()



if __name__ == "__main__":

    #TimemapSinglevideo(sys.argv[1])
    TimeMapfolder(sys.argv[1])

