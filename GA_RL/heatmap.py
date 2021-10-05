#########################################################################################
#
# This script will take pickle files created from transformer predictions
# and generate a heatmap to show how much each cell oscillates
#
#########################################################################################


import pickle, sys, glob, multiprocessing
import cv2
import numpy as np

def generate_singleheatmap(path, processLimiter=multiprocessing.Lock()):

    with processLimiter:
        print("Processing p file "+path)
        data = pickle.load( open( path, "rb"))
        # motors, originals, predictions = data
        motors, predictions = data

        num_preds = len(predictions[0])

        # for each prediction inside the pickle file
        for pred in range(num_preds):

            # for each of xor 4 possibilities 00 10 01 11
            for xor in range(4):
                # create the accumulator to store oscillations and then plot heatmap
                accumulator = np.zeros(25)

                # for each entry/frame of this prediction
                for entry in range(len(predictions[xor][pred])):
                    datarow = predictions[xor][pred][entry]
                    accumulator += datarow

                # normalize accumulator between 0 and 1
                maxa = max(accumulator)
                mina = min(accumulator)
                accumulator -= mina
                accumulator = accumulator / (maxa-mina)

                # now it has gone through all the experiment, we can plot heatmap
                blank_frame = np.zeros((500,500,3), np.uint8)

                # for each of the 25 cells
                for i in range(5):
                    for j in range(5):
                        point1 = (i*100, j*100)
                        point2 = (i*100+100, j*100+100)
                        colour = int(accumulator[i*5+j] * 255 )
                        cv2.rectangle(blank_frame, point1, point2, 
                                [colour,colour,colour], -1)

                # write image into disk
                outimg = path.split(".")[0]+"_"+str(pred)+"_"+str(xor)+".jpeg"
                cv2.imwrite(outimg, blank_frame)


def processfolder(pathtofolder):
    ''' This function will execute the previous one in all the files of a folder.
    '''

    # number of operations in parallel. auchentoshan has 12 cores
    s = multiprocessing.Semaphore(12)
    # we only want to process the p files
    allpfiles = glob.glob(pathtofolder+'*.p')
    
    for pfile in allpfiles:
        p = multiprocessing.Process(target=generate_singleheatmap, args=(pfile,s))
        p.start()


if __name__ == '__main__':

    processfolder(sys.argv[1])
