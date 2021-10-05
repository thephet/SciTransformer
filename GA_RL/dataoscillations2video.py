# If you use TestController.py, save the returned tuple into a pickle file, and then use this script
# to generate a video of the experiments

import pickle, sys, glob, multiprocessing
import cv2
import numpy as np


def calculateFitness(experiment):

    total_sum = np.sum(experiment)
    center_sum = np.sum(experiment[:,12])
    fitness = center_sum * 24 - (total_sum - center_sum)

    return fitness


def generate_singlevideo(path, processLimiter=multiprocessing.Lock()):

    font = cv2.FONT_HERSHEY_SIMPLEX

    with processLimiter:
        print("Processing p file "+path)
        data = pickle.load( open( path, "rb"))
        motors, predictions = data
        num_preds = len(predictions)

        for pred in range(num_preds):
            fitness = int( calculateFitness( predictions[pred] ) )

            fourcc = cv2.VideoWriter_fourcc(*'X264')
            outvideo = path.split(".")[0]+"_"+str(pred)+"_"+str(fitness)+".avi"
            outvideo = cv2.VideoWriter(outvideo, fourcc, 10.0, (500,500))

            accumulator = np.zeros(25)

            for entry in range(len(predictions[pred])):
                blank_frame = np.zeros((500,500,3), np.uint8)

                # in some cases there are more predictions than motors, fix that now
                if entry >= len(motors[pred]):
                    m_entry = len(motors[pred])-1
                else:
                    m_entry = entry
                
                datarow = predictions[pred][entry]
                motor_row = motors[pred][m_entry]
                accumulator += datarow

                for i in range(5):
                    for j in range(5):
                        point1 = (i*100, j*100)
                        point2 = (i*100+100, j*100+100)
                        colour = int(datarow[i*5+j] * 255)
                        cv2.rectangle(blank_frame, point1, point2, 
                                [colour,colour,colour], -1) 

                        speed = (motor_row[i*5+j] * 2) -1
                        cv2.putText(blank_frame, f'M:{ int(speed*10) }',
                                (point1[0]+10, point1[1]+20), font, 0.6, (255, 255, 255))

                        fit = accumulator[i*5+j]
                        cv2.putText(blank_frame, f'O:{ int(fit) }',
                                (point1[0]+10, point1[1]+40), font, 0.6, (255, 255, 255))
                
                outvideo.write(blank_frame)

            outvideo.release()
            cv2.destroyAllWindows()


def processfolder(pathtofolder):
    ''' This function will execute the previous 1 in all the files of a folder.
    '''

    # number of operations in parallel. auchentoshan has 12 cores
    s = multiprocessing.Semaphore(12)
    # we only want to process the p files
    allpfiles = glob.glob(pathtofolder+'*.p')
    
    for pfile in allpfiles:
        p = multiprocessing.Process(target=generate_singlevideo, args=(pfile,s))
        p.start()


if __name__ == '__main__':

    processfolder(sys.argv[1])
