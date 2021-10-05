#########################################################################################
#
# This script will take pickle files created from transformer predictions
# and generate videos to visualize the predictions.
#
#########################################################################################


import pickle, sys, glob, multiprocessing
import cv2
import numpy as np

def generate_singlevideo(path, processLimiter=multiprocessing.Lock()):

    font = cv2.FONT_HERSHEY_SIMPLEX

    with processLimiter:
        print("Processing p file "+path)
        data = pickle.load( open( path, "rb"))
        motors, originals, predictions = data
        #motors, predictions = data
        #predictions = data
        #motors = np.zeros((64,150,25))
        
        num_preds = len(predictions)
        #num_preds = 1

        # for each prediction inside the pickle file
        for pred in range(num_preds):
            # create a video for this prediction
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            outvideo = path.split(".")[0]+"_"+str(pred)+"_pred.avi"
            outvideo = cv2.VideoWriter(outvideo, fourcc, 10.0, (500,500))
            # and an accumulator to store oscillations
            accumulator = np.zeros(25)

            # for each entry/frame of this prediction
            for entry in range(len(predictions[pred])):
            #for entry in range(len(predictions)):

                # start with a blank frame
                blank_frame = np.zeros((500,500,3), np.uint8)
                # get the entry data to paint the cell
                #datarow = originals[pred][entry]
                #datarow = predictions[entry]
                datarow = predictions[pred][entry]
                
                # because we know average non oscillation is 0.3. Set 0.3 to 0
                # and rescale between 0 and 1
                datarow -= 0.30
                datarow = datarow / 0.7
                datarow = datarow.clip(min=0)

                # # in some cases there are more predictions than motors, fix that now
                if entry >= len(motors[pred]):
                    m_entry = len(motors[pred])-1
                else:
                    m_entry = entry

                motor_row = motors[pred][m_entry]
                # #motor_row = motors[m_entry]
                # #accumulator += datarow

                # for each of the 25 cells
                for i in range(5):
                    for j in range(5):
                        # create a rectangle with a color to represent the cell's state
                        point1 = (i*100, j*100)
                        point2 = (i*100+100, j*100+100)
                        colour = int(datarow[i*5+j] * 255) * 2
                        cv2.rectangle(blank_frame, point1, point2, 
                                [colour,colour,int(255-colour/2)], -1) 

                        # writting into the cell different values like motor speed or fit
                        # speed = (motor_row[i*5+j] * 2) -1
                        speed = motor_row[i*5+j]
                        cv2.putText(blank_frame, f'{ int(speed*10) }',
                                (point1[0]+15, point1[1]+60), font, 1.3, (255, 255, 255))
                                #(point1[0]+10, point1[1]+20), font, 0.6, (255, 255, 255))

                        # fit = (accumulator[i*5+j] * 2) -1
                        # cv2.putText(blank_frame, f'O: { int(fit) }', 
                        #         (point1[0]+10, point1[1]+40), font, 0.6, (255, 255, 255)) 
                outvideo.write(blank_frame)

            outvideo.release()
            cv2.destroyAllWindows()


def processfolder(pathtofolder):
    ''' This function will execute the previous one in all the files of a folder.
    '''

    # number of operations in parallel. auchentoshan has 12 cores
    s = multiprocessing.Semaphore(12)
    # we only want to process the p files
    allpfiles = glob.glob(pathtofolder+'*.p')
    print(allpfiles)
    
    for pfile in allpfiles:
        p = multiprocessing.Process(target=generate_singlevideo, args=(pfile,s))
        p.start()


if __name__ == '__main__':

    processfolder(sys.argv[1])
