########################################################################################
#
# This is very similar to the function predict_sequence_seed that you can find in the 
# root folder, in evaluate.py
#
########################################################################################

import numpy as np

def predict_sequence_seed(controller, motors, seed):

    output = seed
    entries = motors.shape[0]
    # number of elements we will generate
    seq_length = motors.shape[1]
    mot_result = np.empty([entries, 0, 25]) # empty to fill in next loop

    for i in range(seq_length-1):
        # make the prediction using the model
        _, predict = controller.t.model.predict([motors, output])
        # round it to 0 or 1
        #predict = np.around(predict)

        # use controller to predict next motors
        new_motors = controller.model.predict([motors, output])
        # add new motor pattern to motor array
        motors = np.concatenate([motors, new_motors], axis=1)
        # and remove the oldest one
        motors = motors[:,1:,:]
        # store results of motors to display later on
        avgmot = np.mean(motors,axis=1).reshape([entries,1,25])
        new_motors = new_motors.reshape([entries,1,25])
        mot_result = np.concatenate([mot_result, new_motors], axis=1)

        # get last entry in sequence
        new_entry = predict[:,-1,:].reshape([entries,1,25])
        # add to predictions last element of the prediction
        output = np.concatenate([output, new_entry], axis=1)
        # remove oldest entry to keep seq_len size
        output = output[:,1:,:]

    return [mot_result, output]
