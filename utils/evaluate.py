#########################################################################################
#
# This script contains a few functions to generate predictions using a transformer model
# keras "predict" will only predict t+1. This script will iterate and predict n
# It also contains a few functions to calculate the accuracy of the predictions
# and plot the results
#
# It also has some function to evluate the quality of the predictions and plot results.
#
#########################################################################################


import numpy as np
import matplotlib.pyplot as plt
import pickle


def predict_sequence(model, inputs):
    ''' Giving inputs (a motor sequence) this function will generate an initial chemical 
    state which will be either all 0s or random numbers, and from there it will generate
    new predictions and append them to generate a full sequence.'''

    # number of diff experiments we will generate
    entries = inputs.shape[0]
    # length of sequence we will generate
    seq_length = inputs.shape[1]
    # all zeros as sequence start.
    #output = np.ones([entries,seq_length,25])*0
    sizeoutput = seq_length*entries*25
    output = np.random.uniform(0.49,0.51,sizeoutput).reshape([entries,seq_length,25])
    result = np.empty([entries, 0, 25]) # empty to fill in next loop

    for i in range(seq_length*2):
        # make the prediction using the model
        _, predict = model.predict([inputs, output])
        # round it to 0 or 1 - only use when binary output - raw output comment out
        # predict = np.around(predict)
        # get last entry in sequence
        new_entry = predict[:,-1,:].reshape([entries,1,25])
        # add to predictions last element of the prediction
        output = np.concatenate([output, new_entry], axis=1)
        # remove oldest entry to keep seq_len size
        output = output[:,1:,:]
        # add to final result
        result = np.concatenate([result, new_entry], axis=1)

    return result


def predict_common_shapes(model):
    ''' predict common shapes such as only center activation, column activation,...'''

    only_center = np.ones([1,1,25])*0.5
    only_center[0,0,12] = 1
    oc_seq = np.repeat(only_center,150,axis=1)

    square = np.ones([1,1,25])*0.5
    square[0,0,6:9] = 1
    square[0,0,11] = 1
    square[0,0,13] = 1
    square[0,0,16:19] = 1
    square_seq = np.repeat(square,150,axis=1)

    left_col = np.ones([1,1,25])*0.5
    left_col[0,0,:5] = 1
    left_seq = np.repeat(left_col, 150, axis=1)

    right_col = np.ones([1,1,25])*0.5
    right_col[0,0,20:] = 1
    right_seq = np.repeat(right_col, 150, axis=1)

    inputs = np.concatenate([oc_seq, square_seq, left_seq, right_seq], axis=0)
    preds = predict_sequence(model, inputs)

    return [inputs, preds]


def predict_random_sequences(model, gen, numSeqs):
    ''' pick random sequences from the dataset and predict them '''

    total_batches = len(gen)
    seq_len = gen[0][0].shape[2]

    motors = np.empty([0, seq_len, 25]) # empty to fill in next loop
    originals = np.empty([0, seq_len, 25]) # empty to fill in next loop
    y_fut = np.empty([0, seq_len, 25]) # empty to fill in next loop

    for _ in range(numSeqs):
        # select a random batch, -20 to be able to fetch future y
        rbatch = np.random.randint(0, total_batches-20)
        xs, ys = gen[ rbatch ]
        # fixed to entry 0 in the batch
        zero_entry = 0
        x_mot = xs[ zero_entry, ..., :25 ]
        x_orig = xs[ zero_entry, ..., 25: ]
        # because sequences are 150, next y is on this position
        y_f = gen[rbatch+18][1][49]
        motors = np.concatenate([motors, x_mot])
        originals = np.concatenate([originals, x_orig])
        y_fut = np.concatenate( [ y_fut, y_f.reshape([1,seq_len,25]) ] )

    preds = predict_sequence_seed(model, motors, originals)

    return [motors, y_fut, preds]


def predict_generator_ids(model, gen, ids):
    ''' Given a TimeSeriesGenerator, and a list of ids, it will fetch those sequences
    from the generator and predict them.'''

    seq_len = gen[0][0].shape[2]
    
    motors = np.empty([0, seq_len, 25]) # empty to fill in next loop
    originals = np.empty([0, seq_len, 25]) # empty to fill in next loop
    y_fut = np.empty([0, seq_len, 25]) # empty to fill in next loop

    for testid in ids:

        # if its bigger than 3000, it can be ignore because it doesnt have future y
        if testid > 3000:
            continue

        # fetch this batch from the generator - it was isolated during training
        xs, ys = gen[testid]
        # fixed to entry 0 in the batch, because it is easier to locate future y
        zero_entry = 0
        x_mot = xs[ zero_entry, ..., :25 ]
        x_orig = xs[ zero_entry, ..., 25: ]
        # because sequences are 150, next y is on this position
        y_f = gen[testid+18][1][49]
        motors = np.concatenate([motors, x_mot])
        originals = np.concatenate([originals, x_orig])
        y_fut = np.concatenate( [ y_fut, y_f.reshape([1,seq_len,25]) ] )

    preds = predict_sequence_seed(model, motors, originals)

    return [motors, y_fut, preds]


def predict_sequence_seed(model, inputs, seed):
    ''' Similar to the first predict_sequence function, but here instead of using 
    random number to generate the initial state of the chemistry, we will pass in a
    initial seed (so a sequence of chemistry state) and generate from there'''

    output = seed
    entries = inputs.shape[0]
    # number of elements we will generate
    seq_length = inputs.shape[1]

    for i in range(seq_length-1):
        # make the prediction using the model
        _, predict = model.predict([inputs, output])
        # round it to 0 or 1
        # predict = np.around(predict)
        # get last entry in sequence
        new_entry = predict[:,-1,:].reshape([entries,1,25])
        # add to predictions last element of the prediction
        output = np.concatenate([output, new_entry], axis=1)
        # remove oldest entry to keep seq_len size
        output = output[:,1:,:]

    return output


def evaluate_quality_preds(preds, window=1):
    ''' preds must be a pickle file generated using the function above
    name predict_random_sequences. This function will calculate how
    similar the predictions are compared to the real y. It will provide
    a % of accuracy per timestep in the sequence.
    The window argument relates to the window of matching between ys and preds.
    A window of 1 means a direct comparison between positions, a window of 2 means
    three positions will be compared [-1, 0, 1]. This is useful because there might
    be a phase difference between prediction and real ys. Probably because the way
    sequences were created during training, and I cant reconstruct the perfect y,
    because for training we only need the next step (t+1) not t+150'''

    # start by loading the pickle file
    data = pickle.load( open(preds, "rb") )
    # we only want the 2nd and 3rd list in data
    _, ys, preds = data
    # create an array where to store accuracy per entry per timestep
    accuracy = np.zeros( [ len(ys), len(ys[0]) ] )

    # set 0 at 0.35
    ys = (ys-0.35) / 0.65
    ys = ys.clip(min=0)
    preds = (preds-0.35) / 0.65
    preds = preds.clip(min=0)
    #print(np.mean(ys), np.quantile(preds, 0.95))

    # now we will iterate comparing
    for i,  y_entry in enumerate(ys):
        for j, y in enumerate(y_entry):

            acc = 99 # just a counter to fill in next loop
            for w in range(-window+1, window): # window=1 would return just 1

                if np.sum(y) == 0:
                    acc = float('nan')
                    break

                # if the position is imposbile, ignore it
                if (j+w)<=0 or (j+w)>=150:
                    continue

                # get current prediction from experiment i at time j
                t_pred = preds[i][j+w]
                # now compare prediction and real value, transform to int
                comp = abs(t_pred - y)
                # calculate the good ones out of the 25
                ratio = np.sum(comp) / 25.

                # update this counter if new ratio is better
                if ratio < acc:
                    acc = ratio

            # store accuracy in array
            accuracy[i][j] = acc

    plot_accuracy(accuracy, np.arange(55,60))
    #return accuracy

    
def plot_accuracy(accuracy, exps):
    ''' accuracy must be a numpy array created with the function above.
    exps must be a list with the experiments to show. Like [1,2,3].'''

    #plt.ylim(0.4, 1.2)
    # because ys with all 0 were set to nan to only focus on oscillations
    not_nan_array = ~ np.isnan(accuracy)

    # from stackoverflow, to generate colors
    def get_cmap(n, name='Set1'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n*2)

    cmap = get_cmap(len(exps))

    for i, exp in enumerate(exps):
        # get the accuracy entry for this experiment. the non-nan elements
        a = accuracy[exp][not_nan_array[exp]]
        # fit a line to see how it evolves
        z = np.polyfit(np.arange(len(a)), a, 1)
        zpoly = np.poly1d(z)
        plt.scatter(np.arange(150), accuracy[exp].T, marker='o', color=cmap(i), s=50)
        #plt.plot(np.arange(150), accuracy[exp].T, color=cmap(i))
        plt.plot( np.arange(150), zpoly(np.arange(150)), color=cmap(i),linewidth=8)
        plt.rcParams["font.size"] = "30"

    plt.rcParams['axes.linewidth'] = 2 #set the value globally
    plt.grid(True)
    
    plt.show()


def evaluate_big_platform(model, motors, newsize):
    ''' motors: sequence of motors to test. it must have size newsize. 
    size: length of the new array. Remember even though the platform is 5b5, we are
    working with 25 (flat) arrays. So if we wanted to have the same size, we would chose 
    25. size can be anything, but if you want to nicely visualize later, chose a square 
    value like 25,36,49,64,81,100,121,...

    BZ is only 5b5, this function will aim to generate a platform size by size.
    To do so, each position in the bigger sizeXsize will be calculated using the 5b5
    prediction and using the center cell as the new cell in sixeXsize.
    So, if size was 100, so 10x10, to fill it, we would do 100 5b5 predictions. '''

    # if you think about cell 0, a 25 flat array prediction centered around it will have
    # a padding of 12 and 12 at each size. So we need to add these slots for extra pad
    total_size = newsize + 24
    # length of the sequence, and number of entries in the batch
    seq_length = motors.shape[1]
    entries = motors.shape[0]
    # initial bz state set to all 0, no oscillations
    # bz_state = np.ones([entries, seq_length, total_size])*0
    sizeo = seq_length*entries*total_size
    bz_state = np.random.uniform(0,1,sizeo).reshape([entries,seq_length,total_size])
    # where to store the final results
    result = np.empty([entries, 0, total_size]) 

    for i in range(seq_length*2):
        print(i)

        # where to store new oscillation
        new_entry = np.zeros([entries,1,total_size])
        # First we will prepare the batches, and then we will do the prediction
        m_batch = np.empty( [0, seq_length, 25])
        bz_batch = np.empty( [0, seq_length, 25])
        
        # original BZ is 25, so we consider padding of 12 to the left
        for cell in range(12, 12 + newsize):

            # if cell is 12, it's the first case, we will use it to also fill the pad 0..11
            if cell == 12:
                m_batch = np.concatenate( [m_batch, motors[:,:,:25] ], axis = 0)
                bz_batch = np.concatenate( [bz_batch, bz_state[:,:,:25] ], axis = 0)
                # # make the prediction using the model
                # _, predict = model.predict([motors[:,:,:25], bz_state[:,:,:25]])
                # # get last and save it
                # new_entry[:,0,:cell+1] = predict[:,-1,:cell+1]

            # last cell, fill padding from this prediction
            elif cell == (12 + newsize - 1):
                lastbz = total_size - 25
                m_batch = np.concatenate( [m_batch, motors[:,:,lastbz:] ], axis = 0)
                bz_batch = np.concatenate( [bz_batch, bz_state[:,:,lastbz:] ], axis = 0)
                
                # _, predict = model.predict([motors[:,:,lastbz:], bz_state[:,:,lastbz:]])
                # new_entry[:,0,total_size-13:] = predict[:,-1,12:]

            else: # any of the other cells in between, we only get and update center cell
                ic = cell - 12
                m_batch = np.concatenate( [m_batch, motors[:,:,ic:25+ic] ], axis = 0)
                bz_batch = np.concatenate( [bz_batch, bz_state[:,:,ic:25+ic] ], axis = 0)
                
                # _, predict = model.predict([motors[:,:,ic:25+ic], bz_state[:,:,ic:25+ic]])
                # new_entry[:,0,cell] = predict[:,-1,12]

        # make the big prediction
        _, predict = model.predict( [m_batch, bz_batch] )

        # now we need to loop through the prediction to fill new_entry vector
        for c, p in enumerate(predict):
            # store center cell of last prediction
            new_entry[:,0,c+12] = p[-1,12]

            # and also fill the padding before and after
            if c == 0: # padding to the left
                new_entry[:,0,:12] = p[-1,:12]
            if c == (newsize-1): # padding to the right
                new_entry[:,0,total_size-12:] = p[-1,13:]

        # add to predictions last element of the prediction
        bz_state = np.concatenate([bz_state, new_entry], axis=1)
        # remove oldest entry to keep seq_len size
        bz_state = bz_state[:,1:,:]
        # add to final result
        result = np.concatenate([result, new_entry], axis=1)

    return result
