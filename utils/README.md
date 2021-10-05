This folder contains a few utils I used to either generate sequences or generate different videos or plots.

Initially they were in the root folder but I moved them to this folder for clarity. You might need to fix the imports.

* evaluate.py is the main script I used to given a Transformer model (a trained model as in a hdf5 file) generate sequences. It contains different options like generating from random states or from given sequences. It also contains functions to evaluate the quality of the generations and also generate results of size N by N as discussed in the last section of the manuscript.
* dataoscillations2video.py is the main function I used to generate the videos. Using evaluate.py you can generate sequences. Put them into a pickle file, and then this file will fetch the results and create a video.
* timemap.py will, given a video as created with the function above, it will generate a 2D image of the video where the frames are 1D flattened and concatenated.