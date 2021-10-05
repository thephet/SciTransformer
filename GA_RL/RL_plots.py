################################################################################
#
# Just generate plots as in the RL section of the manuscript
#
################################################################################

import pickle
import numpy as np
import matplotlib.pyplot as plt



pfile = "outs/conpos/conpos.p"
data = pickle.load( open( pfile, "rb"))
motors, predictions = data
motors = motors[0]
predictions = predictions[0]



size=40
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
plt.rcParams.update(params)


plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.size"] = 20

plt.plot((motors[:, :12]*2)-1, linewidth=2)
plt.plot((motors[:, 13:]*2)-1, linewidth=2)
plt.plot((motors[:, 12]*2)-1, linewidth=9)

#plt.plot(np.power(np.cumsum(predictions[:,:12], axis=0),2), linewidth=2)
#plt.plot(np.power(np.cumsum(predictions[:,13:], axis=0),2), linewidth=2)
#plt.plot(np.power(np.cumsum(predictions[:,12], axis=0),2), linewidth=9)

#plt.plot(np.cumsum(predictions[:,:12], axis=0), linewidth=2)
#plt.plot(np.cumsum(predictions[:,13:], axis=0), linewidth=2)
#plt.plot(np.cumsum(predictions[:,12], axis=0), linewidth=2)

plt.show()
