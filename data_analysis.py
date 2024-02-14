import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import load_data

f,ax = plt.subplots()
f6,ax6 = plt.subplots()
data, labels = load_data('data/')
colors = ['indianred','darkblue','gray']
c0 = []
c1 = []
c2 = []

ff0 = []
ff1 = []
ff2 = []
for i in range(data.shape[0]):
    if labels[i] == 0:
        c0.append(data[i,:])
        ff0.append(np.fft.fft(data[i,:])) 
    elif labels[i] == 1:
        c1.append(data[i,:])
        ff1.append(np.fft.fft(data[i,:])) 
    elif labels[i] == 2:
        c2.append(data[i,:])
        ff2.append(np.fft.fft(data[i,:])) 
    ax.plot(data[i,:],c=colors[labels[i]],linewidth=1)
    ax6.plot(np.abs(np.fft.fft(data[i,:])), c=colors[labels[i]], linewidth=1)

f2,ax2 = plt.subplots()
for i in range(data.shape[0]):
    ax2.plot(data[i,:]-np.mean(data[i,:]),c=colors[labels[i]],linewidth=1)

f3,ax3 = plt.subplots()
ax3.hist(np.array(c0).ravel(),bins=35,color='indianred',alpha=.5)
ax3.hist(np.array(c1).ravel(),bins=35,color='darkblue',alpha=.5)
ax3.hist(np.array(c2).ravel(),bins=35,color='gray',alpha=.5)

f4, ax4 = plt.subplots()
ax4.hist(np.var(np.array(c0),axis=1),bins=35,color='indianred',alpha=.5)
ax4.hist(np.var(np.array(c1),axis=1),bins=35,color='darkblue',alpha=.5)
ax4.hist(np.var(np.array(c2),axis=1),bins=35,color='gray',alpha=.5)

# FFT
f5,ax5 = plt.subplots()
ax3.hist(np.array(ff0).ravel(),bins=35,color='indianred',alpha=.5)
ax3.hist(np.array(ff1).ravel(),bins=35,color='darkblue',alpha=.5)
ax3.hist(np.array(ff2).ravel(),bins=35,color='gray',alpha=.5)

plt.show() 
