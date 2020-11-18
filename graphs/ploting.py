import numpy as np
from scipy.signal import lfilter, filtfilt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
import os

def get_data(path):
    tf_size_guidance = {
        'compressedHistograms': 1,
        'images': 0,
        'scalars': 100*10,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()
    default_acc = event_acc.Tags()['scalars']
    output = {}
    for name in default_acc:
        scalar = event_acc.Scalars(name)
        hist = {'x' : np.zeros(len(scalar)), 'y': np.zeros(len(scalar))}
        for i, event in enumerate(scalar):
            hist['x'][i] = event.step
            hist['y'][i] = event.value
        output[name] = hist.copy()
    return output
    
def plot_val(value, color='b', label=''):
    n=5
    yy = filtfilt([1.0 / n] * n,1,value['y'])
    plt.plot(value['x'], value['y'], alpha=0.3, linewidth=2, color=color, label=label)
    plt.plot(value['x'], yy, linewidth=2, color=color)
    plt.grid()
    plt.xlabel('steps')
