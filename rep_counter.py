import os
from scipy.signal import find_peaks
import numpy as np

class RepCounter:
    '''repetition counter class to count the number of cycle counts on a 1D signal

    Args:
        max_buffer_size: maximum size of the buffer before it resets automatically (not implemented yet)
        min_buffer_size: minimum size of buffer to start finding the peaks in the signal 
        distance: option for scipy find_peaks function. minimum distance between two peaks 
        prominence: option for scipy find_peaks function. minimum vertical distance to neighbours to be considered as peak
        filter_constant: constant for the filter applied to the buffer signal
        use_filter: if true, filter is used otherwise raw data will be used

    count(x):
        return the number of cycles detected in the incoming signal so far

    reset():
        resets the buffer and number of reps 
    '''    
    def __init__(self, max_buffer_size=1000, min_buffer_size=20, 
        distance=10, prominence=0.0, filter_constant=10, use_filter=True):
        self.distance = distance
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.prominence = prominence

        self.use_filter = use_filter
        if use_filter:
            if filter_constant <= 0:
                raise('filter constant cannot be zero')
            else:
                self.filter_constant = filter_constant

        self.reset()

    '''calculate the number of cycle counts on a 1D signal
    
    this function is called every time step and receives x representing the amplitude
    of a signal on a time step. incoming x values in each time step is accumulated 
    in a buffer.
    this can be the y-position of human head in each pose detection
    time step to return the number of reps for an activity

    Args:
        x: amplitude of a signal in a time step (float) 

    Returns:
        rep_count: number of cycles detected in the incoming signal so far
    '''
    def count(self, x):
        self.buffer.append(x)
        self.frame_count += 1

        if len(self.buffer) < self.min_buffer_size:
            return self.rep_count 
        
        else:
            # do the peak counting every n-th frame. here n=self.distance/2
            if self.frame_count % int(self.distance/2) == 0:
                if self.use_filter:
                    self.buffer_filtered = self.loess1d(self.buffer, self.filter_constant)
                    peaks = find_peaks(self.buffer_filtered, distance=self.distance, prominence=self.prominence)
                else:
                    peaks = find_peaks(self.buffer, distance=self.distance, prominence=self.prominence)
            
            else:
                return self.rep_count 
        
        self.rep_count = len(peaks[0])
        return self.rep_count


    def loess1d(self, y, c):
        n = len(y)
        x = np.arange(start=0, stop=n)
        y_filtered = np.zeros(n)

        #calculating the weight
        w = np.array([np.exp(- (x- x[i])**2/(2*c)) for i in range(n)])
    
        for i in range(n):
            weights = w[:, i]
            slopeNumerator = (np.sum(weights * x) * np.sum(weights * y)) - (np.sum(weights) * np.sum(weights * x * y))
            slopeDenominator = (np.sum(weights * x) * np.sum(weights * x)) - (np.sum(weights) * ((np.sum(weights*x))**2))
            A = slopeNumerator / slopeDenominator
            B = (np.sum(weights * y) - (A*np.sum(weights * x))) / np.sum(weights)
            y_filtered[i] = B + A * x[i]

        return y_filtered

    '''
    reset the counter and buffer
    '''
    def reset(self):
        self.rep_count = 0
        self.frame_count = 0
        self.buffer = []
        self.buffer_filtered = []