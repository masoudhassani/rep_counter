from rep_counter import RepCounter
import numpy as np

counter = RepCounter(filter_constant=1, use_filter=True)
data = np.load('test_data.npy')
ref = data[:,21]

rep = 0
rep_prev = 0
for y in ref:
    rep = counter.count(y)
    
print('number of reps:',rep)    
