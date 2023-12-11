import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import numpy as np
import tensorflow as tf

print('--------------------- Start ---------------------')
start_time = time.time()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# with tf.device('/GPU:0'):
#   result = 0
#   for i in range(1_000_000_001):
#     result += i

end_time = time.time()

# elapsed_time = end_time - start_time
# print("Result:", result)
# print("Elapsed Time:", elapsed_time, "seconds")