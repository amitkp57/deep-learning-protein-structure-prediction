import numpy as np
import tensorflow as tf

def get_data_from_filename(filename):
    filename = filename.numpy().decode('utf-8')
    data = np.load(filename, allow_pickle=True)
    return data['primary'], data['secondary']


def get_data_wrapper(filename):
    # Assuming here that both your data and label is float type.
    primary, secondary = tf.py_function(
        get_data_from_filename, [filename], (tf.float32, tf.float32))
    return tf.data.Dataset.from_tensor_slices((primary, secondary))


# Create dataset of filenames.
ds = tf.data.Dataset.from_tensor_slices(['C:\\Users\\amitk\\Downloads\\npz\\protein_21.npz'])
ds = ds.flat_map(get_data_wrapper)
ds = ds.batch(10)

for x, y in ds:
    print(x, y)
    pass