# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Approach 1: loading MNIST dataset from keras datasets into numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# %%
# Approach 2: loading MNIST dataset using TFDS
(ds_train, ds_test), info = tfds.load("mnist", split=['train', 'test'], with_info=True)

# Scale images to the [0, 1] range
def normalize(el):
        return  {
            'image': tf.cast(el['image'], tf.float32)/255,
            'label': el['label']
        }

ds_train = ds_train.map(normalize)
ds_test = ds_test.map(normalize)

print(ds_train.element_spec)

# %%
# Converting approach 1 into approach 2 i.e. NumPy -> tf.data

# First add an extra axis for channel information to match the tfds format
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# We can also change the type of the label from tf.uint8 to tf.int64 to match the
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

# Construct datasets from tensor slices
ds_train_keras = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': y_train})
print(ds_train_keras.element_spec)

# %%
# Converting back to numpy
n_rows = 5
n_cols = 4
fig, axs = plt.subplots(n_rows, n_cols)
axs = axs.flatten()
ds_eights = ds_train.filter(lambda x: x['label'] == 8)
for i, el in enumerate(ds_eights.take(n_rows*n_cols)):
    axs[i].imshow(el['image'].numpy(), cmap='gray')
    axs[i].set_title(el['label'].numpy())
    axs[i].axis('off')

plt.tight_layout()
plt.show()