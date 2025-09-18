# %%
from pathlib import Path
import os
os.environ['JAX_PLATFORMS'] = 'METAL'
import keras
import numpy as np
import pandas as pd
from mnist.io import load_mnist_tfds
from mnist.keras import CNN
import optax
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
# %%
ds_train, ds_test = load_mnist_tfds()

# %%
model = CNN()

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005, ema_momentum=0.9),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    jit_compile=True,
)

# %%
# Could have used the as_supervised=True argument to tfds.load
ds_train = ds_train.map(lambda x: (x['image'], x['label']))
ds_test = ds_test.map(lambda x: (x['image'], x['label']))

# %%
history = model.fit(
    ds_train, epochs=6, validation_data=ds_test, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
)

# %%
model.save('mnist_cnn_keras.keras')
