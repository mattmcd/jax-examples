# %%
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from mnist.io import load_mnist_tfds
from mnist.nnx import ModelTrainer
import optax
import matplotlib.pyplot as plt
import seaborn as sns
# %%
ds_train, ds_test = load_mnist_tfds()

# %%
model_trainer = ModelTrainer.default(ds_train, ds_test)

# %%
model, optimizer, metrics_history = model_trainer.fit()

# %%
df_m = pd.DataFrame(metrics_history).astype(float)

# %%
df_m[['train_loss', 'test_loss']].plot()
plt.show()

# %%
df_m[['train_accuracy', 'test_accuracy']].plot()
plt.show()

# %%
model.save('mnist_cnn_nnx.keras')