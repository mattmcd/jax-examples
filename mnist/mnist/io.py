import keras
import tensorflow as tf
import tensorflow_datasets as tfds


def load_mnist():
    # mnist = keras.datasets.mnist.load_data()
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    return x_train, x_test, y_train, y_test

def load_mnist_tfds():
    # Copied from https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/mnist_tutorial.html#load-the-mnist-dataset
    tf.random.set_seed(0)  # set random seed for reproducibility

    num_epochs = 10
    batch_size = 32

    train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
    test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

    train_ds = train_ds.map(
        lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
        }
    )  # normalize train set
    test_ds = test_ds.map(
        lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label'],
        }
    )  # normalize test set

    # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from
    test_ds = test_ds.shuffle(1024)
    # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    return train_ds, test_ds
