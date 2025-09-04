from flax import nnx
from functools import partial
import optax

# Example https://flax.readthedocs.io/en/latest/mnist_tutorial.html

class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    @classmethod
    def default(cls):
        return cls(rngs=nnx.Rngs(0))


def create_optimizer(model):
    learning_rate = 0.005
    momentum = 0.9

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    return optimizer

def create_metrics():
    metrics = nnx.MultiMetric(
      accuracy=nnx.metrics.Accuracy(),
      loss=nnx.metrics.Average('loss'),
    )
    return metrics


def loss_fn(model: CNN, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']
    ).mean()
    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
      """Train for a single step."""
      grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
      (loss, logits), grads = grad_fn(model, batch)
      metrics.update(loss=loss, logits=logits, labels=batch['label'])
      optimizer.update(grads)


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
      loss, logits = loss_fn(model, batch)
      metrics.update(loss=loss, logits=logits, labels=batch['label'])


def fit(model, optimizer, metrics, train_ds, test_ds):
    train_steps = 1200
    eval_every = 200
    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }

    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        # - The training loss and accuracy batch metrics
        model.train()  # Switch to train mode
        train_step(model, optimizer, metrics, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
            # Log the training metrics.
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            model.eval()  # Switch to eval mode
            for test_batch in test_ds.as_numpy_iterator():
                eval_step(model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
                print(f'test_{metric}: {value:.4f}')
            metrics.reset()  # Reset the metrics for the next training epoch.

    return model, optimizer, metrics_history


class ModelTrainer:
    def __init__(self, model, optimizer, metrics, train_ds, test_ds):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.train_ds = train_ds
        self.test_ds = test_ds

    @classmethod
    def default(cls, train_ds, test_ds):
        model = CNN.default()
        optimizer = create_optimizer(model)
        metrics = create_metrics()
        return cls(model, optimizer, metrics, train_ds, test_ds)

    def fit(self):
        return fit(self.model, self.optimizer, self.metrics, self.train_ds, self.test_ds)
