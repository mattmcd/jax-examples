from keras.layers import Dense, Flatten, Conv2D, AvgPool2D, Input
from keras import Model


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.avg_pool = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        self.flatten = Flatten()
        self.linear1 = Dense(256, activation='relu')
        self.linear2 = Dense(10, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class CVAE(Model):
    def __init__(self):
        super(CVAE, self).__init__()
