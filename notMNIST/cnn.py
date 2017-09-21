import numpy as np
import os
from sklearn.cross_validation import train_test_split
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Merge, Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adadelta
from keras import backend as K
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

batch_size = 128
num_classes = 10
epochs = 20
img_rows, img_cols = 28, 28


# reference : http://enakai00.hatenablog.com/entry/2016/08/02/102917
class NotMNIST:
    def __init__(self):
        images, labels = [], []

        for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']):

            directory = 'C://Users/Choi Seung Yeon/Desktop/2017-2 project/sc/notMNIST_large/%s/' % letter

            files = os.listdir(directory)
            label = np.array([0] * 10)
            label[i] = 1
            for file in files:
                try:
                    im = Image.open(directory + file)
                except:
                    print
                    "Skip a corrupted file: " + file
                    continue
                pixels = np.array(im.convert('L').getdata())
                images.append(pixels / 255.0)
                labels.append(label)

        train_images, test_images, train_labels, test_labels = \
            train_test_split(images, labels, test_size=0.2, random_state=0)

        class train:
            def __init__(self):
                self.images = []
                self.labels = []
                self.batch_counter = 0

            def next_batch(self, num):
                if self.batch_counter + num >= len(self.labels):
                    batch_images = self.images[self.batch_counter:]
                    batch_labels = self.labels[self.batch_counter:]
                    left = num - len(batch_labels)
                    batch_images.extend(self.images[:left])
                    batch_labels.extend(self.labels[:left])
                    self.batch_counter = left
                else:
                    batch_images = self.images[self.batch_counter:self.batch_counter + num]
                    batch_labels = self.labels[self.batch_counter:self.batch_counter + num]
                    self.batch_counter += num

                return (batch_images, batch_labels)

        class test:
            def __init__(self):
                self.images = []
                self.labels = []

        self.train = train()
        self.test = test()

        self.train.images = train_images
        self.train.labels = train_labels
        self.test.images = test_images
        self.test.labels = test_labels


mnist = NotMNIST()

x_train = np.array(mnist.train.images)
y_train = np.array(mnist.train.labels)
x_test = np.array(mnist.test.images)
y_test = np.array(mnist.test.labels)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    
    
model = Sequential()
model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', padding='same', strides=(1,1), input_shape=input_shape))
model.add(MaxPooling2D((2, 2), padding='valid'))

model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same', strides=(1,1)))
model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', padding='same', strides=(1,1)))
model.add(MaxPooling2D((2, 2), padding='valid'))

model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', padding='same', strides=(1,1)))
model.add(MaxPooling2D((2, 2), padding='valid'))

model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', padding='same', strides=(1,1)))
model.add(MaxPooling2D((2, 2), padding='valid'))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=adadelta,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
