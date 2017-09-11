import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import pandas as pd
import numpy as np
from keras.layers import Reshape, Convolution3D, Activation, MaxPooling3D, Dropout, Flatten, Dense, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adadelta
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

batch_size = 500
epochs = 64
nb_classes = 2

# Data parsing
data = pd.read_csv('E://capstone/data.csv', header = None)
data = data.reindex(np.random.permutation(data.index))
train, test = train_test_split(data, train_size=0.9, random_state=0)
x_train, x_test = train.iloc[:,1:].values, test.iloc[:,1:].values
y_train, y_test = np_utils.to_categorical(train.iloc[:,0].values), np_utils.to_categorical(test.iloc[:,0].values)

# input_shape = (none, 50000)
input_shape = (x_train.shape[1], )
output_shape = y_train.shape[1]

# Model (3D-CNN + LSTM)
model = Sequential()
model.add(Reshape(input_shape=input_shape, target_shape=(50, 10, 10, 10, 1)))

model.add(TimeDistributed(Convolution3D(64, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(128, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(256, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(256, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Dropout(0.25)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(128)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(output_shape))

model.summary()


model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training Start
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
