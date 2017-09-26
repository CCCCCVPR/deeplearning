# 3D-CNN + LSTM + Attention
# Yonsei Univ. Computer Science. Team CSdoctor
# Yu Ji-Sang, Yoo Tae-Kwon, Choi Seung-Yeon
# Email : das135@naver.com, Choi Seung-Yeon


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


# Training Data
def trainGenerator():
    for data in pd.read_csv('E://csv_data/ex/train.csv', chunksize=100):
        x_train = data.iloc[:,1:].values
        y_train = np_utils.to_categorical(data.iloc[:,0].values)        
        yield (x_train, y_train)

        
# Test Data        
def testGenerator():
    for data in pd.read_csv('E://csv_data/ex/test.csv', chunksize=100):
        x_test = data.iloc[:,1:].values
        y_test = np_utils.to_categorical(data.iloc[:,0].values)        
        yield (x_test, y_test)

        
        
# input_shape = (none, 50000)
input_shape = (x_train.shape[1], )
output_shape = y_train.shape[1]


# Model (MLP)
# Model Start
final_model = Sequential()
final_model.add(Dense(500, input_shape=input_shape))
final_model.add(Activation('tanh'))
final_model.add(Dense(500))
final_model.add(Activation('tanh'))
final_model.add(Dense(500))
final_model.add(Activation('tanh'))
final_model.add(Dropout(0.25))
final_model.add(Dense(output_shape))
final_model.add(Activation('softmax'))



# Compile
final_model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Training Start
history = final_model.fit_generator(trainGenerator(),
                    steps_per_epoch = 1000, 
                    validation_data=testGenerator(),
                    validation_steps = 1000,
                    verbose=1
                    )


# Result
score = final_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
