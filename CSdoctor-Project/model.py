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



# Model (3D-CNN + LSTM)
# Model Start
model = Sequential()
model.add(Reshape(input_shape=input_shape, target_shape=(50, 8, 8, 8, 1)))

# Attention
attention_model = Sequential()
attention_model.add(model)
attention_model.add((Activation('softmax', name='Attention_softmax')))
middle_model = Sequential()
middle_model.add(Merge([model, attention_model], mode='mul'))

# 3D-CNN
middle_model.add(TimeDistributed(Convolution3D(64, 3, 3, 3, border_mode='valid', activation='tanh')))
middle_model.add(TimeDistributed(Convolution3D(128, 3, 3, 3, border_mode='valid', activation='tanh')))
middle_model.add(TimeDistributed(Convolution3D(256, 3, 3, 3, border_mode='valid', activation='tanh')))
middle_model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))
middle_model.add(TimeDistributed(Activation('tanh')))
middle_model.add(TimeDistributed(Dropout(0.25)))
middle_model.add(TimeDistributed(Flatten()))

# Attention
attention_model = Sequential()
attention_model.add(middle_model)
attention_model.add((Activation('softmax', name='Attention_softmax2')))
final_model = Sequential()
final_model.add(Merge([middle_model, attention_model], mode='mul'))

# LSTM
final_model.add(TimeDistributed(Dense(128)))
final_model.add(LSTM(50, return_sequences=False))
final_model.add(Dense(output_shape))

model.summary()
middle_model.summary()
attention_model.summary()
final_model.summary()

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
