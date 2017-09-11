# 3D-CNN + LSTM + Attention
# Yonsei Univ. Computer Science. Team CSdoctor
# Yu Ji-Sang, Yoo Tae-Kwon, Choi Seung-Yeon


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
# Model Start
model = Sequential()
model.add(Reshape(input_shape=input_shape, target_shape=(50, 10, 10, 10, 1)))

# 3D-CNN
model.add(TimeDistributed(Convolution3D(64, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(128, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(256, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(Convolution3D(256, 3, 3, 3, border_mode='valid', activation='tanh')))
model.add(TimeDistributed(MaxPooling3D(pool_size=(2, 2, 2))))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Dropout(0.25)))
model.add(TimeDistributed(Flatten()))

# Attention
attention_model = Sequential()
attention_model.add(model)
attention_model.add(TimeDistributed(Dense(256, activation='softmax')))
final_model = Sequential()
final_model.add(Merge([model, attention_model], mode='mul'))

# LSTM
final_model.add(TimeDistributed(Dense(128)))
final_model.add(LSTM(50, return_sequences=False))
final_model.add(Dense(output_shape))

model.summary()
attention_model.summary()
final_model.summary()


# Compile
final_model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Training Start
history = final_model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


# Result
score = final_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
