from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix


model = Sequential()
model.add(Convolution2D(32, 6, 3, 3)) 
model.add(Activation('relu'))
model.add(Convolution2D(64, 32, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(256, 64, 3, 3)) 
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(1024, 256))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(256, 1))
#model.add(Activation('softmax'))

# from keras.utils.dot_utils import Grapher
# g = Grapher()
# g.plot(model, 'mymodel.png')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

import numpy as np
# X_train = HDF5Matrix("../../../ConvChess/data/CvC_regression_g05_single_h5/piece.h5", 'data', 0 , 40000000)
# X_test = HDF5Matrix("../../../ConvChess/data/CvC_regression_g05_single_h5/piece.h5", 'data', 40000001 , 44899683)
# Y_train = HDF5Matrix("../../../ConvChess/data/CvC_regression_g05_single_h5/piece.h5", 'label', 0 , 40000000)
# Y_test = HDF5Matrix("../../../ConvChess/data/CvC_regression_g05_single_h5/piece.h5", 'label', 40000001 , 44899683)
X_train = HDF5Matrix("../../../ConvChess/data/sample_regression_h5/piece.h5", 'data', 0 , 40000)
X_test = HDF5Matrix("../../../ConvChess/data/sample_regression_h5/piece.h5", 'data', 40001 , 44083)
Y_train = HDF5Matrix("../../../ConvChess/data/sample_regression_h5/piece.h5", 'label', 0 , 40000)
Y_test = HDF5Matrix("../../../ConvChess/data/sample_regression_h5/piece.h5", 'label', 40001 , 44083)
model.fit(X_train, Y_train, batch_size=1000, nb_epoch=1)

score = model.evaluate(X_test, Y_test, batch_size=16, shuffle=False)