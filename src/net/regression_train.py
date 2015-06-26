from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
import keras
import argparse
parser = argparse.ArgumentParser(description='Train Regression model using Keras')
parser.add_argument('-r', type=float, default=0.8 , help='Ratio in which we need to split the data')
parser.add_argument('-f', type=str, help='H5 File Name')
parser.add_argument('--name', type=str, help='Model Name')
args = parser.parse_args()


def decide_split(h5_file, ratio=0.8):
	import h5py as h5
	f = h5.File(h5_file, 'r')
	size = f['label'].shape[0]
	training_size = int(0.8*size)
	return (0,training_size, training_size+1, size)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

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

#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

h5_file = args.f 
r = args.r
(train_start, train_end, test_start, test_end) = decide_split(h5_file, r)
print (train_start, train_end, test_start, test_end)
X_train = HDF5Matrix(h5_file, 'data', train_start, train_end)
X_test = HDF5Matrix(h5_file, 'data', test_start, test_end)
y_train = HDF5Matrix(h5_file, 'label', train_start, train_end)
y_test = HDF5Matrix(h5_file, 'label', test_start, test_end)

history = LossHistory()
checkpointer = ModelCheckpoint(filepath="/models/regression/weights_%s.hdf5"%args.name, verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=1000, nb_epoch=1, shuffle=False, callbacks=[history, checkpointer])
#print history.losses

#save the last model
import cPickle as pkl
import sys
sys.setrecursionlimit(40000)
pkl.dump(model, open("models/regression/model_%s.pkl"%args.name, "w"))
pkl.dump(history.losses, open("models/regression/losses_%s.pkl"%args.name, "w"))
#evaluate the model
score = model.evaluate(X_test, y_test, batch_size=2048)