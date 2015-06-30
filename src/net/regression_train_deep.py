from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
import keras
import cPickle as pkl
import sys
import argparse
sys.setrecursionlimit(40000)
parser = argparse.ArgumentParser(description='Train Regression model using Keras')
parser.add_argument('-r', type=float, default=0.8 , help='Ratio in which we need to split the data')
parser.add_argument('-f', type=str, help='HDF5 File Name')
parser.add_argument('--name', type=str, help='Model Name')
parser.add_argument('-n', type=int, help='number of epocs', default=1)
parser.add_argument('--cont', type=str, help='model weights to start training from.', default='')
parser.add_argument('-b', type=int, default=1024, help='Batch size while training')
parser.add_argument('--bt', type=int, default=2048, help='Batch size while testing')
args = parser.parse_args()


def decide_split(h5_file, ratio=0.8):
	import h5py as h5
	f = h5.File(h5_file, 'r')
	size = f['label'].shape[0]
	f.close()
	training_size = int(r*size)
	return (0,training_size, training_size+1, size)

if not args.cont:
	model = Sequential()
	model.add(Convolution2D(32, 6, 3, 3, border_mode='full')) 
	model.add(Activation('tanh'))
	
	model.add(Convolution2D(64, 32, 3, 3))
	model.add(Activation('tanh'))

	model.add(Convolution2D(128, 64, 3, 3)) 
	model.add(Activation('tanh'))

	model.add(Convolution2D(256, 128, 3, 3)) 
	model.add(Activation('tanh'))

	model.add(Flatten())
	model.add(Dense(256*4*4, 1024))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(1024, 1))
	model.add(Activation('tanh'))

	model.compile(loss='mean_squared_error', optimizer='rmsprop')
else:
	model = pkl.load(open(args.cont,'r'))

h5_file = args.f 
r = args.r
(train_start, train_end, test_start, test_end) = decide_split(h5_file, r)
print "Training set from %d to %d, Testing set from %d to %d"%(train_start, train_end, test_start, test_end)
X_train = HDF5Matrix(h5_file, 'data', train_start, train_end)
X_test = HDF5Matrix(h5_file, 'data', test_start, test_end)
y_train = HDF5Matrix(h5_file, 'label', train_start, train_end)
y_test = HDF5Matrix(h5_file, 'label', test_start, test_end)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()

import os
if not os.path.isdir("./models/regression"):
	os.mkdir("./models/regression")

model.fit(X_train, y_train, batch_size=args.b, nb_epoch=args.n, shuffle=False, callbacks=[history])

pkl.dump(model, open("./models/regression/model_%s.pkl"%args.name, "w"))
pkl.dump(history.losses, open("./models/regression/losses_%s.pkl"%args.name,"w"))

score = model.evaluate(X_test, y_test, batch_size=args.bt)