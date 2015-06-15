# coding: utf-8
import numpy as np
data = np.load("/data/ConvChess/data/train_data/small_data.pkl.npy")
labels = np.load("/data/ConvChess/data/train_data/small_labels.pkl.npy")
import caffe
solver = caffe.SGDSolver ("solverPIECE.prototxt")
caffe.set_mode_gpu()
solver.net.set_input_arrays (data,labels )
