# ConvChess
Convolutional Neural Networks learn chess moves

The source code consists of three different parts:

1. Data Generation: There are two steps to generating the data (in the form of matrices and labels/scores) from the pgn format set of games:
	- Run get_train_data_large.py with appropriate options. The options list can be viewed by: `python get_train_data_large.py -h `. This will convert the pgn formats to the matrices and labels/scores format into npy files in the specified folder. 
	- Then using the output folder in the generated files, run the file npy_to_hdf5.py. An example case is: `python npy_to_hdf5.py --dir npy_files --regr`.
Now you are done with the dataset generation. You will now have a folder full of hdf5 files that can be used as training data for the convolutional neural networks in one of the next parts.
2.  Training CNNs for move prediction: We divide the task of predicting moves on a given chess board into two parts: Predicting the piece and then predicting the move given the piece. The data generated in part 1 is organized in a way that can be used to train 7 different networks-- one for the piece predictor and rest 6 for each of the piece types. For training, the best way is to run caffe directly using one of the solvers in the directory src/net/solvers. The solvers are named according to their task. To train using caffe: `path/to/caffe/caffe train --solver net/solvers/piece_solver.prototxt --snapshot 5000 --`. Or you can use the file train.py with relevant arguments (check using `python train.py -h`.
3. Training CNNs for move evaluation:  This directly uses regression training on a CNN to learn the evaluation function as specified while generating the data. The code uses Keras (a deep learning library based on Theano). To start the training, you will need to run regression_train.py. You can change the network configuration inside the regression_train.py file.
4. Playing: All the code related to playing chess against a human being or sunfish is present in the src/play directory. The files playN.py contain different scenarios of playing against a computer or a human being. Relevant comments are given in the code itself. 
