# ConvChess
Convolutional Neural Networks learn chess moves

I must admit chess has already been solved and using an approximate solution to evaluate the board or moves is not the best idea. However, I wanted to try how game playing with pattern recognition methods works e.g. in recent success on Go. Feel free to use/reuse the code. There are some good insights in the ipython notebooks so as to investigate the evaluation function and action prediction models. You can always post an issue if you feel something is wrong/outdated/missing.

The description of the model and other relevant details can be found in the docs directory.

The source code consists of four different parts:

1. Data Generation: There are two steps to generating the data (in the form of matrices and labels/scores) from the pgn format set of games:
	- Run get_train_data_large.py with appropriate options. The options list can be viewed by: `python get_train_data_large.py -h `. This will convert the pgn formats to the matrices and labels/scores format into npy files in the specified folder. 
	- Then using the output folder in the generated files, run the file npy_to_hdf5.py. An example case is: `python npy_to_hdf5.py --dir npy_files --regr`.
Now you are done with the dataset generation. You will now have a folder full of hdf5 files that can be used as training data for the convolutional neural networks in one of the next parts.
2.  Training CNNs for move prediction: We divide the task of predicting moves on a given chess board into two parts: Predicting the piece and then predicting the move given the piece. The data generated in part 1 is organized in a way that can be used to train 7 different networks-- one for the piece predictor and rest 6 for each of the piece types. For training, the best way is to run caffe directly using one of the solvers in the directory src/net/solvers. The solvers are named according to their task. To train using caffe: `path/to/caffe/caffe train --solver net/solvers/piece_solver.prototxt --snapshot 5000 --`. Or you can use the file train.py with relevant arguments (check using `python train.py -h`.
3. Training CNNs for move evaluation:  This directly uses regression training on a CNN to learn the evaluation function as specified while generating the data. The code uses Keras (a deep learning library based on Theano). To start the training, you will need to run regression_train.py. You can change the network configuration inside the regression_train.py file.
<img width="641" height="496" alt="image" src="https://github.com/user-attachments/assets/dd891f10-de78-456c-84cf-65f01b654719" />

4. Playing: All the code related to playing chess against a human being or sunfish is present in the src/play directory. The files playN.py contain different scenarios of playing against a computer or a human being. Relevant comments are given in the code itself. 


Some interesting figures demonstrating the capabilties:
<img width="680" height="723" alt="image" src="https://github.com/user-attachments/assets/95642605-3f30-4adf-8c74-1d88091defb2" />
<img width="709" height="733" alt="image" src="https://github.com/user-attachments/assets/7d593d3d-5053-4db1-88ec-952b49b57ade" />
<img width="403" height="624" alt="image" src="https://github.com/user-attachments/assets/90045bca-568e-4fa9-9aef-7005861e3ad6" />

