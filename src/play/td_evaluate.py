import keras

import cPickle as pkl
model = pkl.load(open(pkl_file, 'r'))

#X is the image of the board position 
model.predict_proba(X)