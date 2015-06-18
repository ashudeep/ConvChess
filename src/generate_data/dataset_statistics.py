import numpy as np
import sys
piece_y_re = re.compile("y_[0-9]+_*")
move_y_re = [re.compile("p"+str(i+1)+"_y_*") for i in xrange(6)]
DIR = sys.argv[1]
npy_files = os.listdir(DIR)
npy_files.sort()
piece_label_files = [f for f in npy_files if piece_y_re.match(f)]
move_label_files = [[f for f in npy_files if move_y_re[i].match(f)] for i in xrange(6)]

#initilize out numbers
num_moves = 0

for i in xrange(6):
	for fil in move_label_files[i]:
		y = np.load(fil)
		num_games+=y.shape[0]

print num_games