import numpy as np
import sys
import re, os
import matplotlib.pyplot as plt

piece_y_re = re.compile("y_[0-9]+_*")
move_y_re = [re.compile("p"+str(i+1)+"_y_*") for i in xrange(6)]
DIR = sys.argv[1]
npy_files = os.listdir(DIR)
npy_files.sort()
piece_label_files = [f for f in npy_files if piece_y_re.match(f)]
move_label_files = [[f for f in npy_files if move_y_re[i].match(f)] for i in xrange(6)]
PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
INDEX_TO_PIECE = {0 : 'P', 1 : 'R', 2 : 'N', 3 : 'B', 4 : 'Q', 5 : 'K'}
ranges = ()
num_games=0
for fil in move_label_files[0]:
	num_games = max(int(fil.split('.')[0].split('_')[3]), num_games)
print "%d games"%num_games

#initilize out numbers
num_moves = np.zeros(6, dtype=np.uint64)
for i in xrange(6):
	for fil in move_label_files[i]:
		y = np.load(DIR+"/"+fil)
		if ".npz" in fil:
			num_moves[i]+=y['arr_0'].shape[0]
		else:
			num_moves[i]+=y.shape[0]

for index in INDEX_TO_PIECE:
	print "%s has %d moves"%(INDEX_TO_PIECE[index], num_moves[index]) 

print "Total %d moves"%np.sum(num_moves)

fig, ax = plt.subplots()
ax.bar(np.arange(6), num_moves, color='g')
ax.set_xticklabels(INDEX_TO_PIECE.values()) 
ax.set_xticks(np.arange(6)+0.35)
ax.set_xlabel('Different piece types')
ax.set_ylabel('Number of moves')
fig.suptitle('Total %d moves from %d games'%(np.sum(num_moves), num_games) )
plt.show()