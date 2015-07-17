import numpy as np
import sys
import re, os
import matplotlib.pyplot as plt
plt.rc('text', usetex=False, antialiased=True)
plt.rc('font', family='serif', size=12)

piece_y_re = re.compile("y_[0-9]+_*")
move_y_re = [re.compile("p"+str(i+1)+"_y_*") for i in xrange(6)]

num_moves=np.zeros((len(sys.argv[1:]),6), dtype=np.uint32)
num_games = np.zeros(len(sys.argv[1:]), dtype=np.uint32)

PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}
INDEX_TO_PIECE = {0 : 'Pawn', 1 : 'Rook', 2 : 'Knight', 3 : 'Bishop', 4 : 'Queen', 5 : 'King'}
ranges = ()
fig, ax = plt.subplots()
color_cycle=['lightgrey', 'dodgerblue', 'y', 'r', 'k']
dir_names = ["CvC 2009-14", "FICS 2014"]
for i, DIR in enumerate(sys.argv[1:]):
	print DIR
	npy_files = os.listdir(DIR)
	npy_files.sort()
	piece_label_files = [f for f in npy_files if piece_y_re.match(f)]
	move_label_files = [[f for f in npy_files if move_y_re[k].match(f)] for k in xrange(6)]


	for fil in move_label_files[0]:
		ylim = int(fil.split('.')[0].split('_')[3])
		num_games[i] = max(ylim, num_games[i])
	print "%d games"%num_games[i]

	#initilize out numbers
	for j in xrange(6):
		for fil in move_label_files[j]:
			y = np.load(DIR+"/"+fil)
			if ".npz" in fil:
				num_moves[i,j]+=y['arr_0'].shape[0]
			else:
				num_moves[i,j]+=y.shape[0]

	for index in INDEX_TO_PIECE:
		print "%s has %d moves"%(INDEX_TO_PIECE[index], num_moves[i,index]) 

	print "Total %d moves"%np.sum(num_moves)
	
	ax.bar(np.arange(6)+0.25*i, num_moves[i], width=0.25, color=color_cycle[i], label=dir_names[i])
	ax.legend()

ax.set_xticklabels(INDEX_TO_PIECE.values()) 

ax.set_xticks(np.arange(6)+0.25)
ax.set_xlabel('Piece types')
ax.set_ylabel('Number of moves')
#fig.suptitle('Total %d moves from %d games'%(np.sum(num_moves), num_games) )
plt.show()

