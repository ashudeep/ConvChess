from __future__ import print_function
import numpy as np
import h5py as h5
import sys, os
import timeit
import argparse
parser=argparse.ArgumentParser\
	(description='ConvNet for learning Chess moves')
parser.add_argument('--dir', type=str, default='', help='The data directory')
parser.add_argument('--odir', type=str, default='', help='The output hdf5 data directory')
parser.add_argument('--dtype', type=str, default='', help='The datatype for the h5py data')
parser.add_argument('-v', dest='verbose', action='store_true')
parser.add_argument('--single', dest='single_file', action= 'store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_false')
parser.add_argument('--elo_layer', dest='elo_layer', action='store_true')
parser.set_defaults(verbose=False)
parser.set_defaults(single_file=False)
parser.set_defaults(shuffle=True)
parser.set_defaults(elo_layer=False)
args = parser.parse_args()


if args.elo_layer:
	chunk_size = (100, 7, 8, 8)
	max_shape = (None, 7, 8, 8)
else:
	chunk_size = (100, 6, 8, 8)
	max_shape = (None, 6, 8, 8)

if args.dtype:
	dtype = args.dtype
else: 
	dtype= 'int8'

INPUT_DIR = args.dir
if args.odir:
	OUTPUT_DIR = args.odir
else:
	OUTPUT_DIR = INPUT_DIR+"_h5"

if os.path.isdir(OUTPUT_DIR):
	print("There already exists a folder named %s"%OUTPUT_DIR)
	print("It could be dangerous to remove the previous data")
	print("Either remove manually or rename it before trying again")
	exit(1)
else:
	os.mkdir(OUTPUT_DIR)

npy_files = os.listdir(INPUT_DIR)
npy_files.sort()

print("Using %s as the data source. %s as the destination for hdf5 files."%(INPUT_DIR, OUTPUT_DIR))

import re

piece_re = re.compile("X_[0-9]+_*")
piece_y_re = re.compile("y_[0-9]+_*")
move_re = [re.compile("p"+str(i+1)+"_X_*") for i in xrange(6)]
move_y_re = [re.compile("p"+str(i+1)+"_y_*") for i in xrange(6)]

piece_data_files = [f for f in npy_files if piece_re.match(f)]
piece_label_files = [f for f in npy_files if piece_y_re.match(f)]
move_data_files = [[f for f in npy_files if move_re[i].match(f)] for i in xrange(6)] 
move_label_files = [[f for f in npy_files if move_y_re[i].match(f)] for i in xrange(6)]

if args.verbose:
	print ("PIECE Data and Label files")
	print (piece_data_files,)
	print (piece_label_files,)
	print ("MOVE Data and Label files")
	print (move_data_files,)
	print (move_label_files,)

#sanity checks
assert len(piece_data_files)==len(piece_label_files), "There aren't equal number of files for data and labels"
for i in xrange(6):
	assert len(move_data_files[i])==len(move_label_files[i]), "There aren't equal number of files for data and labels (moves)"

def shuffle_in_unison_inplace(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

if args.shuffle:
	print("Shuffling the data in each file")

if args.single_file:
	#create hdf5 datasets
	start = timeit.default_timer()
	f = h5.File(OUTPUT_DIR+"/piece.h5")
	listf = open(OUTPUT_DIR+"/piece.txt","a")
	listf.write(os.getcwd()+"/"+OUTPUT_DIR+"/piece.h5")
	listf.close()
	piece_data_cur = np.load(INPUT_DIR+"/"+ piece_data_files[0])
	piece_label_cur = np.load(INPUT_DIR+"/"+piece_label_files[0])
	if args.shuffle:
		piece_data_cur, piece_label_cur = shuffle_in_unison_inplace(
			piece_data_cur, piece_label_cur) 
	f.create_dataset('data', data=piece_data_cur, 
		dtype = dtype,
		chunks=chunk_size	, 
		maxshape= max_shape)
	f.create_dataset('label', data=piece_label_cur, 
		dtype = dtype,
		chunks=(100,), maxshape=(None,))
	curr_len = piece_label_cur.shape[0]

	for i in xrange(len(piece_data_files)-1):
		piece_data_cur = np.load(INPUT_DIR+"/"+piece_data_files[i+1])
		piece_label_cur = np.load(INPUT_DIR+"/"+piece_label_files[i+1])
		dataset= f['data']
		dataset.resize((curr_len+piece_label_cur.shape[0], piece_data_cur.shape[1],piece_data_cur.shape[2],piece_data_cur.shape[3]))
		label_dataset = f['label']
		label_dataset.resize((curr_len+piece_label_cur.shape[0],))
		dataset[curr_len:] = piece_data_cur
		label_dataset[curr_len:] = piece_label_cur
		curr_len = label_dataset.shape[0]

	#Cleanup things
	f.close()
	curr_len = None
	label_dataset = None
	dataset = None
	piece_label_cur = None
	piece_data_cur = None
	dataset = None
	end = timeit.default_timer()
	print("PIECE data written into %s in %.2f s"%(OUTPUT_DIR,(end-start)))


	start = timeit.default_timer()
	f = []
	curr_lens = []
	for i in xrange(6):
		f.append(h5.File(OUTPUT_DIR+"/move"+str(i+1)+".h5"))
		listf = open(OUTPUT_DIR+"/move"+str(i+1)+".txt","a")
		listf.write(os.getcwd()+"/"+OUTPUT_DIR+"/move"+str(i+1)+".h5")
		listf.close()
		move_data_cur = np.load(INPUT_DIR+"/"+move_data_files[i][0])
		move_label_cur = np.load(INPUT_DIR+"/"+move_label_files[i][0])
		f[i].create_dataset('data', data=move_data_cur ,
			dtype = dtype ,
			chunks=chunk_size, maxshape= max_shape)
		f[i].create_dataset('label', data=move_label_cur, 
			dtype = dtype,
			chunks= (100,), maxshape=(None,))
		curr_lens.append(move_label_cur.shape[0])


	for i in xrange(6):
		for j in xrange(len(move_data_files[i])-1):
			curr_len = curr_lens[i]

			print("Processing file %s"%INPUT_DIR+"/"+move_data_files[i][j+1])
			#load current data
			move_data_cur = np.load(INPUT_DIR+"/"+move_data_files[i][j+1])
			move_label_cur = np.load(INPUT_DIR+"/"+move_label_files[i][j+1])
			
			#resize the datasets
			dataset= f[i]['data']
			dataset.resize((curr_len+move_label_cur.shape[0], move_data_cur.shape[1],move_data_cur.shape[2],move_data_cur.shape[3]))
			label_dataset = f[i]['label']
			label_dataset.resize((curr_len+move_label_cur.shape[0],))

			#set the current data into dataset
			dataset[curr_len:] = move_data_cur
			label_dataset[curr_len:] = move_label_cur

			#set current length to the new length
			curr_lens[i] = label_dataset.shape[0]

	end = timeit.default_timer()
	print("PIECE data written into %s in %.2fs"%(OUTPUT_DIR,(end-start)))
	print("Writing list of files to list_piece.txt")

else:
	start = timeit.default_timer()
	for i in xrange(len(piece_data_files)):
		piece_data_cur = np.load(INPUT_DIR+"/"+piece_data_files[i])
		piece_label_cur = np.load(INPUT_DIR+"/"+piece_label_files[i])
		f = h5.File(OUTPUT_DIR+'/'+piece_data_files[i].split('.')[0]+'.h5')
		listf = open(OUTPUT_DIR+"/piece.txt","a")
		listf.write(os.getcwd()+"/"+OUTPUT_DIR+"/"+piece_data_files[i].split('.')[0]+'.h5\n')
		listf.close()
		if args.shuffle:
			piece_data_cur, piece_label_cur = shuffle_in_unison_inplace(
				piece_data_cur, piece_label_cur)
		f.create_dataset('data', data=piece_data_cur, 
			dtype = dtype,
			chunks=chunk_size	, 
			maxshape= max_shape)
		f.create_dataset('label', data=piece_label_cur, 
			dtype = dtype,
			chunks=(100,), maxshape=(None,))
		f.close()
	#Cleanup things
	end = timeit.default_timer()
	print("PIECE data written into %s in %.2f s"%(OUTPUT_DIR,(end-start)))
	piece_data_cur = None
	piece_label_cur = None

	start = timeit.default_timer()
	for j in xrange(6):
		for i in xrange(len(move_data_files[j])):
			data_cur = np.load(INPUT_DIR+"/"+move_data_files[j][i])
			label_cur = np.load(INPUT_DIR+"/"+move_label_files[j][i])
			f = h5.File(OUTPUT_DIR+'/'+move_data_files[j][i].split('.')[0]+'.h5')
			listf = open(OUTPUT_DIR+"/move"+str(j+1)+".txt","a")
			listf.write(os.getcwd()+'/'+OUTPUT_DIR+'/'+move_data_files[j][i].split('.')[0]+'.h5\n')
			listf.close()
			if args.shuffle:
				data_cur, label_cur = shuffle_in_unison_inplace(
					data_cur, label_cur)
			f.create_dataset('data', data=data_cur, 
				dtype = dtype,
				chunks=chunk_size	, 
				maxshape= max_shape)
			f.create_dataset('label', data=label_cur, 
				dtype = dtype,
				chunks=(100,), maxshape=(None,))
			f.close()
	#Cleanup things
	end = timeit.default_timer()
	print("MOVE data written into %s in %.2f s"%(OUTPUT_DIR,(end-start)))