"""
This file reads the pgn files one game at a time to 
store the X, y arrays into npz files. 
Its much more suitable for large data which cannot 
fit into memory all at once.

Author: Ashudeep Singh
Date: May 12, 2015
"""
import numpy as np
import pgn
import chess
#import cPickle as npy
import copy
from util import *
import sys, os
import timeit
import argparse
NUM_GAMES = 4000
parser=argparse.ArgumentParser\
	(description='Convert PGN data into numpy arrays of size 6*8*8 with labels (pieces/moves)')
parser.add_argument('--dir', type=str, default='', 
	help='The data directory')
parser.add_argument('--odir', type=str, default='', 
	help='The output npz data directory')
parser.add_argument('-v', dest='verbose', action='store_true')
parser.add_argument('--minelo', type=int, default=1000, 
	help='Minimum ELO rating to be added to training data')
parser.add_argument('--elo', dest='elo_layer', action='store_true',
	help='Whether to include ELO rating layer or not')
parser.add_argument('--C', type=float, default=1255, 
	help='Divide the ELO rating minus Min ELO rating by this value')
parser.add_argument('--partsize', type=int, default=NUM_GAMES, 
	help='Number of games to be dumped into 1 npz file.')
parser.add_argument('--history', type=int, default=1, 
	help='Number of chess boards from history required.')
parser.add_argument('--multi', dest='multiple_layers', action='store_true',
	help='Use multiple layers for a single piece to get (8,8,12) size \
	image per board. Default: False.')
parser.add_argument('--piecelayer', dest='piece_layer', action='store_true',
	help='Append a layer with the piece being played marked as\
	 1 for the move network data.')
parser.add_argument('--resultlayer', dest='result_layer',
	action = 'store_true')
parser.add_argument('--skip', type=int, help='skip first these many games.\
	Ideally a multiple of %d'%NUM_GAMES, default=0)
parser.add_argument('--regr', dest='regression', action='store_true', 
	help='Evaluation of each table according to discounted values\
	 due to win(1), loss(-1), draw(1/2)')
parser.add_argument('-g', dest='gamma', type=float, 
	help='Discount factor for positions', default=0.9)
parser.set_defaults(verbose=False)
parser.set_defaults(elo_layer=False)
parser.set_defaults(multiple_layers=False)
parser.set_defaults(piece_layer=False)
parser.set_defaults(result_layer=False)
parser.set_defaults(regression=False)
args = parser.parse_args()

NUM_GAMES = args.partsize

if args.elo_layer:
	elo_layer = True
else:
	elo_layer = False

min_elo = args.minelo
PGN_DATA_DIR = args.dir
TRAIN_DATA_DIR = args.odir
if not os.path.isdir(TRAIN_DATA_DIR):
	os.mkdir(os.getcwd()+"/"+TRAIN_DATA_DIR)

if args.regression and args.piece_layer:
	raise Exception("Data representation for regression cannot contain piece-layer")
if args.regression and args.result_layer:
	raise Exception("Data representation for regression cannot contain result-layer")

#assign the correct functions from util.py
if args.multiple_layers:
	bitboard_to_image = convert_bitboard_to_image_2
	flip_color = flip_color_2
else:
	bitboard_to_image = convert_bitboard_to_image_1
	flip_color = flip_color_1

print "Started reading PGN files in directory %s"%PGN_DATA_DIR
game_index = 0
for f in os.listdir(PGN_DATA_DIR):
	if ".pgn" in f:
		print "%s file opened...."%f
		for game in pgn.GameIterator(PGN_DATA_DIR+"/"+f):
			if not game:	break
			if game_index < args.skip+1:
				game_index+=1
				if game_index%4000==0:
					print "Skipped %d games"%game_index
				continue
			elif game_index == args.skip+1:
				start = timeit.default_timer()
				X, y = [], []
				p1_X, p2_X, p3_X = [], [], []
				p4_X, p5_X, p6_X = [], [], []
				p1_y, p2_y, p3_y = [], [], []
				p4_y, p5_y, p6_y = [], [], []
			#print PGN_DATA_DIR+"/"+f, game
			board = chess.Bitboard()
			moves = game.moves
			if game_index < args.skip:
				game_index+=1
				continue
			if game_index%NUM_GAMES == 0:
				if game_index!=0:
					end = timeit.default_timer()
					print "Processed %d moves from %d games in %fs"%(len(X), NUM_GAMES,end-start)
					start = timeit.default_timer()
					print "Saving data for %d-%d games.."%(game_index-NUM_GAMES,game_index)

					print "Saving X array..."
					output = TRAIN_DATA_DIR+'/X_%d_%d.npz' % (game_index-NUM_GAMES,game_index)
					X = np.array(X).astype(np.float32)
					np.savez_compressed(output, X)

					print "Saving y array..."
					output = TRAIN_DATA_DIR+'/y_%d_%d.npz' % (game_index-NUM_GAMES,game_index)
					y = np.array(y).astype(np.float32)
					np.savez_compressed(output, y)
					if not args.regression:
						for i in xrange(6):
							output_array = "p%d_X" % (i + 1)
							print "Saving %s array..." % (output_array)
							output_array = eval(output_array)
							output_array = np.array(output_array).astype(np.float32)
							print output_array.shape
							output = TRAIN_DATA_DIR+'/p%d_X_%d_%d.npz' % (i + 1, game_index-NUM_GAMES,game_index) 
							np.savez_compressed(output, output_array)

							output_array = "p%d_y" % (i + 1)
							print "Saving %s array..." % output_array
							output_array = eval(output_array)
							output_array = np.array(output_array).astype(np.float32)
							output = TRAIN_DATA_DIR+'/p%d_y_%d_%d.npz' % (i + 1, game_index-NUM_GAMES,game_index) 
							np.savez_compressed(output, output_array)
					end = timeit.default_timer()
					print "Saved arrays into directory %s in %fs"%(TRAIN_DATA_DIR, end-start)

				#clear the buffers
				start = timeit.default_timer()
				X, y = [], []
				if not args.regression:
					p1_X, p2_X, p3_X = [], [], []
					p4_X, p5_X, p6_X = [], [], []
					p1_y, p2_y, p3_y = [], [], []
					p4_y, p5_y, p6_y = [], [], []
			#increment the game counter
			game_index+=1
			black_elo = int(game.blackelo)
			white_elo = int(game.whiteelo)
			if black_elo < min_elo:
				skip_black = True
			else:
				skip_black = False
			if white_elo < min_elo:
				skip_white = True
			else:
				skip_white = False
			if skip_white and skip_black:	continue
			if elo_layer:
				white_elo_layer = float(white_elo - min_elo)/args.C
				black_elo_layer = float(black_elo- min_elo)/args.C
			if args.result_layer:
				white_result = game.result.split('-')[0]
				if  white_result == '1':
					white_result_layer = np.ones((1,8,8))
					black_elo_layer = np.zeros((1,8,8))
				elif white_result == '0':
					white_result_layer = np.zeros((1,8,8))
					black_result_layer = np.ones((1,8,8))
				elif white_result == '1/2':
					white_result_layer = 0.5*np.ones((1,8,8))
					black_result_layer = 0.5*np.ones((1,8,8))
				else:
					raise Exception("Unknown outcome")
			num_moves = len(moves)-2
			for move_index, move in enumerate(moves):
				if move[0].isalpha(): # check if move is SAN
					from_to_chess_coords = board.parse_san(move)
					from_to_chess_coords = str(from_to_chess_coords)

					from_chess_coords = from_to_chess_coords[:2]
					to_chess_coords = from_to_chess_coords[2:4]
					from_coords = chess_coord_to_coord2d(from_chess_coords)
					to_coords = chess_coord_to_coord2d(to_chess_coords)
								
					if move_index % 2 == 0:
						im = bitboard_to_image(board)
						skip = skip_white
						if elo_layer:
							last_layer = white_elo_layer*np.ones((1,8,8))
						if args.result_layer:
							result_layer = white_result_layer
					else:
						im = flip_image(bitboard_to_image(board))
						im = flip_color(im)
						from_coords = flip_coord2d(from_coords)
						to_coords = flip_coord2d(to_coords)
						skip = skip_black
						if elo_layer:
							last_layer = black_elo_layer*np.ones((1,8,8))
						if args.result_layer:
							result_layer = black_result_layer

					board.push_san(move)

					#don't write if the player<2000 ELO
					if skip:
						continue

					index_piece = np.where(im[from_coords] == 1)
					# index_piece denotes the index in PIECE_TO_INDEX
					if args.multiple_layers:
						index_piece = index_piece[0][0]/2 # ranges from 0 to 5
					else:
						index_piece=index_piece[0][0]

					from_coords = flatten_coord2d(from_coords)
					to_coords = flatten_coord2d(to_coords)

					im = np.rollaxis(im, 2, 0) # to get into form (C, H, W)
					if elo_layer:
						im = np.append(im, last_layer, axis=0)
					if args.result_layer:
						im = np.append(im, result_layer, axis=0)
					
					# Filling the X and y array
					X.append(im)
					if args.regression:
						white_result = game.result.split('-')[0]
						if white_result == '1/2':
							white_result = 0.5
						else:
							white_result = float(white_result)
						white_result = 2*white_result-1 #{1,1/2,0} to {1,0,-1}
						table_score = white_result*args.gamma**((num_moves-move_index)/2)
						if move_index % 2 == 0:
							table_score = table_score
						else:
							table_score = -1*table_score
						y.append(table_score)
					else:
						y.append(from_coords)

						# Filling the p_X and p_y array
						p_X = "p%d_X" % (index_piece + 1)
						p_X = eval(p_X)

						if args.piece_layer:
							piece_layer = np.zeros((1,8,8))
							piece_layer[0, from_coords/8, from_coords%8] = 1
							im = np.append(im, piece_layer,axis=0)
						
						p_X.append(im)

						p_y = "p%d_y" % (index_piece + 1)
						p_y = eval(p_y)
						p_y.append(to_coords)

					end = timeit.default_timer()

print "Processed %d moves from %d games in %fs"%(len(X), game_index%NUM_GAMES,end-start)
start = timeit.default_timer()
print "Saving data for %d-%d games.."%(game_index - game_index%NUM_GAMES,game_index)

print "Saving X array..."
output = TRAIN_DATA_DIR+'/X_%d_%d.npz' % (game_index - game_index%NUM_GAMES,game_index)
X = np.array(X).astype(np.float32)
np.savez_compressed(output, X)

print "Saving y array..."
output = TRAIN_DATA_DIR+'/y_%d_%d.npz' % (game_index - game_index%NUM_GAMES,game_index)
y = np.array(y).astype(np.float32)
np.savez_compressed(output, y)

if not args.regression:
	for i in xrange(6):
		output_array = "p%d_X" % (i + 1)
		print "Saving %s array..." % output_array
		output_array = eval(output_array)
		output_array = np.array(output_array).astype(np.float32)
		output = TRAIN_DATA_DIR+'/p%d_X_%d_%d.npz' % (i + 1, game_index - game_index%NUM_GAMES,game_index) 
		np.savez_compressed(output, output_array)

		output_array = "p%d_y" % (i + 1)
		print "Saving %s array..." % output_array
		output_array = eval(output_array)
		output_array = np.array(output_array).astype(np.float32)
		output = TRAIN_DATA_DIR+'/p%d_y_%d_%d.npz' % (i + 1, game_index - game_index%NUM_GAMES ,game_index) 
		np.savez_compressed(output, output_array)
end = timeit.default_timer()
print "Saved arrays into directory %s in %fs"%(TRAIN_DATA_DIR, end-start)
print "Done with reading %d games"%(game_index+1)