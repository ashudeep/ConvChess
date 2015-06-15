'''
Plays a (or more) games against a human or the sunfish AI.

Adapted from Erik Bern's Chess AI
https://github.com/erikbern/deep-pink

'''
import numpy as np
import sunfish
import chess
import pickle
import random
import time
import traceback
import re
import string
import math
from util import *
#from run import *
from chess import pgn
#from layers import *
#from fast_layers import *
#from classifiers.convnet import *
#from classifier_trainer import ClassifierTrainer
from timeit import default_timer as timer
import caffe
caffe.set_mode_gpu()
import argparse
parser=argparse.ArgumentParser\
    (description='Plays one (or more) games against a human or the sunfish AI.')
parser.add_argument('--dir', type=str, default='models', help='The models directory')
parser.add_argument('--odir', type=str, help='The output to dump \
    gameplays and statistics of the games',
    default='models')
#parser.add_argument('-v', dest='verbose', action='store_true')
parser.add_argument('--elo', type=int, default=3255, 
    help='ELO rating of the player you want to imitate.')
parser.add_argument('--no-elolayer', dest='elo_layer', action='store_false',
    help='Whether to include ELO rating layer or not')
parser.add_argument('--against', type=str, default='sunfish',
    help='The (artificial) intelligence to play against. Currently either sunfish or Human player.')
parser.set_defaults(verbose=False)
parser.set_defaults(elo_layer=True)
args = parser.parse_args()

elo_layer = ((args.elo-2000.0)/1255.0) * np.ones((1,8,8),dtype=np.float32)
#elo_layer = np.ones((1,8,8),dtype=np.float32)
against = args.against

trained_models = {}
INDEX_TO_PIECE_2 = {0 : 'Pawn', 1 : 'R', 2 : 'N', 3 : 'B', 4 : 'Q', 5 : 'K'}
CHECKMATE_SCORE = 10e6  

def load_models(dir):
    model_names = ['piece', 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
    names = ['Piece', 'P', 'R', 'N', 'B', 'Q', 'K']
    for index, model_name in enumerate(model_names):
        model_path = dir+'/%s.caffemodel' % model_name
        net_path = dir+'/common_net.prototxt'
        trained_model = caffe.Net(net_path, model_path,caffe.TEST)
        trained_models[names[index]] = trained_model

def predict(X, model, fn):
    return fn(X, model)

def predictionAccuracy(predictions, label):
    return np.mean(predictions == label)

def scoreToCoordinateIndex(score):
    return (score/8, score%8)

def scoresToBoard(scores):
    return scores.reshape((8, 8))

def boardToScores(board):
    return board.reshape((64))

def predictMove_MaxMethod(img):
    dummy = np.ones((1,), dtype=np.float32)
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype='float32'),dummy)
    res = net.forward()
    probs = res['prob']
    probs = clip_pieces_single(probs, img[0:6])
    probs = probs.flatten()
    pred_piece = np.argmax(probs)

    #pred_piece = int(res['plabel'])
    coordinate = scoreToCoordinateIndex(pred_piece)
    pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, coordinate[0], coordinate[1]])]
    model = trained_models[pieceType]
    model.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res2 = model.forward()
    probs = res2['prob']
    probs = clip_moves(probs, img[0:6], coordinate)
    pred_pos = np.argmax(probs.flatten())
    #pred_pos = int(res2['plabel'])
    return coord2d_to_chess_coord(coordinate)+\
        coord2d_to_chess_coord(scoreToCoordinateIndex(pred_pos))

def predictMove_TopProbMethod(img):
    dummy = np.ones((1,), dtype='float32')
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    probs = res['prob']
    probs = clip_pieces_single(probs, img[0:6])
    #print probs
    probs = probs.flatten()
    cumulative_probs = np.zeros((64,64))
    piece_outputs={} #to cache outputs by the move networks
    for i in xrange(64):
        if probs[i]>0:
            i1,i2 = scoreToCoordinateIndex(i)
            pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
            if pieceType in piece_outputs:
                move_prob= piece_outputs[pieceType]
            else:
                model = trained_models[pieceType] 
                model.set_input_arrays(np.array([img], dtype=np.float32),dummy)
                res2 = model.forward()
                move_prob = res2['prob']
                #print move_prob
                move_prob = clip_moves(move_prob, img[0:6], (i1,i2))
                piece_outputs[pieceType] = move_prob
            #print "Clipped:",move_prob
            cumulative_probs[i] = move_prob*probs[i]
    #print cumulative_probs
    pos = np.argmax(cumulative_probs)
    from_pos, to_pos = pos/64, pos%64

    return coord2d_to_chess_coord(scoreToCoordinateIndex(from_pos))+\
        coord2d_to_chess_coord(scoreToCoordinateIndex(to_pos))

def evaluate_moves(img, moves):
    '''
    For the bitboard representation of the board, returns the
    evaluation function score of the moves.
    im: the 6 layer image representation of the board
    moves: list of moves of the form 'g1f3' 
    '''
    dummy = np.ones((1,), dtype='float32')
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    probs = res['prob']
    probs = clip_pieces_single(probs, img[0:6])
    #print probs
    probs = probs.flatten()
    cumulative_probs = np.zeros((64,64))
    for i in xrange(64):
        if probs[i]>0:
            i1,i2 = scoreToCoordinateIndex(i)
            pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
            model = trained_models[pieceType] 
            model.set_input_arrays(np.array([img], dtype=np.float32),dummy)
            res2 = model.forward()
            move_prob = res2['prob']
            #print move_prob
            move_prob = clip_moves(move_prob, img[0:6], (i1,i2))
            #print "Clipped:",move_prob
            cumulative_probs[i] = move_prob*probs[i]
    #print cumulative_probs
    scores = np.zeros((len(moves),), np.float32)
    for i, move in enumerate(moves):
        fro = flatten_coord2d(chess_coord_to_coord2d(move[0:2]))
        to = flatten_coord2d(chess_coord_to_coord2d(move[2:4]))
        scores[i] = cumulative_probs[fro, to]
    return scores

def evaluate_moves_fast(img, moves):
    '''
    For the bitboard representation of the board, returns the
    evaluation function score of the moves.
    im: the 6 layer image representation of the board
    moves: list of moves of the form 'g1f3' 
    '''
    scores = np.zeros((len(moves),), np.float32)
    net = trained_models['Piece']
    dummy = np.ones((1,), dtype='float32')
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    probs = res['prob']
    probs = clip_pieces_single(probs, img[0:6])
    probs = probs.flatten()
    for i, move in enumerate(moves):
        fro = flatten_coord2d(chess_coord_to_coord2d(move[0:2]))
        to = flatten_coord2d(chess_coord_to_coord2d(move[2:4]))
        prob1 = probs[fro]
        i1, i2 = scoreToCoordinateIndex(i)
        pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
        model = trained_models[pieceType] 
        model.set_input_arrays(np.array([img], dtype=np.float32),dummy)
        res2 = model.forward()
        move_prob = res2['prob']
        #print move_prob
        move_prob = clip_moves(move_prob, img[0:6], (i1,i2))
        move_prob = move_prob.flatten()
        #print "Clipped:",move_prob
        prob2 = move_prob[to]
        scores[i] = prob1*prob2
    return scores

def pos_board_to_bitboard(board):
    strip_whitespace = re.compile(r"\s+")
    board = strip_whitespace.sub('',board)
    board = " ".join(board)
    board = re.sub("(.{16})", "\\1\n", board, 0, re.DOTALL)
    return board

def pos_coords_to_2dcoord(fro):
    i, j = fro/10, fro%10
    i = i-2
    j = j-1
    return (i,j)

def negamax(pos, depth, alpha, beta, color, func, maxm):
    '''
    Derived from deep-pink.
    Currently very slow. Just need to use evaluate moves to get the
    cumulative_probs and then use it to choose the top few moves further.
    '''
    moves = []
    X = [] #will hold the set of possible moves
    pos_children = []
    im = convert_bitboard_to_image(pos_board_to_bitboard(pos.board))
    im = np.rollaxis(im, 2, 0)
    if color == -1:
        im = flip_color(flip_image(im))
    im = np.append(im, elo_layer, axis=0)
    for move in pos.genMoves():
        pos_child = pos.move(move)
        moves.append(move)
        X.append(coord2d_to_chess_coord(pos_coords_to_2dcoord(move[0]))+
            coord2d_to_chess_coord(pos_coords_to_2dcoord(move[1])))
        pos_children.append(pos_child)

    if len(X) == 0:
        return Exception('eh?')

    # Use model to predict scores
    #func = evaluate_moves_fast
    func = evaluate_moves
    #scores = func(X)
    scores = func(im, X)

    for i, pos_child in enumerate(pos_children):
        if color==1 and pos_child.board.find('K') == -1:
            scores[i] = CHECKMATE_SCORE
        elif color==-1 and pos_child.board.find('k') == -1:
            scores[i] = CHECKMATE_SCORE

    child_nodes = sorted(zip(scores, moves), reverse=True)
    #choose only top maxm moves
    if len(child_nodes)>maxm:
        child_nodes = child_nodes[0:maxm]

    best_value = float('-inf')
    best_move = None
    
    for score, move in child_nodes:
        if depth == 1 or score == CHECKMATE_SCORE:
            value = score
        else:
            # print 'ok will recurse', sunfish.render(move[0]) + sunfish.render(move[1])
            pos_child = pos.move(move)
            neg_value, _ = negamax(pos_child, depth-1, -beta, -alpha, -color, func, maxm)
            value = -neg_value

        # value += random.gauss(0, 0.001)

        # crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        # print '\t' * (3 - depth), crdn, score, value

        if value > best_value:
            best_value = value
            best_move = move

        if value > alpha:
            alpha = value

        if alpha > beta:
            break

    return best_value, best_move

def create_move(board, crdn):
    # workaround for pawn promotions
    move = chess.Move.from_uci(crdn)
    if board.piece_at(move.from_square).piece_type == chess.PAWN:
        if int(move.to_square/8) in [0, 7]:
            move.promotion = chess.QUEEN # always promote to queen
    return move

class Player(object):
    def move(self, gn_current):
        raise NotImplementedError()

class MySearch(Player):
    def __init__(self, func=evaluate_moves, maxm=10 , maxd=5):
        self._func = func
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxd = maxd
        self._maxm = maxm

    def move(self, gn_current):
        assert(gn_current.board().turn == 0)

        if gn_current.move is not None:
            # Apply last_move
            crdn = str(gn_current.move)
            move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
            self._pos = self._pos.move(move)

        # for depth in xrange(1, self._maxd+1):
        alpha = float(0)
        beta = float(1)

        depth = self._maxd
        t0 = time.time()
        best_value, best_move = negamax(self._pos, depth, alpha, beta, 1, self._func, maxm=self._maxm)
        crdn = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        print depth, best_value, crdn, time.time() - t0

        self._pos = self._pos.move(best_move)
        crdn = sunfish.render(best_move[0]) + sunfish.render(best_move[1])
        move = create_move(gn_current.board(), crdn)
        
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move


        return gn_new

class Computer(Player):
    def move(self, gn_current):
        bb = gn_current.board()

        im = convert_bitboard_to_image(bb)
        im = np.rollaxis(im, 2, 0)
        
        im = np.append(im, elo_layer, axis=0) 
        #add the dynamic elo bias layer to be the max (=1)

        #move_str = predictMove_TopProbMethod(im)
        move_str = predictMove_MaxMethod(im)
        move = chess.Move.from_uci(move_str)

        if move not in bb.legal_moves:
            print "NOT A LEGAL MOVE"

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move
        
        return gn_new

class Human(Player):
    def move(self, gn_current):
        bb = gn_current.board()

        #print bb

        def get_move(move_str):
            try:
                move = chess.Move.from_uci(move_str)
            except:
                print 'cant parse'
                return False
            if move not in bb.legal_moves:
                print 'not a legal move'
                return False
            else:
                return move

        while True:
            print 'your turn:'
            move = get_move(raw_input())
            if move:
                break

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        print move
        gn_new.move = move
        
        return gn_new

class Sunfish(Player):
    def __init__(self, maxn=1e4):
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxn = maxn

    def move(self, gn_current):
        import sunfish

        assert(gn_current.board().turn == 1)

        # Apply last_move
        crdn = str(gn_current.move)
        move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
        self._pos = self._pos.move(move)

        #t0 = time.time()
        move, score = sunfish.search(self._pos, maxn=self._maxn)
        #print time.time() - t0, move, score
        self._pos = self._pos.move(move)

        crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        move = create_move(gn_current.board(), crdn)
        
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

def krk_end_game():
    gn = chess.pgn.Game()


def game():
    gn_current = chess.pgn.Game()
    #gn_current = krk_end_game()

    maxn = 10#00#10 ** (1.0 + random.random() * 1.0) # max nodes for sunfish
    maxd = 2#random.randint(2,3)
    maxm = random.randint(5,10)
    print 'maxn %f' % (maxn)
    print 'maxm %d' % (maxm)
    print 'maxd %d'% (maxd)

    player_a = Computer()
    #player_a = MySearch(maxd=maxd, maxm=maxm)
    if against=="human":
        player_b = Human()
    elif against=="sunfish":
        player_b = Sunfish(maxn=maxn)
    else:
        print "Only sunfish and human players are supported currently"
        exit(1)

    times = {'A' : 0.0, 'B' : 0.0}

    while True:
        for side, player in [('A', player_a), ('B', player_b)]:
            start = timer()
            try:
                gn_current = player.move(gn_current)
            except KeyboardInterrupt:
                return
            except:
                traceback.print_exc()
                return side + '-exception'

            end = timer()
            times[side] += end-start
            print '=========== Player %s: %s' % (side, gn_current.move)
            s = str(gn_current.board())
            print s
	    print times[side]
            if gn_current.board().is_checkmate():
                return side
            elif gn_current.board().is_stalemate():
                return '-'
            elif gn_current.board().can_claim_fifty_moves():
                return '-' 
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                return side

def play():
    side = game()
    f = open(args.odir+'/stats.txt', 'a')
    f.write('%s\n' % (side))
    f.close()

if __name__ == '__main__':
    load_models(args.dir)
    play()
