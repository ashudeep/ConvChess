'''
Plays one (or more) games against a human or the sunfish AI.

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
from chess import pgn
from timeit import default_timer as timer
import caffe
import operator
caffe.set_mode_gpu()
import argparse
#import sunfish_mod
import sunfish_mod2
#import sunfish_mod3
#import functools

parser=argparse.ArgumentParser\
    (description='Plays one (or more) games against a human or the sunfish AI.')

def add_arguments(parser):
    parser.add_argument('--dir', type=str, default='models', help='The models directory')
    parser.add_argument('--odir', type=str, help='The output to dump \
        gameplays and statistics of the games',
        default='models')
    parser.add_argument('--no-piecelayer', dest='piecelayer', action='store_false',
        help='Whether to include piece layer or not.')
    parser.add_argument('--no-multilayer', dest='multilayer', action='store_false',
        help='Whether to include multiple layers for enemies or not.')
    #parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('--elo', type=int, default=3255, 
        help='ELO rating of the player you want to imitate.')
    parser.add_argument('--no-elolayer', dest='elo_layer', action='store_false',
        help='Whether to include ELO rating layer or not')
    parser.add_argument('--against', type=str, default='sunfish',
        help='The (artificial) intelligence to play against. Currently either sunfish or Human player.')
    parser.set_defaults(verbose=False)
    parser.set_defaults(elo_layer=True)
    parser.set_defaults(piecelayer=True)
    parser.set_defaults(multilayer=True)

add_arguments(parser)
args = parser.parse_args()

if args.elo_layer:
    elo_layer = ((3255-2000.0)/1255.0) * np.ones((1,8,8),dtype=np.float32)
#elo_layer = np.ones((1,8,8),dtype=np.float32)
against = args.against
#against = 'sunfish'
trained_models = {}
INDEX_TO_PIECE_2 = {0 : 'Pawn', 1 : 'R', 2 : 'N', 3 : 'B', 4 : 'Q', 5 : 'K'}
CHECKMATE_SCORE = 1e6  
TOP_MOVES_CACHE = {}
CACHING = False
def load_models(dir):
    model_names = ['piece', 'pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
    names = ['Piece', 'P', 'R', 'N', 'B', 'Q', 'K']
    for index, model_name in enumerate(model_names):
        model_path = dir+'/%s.caffemodel' % model_name
        if model_name == 'piece':
            net_path = dir+'/piece.prototxt'
        else:
            net_path = dir+'/move.prototxt'
        trained_model = caffe.Net(net_path, model_path,caffe.TEST)
        trained_models[names[index]] = trained_model

# if trained_models == {}:
#     load_models(args.dir)

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

def pos_board_to_bitboard(board):
    strip_whitespace = re.compile(r"\s+")
    board = strip_whitespace.sub('',board)
    board = " ".join(board)
    board = re.sub("(.{16})", "\\1\n", board, 0, re.DOTALL)
    return board

def predictMove_MaxMethod(img):
    dummy = np.ones((1,), dtype=np.float32)
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype='float32'),dummy)
    res = net.forward()
    probs = res['prob']
    if args.multilayer:
        probs = clip_pieces_single_2(probs, img[0:12])
    else:
        probs = clip_pieces_single(probs, img[0:6])
    probs = probs.flatten()
    pred_piece = np.argmax(probs)
    coordinate = scoreToCoordinateIndex(pred_piece)
    #pred_piece = int(res['plabel'])
    if args.multilayer:
        pieceType = INDEX_TO_PIECE[np.argmax(img[0:12, coordinate[0], coordinate[1]])/2]
    else:
        pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, coordinate[0], coordinate[1]])]
    #print pieceType
    model = trained_models[pieceType]
    
    if args.piecelayer:
        piece_layer = np.zeros((1,8,8))
        piece_layer[0,coordinate[0],coordinate[1]] = 1
        img =  np.append(img, piece_layer, axis=0)

    model.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res2 = model.forward()
    probs = res2['prob']
    #print probs
    if args.multilayer:
        probs = clip_moves_2(probs, img[0:12], coordinate)
    else:
        probs = clip_moves(probs, img[0:6], coordinate)
    #print probs
    pred_pos = np.argmax(probs.flatten())
    #pred_pos = int(res2['plabel'])
    return coord2d_to_chess_coord(coordinate)+\
        coord2d_to_chess_coord(scoreToCoordinateIndex(pred_pos))

def topk(a,k, vals=False, threshold=float('-inf')):
    top_ids = np.argpartition(a, -k)[-k:]
    if vals:
        if threshold!=float('-inf'):
            return [(i,j) for i,j in zip(top_ids, a[top_ids]) if j>threshold]
        else:
            return zip(top_ids,a[top_ids])
    else:
        return top_ids

def getim(bb):
    #bb is the bitboard
    if args.multilayer:
        im = convert_bitboard_to_image_2(bb)
    else:
        im = convert_bitboard_to_image(bb)
    im = np.rollaxis(im, 2, 0)
    if args.elo_layer:
        elo_layer = ((args.elo-2000.0)/1255.0) * np.ones((1,8,8),dtype=np.float32)
        im = np.append(im, elo_layer, axis=0)
        #the piece layer is appended in the functions separately
    return im

def predictMove_TopProbMethod(img, maxwidth=10):
    dummy = np.ones((1,), dtype='float32')
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    probs = res['prob']
    if args.multilayer:
        probs = clip_pieces_single_2(probs, img[0:12])
    else:
        probs = clip_pieces_single(probs, img[0:6])
    #print probs
    probs = probs.flatten()
    cumulative_probs = np.zeros((64,64))    
    for i, piece_pos in enumerate(topk(probs,maxwidth)):
        if probs[piece_pos]>0:
            i1,i2 = scoreToCoordinateIndex(piece_pos)
            if args.multilayer:
                pieceType = INDEX_TO_PIECE[np.argmax(img[0:12, i1, i2])/2]
            else:
                pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
            if args.piecelayer:
                piece_layer = np.zeros((1,8,8))
                piece_layer[0,i1,i2] = 1
                img2 = np.append(img, piece_layer, axis=0)
            else:
                img2 = img
            model = trained_models[pieceType]
            model.set_input_arrays(np.array([img2], dtype=np.float32),dummy)
            res2 = model.forward()
            move_prob = res2['prob']
            #print move_prob
            if args.multilayer:
                move_prob = clip_moves_2(move_prob, img2[0:12], (i1,i2))
            else:
                move_prob = clip_moves(move_prob, img2[0:6], (i1,i2))
            #print move_prob
            cumulative_probs[piece_pos] = move_prob*probs[piece_pos]
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
    scores: scores of each of those moves in the same order as in moves
    '''
    dummy = np.ones((1,), dtype='float32')
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    probs = res['prob']
    probs = clip_pieces_single(probs, img[0:6])
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
            move_prob = clip_moves(move_prob, img[0:6], (i1,i2))
            cumulative_probs[i] = move_prob*probs[i]
    scores = np.zeros((len(moves),), np.float32)
    for i, move in enumerate(moves):
        fro = flatten_coord2d(chess_coord_to_coord2d(move[0:2]))
        to = flatten_coord2d(chess_coord_to_coord2d(move[2:4]))
        scores[i] = cumulative_probs[fro, to]
    return scores


def get_top_moves(img, k, vals=True, valType='prob', clipping=True):
    #valType can be 'prob' or 'fc1'
    global CACHING
    global TOP_MOVES_CACHE
    if hash(img.tostring()) in TOP_MOVES_CACHE and CACHING:
        return TOP_MOVES_CACHE[hash(img.tostring())]
    global trained_models
    dummy = np.ones((1,), dtype='float32')
    net = trained_models['Piece']
    net.set_input_arrays(np.array([img], dtype=np.float32),dummy)
    res = net.forward()
    if valType=='prob':
        probs = res['prob']
        if args.multilayer and clipping:
            probs = clip_pieces_single_2(probs, img[0:12])
        elif clipping:
            probs = clip_pieces_single(probs, img[0:6])
        #print probs
        probs = probs.flatten()
        cumulative_probs = np.zeros((64,64))    
        for i, piece_pos in enumerate(topk(probs,k)):
            if probs[piece_pos]>0:
                i1,i2 = scoreToCoordinateIndex(piece_pos)
                if args.multilayer:
                    pieceType = INDEX_TO_PIECE[np.argmax(img[0:12, i1, i2])/2]
                else:
                    pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
                if args.piecelayer:
                    piece_layer = np.zeros((1,8,8))
                    piece_layer[0,i1,i2] = 1
                    img2 = np.append(img, piece_layer, axis=0)
                else:
                    img2 = img
                model = trained_models[pieceType]
                model.set_input_arrays(np.array([img2], dtype=np.float32),dummy)
                res2 = model.forward()
                move_prob = res2['prob']
                #print move_prob
                if args.multilayer and clipping:
                    move_prob = clip_moves_2(move_prob, img2[0:12], (i1,i2))
                elif clipping:
                    move_prob = clip_moves(move_prob, img2[0:6], (i1,i2))
                #print move_prob
                cumulative_probs[piece_pos] = move_prob*probs[piece_pos]
        pos = topk(cumulative_probs.flatten(), k)
    elif valType=='fc1':
        fcvals = net.blobs['fc1'].data
        if args.multilayer and clipping:
            fcvals = clip_pieces_single_2(fcvals, img[0:12], normalize=False)
        elif clipping:
            fcvals = clip_pieces_single(fcvals, img[0:6], normalize=False)
        fcvals = fcvals.flatten()
        cumulative_vals = np.zeros((64,64))
        for i, piece_pos in enumerate(topk(fcvals,k)):
            i1 , i2 = scoreToCoordinateIndex(piece_pos)
            if args.multilayer:
                pieceType = INDEX_TO_PIECE[np.argmax(img[0:12, i1, i2])/2]
            else:
                pieceType = INDEX_TO_PIECE[np.argmax(img[0:6, i1, i2])]
            if args.piecelayer:
                piece_layer = np.zeros((1,8,8))
                piece_layer[0,i1,i2] = 1
                img2 = np.append(img, piece_layer, axis=0)
            else:
                img2 = img
            model = trained_models[pieceType]
            model.set_input_arrays(np.array([img2], dtype=np.float32),dummy)
            res2 = model.forward()
            move_vals = model.blobs['fc1'].data
            #print move_prob
            if args.multilayer and clipping:
                move_vals = clip_moves_2(move_vals, img2[0:12], (i1,i2), normalize=False)
            elif clipping:
                move_vals = clip_moves(move_vals, img2[0:6], (i1,i2), normalize=False)
            #print move_prob
            #print move_vals, cumulative_vals
            cumulative_vals[piece_pos] = move_vals+cumulative_vals[piece_pos]
        pos = topk(cumulative_vals.flatten(), k)
        cumulative_probs = cumulative_vals
        # if args.multilayer:
        #raise NotImplementedError("typeVal=fc1 is still unimplemented.")
    #print cumulative_probs
    
    moves = [(p/64,p%64) for p in pos]
    str_moves = [coord2d_to_chess_coord(scoreToCoordinateIndex(move[0]))+\
        coord2d_to_chess_coord(scoreToCoordinateIndex(move[1])) for move in moves]
    moves_vals = zip(str_moves, cumulative_probs.flatten()[pos])
    moves_vals.sort(key=operator.itemgetter(1), reverse=True)
    if vals:
        if CACHING: TOP_MOVES_CACHE[hash(img.tostring())] = moves_vals
        return moves_vals   
    else:
        str_moves = [move for (move, val) in moves_vals]
        if CACHING: TOP_MOVES_CACHE[hash(img.tostring())] = str_moves
        #print str_moves
        return str_moves


#@functools.lru_cache(maxsize=None)
def negamax(im, depth, alpha, beta, color, maxm):
    '''
    Derived from deep-pink.
    Currently very slow. Just need to use evaluate moves to get the
    cumulative_probs and then use it to choose the top few moves further.
    '''
    #print pos.board
    # if args.multilayer:
    #     im = convert_bitboard_to_image_2(pos_board_to_bitboard(pos.board))
    # else:
    #     im = convert_bitboard_to_image(pos_board_to_bitboard(pos.board))
    # im = flip_image(im)
    # if args.multilayer:
    #     im = flip_color_2(im)
    # else:
    #     im = flip_color_1(im)
    # im = np.rollaxis(im,2,0)
    if color == -1:
        im = im[:,:,::-1]
    # if args.elo_layer:
    #     im = np.append(im, elo_layer, axis=0)
    top_moves = get_top_moves(im, maxm)
    #print top_moves, depth
    best_value = float('-inf')
    best_move = None
    for move, val in top_moves:
        #print move, val
        if depth == 1:
            value = val
            if val == 0: #no move possible
                value = value/CHECKMATE_SCORE
        else:
            # crdn_move =  (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
            #print crdn_move, move, val, color
            try:
                fro = chess_coord_to_coord2d(move[0:2])
                to = chess_coord_to_coord2d(move[2:4])
                #pos_child=pos.move(move)
                #print fro, to
                if args.multilayer:
                    which_layer = np.where(im[0:12,fro[0],fro[1]]==1)[0]
                    # print fro
                    # print which_layer
                    if which_layer.size == 0: 
                        continue
                    which_layer=which_layer[0]
                    try:
                        if_opponent_piece = np.where(im[0:12,to[0],to[1]]==1)[0][0]
                    except IndexError:
                        if_opponent_piece = None
                else:
                    which_layer = np.where(im[0:6, fro[0],fro[1]]==1)[0]
                    if which_layer.size == 0: continue
                    which_layer=which_layer[0]
                    try:
                        if_opponent_piece = np.where(im[0:6,to[0],to[1]]==1)[0][0]
                    except IndexError:
                        if_opponent_piece = None
                #make move
                im2 = np.copy(im)
                im2[which_layer,to[0],to[1]]=1
                im2[which_layer,fro[0],fro[1]]=0
                if if_opponent_piece:
                    im2[if_opponent_piece,to[0],to[1]]=0
                temp = np.zeros((8,8))
                for i in xrange(im2.shape[0]/2):
                    temp[:] = im2[2*i,:,:]
                    im2[2*i,:,:] = im2[2*i+1,:,:]
                    im2[2*i+1,:,:] = temp[:]
                pos_child = np.copy(im2)
                #print pos_child
                #pos_child = pos.move(crdn_move)
            except KeyError:
                #means the move isn't possible
                continue
            if args.multilayer: kings_layer = 10
            else:   kings_layer=5        
            if not np.any(pos_child[kings_layer,:,:]):
                value = 1.0/CHECKMATE_SCORE
                print "checkmate is going to happen"
            neg_value, _ = negamax(pos_child, depth-1, -beta, -alpha, -color, maxm)
            value = val/neg_value
        if value > best_value:
            best_value = value
            best_move = move

        # if value > alpha:
        #     alpha = value

        # if alpha > beta:
        #     break
    #print "NEGAMAX END with %s %f"%(best_move, best_value)
    return best_value, best_move

    # for move in pos.genMoves():
    #     pos_child = pos.move(move)
    #     moves.append(move)
    #     X.append(coord2d_to_chess_coord(pos_coords_to_2dcoord(move[0]))+
    #         coord2d_to_chess_coord(pos_coords_to_2dcoord(move[1])))
    #     pos_children.append(pos_child)

    # if len(X) == 0:
    #     return Exception('eh?')

    # # Use model to predict scores
    # #func = evaluate_moves_fast
    # func = evaluate_moves
    # #scores = func(X)
    # scores = func(im, X)

    #XXX
    # for i, pos_child in enumerate(pos_children):
    #     if color==1 and pos_child.board.find('K') == -1:
    #         scores[i] = CHECKMATE_SCORE
    #     elif color==-1 and pos_child.board.find('k') == -1:
    #         scores[i] = CHECKMATE_SCORE

    #child_nodes = sorted(zip(scores, top_moves), reverse=True)
    #choose only top maxm moves
    # if len(child_nodes)>maxm:
    #     child_nodes = child_nodes[0:maxm]


    
    # for move,score in top_moves:
    #     if depth == 1 or score == CHECKMATE_SCORE:
    #         value = score
    #     else:
    #         # print 'ok will recurse', sunfish.render(move[0]) + sunfish.render(move[1])
    #         crdn_move =  (sunfish.parse(move[0:2]), sunfish.parse(move[2:4]))
    #         pos_child = pos.move(crdn_move)
    #         neg_value, _ = negamax(pos_child, depth-1, -beta, -alpha, -color, maxm)
    #         value = -neg_value

        # value += random.gauss(0, 0.001)

        # crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        # print '\t' * (3 - depth), crdn, score, value



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
    def __init__(self, maxm=10 , maxd=2):
        #self._func = func
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
        alpha = float('-inf')
        beta = float('inf')

        depth = self._maxd
        t0 = time.time()
        bb = pos_board_to_bitboard(self._pos.board)
        if args.multilayer:
            im = convert_bitboard_to_image_2(bb)
        else:
            im = convert_bitboard_to_image(bb)
        im = np.rollaxis(im,2,0)
        if args.elo_layer:
            im = np.append(im,elo_layer,axis=0)
        #print im.shape
        best_value, best_move = negamax(im, depth, alpha, beta, 1, maxm=self._maxm)
        best_move = (sunfish.parse(best_move[0:2]), sunfish.parse(best_move[2:4]))
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
        # if args.multilayer:
        #     im = convert_bitboard_to_image_2(bb)
        # else:
        #     im = convert_bitboard_to_image(bb)
        # im = np.rollaxis(im, 2, 0)
        # if args.elo_layer:
        #     im = np.append(im, elo_layer, axis=0) 
        # #add the dynamic elo bias layer to be the max (=1)
        im = getim(bb)
        move_str = predictMove_TopProbMethod(im)
        #move_str = predictMove_MaxMethod(im)
        move = chess.Move.from_uci(move_str)

        if move not in bb.legal_moves:
            print "%s is NOT A LEGAL MOVE"%move
            #print list(bb.legal_moves)
            #exit(1)

        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move
        
        return gn_new

def get_move_prediction(im, method='TopProb'):
    #method can be one of 'TopProb', 'MaxProb', 'InterleavedSearch'
    if method=='TopProb':
        move_str = predictMove_TopProbMethod(im)
    elif method=='MaxMethod':
        move_str=predictMove_MaxMethod(im)
    elif method=='InterleavedSearch':
        raise NotImplementedError('Not yet implemented')
    else:
        print "Please use a supported method"
        exit(1)
    from_chess_coords = move_str[:2]
    to_chess_coords = move_str[2:4]
    from_coords = chess_coord_to_coord2d(from_chess_coords)
    to_coords = chess_coord_to_coord2d(to_chess_coords)
    return (from_coords[0]*8+from_coords[1], to_coords[0]*8+to_coords[1])


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

class Sunfish_Mod(Player):
    def __init__(self, maxn=1e4, k=10):
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxn = maxn
        self._k = k
        #self.first_move = True

    def move(self, gn_current):
        if gn_current.board().turn == 1 :

            # Apply last_move
            crdn = str(gn_current.move)
            move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
            self._pos = self._pos.move(move)
            self.first_move = False
            #t0 = time.time()
            move, score = sunfish_mod.search(self._pos, maxn=self._maxn, k=self._k)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        else:
            if gn_current.move is not None:
                # Apply last_move
                crdn = str(gn_current.move)
                move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
                self._pos = self._pos.move(move)
            #t0 = time.time()
            move, score = sunfish_mod.search(self._pos, maxn=self._maxn, k=self._k)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        move = create_move(gn_current.board(), crdn)  
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

class Sunfish_Mod2(Player):
    def __init__(self, maxn=1e4):
        self._pos = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxn = maxn
        #self.first_move = True

    def move(self, gn_current):
        if gn_current.board().turn == 1 :

            # Apply last_move
            crdn = str(gn_current.move)
            move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
            self._pos = self._pos.move(move)
            #self.first_move = False
            #t0 = time.time()
            move, score = sunfish_mod2.search(self._pos, maxn=self._maxn)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        else:
            if gn_current.move is not None:
                # Apply last_move
                crdn = str(gn_current.move)
                move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
                self._pos = self._pos.move(move)
            #t0 = time.time()
            move, score = sunfish_mod2.search(self._pos, maxn=self._maxn)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        move = create_move(gn_current.board(), crdn)  
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new

class Sunfish_Mod3(Player):
    def __init__(self, maxn=1e4):
        self._pos = sunfish_mod3.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        self._maxn = maxn
        #self.first_move = True

    def move(self, gn_current):
        if gn_current.board().turn == 1 :

            # Apply last_move
            crdn = str(gn_current.move)
            move = (sunfish.parse(crdn[0:2]), sunfish.parse(crdn[2:4]))
            self._pos = self._pos.move(move)
            #self.first_move = False
            #t0 = time.time()
            move, score = sunfish_mod3.search(self._pos, maxn=self._maxn)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(119-move[0]) + sunfish.render(119 - move[1])
        else:
            if gn_current.move is not None:
                # Apply last_move
                crdn = str(gn_current.move)
                move = (119 - sunfish.parse(crdn[0:2]), 119 - sunfish.parse(crdn[2:4]))
                self._pos = self._pos.move(move)
            #t0 = time.time()
            move, score = sunfish_mod3.search(self._pos, maxn=self._maxn)
            #print time.time() - t0, move, score
            self._pos = self._pos.move(move)

            crdn = sunfish.render(move[0]) + sunfish.render(move[1])
        move = create_move(gn_current.board(), crdn)  
        gn_new = chess.pgn.GameNode()
        gn_new.parent = gn_current
        gn_new.move = move

        return gn_new


def game():
    gn_current = chess.pgn.Game()

    maxn1 =  10 ** (1.0 + random.random() * 2.0) # max nodes for sunfish
    maxn2 =  10 ** (1.0 + random.random() * 2.0) # max nodes for sunfish_mod2
    
    # maxd = 3#random.randint(2,5)
    # maxm = 5#random.randint(1,10)
    #k = random.randint(3, 30)
    print 'maxn: %f %f' % (maxn1, maxn2)
    # print 'maxm %d' % (maxm)
    # print 'maxd %d'% (maxd)
    f = open(args.odir+'/stats.txt', 'a')
    f.write('%d %d' %(maxn1, maxn2))
    f.close()
    #load_models(args.dir)
    #player_a = Computer()
    #player_a = Sunfish(maxn=maxn)
    player_a = Sunfish_Mod2(maxn=maxn1)
    player_b = Sunfish(maxn=maxn2)
    #player_a = MySearch(maxd=maxd, maxm=maxm)
    # if against=="human":
    #     player_b = Human()
    # elif against=="sunfish":
    #     player_b = Sunfish(maxn=maxn)
    # elif against=="sunfish_mod":
    #     player_b = Sunfish_Mod(maxn=maxn, k=k)
    # else:
    #     print "Only sunfish and human players are supported currently"
    #     exit(1)

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
                print "Checkmate"
                return side
            elif gn_current.board().is_stalemate():
                print "Stalemate"
                return '-'
            elif gn_current.board().can_claim_fifty_moves():
                print "Stalemate by 50-move-rule"
                return '-' 
            elif s.find('K') == -1 or s.find('k') == -1:
                # Both AI's suck at checkmating, so also detect capturing the king
                print "King killed"
                return side

def play():
    side = game()
    if side == '-':
        print "Game Drawn"
    else:
        print "Player %s won the game."%side
    f = open(args.odir+'/stats.txt', 'a')
    f.write('%s\n' % (side))
    f.close()

if __name__ == '__main__':
    #load_models(args.dir)
    for i in xrange(10000):
        play()
