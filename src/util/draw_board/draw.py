#! /bin/python
"""
Draws a chess board for a bitmap of size (6x8x8) 
where each kind of piece is in a different channel
Author: Ashudeep Singh (17-05-2015)
"""

import numpy as np
import cv2
pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']

def blank_board(img_size):
	board = np.zeros((img_size, img_size), dtype=np.uint8)
	piece_size = img_size/8
	for i in xrange(8):
		for j in xrange(8):
			if (i+j)%2 == 0:
				board[i*piece_size:(i+1)*piece_size,j*piece_size:(j+1)*piece_size] = 255*np.ones((piece_size, piece_size), dtype=np.uint8)
	return board

def draw_board(bitmap, img_size=344, highlight=None):
	piece_size = img_size/8

	#initialize blank board
	board = blank_board(img_size)
	bw = cv2.CV_LOAD_IMAGE_GRAYSCALE
	#load piece images
	wpawnw = cv2.imread("elements/wpawnw.png", bw)
	wpawnb = cv2.imread("elements/wpawnb.png", bw)
	bpawnw = abs(255-wpawnb)
	bpawnb = abs(255-wpawnw)
	wrookw = cv2.imread("elements/wrookw.png", bw)
	wrookb = cv2.imread("elements/wrookb.png", bw)
	brookw = abs(255-wrookb)
	brookb = abs(255-wrookw)
	wknightw = cv2.imread("elements/wknightw.png", bw)
	wknightb = cv2.imread("elements/wknightb.png", bw)
	bknightw = abs(255-wknightb)
	bknightb = abs(255-wknightw)
	wbishopw = cv2.imread("elements/wbishopw.png", bw)
	wbishopb = cv2.imread("elements/wbishopb.png", bw)
	bbishopw = abs(255-wbishopb)
	bbishopb = abs(255-wbishopw)
	wqueenw = cv2.imread("elements/wqueenw.png", bw)
	wqueenb = cv2.imread("elements/wqueenb.png", bw)
	bqueenw = abs(255-wqueenb)
	bqueenb = abs(255-wqueenw)
	wkingw = cv2.imread("elements/wkingw.png", bw)
	wkingb = cv2.imread("elements/wkingb.png", bw)
	bkingw = abs(255-wkingb)
	bkingb = abs(255-wkingw)

	for p in xrange(bitmap.shape[0]):
		for i in xrange(bitmap.shape[1]):
			for j in xrange(bitmap.shape[2]):
				if bitmap[p,i,j] == 1:
					piece = "w"
				elif bitmap[p,i,j] == -1:
					piece = "b"
				else:
					continue
				piece = piece+pieces[p]
				if (i+j)%2 == 0:
					piece = piece+"w"
				else:
					piece = piece+"b"
				piece = eval(piece)
				piece = cv2.resize(piece, (piece_size, piece_size)) 
				board[i*piece_size:(i+1)*piece_size,j*piece_size:(j+1)*piece_size] = piece
	if highlight:
		i, j = highlight
		stripes = 5
		patt = np.eye(piece_size/stripes, dtype=np.uint8)
		if (i+j)%2==0:
			#background is white
			patt = abs(255-patt*125)
		else:
			patt = abs(patt*125)
		filtr = np.tile(patt, (stripes, stripes))
		filtr = cv2.resize(filtr, (piece_size, piece_size))
		board[i*piece_size:(i+1)*piece_size,j*piece_size:(j+1)*piece_size] = board[i*piece_size:(i+1)*piece_size,j*piece_size:(j+1)*piece_size]/2 + filtr/2
	return board

#cv2.startWindowThread()
#cv2.namedWindow("board")
#cv2.imshow("board",board)
if __name__ == '__main__':
	bitmap = np.load("sample_bitmap.npy")
	board = draw_board(bitmap, size = 480, highlight=(4,3))
	cv2.imwrite("board.png", board)
