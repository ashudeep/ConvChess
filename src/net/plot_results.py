import numpy as np 
import matplotlib.pyplot as plt
import caffe
import argparse
parser=argparse.ArgumentParser\
	(description='Plot results for training and testing losses and accuracies')
parser.add_argument('--file', type=str, help='File generated during training')
parser.add_argument('--no-header',dest='header',action='store_false',help='skip the first row as a header')
parser.add_argument('--output',type=str, help='directory for output plots')
parser.add_argument('--mode', choices= ['test','train'])
parser.add_argument('--filetype',choices=['txt','pkl'])
parser.add_argument('--modelname',choices=['Piece','Pawn', 'Rook',\
	'Knight','Bishop','Queen','King'])
parser.add_argument('--ld', type=float, help='density of the line in training plot..use values from 0 to 1.',default=0.1)
parser.set_defaults(header=True)
parser.set_defaults(mode='test')
parser.set_defaults(filetype='txt')
parser.set_defaults(modelname='Piece')
args = parser.parse_args()
if args.filetype == 'pkl':
 	import cPickle as pkl

skip_header = args.header
if args.mode == 'test':
	x=[]
	y1=[]
	y2=[]
	y3=[]
	y4=[]
	y5=[]
	y6=[]
	with open(args.file,'r') as results_file:
		for line in results_file:
			if skip_header:
				skip_header=False
				continue
			[niter,train_loss,test_loss,acc_1,acc_3,acc_5,acc_10]=line.strip().split("\t")
			#print line
			x.append(int(niter))
			y1.append(float(train_loss))
			y2.append(float(test_loss))
			y3.append(float(acc_1))
			y4.append(float(acc_3))
			y5.append(float(acc_5))
			y6.append(float(acc_10))
	#print x,y1,y2
	fig = plt.figure(1)
	ax1 = plt.subplot(131)
	ax1.plot(x,y1)
	ax1.set_xlabel('Number of iterations',family='serif')
	ax1.set_ylabel('Training Loss',family='serif')
	ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax2 = plt.subplot(132, sharey=ax1)
	ax2.plot(x,y2)
	ax2.set_xlabel('Number of iterations',family='serif')
	ax2.set_ylabel('Test Loss',family='serif')
	ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax3 = plt.subplot(133)
	ax3.plot(x,y3, 'g', label='k=1')
	ax3.plot(x,y4, 'g--', label='k=3')
	ax3.plot(x,y5,'b--', label='k=5')
	ax3.plot(x,y6,'b', label='k=10')
	ax3.set_xlabel('Number of iterations',family='serif')
	ax3.set_ylabel('Test Accuracies',family='serif')
	ax3.yaxis.tick_right()
	ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	ax3.yaxis.set_label_position("right")
	legend = ax3.legend(loc='lower right', shadow=True,numpoints=1)
	#fig.suptitle('Results for %s model'%args.modelname, family='serif',
	#	fontsize=20)
	plt.show()
	if args.output:
		plt.savefig(args.output+'/plot.png')

elif args.mode =='train':
	if args.filetype=='txt':
		losses = []
		with open(args.file,'r') as results_file:
			for line in results_file:
				if skip_header:
					skip_header=False
					continue
				loss = float(line.strip())
				losses.append(loss)
	elif args.filetype=='pkl':
		losess = pkl.load(open(args.file,'r'))
	else:
		print "File %s type not supported"%args.filetype
	plt.plot(range(len(losses))[::int(1.0/args.ld)],losses[::int(1.0/args.ld)])
	#plt.suptitle('Training Loss vs Number of iterations for %s model'%args.modelname, family='serif', fontsize=20)
	plt.xlabel('Number of iterations')
	plt.ylabel('Training Loss')
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.show()
	if args.output:
		plt.savefig(args.output+'/plot.png')





