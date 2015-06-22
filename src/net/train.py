import numpy as np 
import matplotlib.pyplot as plt
import caffe
import argparse
import os
import cPickle as pkl
parser=argparse.ArgumentParser\
	(description='ConvNet for learning Chess moves')
#parser.add_argument('--data', type=str, default='', help='The data file')
parser.add_argument('--testing', action='store_true', help='Testing Mode gets on')
parser.add_argument('--net', choices= ['PIECE','P', 'R', 'B', 'N', 'Q', 'K'])
parser.add_argument('--cpu', action='store_true')
#parser.add_argument('--labels', type=str, default='', help='Labels for training.')
# parser.add_argument('--test-data', type=str, default='',
# 	dest='test_data', help='The test data file')
# parser.add_argument('--test-labels', type=str, default='', 
# 	dest='test_labels', help='The test label file')
parser.add_argument('--resume-state', type=str, default='',
	dest='resume_state', help='Solver state file to resume the training from')
parser.add_argument('--resume-model', type=str, default='',
	dest='resume_model', help='CaffeModel to resume training from')
parser.add_argument('--niters', type=int, default=300000,
	dest='niters', help='Number of iterations to train')
parser.add_argument('--test-int', type=int, default=10000,
	dest='test_interval', help='Interval to test the network after')
parser.add_argument('--save-int', type=int, default=10000,
	dest='save_interval', help='Interval to save the results')
parser.add_argument('--solver', type=str, default='',
	dest='solver', help='File that contains the solver configuration')
parser.add_argument('-r','--regr', dest='regression', action='store_true',
	help='Whether to train a regression or not. Default: False' )
parser.set_defaults(cpu=False)
parser.set_defaults(testing=False)
parser.set_defaults(regression=False)
args = parser.parse_args()

# if args.labels == '':
# 	print "Please choose --labels <labels_file>"
# 	exit(1)

# data = np.load(args.data)
# labels = np.load(args.labels)
# length = data.shape[0] - (data.shape[0]%1000)
# data = data[0:length]
# labels = labels[0:length]
# data = data.astype('float32')
# labels = labels.astype('float32')
#test_data = data[-3000:]#np.load(args.test_data)
#test_labels = labels[-3000:]#np.load(args.test_labels)
#data = data[0:length-3000]
#labels = labels[0:length-3000]

if args.cpu:
	print "CPU Mode set"
	caffe.set_mode_cpu()
else:
	print "GPU mode set"
	caffe.set_mode_gpu()
	caffe.set_device(0)

niters = args.niters
test_interval = args.test_interval
save_interval = args.save_interval
train_losses = []
test_iter_ids = []
test_losses = []
model_name = args.solver.split('/')[1].split('_')[0]
if not os.path.isdir('models/%s'%model_name):
	os.mkdir('models/%s'%model_name)
accuracies = {'accuracy@1':[], 'accuracy@3':[], 'accuracy@5':[], 'accuracy@10':[]}

if not args.testing and not args.regression: # Training
	solver = caffe.SGDSolver(args.solver)
	if args.resume_state !='' :	
		solver.restore(args.resume_state)
	else:
		f = open("models/%s/results.txt"%model_name,"a")
		f.write("iter\ttrain_loss\ttest_loss\taccuracy@1\taccuracy@3\taccuracy@5\taccuracy@10\n")
		f.close()
	for it in xrange(niters):
		solver.step(1)
		train_losses.append(solver.net.blobs['loss'].data.flat[0])

		solver.test_nets[0].forward(start='conv1')

		if it % test_interval == 0 :
			print 'Iteration %d testing...'%it
			test_iter_ids.append(it)
			test_losses.append(solver.test_nets[0].blobs['loss'].data.flat[0])
			for key in accuracies:
				accuracies[key].append(solver.test_nets[0].blobs[key].data.flat[0]*100)

		if it%save_interval == 0:
			print "Saving the loss and accuracy values"
			# np.savetxt('models/%s/train_losses.txt'%model_name, train_losses,
			# 	delimiter=',', fmt='%1.4e', header='training_loss')
			# np.savetxt('models/%s/test_losses.txt'%model_name, np.vstack((test_iter_ids,\
			# 	test_losses,\
			# 	accuracies['accuracy@1'],accuracies['accuracy@3'],\
			# 	accuracies['accuracy@5'],accuracies['accuracy@10'])).reshape(len(test_iter_ids),6),
			# 	header="iter, test_loss, accuracy@1, accuracy@3,accuracy@5, accuracy@10",
			# 	delimiter=',', fmt='%d,%1.4e,%2.4f,%2.5f,%2.4f,%2.4f')
			
			f = open("models/%s/results.txt"%model_name,"a+")
			f.write("%d\t%1.4e\t%1.4e\t%2.4f\t%2.4f\t%2.4f\t%2.4f\n"%(it,train_losses[-1],test_losses[-1],\
				accuracies['accuracy@1'][-1],accuracies['accuracy@3'][-1],\
				accuracies['accuracy@5'][-1],accuracies['accuracy@10'][-1]))
			f.close()
			if it%(10*save_interval)==0:
				#save pickles 10 times less often
				pkl.dump(train_losses, open('models/%s/train_loss.pkl'%model_name,'w'))
				pkl.dump(test_losses, open('models/%s/test_loss.pkl'%model_name,'w'))
				pkl.dump(accuracies, open('models/%s/accuracies.pkl'%model_name,'w'))

elif args.regression and not args.testing:
	solver = caffe.SGDSolver(args.solver)
	if args.resume_state !='' :	
		solver.restore(args.resume_state)
	else:
		f = open("models/%s/results.txt"%model_name,"a")
		f.write("iter\ttrain_loss\ttest_loss\n")
		f.close()
	for it in xrange(niters):
		solver.step(1)
		train_losses.append(solver.net.blobs['loss'].data.flat[0])

		solver.test_nets[0].forward(start='conv1')

		if it % test_interval == 0 :
			print 'Iteration %d testing...'%it
			test_iter_ids.append(it)
			test_losses.append(solver.test_nets[0].blobs['loss'].data.flat[0])
		if it%save_interval == 0:
			print "Saving the loss and accuracy values"
			f = open("models/%s/results.txt"%model_name,"a+")
			f.write("%d\t%1.4e\t%1.4e\n"%(it,train_losses[-1],test_losses[-1]))
			f.close()
else: # Testing
	print "Testing on %d inputs." % inputs.shape[0]
	classifier = caffe.Classifier("move.prototxt", "%s_train.caffemodel" % args.net, gpu=abs(1-args.cpu))
	prediction = classifier.predict(data)
	if args.labels:
		print "Accuracy is %f"% np.mean(prediction == labels)
	print prediction
