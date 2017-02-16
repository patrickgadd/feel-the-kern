import theano.tensor as T
import theano
import sys
sys.path.insert(0,'./Lasagne') # local checkout of Lasagne
import lasagne

import theano.tensor.nnet

import numpy as np
from os import remove, makedirs
from os.path import isfile, exists
import time

from nn_classes.mk1 import build_network
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from generate_training_data import get_batch

def main():
	z_dim = 10
	n_gram_cnt = 2 # number of chars in the N-grams
	batch_size = 32
	l_r = 0.4 / 128.0 * batch_size

	alphabet = 'abcdefghijklmnopqrstuvwxyz'.upper() + '-'
	alphabet_size = len(alphabet)

	val_cnt = batch_size*10

	output_shape = (65,41)

	(X_train, Y_train) = get_batch(batch_size, n_gram_cnt, alphabet, z_dim, output_shape)
	(X_val, Y_val) = get_batch(val_cnt, n_gram_cnt, alphabet, z_dim, output_shape)

	img_height = Y_train.shape[2]
	img_width = Y_train.shape[3]
	z_shape = X_train.shape[2]

	assert(batch_size == X_train.shape[0])
	assert(X_train.shape[0] == Y_train.shape[0])
	assert(z_shape == alphabet_size*n_gram_cnt + z_dim)
	assert(X_train.shape[2] == alphabet_size*n_gram_cnt + z_dim)
	assert(X_val.shape[2] == alphabet_size*n_gram_cnt + z_dim)

	output_var = T.tensor4('X')
	conv_nonlinearity = lasagne.nonlinearities.rectify
	dense_nonlinearity = lasagne.nonlinearities.rectify

	l_in, l_out = build_network(
		batch_size,
		z_shape,
		img_height,
		img_width,
		conv_nonlinearity,
		dense_nonlinearity)

	print('Parameter count in the network: {0:.2f} million'.format(lasagne.layers.helper.count_params(l_out)/float(10**6)))
	print "output shape #X:",l_out.output_shape

	# Verify that the image sizes matches the expected
	assert(Y_train.shape[1:] == l_out.output_shape[1:])

	def abs_error(a, b):
		return abs(a - b)

	all_params = lasagne.layers.get_all_params(l_out, trainable=True)
	prediction = lasagne.layers.get_output(l_out)
	loss = abs_error(prediction, output_var)
	aggregated_loss = lasagne.objectives.aggregate(loss)# / img_blackness(output_var)

	updates = lasagne.updates.nesterov_momentum(aggregated_loss, all_params, l_r)
	train_fn = theano.function([l_in.input_var, output_var], loss, updates=updates)

	predict_fn = theano.function([l_in.input_var], lasagne.layers.get_output(l_out, deterministic=True))

	# Create a validation function
	val_prediction = lasagne.layers.get_output(l_out)

	val_loss = abs_error(val_prediction, output_var)

	# # Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([l_in.input_var, output_var], val_loss)

	min_mva_loss = 9143.44

	# print('Loading network...')
	# with np.load('../data/networks/network-model_{0:.2f}.npz'.format(min_mva_loss)) as f:
	# 	param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(l_out, param_values)

	print('Starting training...')

	loss_list = []
	mva_length = 40
	mva_loss = -1
	start_time = -1
	timer_cnt = 5
	start_iter = 0


	for iter in range((start_iter / batch_size) +1, 200000 +1):
		# print('iter: {0}'.format(iter))
		if iter == start_iter/batch_size+1:
			start_time = time.time()
		if iter == start_iter/batch_size+1+timer_cnt:
			print('Time per data point: {0} s'.format((time.time()-start_time) / timer_cnt / batch_size ))

		(X_train_batch, Y_train_batch) = get_batch(batch_size, n_gram_cnt, alphabet, z_dim, output_shape)

		losses = train_fn(X_train_batch, Y_train_batch)
		loss_per_pixel = np.sum(losses)*1000 / (batch_size*img_height*img_width)

		loss_list.append(loss_per_pixel)

		if len(loss_list) > mva_length:
			loss_list.pop(0)
		if len(loss_list) == mva_length:
			mva_loss = sum(loss_list) / mva_length

		# Run against the validation set and dump to logs
		if (iter*batch_size % (256*40) == 0) and (len(loss_list) == mva_length):
			val_loss_sum = 0
			for i in range(int(val_cnt/batch_size)):
				X_val_batch = X_val[i*batch_size: (i+1)*batch_size, :, :]
				Y_val_batch = Y_val[i*batch_size: (i+1)*batch_size, :, :, :]
				val_losses = val_fn(X_val_batch, Y_val_batch)
				val_loss_per_pixel = np.sum(val_losses)*1000 / (batch_size*img_height*img_width)
				val_loss_sum += val_loss_per_pixel

			val_loss_avg = val_loss_sum / float(val_cnt/batch_size)

			with open('validation_error_log.txt', "a") as myfile:
				myfile.write('{0}\t{1:.4f}\n'.format(iter*batch_size, val_loss_avg))
			with open('training_error_log.txt', "a") as myfile:
				myfile.write('{0}\t{1:.4f}\n'.format(iter*batch_size, mva_loss))

		if ((iter % 20) == 0) and (len(loss_list) == mva_length):
			print('Data points processed: # {0}. Loss per pixel (*10^3): {1:.2f} ({2}-step moving average)'.format(iter*batch_size, mva_loss, mva_length))

		if (iter % 50 == 0) and (len(loss_list) == mva_length):
			# SAVING THE NETWORK
			if (mva_loss < min_mva_loss):
				previous_min_loss = min_mva_loss
				min_mva_loss = mva_loss
				dump_dir = '../data/networks/'
				network_path = dump_dir + 'network-model_{0:.2f}.npz'.format(mva_loss)
				np.savez(network_path, *lasagne.layers.get_all_param_values(l_out))
				previous_filename = dump_dir + 'network-model_{0:.2f}.npz'.format(previous_min_loss)
				if isfile(previous_filename) and (not (network_path == previous_filename)):
					remove(previous_filename)
				# winsound.PlaySound('hold-it.wav', winsound.SND_FILENAME)
				print('Saved the network at {0}'.format(network_path))

			print("iter*batch_size %d: Loss per pixel (*10^3) %g" % (iter*batch_size, mva_loss))

		if len(loss_list) == mva_length and iter != 0 and (iter % 200 == 0):
			print('Saving plots')

			vis_amount = 64
			imgs_per_row = 8
			vis_array = []
			assert(vis_amount % imgs_per_row == 0)
			assert( (vis_amount/imgs_per_row) <= val_cnt)
			assert( (batch_size % vis_amount == 0) or (vis_amount % batch_size == 0))

			# TODO: old and terribly inefficient, but hey - it dumps the output of the network
			for i in range(int(vis_amount/imgs_per_row)):
				x_chunk = X_val[i*imgs_per_row:(i)*imgs_per_row+batch_size,:,:]
				y_chunk = Y_val[i*imgs_per_row:(i)*imgs_per_row+batch_size,:,:,:]
				result = predict_fn(x_chunk)
				for z in range(1, int(imgs_per_row/batch_size)):
					x_chunk_ = X_val[i*imgs_per_row+z*batch_size:i*imgs_per_row+(z+1)*batch_size,:,:]
					y_chunk_ = Y_val[i*imgs_per_row+z*batch_size:i*imgs_per_row+(z+1)*batch_size,:,:,:]
					result_ = predict_fn(x_chunk_)
					y_chunk = np.concatenate((y_chunk, y_chunk_))
					result = np.concatenate((result, result_))

				vis1 = np.hstack([y_chunk[j,0,:,:] for j in range(imgs_per_row)])
				vis2 = np.hstack([result[j,0,:,:] for j in range(imgs_per_row)])
				vis_array.append(vis1)
				vis_array.append(vis2)

			fig = plt.figure(frameon=False)
			ax = plt.Axes(fig, [0., 0., 1., 1.])
			ax.set_axis_off()
			fig.add_axes(ax)
			ax.imshow(np.vstack(vis_array), cmap='Greys_r', aspect='equal')
			fig.savefig('../data/nn_train_imgs/{0}_{1:.2f}.png'.format(iter*batch_size, mva_loss), dpi=200, bbox_inches='tight', pad_inches=0)
			plt.close()

	print "done."

	# import ipdb; ipdb.set_trace()
if __name__ == "__main__":
	if not exists('../data/nn_train_imgs/'):
		makedirs('../data/nn_train_imgs')
		print('Created folder for storing images of the output of the network during training')
	if not exists('../data/networks/'):
		makedirs('../data/networks/')
		print('Created folder for storing networks')

	main()

