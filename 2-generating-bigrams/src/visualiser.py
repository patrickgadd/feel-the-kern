import theano.tensor as T
import theano
import sys
sys.path.insert(0,'./Lasagne') # local checkout of Lasagne
import lasagne

from PIL import Image

import theano.tensor.nnet


from os.path import exists
from os import makedirs

import numpy as np
from os import remove
from os.path import isfile
import time

from nn_classes.mk1 import build_network
from generate_training_data import get_batch
from random import sample
from math import ceil

# This function generates bigrams of a sentence in different fonts corresponding to interpolations between some of the original styles
# The generated bigrams are stored under "/visualisations/sentence-bigrams/$SENTENCE/font-$X/" for matching in "Step 3"
def plot_bigram_interpolations(zs, batch_size, alphabet, sentence, interpol_steps_cnt, no_of_fonts, predict_fn, visualisation_dir):
	def generate_bigram_imgs(sentence, alphabet, z, batch_size, interpol_steps_cnt, predict_fn):
		one_hots = []
		bigrams = []
		for i in range(len(sentence)-1):
			bigram = sentence[i:i+2]
			bigrams.append(bigram)
			one_hot = []
			for c in bigram:
				one_hot.extend([1 if c == alphabet[i] else 0 for i in range(len(alphabet))])

			one_hots.append(one_hot)

		# Generate the inputs to the network
		xs = [one_hot + z for one_hot in one_hots]
		while len(xs) < batch_size:
			xs.append(xs[0])
		xs = np.asarray(xs).astype('float32')
		xs = np.reshape(xs, (batch_size, 1, xs.shape[1]))

		# Have the network produce bigrams
		nn_imgs = predict_fn(xs)

		# Store the images in a list and return this with the corresponding bigrams
		bigram_imgs = []
		for i in range(len(sentence)):
			bigram_img = nn_imgs[i, 0,:,:]
			bigram_img = bigram_img - np.min(bigram_img)
			bigram_img = bigram_img / np.max(bigram_img) * 255.0
			bigram_img = bigram_img.astype('uint8')
			bigram_imgs.append(bigram_img)

		return [bigram_imgs, bigrams]


	assert((len(sentence) < batch_size) and len(sentence) >= 2)
	assert(no_of_fonts > 2)
	# Verify that the sentence can be produced with the alphabet
	for c in sentence:
		assert(c in alphabet)

	# alphabet_imgs = []
	plt_cnt = 0 # For using a different folder for each type of font

	for i, z in enumerate(zs):
		if i % 10 == 0:
			print('Generating bigrams no. {0} of {1}'.format(i, len(zs)))

		plt_cnt += 1

		dump_dir = visualisation_dir + 'sentence-bigrams/{0}/font-{1}/'.format(sentence,str(plt_cnt).zfill(3))
		if not exists(dump_dir):
			makedirs(dump_dir)

		bigram_imgs, bigrams = generate_bigram_imgs(sentence, alphabet, z, batch_size, interpol_steps_cnt, predict_fn)

		# Store the bigrams for later matching and merging in "step 3"
		for i, bigram in enumerate(bigrams):
			img = Image.fromarray(bigram_imgs[i])
			img.save(dump_dir + bigram + '.jpg')



def interpolate_zs(z_1, z_2, interpol_cnt):
	z_dim = len(z_1)

	linspaces = np.linspace(0,1,num=interpol_cnt)
	zs = []
	# Beautifully inefficient interpolation
	for i, intp in enumerate(linspaces):
		z_i = []
		for j in range(z_dim):
			z_ij = z_1[j]*(1-intp) + z_2[j]*(intp)
			z_i.append(z_ij)

		zs.append(z_i)

	return zs

def interpolate_pair(bigram, z_1, z_2, interpol_cnt, predict_fn):
	zs = interpolate_zs(z_1, z_2, interpol_cnt)

	# Generate the interpolated inputs
	X_intp = []
	for i in range(interpol_cnt):
		x = bigram.tolist()
		x.extend(zs[i])
		X_intp.append(x)

	# TODO: handle if more than batch_size?
	assert(interpol_cnt <= batch_size)

	while len(X_intp) < batch_size:
		X_intp.append(X_intp[-1])

	X_intp = np.asarray(X_intp)
	X_intp = np.reshape(X_intp, (batch_size, 1, X_intp.shape[1]))
	X_intp = X_intp.astype('float32')

	result = predict_fn(X_intp)
	result = np.reshape(result, (result.shape[0], result.shape[2], result.shape[3]) )
	img_height = result.shape[1]
	img_width = result.shape[2]

	# interpol_imgs = result[:interpol_cnt,:,:,:] #np.zeros( (interpol_cnt, img_height, img_width) )
	# interpol_imgs = np.reshape(interpol_imgs, (interpol_cnt, img_height, img_width))
	interpol_img = np.zeros( (img_height, img_width*interpol_cnt) ).astype('float32')
	for i in range(interpol_cnt):
		interpol_img[:, i*img_width: (1+i)*img_width] = result[i,:,:]

	interpol_img = interpol_img - np.min(interpol_img)
	interpol_img = interpol_img / np.max(interpol_img) * 255.0
	interpol_img = interpol_img.astype('uint8')

	return interpol_img

from random import uniform

def plot_alphabet(z, batch_size, alphabet_size, predict_fn):
	assert(alphabet_size <= batch_size)

	z_dim = len(z)

	X = np.ones( (batch_size, 1, alphabet_size+z_dim) )
	for a in range(alphabet_size):
		# Encode the character in one-hot
		one_hot = np.asarray([ 1 if n == a else 0 for n in range(alphabet_size) ])

		X[a, :, 0:alphabet_size] = one_hot
		X[a, :, -z_dim:] = np.copy(z)

	# Have the network generate images
	X = X.astype('float32')
	result = predict_fn(X)

	# Merge the images
	img_height = result.shape[2]
	img_width = result.shape[3]
	imgs_per_row = 8
	row_cnt = int(ceil(alphabet_size / float(imgs_per_row)))

	alphabet_img = np.ones( (img_height*row_cnt, img_width*imgs_per_row) )

	for i in range(alphabet_size):
		x = i % imgs_per_row
		y = (i - x) / imgs_per_row
		alphabet_img[y*img_height:(y+1)*img_height, x*img_width:(x+1)*img_width] = result[i,0,:,:]

	# Store the image
	alphabet_img = alphabet_img - np.min(alphabet_img)
	alphabet_img = alphabet_img / np.max(alphabet_img) * 255.0
	alphabet_img = alphabet_img.astype('uint8')
	return alphabet_img

def interpolate_alphabet(zs, z_dim, batch_size, alphabet_size, interpol_steps_cnt, no_of_fonts, predict_fn, visualisation_dir):
	img_cnt = 0

	for i, z in enumerate(zs):
		if i % 10 == 0:
			print('plotting and dumping alphabet no. {0} of {1}'.format(i, len(zs)))

		alphabet_img = plot_alphabet(z, batch_size, alphabet_size, predict_fn)

		alphabet_img = Image.fromarray(alphabet_img)
		alphabet_img.save(visualisation_dir + 'alphabets/{0}.jpg'.format(str(img_cnt).zfill(3)))
		img_cnt += 1

def extract_zs(X, no_of_fonts, interpol_steps_cnt, transition_similarity):
	# This functions extracts "no_of_fonts" style-Zs, finds somewhat similar ones and interpolates between them
	original_zs = []
	batch_size = X.shape[0]

	# Extract all the initial styles
	for i in range(batch_size):
		z = X[i, 0, -z_dim:]
		original_zs.append(z)

	# Choose "no_of_fonts" somewhat similar styles

	squared_dists = np.ones( (batch_size, batch_size) )
	for i in range(batch_size):
		for j in range(i+1,batch_size):
			zi = original_zs[i]
			zj = original_zs[j]
			sq_dist = zi - zj
			sq_dist = np.square(sq_dist)
			sq_dist = np.sum(sq_dist)

			squared_dists[i,j] = sq_dist
			squared_dists[j,i] = sq_dist

	dists_mean = np.mean(squared_dists)
	dists_std = np.std(squared_dists)

	prev_style_idx = 0 # used to find similar font styles to the last in the list of chosen styles, for a (hopefully) nice animation
	chosen_style_idxs = [prev_style_idx] # Has to start with a font, to it will be the first
	remaining_style_idxs = range(1,batch_size)

	# Select some styles that are somewhat similar, but preferably not too much nor too little
	while len(chosen_style_idxs) < (no_of_fonts):
		pref_min_dist = dists_mean - dists_std*transition_similarity
		pref_max_dist = dists_mean - dists_std*transition_similarity

		chosen_style_idx = -1

		# Find a style somewhat similar to the latest chosen one
		while chosen_style_idx == -1:
			for style_idx in remaining_style_idxs:
				dist = squared_dists[prev_style_idx, style_idx]

				# If the style is within the preferred range, choose it
				if pref_min_dist <= dist and dist <= pref_max_dist:
					chosen_style_idx = style_idx
					break

			if chosen_style_idx == -1: # Become less picky about the similarity range
				pref_min_dist -= dists_mean*dists_std*0.05
				pref_max_dist += dists_mean*dists_std*0.05

		chosen_style_idxs.append(chosen_style_idx)
		prev_style_idx = chosen_style_idx

		# Remove the chosen style-idx from the list of remaining possible styles (we don't want to repeat ourselves here)
		remaining_style_idxs = [idx for idx in remaining_style_idxs if idx not in chosen_style_idxs]


	# add the first style at the end of chosen_style_idxs to make the style iterations loop
	chosen_style_idxs.append(chosen_style_idxs[0])

	zs_interpolated = []
	for i in range(no_of_fonts):
		z_1 = original_zs[chosen_style_idxs[i]]
		z_2 = original_zs[chosen_style_idxs[i+1]]

		zs_interpolated.extend(interpolate_zs(z_1, z_2, interpol_steps_cnt))

	return zs_interpolated



if __name__ == "__main__":
	z_dim = 10
	n_gram_cnt = 2 # number of chars in the N-gram
	batch_size = 32

	visualisation_dir = '../../visualisations/'

	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	alphabet = alphabet.upper()
	alphabet = alphabet + '-'
	alphabet_size = len(alphabet)

	val_cnt = batch_size*10
	output_shape = (65,41)

	(X, Y) = get_batch(batch_size, n_gram_cnt, alphabet, z_dim, output_shape)

	img_height = Y.shape[2]
	img_width = Y.shape[3]
	z_shape = X.shape[2]

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

	predict_fn = theano.function([l_in.input_var], lasagne.layers.get_output(l_out, deterministic=True))

	min_mva_loss = 59.79

	print('Loading network...')
	with np.load('../data/networks/network-model_{0:.2f}.npz'.format(min_mva_loss)) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(l_out, param_values)

	#########################################################################
	### Plot the characters and a few interpolations for a nice animation ###
	#########################################################################

	# interpol_steps_cnt = 10
	# no_of_fonts = 20
	# transition_similarity = 1.0 # Choose a number in the range 0 to about 3 (0 is completely random order of styles, 3 is picking the most similar transition every time)
	# zs = extract_zs(X, no_of_fonts, interpol_steps_cnt, transition_similarity)
	#
	# interpolate_alphabet(zs, z_dim, batch_size, alphabet_size, interpol_steps_cnt, no_of_fonts, predict_fn, visualisation_dir)

	#########################################################
	### Plot bi-grams of a sentence in interpolated fonts ###
	#########################################################

	sentence = 'MACHINE-LEARNING'
	interpol_steps_cnt = 4
	no_of_fonts = 5
	assert(n_gram_cnt == 2)

	transition_similarity = 1.0 # Choose a number in the range from about -3 to about 3 (higher number, more similar fonts at every transition)
	zs = extract_zs(X, no_of_fonts, interpol_steps_cnt, transition_similarity)
	plot_bigram_interpolations(zs, batch_size, alphabet, sentence, interpol_steps_cnt, no_of_fonts, predict_fn, visualisation_dir)
