import lasagne
from lasagne.layers import InputLayer, DenseLayer, ConcatLayer, TransposedConv2DLayer, ReshapeLayer, Conv2DLayer
from lasagne.layers import TransposedConv2DLayer, batch_norm, ElemwiseSumLayer

def build_network(batch_size, z_shape, img_height, img_width, conv_nonlinearity, dense_nonlinearity):
	# Draws heavy inspiration from ResNet
	num_filters = 32
	filter_shape = (5,5)

	l_in = InputLayer((batch_size, 1, z_shape))

	dense_nonlinearity = lasagne.nonlinearities.rectify
	conv_nonlinearity = lasagne.nonlinearities.rectify

	config = {
		'conv_1_repeats': 0,
		'conv_2_repeats': 0,
		'conv_3_repeats': 0,
		'conv_4_repeats': 0
	}

	#####################
	### Decoding half ###
	#####################
	h_test = 2
	w_test = 5

	dec_2_size = h_test * w_test * num_filters * 8

	l_hid_dec_2 = batch_norm(DenseLayer(l_in, dec_2_size,nonlinearity=dense_nonlinearity))
	l_dec_reshape = ReshapeLayer(l_hid_dec_2, [batch_size, dec_2_size/h_test/w_test, h_test, w_test])

	conv_1 = batch_norm(TransposedConv2DLayer(l_dec_reshape, num_filters*8, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True))
	for _ in range(config['conv_1_repeats']):
		conv_1 = batch_norm(TransposedConv2DLayer(conv_1, num_filters*8, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True,crop='same'))

	conv_2 = batch_norm(TransposedConv2DLayer(conv_1, num_filters*4, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True, stride=(2,2),crop='same'))
	for _ in range(config['conv_2_repeats']):
		conv_2 = batch_norm(TransposedConv2DLayer(conv_2, num_filters*4, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True,crop='same'))

	conv_3 = batch_norm(TransposedConv2DLayer(conv_2, num_filters*2, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True, stride=(2,2),crop='same'))
	for _ in range(config['conv_3_repeats']):
		conv_3 = batch_norm(TransposedConv2DLayer(conv_3, num_filters*2, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True, crop='same'))

	conv_4 = batch_norm(TransposedConv2DLayer(conv_3, num_filters, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True, stride=(2,2),crop='same'))
	for _ in range(config['conv_4_repeats']):
		conv_4 = batch_norm(TransposedConv2DLayer(conv_4, num_filters, filter_shape, nonlinearity=conv_nonlinearity, untie_biases=True, crop='same'))

	l_out = batch_norm(TransposedConv2DLayer(conv_4, 1, filter_shape, nonlinearity=lasagne.nonlinearities.sigmoid, untie_biases=True,crop='same'))

	return l_in, l_out

