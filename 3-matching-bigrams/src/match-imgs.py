import numpy as np
from PIL import Image
from random import uniform
from math import exp, log
from os.path import exists
from os import makedirs
import json

global_orig_img_width = 65
global_orig_img_height = 41

char_widths = {
		'I': 12.0,
		'J': 15.0,
		'-': 17.0,
		'L': 22.0,
		'E': 25.0,
		'F': 25.0,
		'S': 25.0,
		'Z': 25.0,
		'T': 26.0,
		'Y': 26.0,
		'K': 27.0,
		'X': 27.0,
		'P': 27.0,
		'R': 28.0,
		'B': 28.0,
		'C': 28.0,
		'V': 30.0,
		'A': 31.0,
		'D': 32.0,
		'U': 33.0,
		'G': 33.0,
		'H': 32.0,
		'N': 34.0,
		'O': 35.0,
		'Q': 36.0,
		'M': 45.0,
		'W': 47.0
	}



class State:
	def __init__(self, img_height, img_width):
		self.img_height = img_height
		self.img_width = img_width
		self.x_translation = int(img_width * 0.5) #int(img_width * 1.5)
		self.y_translation = img_height

	def get_clone(self):
		clone = State(self.img_height, self.img_width)
		clone.x_translation = self.x_translation
		clone.y_translation = self.y_translation

		return clone

	def to_obj(self):
		return {
			'img_height': self.img_height,
			'img_width': self.img_width,
			'x_translation': self.x_translation,
			'y_translation': self.y_translation
		}

class Action:
	def __init__(self, type, axis, direction):
		self.type = type
		self.axis = axis
		self.direction = direction

	def to_obj(self):
		return {
			'type': self.type,
			'axis': self.axis,
			'direction': self.direction
		}

def sign(n):
	if n == 0:
		return 0
	return 1 if n > 0 else -1

def get_random_action():
	magnitude = int(uniform(1,20))

	return Action(
		'translation' if uniform(0,1) < 0.5 else 'scale',
		'x' if uniform(0,1) < 0.5 else 'y',
		magnitude if uniform(0,1) < 0.5 else -magnitude
	)

def apply_action(state, action):
	new_state = state.get_clone()

	if action.type == 'translation':
		if action.axis == 'x': # TODO: wrote this when tired, give it an extra look
			new_state.x_translation += action.direction # max(int(abs(action.magnitude) * global_orig_img_width), 1) * sign(action.magnitude)
		else:
			new_state.y_translation += action.direction #max(int(abs(action.magnitude) * global_orig_img_height), 1) * sign(action.magnitude)
	else: # action.type == 'scaling'
		if action.axis == 'x':
			new_state.img_width += action.direction #int(new_state.img_width * (1.0 + action.magnitude))
		else:
			new_state.img_height += action.direction #int(new_state.img_height * (1.0 + action.magnitude))

	return new_state

def test_action_legality(state, action, tri1, tri2):
	# TODO: check wether img2 will be within img1
	test_state = state.get_clone()
	test_state = apply_action(test_state, action)
	fail_val = [False, None]

	# check that image isn't too small to be meaningful
	min_scale_limit = 0.5
	if test_state.img_height < min_scale_limit * global_orig_img_height:
		return fail_val
	if test_state.img_width < min_scale_limit * global_orig_img_width:
		return fail_val

	# Check that the action doesn't move the image outside of the upper left corner
	if test_state.y_translation < 0:
		return fail_val

	# TODO: make this based on the width of the first character?
	c1_width = char_widths[tri1[0]]
	c2_width = char_widths[tri1[1]]
	char_width_est = c1_width + c2_width
	est_offset = c1_width / char_width_est
	est_offset *= 0.6 # Allow some margin of error in estimated widths

	# TODO: This offset is to avoid errors from tri2="in" and tri1="hi", where the lowest error would be "i" overlapping with "h"
	# TODO: trigram if test_state.x_translation < global_orig_img_width * 0.15:
	if test_state.x_translation < global_orig_img_width * est_offset:
		return fail_val
	# if test_state.x_translation < 0:
	# 	return fail_val

	# Check that the action doesn't move the image outside of the lower right corner
	if test_state.y_translation + test_state.img_height >= 3*global_orig_img_height:
		return fail_val
	if test_state.x_translation + test_state.img_width >= 2*global_orig_img_width:
		return fail_val

	# Should be all good
	return [True, test_state]

# TODO: def is_action_legal(state, action):
# TODO: def perform_action(state, action):

def compute_loss(state, img1_array, img2):
	img1_ = img1_array[
			state.y_translation:state.y_translation+state.img_height,
			state.x_translation:state.x_translation+state.img_width]
	img1_ = np.copy(img1_)
	img2_ = np.asarray(img2.resize((state.img_width, state.img_height), Image.LANCZOS)).astype('float32')

	diff_img = img2_ - img1_
	loss = np.sum(np.square(diff_img))
	loss = loss / (img2_.shape[0] * img2_.shape[1])

	return loss # It's the mean squared error

def visualise_state(state, img1_array, img2):
	img2 = np.asarray(img2.resize((state.img_width, state.img_height), Image.LANCZOS)).astype('float32')

	img1_ = img1_array[
			state.y_translation:state.y_translation+state.img_height,
			state.x_translation:state.x_translation+state.img_width]
	img1_ = np.copy(img1_)

	img = np.minimum(img1_, img2)
	img1_array_ = np.copy(img1_array)

	img1_array_[state.y_translation:state.y_translation+state.img_height,
			state.x_translation:state.x_translation+state.img_width] = img

	return img1_array_

def match_trigram_pair(img1, img2, tri1, tri2):
	# Assume that images are shaped the same way, as this will otherwise influence later assumptions
	# (and I feel it's a fair assumption to make for fonts outputted by autoencoders)
	assert(img1.size == img2.size)
	(width, height) = img1.size
	# TODO: trigram tetragram = tri1 + tri2[2]
	tetragram = tri1 + tri2[1]


	# TODO: pad img1 with 2x its width and height in white space to allow for img2 to scale and translate

	img1_array = np.ones((height*3, width*2)).astype('float32') * 255.0
	img1_array[height:2*height, 0:width] = np.asarray(img1)

	state = State(height, width)
	prev_loss = compute_loss(state, img1_array, img2)

	state_history = []

	start_temp = 5.0 * 10**3 * 0.2
	temp_loss_rate = 0.00005 # 92102 iterations at starting = 500, loss_rate = 0.00005, min_temp = 5
	min_temp = 5
	dump_cnt = 10

	m = log(min_temp/start_temp) / log(1 - temp_loss_rate)
	dump_ks = [int(m*dump_k/float(dump_cnt)) for dump_k in range(dump_cnt)]

	min_loss = float('Inf')
	min_state = None

	k = 0
	temp = start_temp
	while temp > min_temp:
		action = get_random_action()

		action_legality, new_state = test_action_legality(state, action, tri1, tri2)
		if not (action_legality == True):
			continue

		k += 1
		temp = temp * (1 - temp_loss_rate)

		loss = compute_loss(new_state, img1_array, img2)

		# Simulated annealing
		delta = loss - prev_loss
		if delta < 0:
			state = new_state
			prev_loss = loss
		else:
			p = exp(-delta / temp)
			if uniform(0,1) < p:
				state = new_state
				prev_loss = loss

		# if k % 1000 == 0:
		if k in dump_ks:
			print('Iteration # {0}, temp: {1:.3f}. prev_loss: {2}'.format(k, temp, prev_loss))
			print('p = exp(-delta / temp): {0}'.format(exp(-delta / temp)))
			historic_state = state.to_obj()
			historic_state['loss'] = prev_loss
			historic_state['iteration'] = k
			state_history.append(historic_state)

		if loss < min_loss:
			min_loss = loss
			min_state = new_state


		assert(action_legality)

	historic_state = state.to_obj()
	historic_state['loss'] = prev_loss
	historic_state['iteration'] = k
	state_history.append(historic_state)

	if min_loss < prev_loss:
		print('Simulated annealing kind of failed. The minimum found in the search had an loss of {0}'.format(min_loss))
		historic_state = min_state.to_obj()
		historic_state['loss'] = min_loss
		historic_state['iteration'] = k+1
		state_history.append(historic_state)

	return state_history


def match_bigrams(sentence, img_dir, font_name):
	finished_trigrams = []

	for n in range(len(sentence)-2):
		bi1 = sentence[n:n+2]
		bi2 = sentence[n+1:n+3]
		trigram = bi1 + bi2[1]

		print('trigram: {0}, n: {1}'.format(trigram, n))
		if trigram in finished_trigrams:
			continue

		# Load images as arrays of float32
		img1 = Image.open(img_dir + bi1 + '.jpg').convert('L')
		img2 = Image.open(img_dir + bi2 + '.jpg').convert('L')

		state_history = match_trigram_pair(img1, img2, bi1, bi2)

		dump_dir = '../data/state-histories/{0}/{1}/'.format(sentence, font_name)
		if not exists(dump_dir):
			makedirs(dump_dir)
		history_path = dump_dir + '{0}.json'.format(trigram)
		with open(history_path, 'w') as f:
			json.dump(state_history, f)
			print('Saved history in ' + history_path)

		finished_trigrams.append(trigram)

if __name__ == "__main__":
	font_idxs = range(1, 21)
	sentence = 'MACHINE-LEARNING'

	for font_idx in font_idxs:
		font_name = 'font-{0}'.format(str(font_idx).zfill(3))
		img_dir = '../../visualisations/sentence-bigrams/{0}/{1}/'.format(sentence, font_name)

		match_bigrams(sentence, img_dir, font_name)