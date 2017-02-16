import numpy as np
from PIL import Image
import json
from os.path import exists
from os import makedirs

global_orig_img_width = 65
global_orig_img_height = 41

field_y_trans = 'y_translation'
field_x_trans = 'x_translation'
field_real_y_trans = 'real_y_translation'
field_real_x_trans = 'real_x_translation'
field_img_height = 'img_height'
field_img_width = 'img_width'
field_real_composite_shape = 'real_composite_shape'
field_float32 = 'float32'
field_x_scale = 'x_scale'
field_y_scale = 'y_scale'


def merge_trigram_in(composite_array, tri_img, tri_state_dict, prev_state_dict):
	print(tri_state_dict)
	print(prev_state_dict)

	y_scale = prev_state_dict[field_y_scale]
	x_scale = prev_state_dict[field_x_scale]

	tri_height = int(tri_state_dict[field_img_height] * y_scale)
	tri_width = int(tri_state_dict[field_img_width] * x_scale)
	# print(tri_img.size)
	tri_img = tri_img.resize((tri_width, tri_height), Image.LANCZOS)
	# print(tri_img.size)
	tri_img = np.asarray(tri_img).astype(field_float32)

	left_x = prev_state_dict[field_x_trans] + int(tri_state_dict[field_x_trans] * x_scale)
	top_y = prev_state_dict[field_y_trans] + int( (tri_state_dict[field_y_trans] - global_orig_img_height) * y_scale ) # Due to the offset in the matching of the trigrams

	tri_img = np.minimum(composite_array[top_y:top_y+tri_height , left_x:left_x+tri_width], tri_img)
	composite_array[top_y:top_y+tri_height , left_x:left_x+tri_width] = tri_img

	return composite_array


def merge_trigram_in2(composite_array, tri_img, tri_state_dict, prev_state_dict):
	real_y_trans = prev_state_dict[field_real_y_trans] + tri_state_dict[field_y_trans] - global_orig_img_height
	real_x_trans = prev_state_dict[field_real_x_trans] + tri_state_dict[field_x_trans]
	int_y_trans = int(real_y_trans)
	int_x_trans = int(real_x_trans)

	tri_height = tri_state_dict[field_img_height]
	tri_width = tri_state_dict[field_img_width]

	composite_array = np.copy(np.asarray(composite_array)) # Because otherwise it won't let me write to it
	tri_img = tri_img.resize((tri_width, tri_height), Image.LANCZOS)

	tri_x_scale = tri_state_dict[field_img_width] / float(global_orig_img_width)
	tri_y_scale = tri_state_dict[field_img_height] / float(global_orig_img_height)

	# new_composite_size
	new_real_y_trans = real_y_trans / tri_y_scale
	new_real_x_trans = real_x_trans / tri_x_scale

	new_real_composite_shape = (prev_state_dict[field_real_composite_shape][0] / tri_y_scale, prev_state_dict[field_real_composite_shape][1] / tri_x_scale)

	composite_overlay = composite_array[int_y_trans:int_y_trans+tri_height , int_x_trans:int_x_trans+tri_width]

	# print(tri_img.size)
	# print( (tri_height, tri_width) )
	# print( '(int_y_trans,int_y_trans+tri_height): {0}'.format((int_y_trans,int_y_trans+tri_height)) )
	# print( '(int_x_trans, int_x_trans+tri_width): {0}'.format((int_x_trans, int_x_trans+tri_width)) )
	# print(composite_array[int_y_trans:int_y_trans+tri_height , int_x_trans:int_x_trans+tri_width].shape)
	# print(composite_array.shape)
	# If this fails, it's possibly due to setting the size of the composite_img too small
	assert((tri_img.size[0] == composite_overlay.shape[1]) and tri_img.size[1] == composite_overlay.shape[0])

	tri_img = np.minimum(composite_overlay, tri_img)
	composite_array[int_y_trans:int_y_trans+tri_height , int_x_trans:int_x_trans+tri_width] = tri_img


	composite_array = Image.fromarray(composite_array).resize( (int(new_real_composite_shape[1]), int(new_real_composite_shape[0])) , Image.LANCZOS)
	composite_array = np.asarray(composite_array).astype(field_float32)
	# plot.imshow(composite_array, cmap='gray')
	# plot.show()
	# plot.close()

	new_state_dict = {
		field_real_y_trans: new_real_y_trans,
		field_real_x_trans: new_real_x_trans,
		field_real_composite_shape: new_real_composite_shape
	}

	return composite_array, new_state_dict


def match_sentence_bigrams(img_dir, font_name, sentence):
	composite_array = np.ones((global_orig_img_height*3, global_orig_img_width*9)).astype(field_float32) * 255.0
	img1_y = global_orig_img_height
	img1_x = 0

	current_state_dict = {
		field_real_y_trans: float(img1_y),
		field_real_x_trans: float(img1_x),
		field_real_composite_shape: (float(composite_array.shape[0]), float(composite_array.shape[1]) )
	}

	for n in range(len(sentence)-2):
		bi1 = sentence[n:n+2]
		bi2 = sentence[n+1:n+3]
		trigram = bi1 + bi2[1]


		if n == 0:
			img1 = Image.open(img_dir + bi1 + '.jpg').convert('L')
			img1 = np.asarray(img1).astype(field_float32)
			composite_array[img1_y:img1_y+global_orig_img_height, img1_x:img1_x+global_orig_img_width] = img1

		img2 = Image.open(img_dir + bi2 + '.jpg').convert('L')

		print('trigram: {0}'.format(trigram))

		history_path = '../data/state-histories/{0}/{1}/{2}.json'.format(sentence, font_name,trigram.replace(' ', '-'))
		with open(history_path, 'r') as f:
			state_history = json.load(f)

		tri2_final_state = state_history[-1]

		composite_array, new_state_dict = merge_trigram_in2(composite_array, img2, tri2_final_state, current_state_dict)
		current_state_dict = new_state_dict

	# Get the image into the range [0,255]
	composite_array = composite_array - np.min(composite_array)
	median = np.percentile(composite_array, 50)
	composite_array = composite_array / median * 255
	composite_array = np.minimum(composite_array, 255)
	composite_array = composite_array.astype('uint8')

	# Crop image
	def crop_img(img_array):
		cutoff_val = 200
		top = 0
		left = 0
		bottom = img_array.shape[0]
		right = img_array.shape[1]

		for i in range(img_array.shape[0]):
			top = i
			row_min_val = np.min(img_array[i,:])
			if row_min_val < cutoff_val:
				break

		for i in range(img_array.shape[0]-1, 0, -1):
			bottom = i
			row_min_val = np.min(img_array[i,:])
			if row_min_val < cutoff_val:
				break

		for i in range(img_array.shape[1]):
			left = i
			col_min_val = np.min(img_array[:,i])
			if col_min_val < cutoff_val:
				break

		for i in range(img_array.shape[1]-1, 0, -1):
			right = i
			col_min_val = np.min(img_array[:,i])
			if col_min_val < cutoff_val:
				break


		img_array = img_array[top:bottom, left:right]
		return img_array

	composite_array = crop_img(composite_array)

	composite_img = Image.fromarray(composite_array, 'L')
	composite_img = composite_img.resize( (int(global_orig_img_width*7/2)+1, global_orig_img_height+1), Image.LANCZOS )

	if not exists('../../visualisations/sentences/'):
		makedirs('../../visualisations/sentences/')

	composite_img.save('../../visualisations/sentences/{0}-{1}.jpg'.format(sentence, font_name), quality=95)


if __name__ == "__main__":
	font_idxs = range(1, 21)
	sentence = 'MACHINE-LEARNING'

	for font_idx in font_idxs:
		font_name = 'font-{0}'.format(str(font_idx).zfill(3))
		img_dir = '../../visualisations/sentence-bigrams/{0}/{1}/'.format(sentence, font_name)
		match_sentence_bigrams(img_dir, font_name, sentence)