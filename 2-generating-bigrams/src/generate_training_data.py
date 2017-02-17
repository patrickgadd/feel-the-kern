from auxiliary import get_font_infos, get_z_space_dict

from random import uniform, sample
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import textwrap

font_infos = get_font_infos()

def crop_img(img, border=10):
	img = np.asarray(img)
	threshold = 220

	# crop from left and top first
	img_res = img.shape
	left_border = 0
	top_border = 0
	for x in range(0, img_res[1]):
		col_min_val = np.min(img[:,x])
		if col_min_val < threshold:
			left_border = max(0, x-border)
			break

	for y in range(0, img_res[0]):
		col_min_val = np.min(img[y, :])
		if col_min_val < threshold:
			top_border = max(0,y-border)
			break

	img = img[:,left_border:img_res[1]]
	img = img[top_border:img_res[0], :]

	# crop from right and bottom afterwards
	img_res = img.shape
	right_border = img_res[1]
	bottom_border = img_res[0]
	xs = range(0,img_res[1])
	for x in xs[::-1]:
		col_min_val = np.min(img[:,x])
		if col_min_val < threshold:
			right_border = min(x+border, img_res[1])
			break

	ys = range(0,img_res[0])
	for y in ys[::-1]:
		col_min_val = np.min(img[y,:])
		if col_min_val < threshold:
			bottom_border = min(y+border, img_res[0])
			break

	img = img[:, 0:right_border]
	img = img[0:bottom_border, :]
	img = Image.fromarray(img)

	return img

def render_text(text, output_shape, font_path):
	scale = 1 # Increase for higher quality (however, run-time is proportional with O(scale^2))
	font_size = 65 * scale
	font = ImageFont.truetype(font_path, font_size)

	h = 400 * scale
	w = 400 * scale
	orig_size = (w, h)
	background_color = 255
	font_color = 0

	image = Image.new("L", orig_size, background_color)
	draw = ImageDraw.Draw(image)

	lines = textwrap.wrap(text, width=8)#width=13)

	y_offset = 0 * scale
	for line in lines:
		width, height = font.getsize(line)
		draw.text(((w - width) / 2, y_offset), line, font=font, fill=font_color) # Centered
		y_offset += height


	image = ImageOps.equalize(image)
	image = crop_img(image, border=4)
	image = image.resize(output_shape, Image.LANCZOS)

	return image

def generate_data(data_cnt, n_gram_n, alphabet, z_dim, output_shape=(68,40), mode='return'):
	zs = [] # will be merged with one_hots into X
	one_hots = [] # will be merged with zs into X
	Y = []
	z_dict = get_z_space_dict(z_dim)

	font_ids = z_dict.keys()

	batch_font_ids = []
	if len(font_ids) < data_cnt:
		# very inefficient approach, but ...
		while len(batch_font_ids) < data_cnt:
			batch_font_ids.append(sample(font_ids,1)[0])
	else:
		batch_font_ids = sample(font_ids, data_cnt)

	counter = 0
	for font_id in batch_font_ids:
		# Again, not efficient by any means, but in the scope of things, it's fine
		font_info = [info for info in font_infos if info['id'] == font_id]
		assert(len(font_info) == 1)
		font_info = font_info[0]

		z = z_dict[font_id]

		counter += 1

		# Generate the bigram text
		text = ''
		char_idxs = []
		for i in range(n_gram_n):
			char_idx = int(uniform(0,len(alphabet)))
			char_idxs.append(char_idx)
			text += alphabet[char_idx]

		font_path = font_info['font_path']
		font_path = '../../shared-data/fonts/' + font_path.split('/')[-1]

		image = render_text(text, output_shape, font_path)

		if mode == 'dump-images':
			img_path = '../data/text-imgs/{0}_{1}.png'.format(font_id,text)
			# img_path = '../data/text-imgs/{0}_{1}.png'.format(font_id,text)
			image.save(img_path, "PNG")
		elif mode == 'npz' or mode == 'return':
			image = np.asarray(image).astype('float32') / 255.0
			Y.append(image)
			zs.append(z)


			one_hot = []
			# TODO: char_idxs to one-hot encodings
			for char_idx in char_idxs:
				one_hot.extend( [1 if i == char_idx else 0 for i in range(len(alphabet))] )

			one_hots.append(one_hot)

	if mode == 'npz' or mode == 'return':
		zs = zs - np.min(zs,axis=0) # zs now in the range [0;?]
		zs = zs / np.percentile(zs, 95, axis=0) # zs now almost in the range [0;1] (deliberately didn't divide by np.max due to potential outliers)

		X = []
		for i, z in enumerate(zs):
			one_hot = one_hots[i]
			x = one_hot
			x.extend(z)
			X.append(x)
		X = np.asarray(X).astype('float32')

		if mode == 'npz':
			dump_path = '../data/training-data_cnt{0}_{1}z_dim.npz'.format(counter, z_dim)
			print('dumping files..')
			np.savez_compressed(dump_path, X, Y)


	if mode =='return':
		return [np.asarray(X).astype('float32'), np.asarray(Y).astype('float32')]

def get_batch(data_cnt, n_gram_n, alphabet, z_dim, output_shape):
	(X, Y) = generate_data(data_cnt, n_gram_n, alphabet, z_dim, output_shape, mode='return')
	X = np.reshape(X, (data_cnt, 1, X.shape[1]))
	Y = np.reshape(Y, (data_cnt, 1, Y.shape[1], Y.shape[2]))

	return (X, Y)


if __name__ == "__main__":
	(X, Y) = generate_data(3, 2, 'ASDF', 10, (65, 41))
	print(X)


