from auxiliary import get_font_infos, font_path_to_hex_hash

from os.path import exists
from os import makedirs

import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import textwrap

def crop_img(img, border=10):
	img = np.asarray(img)
	threshold = 220

	# TODO: crop from left and top first
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

	# TODO: crop from right and bottom afterwards
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
		# print(y)
		col_min_val = np.min(img[y,:])
		if col_min_val < threshold:
			bottom_border = min(y+border, img_res[0])
			break
	# print(bottom_border)
	img = img[:, 0:right_border]
	img = img[0:bottom_border, :]

	img = Image.fromarray(img)

	return img

def render_text(text, output_shape, font_path):
	scale = 3
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

if __name__ == "__main__":
	font_infos = get_font_infos()

	text = 'Aon'
	output_shape = (48, 48)

	if not exists('../data/text-imgs/'):
		makedirs('../data/text-imgs/')
		print('Created the required folders.')

	hash_dict = {}
	font_count = len(font_infos)
	for i, font_info in enumerate(font_infos):
		if i % 50 == 0:
			print('{0} of {1} fonts processed.'.format(i, font_count))

		font_path = font_info['font_path']

		if not exists(font_path):
			print('Warning, no font found at {0}. Skipping this...'.format(font_path))
			continue

		font_id = font_info['id']

		image = render_text(text, output_shape, font_path)

		img_path = '../data/text-imgs/{0}.png'.format(font_id)
		image.save(img_path, "PNG")