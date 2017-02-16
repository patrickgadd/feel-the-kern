from os import listdir
from os.path import isfile, join

import json
import math
import numpy as np
from tsne_python.tsne import tsne

import matplotlib.pyplot as plot
from PIL import Image


def get_filenames(dir_path):
	return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

def read_img_matrix(img_folder, img_res, img_count_limit):
	img_names = get_filenames(img_folder)
	assert(len(img_names) > 0)

	img_shape = (img_res, img_res)

	if img_count_limit > 0:
		img_matrix = np.zeros([img_count_limit, img_res*img_res])
	else:
		img_matrix = np.zeros([len(img_names), img_res*img_res])



	for i, img_name in enumerate(img_names): # Do this for all files
		if i == img_count_limit and not (img_count_limit == 0):
			break

		img = Image.open(img_folder + img_name).convert('L').resize((img_res, img_res), Image.LANCZOS)

		# img = equalize(img) # TODO: bad results seemingly arise from this
		img = np.asarray(img, dtype="float32")
		img = img.flatten()
		img = img / np.max(img)
		img_matrix[i,:] = img

		if i % 100 == 0:
			print('{0} of {1} images loaded'.format(i, len(img_names)))

	return [img_matrix, img_names]

def plot_tsne_grid(img_matrix, resized_res, Z, grid_plot_count, data_dir):
	assert(Z.shape[1] == 2)

	# grid_plot_count: number of images to plot along each axis

	d1 = Z[:,0]
	d2 = Z[:,1]
	d1_min = np.min(d1)
	d1_max = np.max(d1)
	print('d1_min: {0}, d1_max: {1}'.format(d1_min, d1_max))
	# d1_min = -40
	# d1_max = 40
	d1_range = d1_max - d1_min
	d2_min = np.min(d2)
	d2_max = np.max(d2)
	print('d2_min: {0}, d2_max: {1}'.format(d2_min, d2_max))
	# d2_min = -40
	# d2_max = 40
	d2_range = d2_max - d2_min

	d1_cell_width = d1_range / float(grid_plot_count)
	d1_cell_centers = [d1_min + (i+0.5)*d1_cell_width for i in range(grid_plot_count)]

	d2_cell_width = d2_range / float(grid_plot_count)
	d2_cell_centers = [d2_min + (i+0.5)*d2_cell_width for i in range(grid_plot_count)]

	grid_img = np.zeros((resized_res*grid_plot_count, resized_res*grid_plot_count))

	print(grid_img.shape)

	for i, grid_x in enumerate(d1_cell_centers):
		if i % 10 == 0:
			print('Plotting image. At row {0} of {1}'.format(i, len(d1_cell_centers)))

		for j, grid_y in enumerate(d2_cell_centers):
			# TODO:
			# Iterate over all grid-centroids
			# For each centroid find the NN in the Y-space (nn_idx)
			# The plot the corresponding image from img_matrix

			nn_idx = -1
			nn_dist = float('inf')
			# TODO: find nearest neighbour in Y
			for n in range(len(d1)):
				dist = math.pow(d1[n] - grid_x,2) + math.pow(d2[n] - grid_y,2)
				if dist < nn_dist:
					nn_dist = dist
					nn_idx = n
					# print('New nearest neighbour.\nnn_idx: {0}\nnn_dist: {1}'.format(nn_idx, nn_dist))

			img = img_matrix[nn_idx,:]
			img = np.multiply(img, 255.0)
			img = img.reshape((resized_res, resized_res))
			grid_img[j*resized_res:(1+j)*resized_res, i*resized_res:(1+i)*resized_res] = img

	print(grid_img.shape)
	print(np.max(grid_img))
	print(np.min(grid_img))
	grid_img = grid_img.astype('uint8')
	grid_img = Image.fromarray(grid_img)

	if grid_img.mode != 'RGB':
		grid_img = grid_img.convert('RGB')

	grid_img.save(data_dir + 't-sne-grid.jpg')

	plot.scatter(Z[:,0], Z[:,1], 20)
	plot.savefig(data_dir + 't-sne.png')
	plot.close()


def dump_data(data_dump_path, Z, img_res, img_count_limit, img_names):
	data_dict = {
		'Z': Z.tolist(),
		'img_res': img_res,
		'img_count_limit': img_count_limit,
		'img_names': img_names
	}

	with open(data_dump_path, 'w') as data_file:
		json.dump(data_dict, data_file)
		print('Dumped the data')

def load_data(data_dump_path):
	with open(data_dump_path, 'r') as data_file:
		data_dict = json.load(data_file)
	return data_dict

def main():
	data_dir = '../../../shared-data/'
	img_dir = '../../data/text-imgs/'

	dimensionality = 10
	img_res = 48
	data_dump_path = data_dir + 'tsne_dump_{0}d_{1}px.json'.format(dimensionality, img_res)

	grid_plot_count = 30 # No. of images per axis in plot of the images in the 2-D space
	img_count_limit = 0

	use_stored_data = False
	img_matrix, img_names = read_img_matrix(img_dir, img_res, img_count_limit)

	if use_stored_data == True:
		data_dict = load_data(data_dump_path)
		Z = np.asarray(data_dict['Z'])
	else:
		max_iter = 500
		num_pcs = 300
		perplexity = 20.0 # Originally 20.0
		Z = tsne(img_matrix, dimensionality, num_pcs, perplexity, max_iter)
		dump_data(data_dump_path, Z, img_res, img_count_limit, img_names)

	if dimensionality == 2:
		plot_tsne_grid(img_matrix, img_res, Z, grid_plot_count, data_dir)


if __name__ == "__main__":
	main()