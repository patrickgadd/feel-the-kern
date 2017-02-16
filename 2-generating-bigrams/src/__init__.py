import json
import hashlib

def get_font_paths():
	config_dict = {}
	with open('../../shared-data/font_paths.json', 'r') as f:
		config_dict = json.load(f)

	return [config_dict['good_font_paths'], config_dict['good_font_names']]

def font_path_to_hex_hash(font_path):
	def baseN(num,b,numerals):
		return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

	original_hash = hashlib.sha1(font_path).hexdigest()

	numerals = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' # len = 62
	large_num_almost_prime = 15485867 * 32452843
	# large_num_almost_prime = 202357 * 202361
	hash_int = int(original_hash, 16) % large_num_almost_prime
	truncated_hash = baseN(hash_int, len(numerals), numerals)

	return truncated_hash

# def get_z_space_dict(dimensionality):
# 	# This returns a dict which maps from the img-hash to it's coordinates in the Z space
# 	z_space_dict = {}
#
# 	with open('../data/tsne_dump_{0}d.json'.format(dimensionality), 'r') as f:
# 		tsne_dump = json.load(f)
#
# 	Z = tsne_dump['Z']
# 	img_names = tsne_dump['img_names']
# 	img_hashes = [name[:-4] for name in img_names]
#
#
# 	for i, img_hash in enumerate(img_hashes):
# 		z_space_dict[img_hash] = Z[i]
#
# 	return z_space_dict


