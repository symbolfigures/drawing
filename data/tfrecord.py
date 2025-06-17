import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from PIL import Image
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def sample(image_string, image_shape):
	feature = {
		'image_bytes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
		'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))


def worker(args):
	dir_in, group_i, group, dir_out = args

	writer = None

	for im_path in group:
		with open(im_path, 'rb') as f:
			im = Image.open(f)
			mode = im.mode
			im = np.array(im)
			# for pixels with 3-4 channels (RGB or RGBA) the array has shape (3-4, X)
			# for pixels with 1 channel (L or 1) the array has shape (X) and must be expanded to (1, X)
			if mode == 'L' or mode == '1':
				im = np.expand_dims(im, axis=-1)
			im_shape = im.shape
			im_tensor = tf.convert_to_tensor(im, dtype=tf.uint8)
			im_string = tf.io.encode_png(im_tensor).numpy()

		im_size = len(im_string)

		if not writer:
			shard_path = os.path.join(dir_out, f'{group_i}.tfrecord')
			writer = tf.io.TFRecordWriter(shard_path)

		tf_example = sample(im_string, im_shape)
		writer.write(tf_example.SerializeToString())

	if writer:
		writer.close()


def gs_estimate(filepaths, T):
	# adjust group size so .tfrecord files are between 100 and 500 MB
	# target size of .tfrecord file is 300 MB
	# adjust for very small datasets so there is more than one file
	C = min(len(filepaths), 1000) # number of .png files to average
	P = 0 # total size of .png files
	for fp in filepaths[:1000]:
		P += os.path.getsize(fp)
	# T = G * P / C # .tfrecord size = group size * average .png size
	G = T * C / P # group size
	return int(G)


def main(dir_in, dir_out, T):
	os.makedirs(dir_out, exist_ok=True)
	filepaths = []
	# dir_in must follow folder tree structure created by tile.py
	for subdir in os.listdir(dir_in):
		for filename in os.listdir(f'{dir_in}/{subdir}'):
			filepaths.append(f'{dir_in}/{subdir}/{filename}')
	group_size = gs_estimate(filepaths, T)
	groups = [filepaths[i:i + group_size] for i in range(0, len(filepaths), group_size)]
	args = [(dir_in, x_i, x, dir_out) for x_i, x in enumerate(groups)]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='folder of source images e.g. tile')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='tfrecord',
		help='output folder')
	parser.add_argument(
		'--size',
		type=int,
		default=300000000,
		help='output .tfrecord size in bytes')
	args = parser.parse_args()
	main(args.dir_in, args.dir_out, args.size)













































