# to validate the .tfrecord files
# should produce the same .png files that the .tfrecord files were made from
# however the file names and paths are not preserved
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


feature_description = { # must match that which created the tf records
	'image_bytes': tf.io.FixedLenFeature([], tf.string),
	'image_shape': tf.io.FixedLenFeature([3], tf.int64), # 3: height, width, channels
}


# parse record data
def parse_function(proto): # proto: serialized protobuf string
	return tf.io.parse_single_example(proto, feature_description)


# decode image from parsed record
def decode_image(parsed_record):
	image = tf.io.decode_png(
		parsed_record['image_bytes']
	)
	return image


def worker(args):
	dir_in, dir_out, i, filename = args
	filepath = f'{dir_in}/{filename}'
	os.makedirs(f'{dir_out}/{i}', exist_ok=True)
	raw_dataset = tf.data.TFRecordDataset(filepath)

	# apply transformations per image datum
	parsed_dataset = raw_dataset.map(parse_function)
	decoded_dataset = parsed_dataset.map(decode_image)

	# access data
	for j, image in enumerate(decoded_dataset):
		filepath = f'{dir_out}/{i}/{j}'
		tf.io.write_file(filepath, tf.io.encode_png(image))


def main(dir_in, dir_out):
	filenames = os.listdir(dir_in)
	args = [(dir_in, dir_out, i, f) for i, f in enumerate(filenames)]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='folder of .tfrecord files')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='tfrecord_reverse',
		help='output folder')
	args = parser.parse_args()
	main(args.dir_in, args.dir_out)










