'''
input a folder of subfolders
each subfolder has all the tiles for one page
tiles are rotated and flipped to multiply tile count by 8
'''
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from PIL import Image


def rotateflip(args):
	dir_in, page = args
	dir_in = f'{dir_in}/{page}'
	# postfix existing images with _rf00
	images = os.listdir(dir_in)
	for image in images:
		image2 = image.split('.')[0] + '_rf00.' + image.split('.')[-1]
		os.rename(f'{dir_in}/{image}', f'{dir_in}/{image2}')
	# rotate and flip
	for tile in [f'{i:02}' for i in range(len(os.listdir(dir_in)))]:
		source = f'{dir_in}/{page}_t{tile:02}_rf00.png'
		image = Image.open(source)
		for f in range(2):
			for r in range(4):
				image = image.rotate(90)
				target = f'{dir_in}/{page}_t{tile:02}_rf{f}{r}.png'
				image.save(target)
			image = image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)


def main(dir_in):
	pages = os.listdir(dir_in)
	args = [(dir_in, page) for page in pages]
	with ProcessPoolExecutor() as executor:
		executor.map(rotateflip, args)
	#rotateflip(args[0]) # debug


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		default='tile',
		help='input folder')
	args = parser.parse_args()
	main(args.dir_in)















