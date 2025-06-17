'''
rgb to rgba
input a folder full of rgb images
output a copy in rgba
 white --> clear
 black --> opaque black
or with option --color
 black --> opaque color
'''
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import os
from PIL import Image, ImageDraw, ImageColor


def worker(args):
	filename, dir_in, res, rgb, dir_out = args
	filepath = f'{dir_in}/{filename}'
	im = Image.open(filepath)
	im2 = Image.new('RGBA', (res, res))
	draw = ImageDraw.Draw(im2)
	for x in range(res):
		for y in range(res):
			(r, g, b) = im.getpixel((x, y))
			alpha = (r + g + b) // 3 # 3 channels to 1
			alpha = 255 - alpha # invert
			rgba = (rgb[0], rgb[1], rgb[2], alpha)
			draw.point((x, y), fill=rgba)
	im2.save(f'{dir_out}/{filename}')


def main(dir_in, color):
	dir_out = f'{dir_in}_rgba_{color}'
	os.makedirs(dir_out, exist_ok=True)

	rgb = ImageColor.getrgb(color)

	filenames = os.listdir(dir_in)
	im = Image.open(f'{dir_in}/{filenames[0]}')
	res = im.size[0]

	args = [(f, dir_in, res, rgb, dir_out) for f in filenames]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)
	#worker(args[0]) # debug


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='input folder')
	parser.add_argument(
		'--color',
		type=str,
		default='#000000',
		help='input folder')
	args = parser.parse_args()
	main(args.dir_in, args.color)







			

























