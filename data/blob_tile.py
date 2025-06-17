# makes .png images (cuts tiles) of a standard size for training
# for each scan, tiles are evenly spaced on a grid
# each tile is rotated at a random angle and perchance flipped
# hence only works for source material that need not preserve asymmetric features
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from PIL import Image
import random
import time

Image.MAX_IMAGE_PIXELS = 277813800


def worker(args):
	i, adj, dir_in, dir_out, dpi, res, rows, cols, steps = args

	# tiles are cut within the grid set by grid.py
	# each square in the grid is 1 unit
	# 1 unit = 256 / 300 square inches.
	# - if dpi=300, then each unit is 256 square pixels and 1 square inch
	# - if dpi=600, then each unit is 512 square pixels and 1 square inch
	# - if dpi=1200, then each unit is 1024 square pixels and 1 square inch
	unit = int((dpi / 300) * 256)
	grid = (cols * unit, rows * unit)

	# consider each cell of the grid is a tile to be cut
	# but the move from one cell to the next is actually gradual
	# the move is 1/{steps} units
	# - if steps=1, then the tile is cropped and takes 1 step to the adjacent cell
	# - if steps=2, then the tile takes 2 steps, so it is cropped halfway between the two cells
	# - if steps=16, then the tile takes 16 steps and the adjacent tiles mostly overlap
	# 1 step is measured in pixels.
	step = unit // steps

	# adjustment to the grid placement.
	adj_x = adj[i]['x'] * unit
	adj_y = adj[i]['y'] * unit

	# each tile is rotated randomly around its center before the crop
	# the grid border needs an additional margin for the diagonally-oriented tiles near the border
	# so that they still only capture the intended area of the page
	# the margin is the radius of the circle it rotates within
	# any coordinates within this padded border can define the center of a tile
	pad = math.sqrt(res**2 + res**2) / 2

	# boundary of crop coordinates.
	box = [
		int(adj_x + pad), # left
		int(adj_y + pad), # top
		int(adj_x + grid[0] - pad), # right
		int(adj_y + grid[1] - pad) # bottom
	]

	os.makedirs(f'{dir_out}/{i}', exist_ok=True)
	img = Image.open(f'{dir_in}/{i}.png')

	# iterate through every row and column.
	count = 0
	for y in range(box[1], box[3], step):
		for x in range(box[0], box[2], step):
			# crop an area large enough for the tile to rotate within
			left = x - pad
			right = x + pad
			top = y - pad
			bottom = y + pad
			scope = img.crop((left, top, right, bottom))

			# rotate and flip randomly
			theta = random.randrange(360)
			scope = scope.rotate(theta)
			if random.randrange(1) == 1:
				scope = scope.transpose.TRANSPOSE

			# cut the tile
			tx = scope.width / 2
			ty = scope.height / 2
			left = tx - (res / 2)
			right = tx + (res / 2)
			top = ty - (res / 2)
			bottom = ty + (res / 2)
			tile = scope.crop((left, top, right, bottom))

			tile.save(f'{dir_out}/{i}/{count}.png')
			count += 1


def main(args):
	with open('adjustment.json', 'r') as json_file:
		adj = json.load(json_file)
	args = [(
		i,
		adj,
		args.dir_in,
		args.dir_out,
		args.dpi,
		args.resolution,
		args.rows,
		args.cols,
		args.steps
	) for i in range(len(adj))]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)
	#worker(args[0]) # debug


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'dir_in',
		type=str,
		help='folder of source images e.g. scan/thirdstudy_300_rgb')
	parser.add_argument(
		'dpi',
		type=int,
		help='dpi of scans as determined by the scanner')
	parser.add_argument(
		'resolution',
		type=int,
		choices=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
		help='how many square pixels each tile will have. power of 2 between 4 and 1024')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='tile',
		help='output folder')
	parser.add_argument(
		'--rows',
		type=int,
		default=12,
		help='rows in the grid')
	parser.add_argument(
		'--cols',
		type=int,
		default=18,
		help='columns in the grid')
	parser.add_argument(
		'--steps',
		type=int,
		default=2,
		help='positive integer. inverse of the fraction that tiles overlap.' +
		'more steps -> more overlap. value of 1 means they don\'t overlap.')

	args = parser.parse_args()
	main(args)








