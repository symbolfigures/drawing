'''
Creates copies of scans with a grid superimposed to show how they will be cut into tiles.
The grid is offset by (x,y) values in adjustment.json.
It needs to be centered over the drawing area.
Edit adjustment.json directly to get the fit right for each scan.
Use the --index flag to work a particular scan, otherwise they are all worked.
'''
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from PIL import Image, ImageDraw


def load_adjustment(dir_in):
	try:
		with open('adjustment.json', 'r') as f:
			data = json.load(f)
	except FileNotFoundError:
		data = []
		for i in range(len(os.listdir(dir_in))):
			entry = {
				'index': i,
				'x': 0.0,
				'y': 0.0
			}
			data.append(entry)		
		with open('adjustment.json', 'w') as f:
			json.dump(data, f, indent=4)
	return data


def worker(args):
	dir_in, dpi, dir_out, i, rows, cols, adj = args
	filename = f'{i}.png'
	unit = (dpi // 300) * 256
	margin_x = adj[i]['x'] * unit
	margin_y = adj[i]['y'] * unit
	im = Image.open(f'{dir_in}/{filename}').convert('RGB')
	draw = ImageDraw.Draw(im)
	w, h = im.size
	for row in range(rows + 1):
		y = margin_y + row * unit
		draw.line([(0, y), (w, y)], fill=(255,0,0), width=3)
	for col in range(cols + 1):
		x = margin_x + col * unit
		draw.line([(x, 0), (x, h)], fill=(255,0,0), width=3)
	im.save(f'{dir_out}/{filename}')


def main(dir_in, dpi, dir_out, index, rows, cols):
	os.makedirs('grid', exist_ok=True)
	adj = load_adjustment(dir_in)
	if index:
		args = [(dir_in, dpi, dir_out, index, rows, cols, adj)]
	else:
		args = [(dir_in, dpi, dir_out, i, rows, cols, adj) for i in range(len(adj))]
	with ProcessPoolExecutor() as executor:
		executor.map(worker, args)
	#worker(args[0]) # debug


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='folder of scans e.g. scan/field_dpi300_rgba')
	parser.add_argument(
		'dpi',
		type=int,
		help='dpi of scans as determined by the scanner')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='grid',
		help='output folder')
	parser.add_argument(
		'--index',
		type=int,
		default=None,
		help='specify a specific index to work')
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
	args = parser.parse_args()
	main(args.dir_in, args.dpi, args.dir_out, args.index, args.rows, args.cols)




