import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import numpy as np
import os
from PIL import Image, ImageDraw
from random import randrange, shuffle
import sys
import time

sys.setrecursionlimit(10000000)
colors = []


def get_colors(palette):
	img = Image.open(palette)
	for x in range(img.width):	
		for y in range(img.height):
			colors.append(img.getpixel((x, y)))
	return colors


def bitmap(img):
	img = img.convert('L')
	array = np.array(img)
	# bitmap
	array = np.where(array > 208, 255, 0).astype(np.uint8)
	# white out pixels near edge
	margin = 32
	h, w = array.shape
	mask = np.zeros_like(array, dtype=bool)
	mask[:margin, :] = True # top
	mask[-margin:, :] = True # bottom
	mask[:, :margin] = True # left
	mask[:, -margin:] = True # right
	array[(array == 0) & mask] = 255
	return Image.fromarray(array.astype(np.uint8))


def get_max_radius1(img, origin):
	x = 0
	y = origin[1]
	while(img.getpixel((x, y)) == (255, 255, 255)):
		x += 1
	rad_x = origin[0] - x
	y = 0
	x = origin[0]
	while(img.getpixel((x, y)) == (255, 255, 255)):
		y += 1
	rad_y = origin[1] - y
	return max(rad_x, rad_y)


def get_max_radius(img, origin, margin=50):
	xn, xp, yn, yp = 0, 0, 0, 0
	(x, y) = origin
	p = 0
	while(p < origin[0] - margin):
		if img.getpixel((x-p, y)) == (0, 0, 0):
			xn = p
		p += 1
	p = 0
	while(p < origin[0]):
		if img.getpixel((x+p, y)) == (0, 0, 0):
			xp = p
		p += 1
	p = 0
	while(p < origin[1]):
		if img.getpixel((x, y-p)) == (0, 0, 0):
			yn = p
		p += 1
	p = 0
	while(p < origin[1]):
		if img.getpixel((x, y+p)) == (0, 0, 0):
			yp = p
		p += 1
	return max(xn, xp, yn, yp)


def gen_points(o, r, spacing=1):
	C4 = (2 * math.pi * r) / 4 # C4 = circumference divided by 4
	num_points = max(1, int(C4 / spacing)) # at least 1 point, num_points proportional to C4
	points = []
	for i in range(num_points):
		theta = (math.pi / 2) * (i / num_points)
		x = o[0] - r * math.cos(theta)
		y = o[1] - r * math.sin(theta)
		points.append((int(x), int(y)))
	return points


def fill_shape(point, color, img, draw):
	(x, y) = point
	pix = []
	pix_set = set()

	def gather_pix(x, y):
		if img.getpixel((x, y)) != (255, 255, 255):
			return
		pix.append((x, y))
		pix_set.add((x, y))
		if x > 0 and (x - 1, y) not in pix_set:
			gather_pix(x - 1, y)
		if x < img.width - 1 and (x + 1, y) not in pix_set:
			gather_pix(x + 1, y)
		if y > 0 and (x, y - 1) not in pix_set:
			gather_pix(x, y - 1)
		if y < img.height - 1 and (x, y + 1) not in pix_set:
			gather_pix(x, y + 1)

	gather_pix(x, y)
	if pix != []:
		for p in pix:
			draw.point(p, fill=color)

	return pix


def neighborhood(q, img): # is every pixel in the neighborhood white
	(x, y) = q
	neighbors = []
	neighbors.extend([
		(x-2, y),
		(x-1, y-1), (x-1, y), (x-1, y+1),
		(x, y-2), (x, y-1), (x, y), (x, y+1), (x, y+2),
		(x+1, y-1), (x+1, y), (x+1, y+1),
		(x+2, y)
	])
	for p in neighbors:
		if img.getpixel(p) != (255, 255, 255):
			return False
			break
	return True


def fill_if_symmetric(p, img, draw, origin):
	(xn, yn) = p
	xp = xn + 2 * (origin[0] - xn)
	yp = yn + 2 * (origin[1] - yn)
	quads = [
		(xn, yn),
		(xp, yn),
		(xn, yp),
		(xp, yp)
	]
	symmetric = True
	for q in quads:
		if not neighborhood(q, img):
			symmetric = False
			break
	if symmetric:
		color = colors[randrange(len(colors))]
		for q in quads:
			fill_shape(q, color, img, draw)


def fill_line(x, y, bb, img, draw):
	z = 0
	while True:
		z += 1
		box = [
			(x - z, y - z),
			(x, y - z),
			(x + z, y - z),
			(x - z, y),
			(x + z, y),
			(x - z, y + z),
			(x, y + z),
			(x + z, y + z)]
		shuffle(box)
		for (i, j) in box:
			if bb[0] <= i < bb[1] and bb[2] <= j < bb[3]:
				color = img.getpixel((i, j))
				if color != (0, 0, 0):
					draw.point((x, y), fill=color)
					return


def blend(x, y, bb, img):
	box = [
		(x-1, y-1), (x-1, y), (x-1, y+1),
		(x, y-1), (x, y), (x, y+1),
		(x+1, y-1), (x+1, y), (x+1, y+1)]
	r, g, b = 0, 0, 0
	for p in box:
		c = img.getpixel(p)
		r += c[0]
		g += c[1]
		b += c[2]
	color = (r//9, g//9, b//9)
	return color


def process(args):
	dir_in, dir_out, blend, filename = args

	img = Image.open(os.path.join(dir_in, filename))
	img = bitmap(img)
	img = img.convert('RGB')
	draw = ImageDraw.Draw(img)

	origin = (img.width // 2, img.height // 2) # set origin at center of page
	#max_radius = get_max_radius(img) + 10
	max_radius = get_max_radius(img, origin) + 10
	bb = (
		origin[0] - max_radius,
		origin[0] + max_radius,
		origin[1] - max_radius,
		origin[1] + max_radius)

	# mask environment (inside and outside of dilly)
	mask = []
	for p in [(0, 0), (origin[0], origin[1])]:
		mask.extend(fill_shape(p, (0, 255, 0), img, draw))

	# iterate over points on the dilly
	for radius in range(4, max_radius, 1):
		points = gen_points(origin, radius)
		for p in points:
			#draw.point(p, fill=(255, 0, 0)) # test
			if img.getpixel(p) == (255, 255, 255):
				fill_if_symmetric(p, img, draw, origin)

	# iterate over points that are still white
	for x in range(bb[0], bb[1], 1):
		for y in range(bb[2], bb[3], 1):
			if img.getpixel((x, y)) == (255, 255, 255):
				color = colors[randrange(len(colors))]
				fill_shape((x, y), color, img, draw)

	# unmask environment
	for p in mask:
		draw.point(p, fill=(255, 255, 255))

	# fill black lines
	for x in range(bb[0], bb[1], 1):
		for y in range(bb[2], bb[3], 1):
			if img.getpixel((x, y)) == (0, 0, 0):
				fill_line(x, y, bb, img, draw)

	# blend
	if blend:
		trace = img
		for x in range(bb[0], bb[1], 1):
			for y in range(bb[2], bb[3], 1):
				color = blend(x, y, bb, trace)
				draw.point((x, y), color)

	img.save(os.path.join(dir_out, filename))


def main(dir_in, palette, dir_out, blend):
	get_colors(palette)
	os.makedirs(dir_out, exist_ok=True)
	filenames = os.listdir(dir_in)
	args = [(dir_in, dir_out, blend, f) for f in filenames]
	with ProcessPoolExecutor() as executor:
		executor.map(process, args)
	#process(args[0]) # debug


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='path to source images folder')
	parser.add_argument(
		'palette',
		type=str,
		help='image showing the desired colors and their respective proportions')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='out',
		help='output folder')
	parser.add_argument(
		'--blend',
		action='store_true',
		default=False,
		help='blend with no GPU support')
	args = parser.parse_args()
	main(args.dir_in, args.palette, args.dir_out, args.blend)

























