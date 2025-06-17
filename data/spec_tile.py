'''
for a set of scanned drawings such that
 drawings are dark on light background
 each page has multiple drawings roughly arranged in a grid
 the grid must be precise enough that a vertical line may be drawn
  to separate any pair of adjacent columns
  likewise for rows
false-flag checks avoid mistaking tiny specs as drawings
 thresholds scale according to resolution and may require fine tuning
'''
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from PIL import Image, ImageDraw

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Image.MAX_IMAGE_PIXELS = 277813800


def rgb_avg(img, x, y):
	pix = img.getpixel((x, y))
	return (pix[0] + pix[1] + pix[2]) // 3


def crop_page(img, margin=40):
	# crop out the scanner bed surrounding the paper
	w, h = img.size
	left = 0
	right = 0
	top = 0
	bottom = 0

	y = h // 2
	x = 0
	while(rgb_avg(img, x, y) < 128):
	    x += 1
	left = x

	x = w - 1
	while(rgb_avg(img, x, y) < 128):
	    x -= 1
	right = x

	x = w // 2
	y = 0
	while(rgb_avg(img, x, y) < 128):
	    y += 1
	top = y

	y = h - 1
	while(rgb_avg(img, x, y) < 128):
	    y -= 1
	bottom = y

	bounding_box = (
		left + margin,
		top + margin,
		right - margin,
		bottom - margin
	)
	img = img.crop(bounding_box)
	return img


def false_flag(img, a, side, b1, b2, thresh=10):
	dir = 1
	if side == 'right' or side == 'bottom':
		dir = -1
	a_td = a + thresh * dir
	if side == 'left' or side == 'right':
		if a_td <= 0 or a_td >= img.size[0]:
			return True
		x1 = a
		x2 = x1
		px_white = True
		for y in range(b1, b2):
			px = rgb_avg(img, a_td, y)#img.getpixel((a_td, y))
			# if any pixels are dark, it's not a false flag
			if px < 128:
				px_white = False
				break
		if not px_white:
			return False
		return True
	if side == 'top' or side == 'bottom':
		if a_td <= 0 or a_td >= img.size[1]:
			return True
		y1 = a
		y2 = y1
		px_white = True
		for x in range(b1, b2):
			px = rgb_avg(img, x, a_td)#img.getpixel((x, a_td))
			# if any pixels are dark, it's not a false flag
			if px < 128:
				px_white = False
				break
		if not px_white:
			return False
		return True


def get_columns(img, w, h, s):
	columns = []
	x = 100 * s
	px_white = True
	while(x < w):
		# identify left side of black section
		while px_white and x < w:
			for y in range(h):
				px = rgb_avg(img, x, y)
				if (px < 128 and
					not false_flag(img, x, 'left', 0, h, 2 * s) and
					not false_flag(img, x, 'left', 0, h, 5 * s) and
					not false_flag(img, x, 'left', 0, h, 10 * s) and
					not false_flag(img, x, 'left', 0, h, 15 * s)):
					px_white = False
					break
			x += 1
		columns.append(x - 1)
		# cross black section from left to right
		x += 10 * s # prevents false flags
		while not px_white and x < w:
			px_white = True
			for y in range(h):
				px = rgb_avg(img, x, y)
				if px < 128:
					px_white = False
					break
			x += 1
	print('columns:', columns)
	return columns


def get_rows(img, w, h, s):
	rows = []
	y = 100 * s
	px_white = True
	while(y < h):
		# identify top side of black section
		while px_white and y < h:
			for x in range(w):
				px = rgb_avg(img, x, y)
				if (px < 128 and
					not false_flag(img, y, 'top', 0, w, 2 * s) and
					not false_flag(img, y, 'top', 0, w, 5 * s) and
					not false_flag(img, y, 'top', 0, w, 10 * s) and
					not false_flag(img, y, 'top', 0, w, 15 * s)):
					px_white = False
					break
			y += 1
		rows.append(y - 1)
		# cross black section from top to bottom
		y += 10 * s # prevents false flags
		while not px_white and y < h:
			px_white = True
			for x in range(w):
				px = rgb_avg(img, x, y)
				if px < 128:
					px_white = False
					break
			y += 1
	print('rows:', rows)
	return rows


def fit(img, l1, r1, t1, b1):
	#left
	x = l1
	px_white = True
	while px_white:
		x += 1
		for y in range(t1, b1):
			#print(f'(x, y): ({x}, {y})')
			px = rgb_avg(img, x, y)
			if px < 128 and not false_flag(img, x, 'left', t1, b1):
				px_white = False
				break
	l2 = x
	#right
	x = r1
	px_white = True
	while px_white:
		x -= 1
		for y in range(t1, b1):
			px = rgb_avg(img, x, y)
			if px < 128 and not false_flag(img, x, 'right', t1, b1):
				px_white = False
				break
	r2 = x
	#top
	y = t1
	px_white = True
	while px_white:
		y += 1
		for x in range(l1, r1):
			px = rgb_avg(img, x, y)
			if px < 128 and not false_flag(img, y, 'top', l1, r1):
				px_white = False
				break
	t2 = y
	#bottom
	y = b1
	px_white = True
	while px_white:
		y -= 1
		for x in range(l1, r1):
			px = rgb_avg(img, x, y)
			if px < 128 and not false_flag(img, y, 'bottom', l1, r1):
				px_white = False
				break
	b2 = y
	return l2, r2, t2, b2


def crop_box(l, r, t, b, res):
	w = r - l
	h = b - t
	x = l + w / 2
	y = t + h / 2
	l = x - res / 2
	r = x + res / 2
	t = y - res / 2
	b = y + res / 2
	return l, r, t, b


def worker(args):
	dir_in, res, dir_out, dir_out_grid, pre_res, file = args

	pageno = file.split('.')[0]
	dir_out_tile = f'{dir_out}/p{pageno}'
	os.makedirs(dir_out_tile, exist_ok=True)
	print(file)

	img = Image.open(f'{dir_in}/{file}')
	img = crop_page(img)
	w, h = img.size

	s = res // 256
	columns = get_columns(img, w, h, s)
	rows = get_rows(img, w, h, s)

	# draw grid
	copy = img.copy()
	draw = ImageDraw.Draw(copy)
	for x in columns:
		draw.line([(x, 0), (x, h)], fill='#ff0000', width=5)
	for y in rows:
		draw.line([(0, y), (w, y)], fill='#ff0000', width=5)
	copy.save(f'{dir_out_grid}/{file}')

	# identify and crop
	tileno = 0
	for col in range(len(columns) - 1):
		l1 = columns[col]
		r1 = columns[col + 1]
		for row in range(len(rows) - 1):
			t1 = rows[row]
			b1 = rows[row + 1]
			# fit box to exact with and height
			l2, r2, t2, b2 = fit(img, l1, r1, t1, b1)
			# skip blank spaces
			if l2 > r2 and t2 > b2:
				continue
			# find center and standard size box
			l3, r3, t3, b3 = crop_box(l2, r2, t2, b2, pre_res)
			crop = img.crop((l3, t3, r3, b3))
			crop = crop.resize((res, res))
			filename = f'p{pageno}_t{tileno:02}.png'
			crop.save(f'{dir_out_tile}/{filename}')
			tileno += 1


def main(dir_in, res, dir_out, dir_out_grid):

	pre_res = res + res // 2
	os.makedirs(dir_out_grid, exist_ok=True)
	files = os.listdir(dir_in)

	args = [(dir_in, res, dir_out, dir_out_grid, pre_res, f) for f in files]
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
		'resolution',
		type=int,
		choices=[256, 512, 1024],
		help='resolution of square tiles. 256px <=> dpi300')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='tile',
		help='output folder')
	parser.add_argument(
		'--dir_out_grid',
		type=str,
		default='grid',
		help='output folder for grid')
	args = parser.parse_args()
	main(args.dir_in, args.resolution, args.dir_out, args.dir_out_grid)








