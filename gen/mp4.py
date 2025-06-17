import argparse
import ffmpeg
import os
from PIL import Image


def png_to_mp4(dir_in):
	'''
	converts a file full of images to mp4
	'''
	dir_out = f'{dir_in}.mp4'
	(
		ffmpeg
			.input(f'{dir_in}/*.png', pattern_type='glob', framerate=30)
			.output(dir_out, vcodec='libx264', pix_fmt='yuv420p')
			.run()
	)


def png_to_mp4_rgba(dir_in, dir_in_bg, fps=30):
	'''
	if images are rgba
	convert to mp4 with specified background image or video
	'''
	sample = f'{dir_in}/{os.listdir(dir_in)[0]}'
	assert Image.open(sample).mode == 'RGBA', 'convert foreground to RGBA first!'
	dir_out = f'{dir_in}.mp4'
	background = (
		# make a video of a background color
		# duration t is a dummy overriden by overlay
		ffmpeg
		.input(dir_in_bg, t=1)
		.filter('fps', fps=fps) # framerate consistency
	)
	foreground = (
		# make a video from a sequence of .png images
		ffmpeg
		.input(f'{dir_in}/*.png', pattern_type='glob', framerate=fps)
		.filter('format', 'rgba')
	)
	overlay = (
		ffmpeg
		.overlay(background, foreground)
		.output(dir_out, vcodec='libx264', pix_fmt='yuv420p')
	)
	overlay.run()


def main(dir_in, dir_in_bg):
	if dir_in_bg is None:
		png_to_mp4(dir_in)
	else:
		png_to_mp4_rgba(dir_in, dir_in_bg)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='input folder')
	parser.add_argument(
		'--background',
		type=str,
		default=None,
		help='path to background image or video')
	args = parser.parse_args()
	main(args.dir_in, args.background)















