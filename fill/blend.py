import argparse
import ctypes
import numpy as np
import os
from PIL import Image

# Load the CUDA shared library
blend_lib = ctypes.CDLL('./blend.so')

# Define the function signature
blend_lib.blendImageCUDA.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # Input image pointer
    ctypes.POINTER(ctypes.c_ubyte),  # Output image pointer
    ctypes.c_int,                   # Image width
    ctypes.c_int                    # Image height
]


def main(dir_in, dir_out):
	os.makedirs(dir_out, exist_ok=True)
	filenames = os.listdir(dir_in)

	for filename in filenames:
		filepath = f'{dir_in}/{filename}'

		# Load the image
		im = Image.open(filepath).convert('RGB')
		im_array = np.array(im, dtype=np.uint8)
		h, w, c = im_array.shape

		# Prepare input and output buffers
		in_im = im_array.ravel()
		out_im = np.zeros_like(in_im, dtype=np.uint8)

		# Create pointers to the input and output arrays
		in_ptr = in_im.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
		out_ptr = out_im.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

		# Call the CUDA function
		blend_lib.blendImageCUDA(in_ptr, out_ptr, w, h)

		# Reshape the output and save the result
		out_im = out_im.reshape((h, w, c))
		out_im = Image.fromarray(out_im, mode='RGB')
		out_im.save(f'{dir_out}/{filename}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'dir_in',
		type=str,
		help='path to source images folder')
	parser.add_argument(
		'--dir_out',
		type=str,
		default='out',
		help='output folder')
	args = parser.parse_args()
	main(args.dir_in, args.dir_out)










