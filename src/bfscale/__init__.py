# Copyright at end of file.

from math import floor, sqrt
import numpy as np

def scale(img: np.ndarray, scale: int):
	if img.shape[0] % scale != 0 or img.shape[1] % scale != 0:
		raise ValueError(f"Scale ({scale}) is not an integer factor of image shape ({img.shape})")

	# Split image into chunks of size: scale by scale
	data = np.concatenate(np.array_split(img, img.shape[1]//scale, axis=1), axis=-1)
	data = np.concatenate(np.array_split(data, img.shape[0]//scale, axis=0), axis=-1)
	data = data.reshape(scale ** 2, -1)

	# Create indices for fitting
	indices = np.linspace(0, scale, scale+1)[0:-1]
	X = np.tile(indices, scale)
	Y = np.repeat(indices, scale)
	yx_indices = np.column_stack(((1-Y)*(1-X), (1-X)*Y, X*(1-Y), X*Y))

	# Fit the data
	parameters, _, _, _ = np.linalg.lstsq(yx_indices, data, rcond=None)

	# Reshape fit result parameters into image
	out_img = parameters.T
	out_img = out_img.reshape(img.shape[0]//scale, img.shape[1]//scale, img.shape[2], 4)
	out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values=-1)

	# Shift to compensate for edge pixels
	out_img[:, 1:, :, 1] = out_img[:, :-1, :, 1]
	out_img[1:, :, :, 2] = out_img[:-1, :, :, 2]
	out_img[1:, 1:, :, 3] = out_img[:-1, :-1, :, 3]
	out_img[:, 0, :, 1] = -1
	out_img[0, :, :, 2] = -1
	out_img[0, 0, :, 3] = -1

	# Combine result parameters of each chunk
	out_img = np.ma.median(np.ma.masked_values(out_img, -1), axis=-1)
	out_img = np.ma.getdata(out_img)

	# Convert to img dtype
	if issubclass(img.dtype.type, np.integer):
		dtype_min = np.iinfo(img.dtype).min
		dtype_max = np.iinfo(img.dtype).max
		out_img = np.round(out_img)
	if issubclass(img.dtype.type, np.floating):
		dtype_min = 0
		dtype_max = 1
	out_img = np.clip(out_img, dtype_min, dtype_max)
	out_img = out_img.astype(img.dtype)

	return out_img

"""
Copyright (c) 2023 GroveDG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""