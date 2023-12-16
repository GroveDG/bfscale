# Copyright at end of file.
from math import floor, ceil
import numpy as np
from itertools import product
try:
	from itertools import pairwise
except:
	from itertools import tee
	def pairwise(iterable):
		"s -> (s0,s1), (s1,s2), (s2, s3), ..."
		a, b = tee(iterable)
		next(b, None)
		return zip(a, b)

from typing import Tuple
def _create_yx_indices(shape: Tuple[int, int]):
	X = np.tile(np.linspace(0, shape[1], shape[1]), shape[0])
	Y = np.repeat(np.linspace(0, shape[0], shape[0]), shape[1])
	yx_indices = np.column_stack(((1-Y)*(1-X), (1-X)*Y, X*(1-Y), X*Y))
	return yx_indices

def _divvy_up_by_shape(data: list[np.ndarray], img_size: Tuple[int, int, int], target_size: Tuple[int, int]):
	chunk_mapping = {
		(floor(img_size[0]/target_size[0]), floor(img_size[1]/target_size[1]), img_size[2]),
		(floor(img_size[0]/target_size[0]), ceil(img_size[1]/target_size[1]), img_size[2]),
		(ceil(img_size[0]/target_size[0]), floor(img_size[1]/target_size[1]), img_size[2]),
		(ceil(img_size[0]/target_size[0]), ceil(img_size[1]/target_size[1]), img_size[2])
	}
	num_shapes = len(chunk_mapping)
	chunk_mapping = {shape: i for i, shape in enumerate(chunk_mapping)}
	all_chunks = [[] for _ in range(num_shapes)]
	inverse = []
	if num_shapes == 1:
		all_chunks[0] = data
		inverse = [0] * len(data)
	else:
		for chunk in data:
			chunk_shape_ind = chunk_mapping[chunk.shape]
			all_chunks[chunk_shape_ind].append(chunk)
			inverse.append(chunk_shape_ind)
	return all_chunks, chunk_mapping, inverse


def scale(img: np.ndarray, target_size: Tuple[int, int]):
	assert target_size[0] < img.shape[0], ValueError("Target size must be less than image size")
	assert target_size[1] < img.shape[1], ValueError("Target size must be less than image size")

	# Split image into chunks of size: scale by scale
	target_size = (target_size[0]-1, target_size[1]-1)
	indices_y = np.round(np.linspace(0, img.shape[0], target_size[0]+1)).astype(np.int16)
	indices_x = np.round(np.linspace(0, img.shape[1], target_size[1]+1)).astype(np.int16)
	indices_x = [slice(i,j) for i, j in pairwise(indices_x)]
	indices_y = [slice(i,j) for i, j in pairwise(indices_y)]
	data = [img[slice_y, slice_x] for slice_y, slice_x in product(reversed(indices_y), reversed(indices_x))]

	# Separate out chunks by size
	all_chunks, chunk_mapping, inverse = _divvy_up_by_shape(data, img.shape, target_size)
	all_chunks = [np.concatenate(chunks, axis=-1).reshape(shape[0]*shape[1], -1) for chunks, shape in zip(all_chunks, chunk_mapping.keys())]
	
	# Create indices for fitting
	all_yx_indices = [_create_yx_indices(shape) for shape in chunk_mapping.keys()]

	# Fit the data
	all_parameters = []
	for yx_indices, chunks, chunk_shape in zip(all_yx_indices, all_chunks, chunk_mapping.keys()):
		if chunk_shape[0] == 1 and chunk_shape[1] == 1:
			parameters = chunks.repeat(4, 0)
		else:
			parameters, _, _, _ = np.linalg.lstsq(yx_indices, chunks, rcond=None)
			if chunk_shape[0] == 1:
				parameters = np.tile(chunks, (2, 1))
			if chunk_shape[1] == 1:
				parameters = chunks.repeat(2, 0)
		all_parameters.append(parameters.T)
		

	# Reshape fit result parameters into image
	for i, parameters in enumerate(all_parameters):
		all_parameters[i] = np.split(parameters, parameters.shape[0]//img.shape[2], axis=0)
	data = [all_parameters[chunk_shape_ind].pop(-1) for chunk_shape_ind in inverse]
	data = np.stack(data, axis=0)
	out_img = data.reshape(target_size[0], target_size[1], img.shape[2], 4)
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

	# Convert output to original image dtype
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