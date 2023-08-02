# Copyright information at end of file.

import os
from pathlib import Path
from math import floor, sqrt
from enum import *

import numpy as np
from numba import njit, prange
from imageio.v3 import imread, imwrite, improps
import PySimpleGUI as gui

# GUI Window Layout
layout = [[gui.Text("Enter the image you wish to process")],
		  [gui.Input(key='-IN_PATH-', enable_events=True), gui.FileBrowse()],
		  [gui.Frame("Save As...", [
			[gui.Text("Name:", pad=((6,4),(1,5))), gui.Input(key='-OUT_NAME-', size=(10,1), expand_x=True, pad=((0,6),(1,5)))],
			], expand_x=True)],
		  [gui.Frame("Scaling", [
			[gui.Text("Image Downscale:", pad=((6,4),(1,5))), gui.Combo([1], default_value = 1, key='-SCALE-', enable_events=True, pad=((0,6),(1,5))), gui.Text("Final Size:", key='-SIZE-', pad=((6,4),(1,5)))],			], expand_x=True)],
		  [gui.Text(size=(40,1), key='-OUTPUT-')],
		  [gui.Button('Ok'), gui.Button('Quit')]]

# Better when parallelization is not available (and it's prettier)
# @njit
# def flatten_data(data, out_img_shape, img, scale):
# 	out_img_size = out_img_shape[0] * out_img_shape[1] * out_img_shape[2]
# 	for ind, coord in enumerate(np.ndindex(out_img_shape[0:3])):
# 		data[ind] = img[
# 			int((coord[0]+0.5)*scale):int((coord[0]+1.5)*scale),
# 			int((coord[1]+0.5)*scale):int((coord[1]+1.5)*scale),
# 			coord[2]
# 		].flatten()
# 	return data.transpose()

@njit(parallel=True)
def _flatten_data(data, out_img_shape, img, scale):
	for y in prange(out_img_shape[0]):
		y_ind = y * out_img_shape[1] * out_img_shape[2]
		for x in prange(out_img_shape[1]):
			x_ind = x * out_img_shape[2]
			for c in prange(out_img_shape[2]):
				ind = y_ind + x_ind + c
				img_section = img[int((y+0.5)*scale):int((y+1.5)*scale), int((x+0.5)*scale):int((x+1.5)*scale), c]
				data[ind] = img_section.flat
	return data.transpose()

def resize(img, scale):
	out_img_shape = (img.shape[0]//scale-1, img.shape[1]//scale-1, img.shape[2], 4)

	data = np.zeros((out_img_shape[0] * out_img_shape[1] * out_img_shape[2], scale ** 2), dtype=np.float32) # Shape: (number of fits, number of points)
	data = _flatten_data(data, out_img_shape, img, scale)

	indicies = np.linspace(0, scale, scale+1)[0:-1]
	X = np.tile(indicies, scale)
	Y = np.repeat(indicies, scale)

	yx_indices = np.column_stack(((1-Y)*(1-X), (1-X)*Y, X*(1-Y), X*Y))

	parameters, _, _, _ = np.linalg.lstsq(yx_indices, data, rcond=None)

	out_img = np.pad(parameters.transpose().reshape(out_img_shape), ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values=-1)
	# This ↓↓↓ is what this line ↑↑↑ is doing
	# ---------------------------------------
	# out_img = out_img.transpose()
	# out_img = out_img.reshape(out_img_shape)
	# out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values=-1)

	out_img[:, 1:, :, 1] = out_img[:, :-1, :, 1]
	out_img[1:, :, :, 2] = out_img[:-1, :, :, 2]
	out_img[1:, 1:, :, 3] = out_img[:-1, :-1, :, 3]
	out_img[:, 0, :, 1] = -1
	out_img[0, :, :, 2] = -1
	out_img[0, 0, :, 3] = -1

	out_img = np.ma.median(np.ma.masked_values(out_img, -1), axis=-1)
	out_img = np.uint8(np.clip(np.round(np.ma.getdata(out_img)), 0, 255))
	return out_img

def _scales_of(size):
	scales = []
	for n in range(1, floor(sqrt(size[0]))+1):
		if size[0] % n == 0 and size[1] % n == 0:
			scales.append(n)
	return scales

def _update_scale(window, filepath, scale):
	size = improps(filepath).shape
	window['-SIZE-'].update("Final Size: {x}, {y}".format(x = floor(size[0] / scale),y = floor(size[1] / scale)))

def _get_available_scales(filepath):
	size = improps(filepath).shape
	return _scales_of(size)

def _write_img(filepath, savename, img):
	savepath = Path(filepath)
	if savename != "":
		savepath = savepath.with_stem(savename)
	imwrite(savepath, img)

# ================== START ================== #
def open_window():
	window = gui.Window('Best Fit Scale', layout) # Create the window

	prev_filepath = ""
	while True: # Event loop
		event, values = window.read()
		
		if event == gui.WINDOW_CLOSED or event == 'Quit':
			break # If user wants to quit or window was closed

		filepath = values['-IN_PATH-']
		scale = values['-SCALE-']
		match event:
			case 'Ok':
				try:
					img = imread(filepath, index=None)
				except Exception as e:
					print(e)
					window['-OUTPUT-'].update('Invalid image: ' + values['-IN_PATH-'])
					continue
				out_img = resize(img, scale)
				_write_img(filepath, values['-OUT_NAME-'], out_img)
			case '-IN_PATH-':
				if os.path.exists(filepath):
					savepath = Path(filepath)
					window['-OUT_NAME-'].update(savepath.stem + "-scaled")
					window['-SCALE-'].update(values = _get_available_scales(filepath), value = 1)
					_update_scale(window, filepath, 1)
			case '-SCALE-':
				_update_scale(window, filepath, scale)

		prev_filepath = filepath

	window.close()

if __name__ == '__main__':
	open_window()

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