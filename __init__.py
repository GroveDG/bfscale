# Copyright information at end of file.

import os
from pathlib import Path
from math import floor, sqrt
from multiprocessing import Pool, shared_memory
from enum import *
from itertools import tee
from time import time

import numpy as np
from imageio.v3 import imread, imwrite, improps # Image IO
from scipy.optimize import curve_fit			# CPU fitting
from numba import njit, prange

class ResizeDevice(Enum):
	CPU_SINGLE = auto()
	CPU_MULTI = auto()
	GPU = auto()

available_devices = [ResizeDevice.CPU_SINGLE]
# try:
# 	import pygpufit.gpufit as gf 					# GPU fitting (a fun compile-it-yourself project for the whole family!)
# 	available_devices.append(ResizeDevice.GPU)
# except ImportError:
# 	print('Module "gpufit" not found: GPU unavailable')

import PySimpleGUI as gui 						# GUI

# GUI Window Layout
layout = [[gui.Text("Enter the image you wish to process")],
		  [gui.Input(key='-IN_PATH-', enable_events=True), gui.FileBrowse()],
		  [gui.Frame("Save As...", [
			[gui.Text("Name:", pad=((6,4),(1,5))), gui.Input(key='-OUT_NAME-', size=(10,1), expand_x=True, pad=((0,6),(1,5)))],
			], expand_x=True)],
		  [gui.Frame("Scaling", [
			[gui.Text("Image Downscale:", pad=((6,4),(1,5))), gui.Combo([1], default_value = 1, key='-SCALE-', enable_events=True, pad=((0,6),(1,5))), gui.Text("Final Size:", key='-SIZE-', pad=((6,4),(1,5)))],
			[gui.Text("Scaling Device:", pad=((6,4),(1,5))), gui.Combo([device.name for device in available_devices], key='-SCALE_DEVICE-', pad=((6,4),(1,5)))]
			], expand_x=True)],
		  [gui.Text(size=(40,1), key='-OUTPUT-')],
		  [gui.Button('Ok'), gui.Button('Quit')]]

# Better when parallelization is not available (and it's prettier)
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
def flatten_data(data, out_img_shape, img, scale):
	for y in prange(out_img_shape[0]):
		y_ind = y * out_img_shape[1] * out_img_shape[2]
		for x in prange(out_img_shape[1]):
			x_ind = x * out_img_shape[2]
			for c in prange(out_img_shape[2]):
				ind = y_ind + x_ind + c
				img_section = img[int((y+0.5)*scale):int((y+1.5)*scale), int((x+0.5)*scale):int((x+1.5)*scale), c]
				data[ind] = img_section.flat
	return data.transpose()

def RESIZE_CPU(img, scale, SCALE_DEVICE):
	out_img_shape = (img.shape[0]//scale-1, img.shape[1]//scale-1, img.shape[2], 4)

	data = np.zeros((out_img_shape[0] * out_img_shape[1] * out_img_shape[2], scale ** 2), dtype=np.float32) # Shape: (number of fits, number of points)
	data = flatten_data(data, out_img_shape, img, scale)

	indicies = np.linspace(0, scale, scale+1)[0:-1]
	X = np.tile(indicies, scale)
	Y = np.repeat(indicies, scale)

	yx_indices = np.column_stack(((1-Y)*(1-X), (1-X)*Y, X*(1-Y), X*Y))

	parameters, _, _, _ = np.linalg.lstsq(yx_indices, data, rcond=None)

	out_img = np.pad(np.clip(parameters, 0, 255).transpose().reshape(out_img_shape), ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values=-1)
	# This ↓↓↓ is what this line ↑↑↑ is doing
	# ---------------------------------------
	# out_img = np.clip(parameters, 0, 255)
	# out_img = out_img.transpose()
	# out_img = out_img.reshape(out_img_shape)
	# out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values=-1)

	out_img[:, 1:, :, 1] = out_img[:, :-1, :, 1]
	out_img[1:, :, :, 2] = out_img[:-1, :, :, 2]
	out_img[1:, 1:, :, 3] = out_img[:-1, :-1, :, 3]
	out_img[:, 0, :, 1] = -1
	out_img[0, :, :, 2] = -1
	out_img[0, 0, :, 3] = -1

	return out_img # Not a real image yet

def best_fit(img, scale, SCALE_DEVICE):
	out_img = None

	match ResizeDevice[SCALE_DEVICE]:
		case ResizeDevice.CPU_SINGLE:
			out_img = RESIZE_CPU(img, scale, SCALE_DEVICE)

	out_img = np.ma.median(np.ma.masked_values(out_img, -1), axis=-1)
	out_img = np.uint8(np.round(np.ma.getdata(out_img)))
	return out_img

def scales_of(size):
	scales = []
	for n in range(1, floor(sqrt(size[0]))+1):
		if size[0] % n == 0 and size[1] % n == 0:
			scales.append(n)
	return scales

def update_scale(filepath, scale):
	size = improps(filepath).shape
	window['-SIZE-'].update("Final Size: {x}, {y}".format(x = floor(size[0] / scale),y = floor(size[1] / scale)))

def get_available_scales(filepath):
	size = improps(filepath).shape
	return scales_of(size)

def write_img(filepath, savename, img):
	savepath = Path(filepath)
	if savename != "":
		savepath = savepath.with_stem(savename)
	imwrite(savepath, img)

# ================== START ================== #
def main():

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
				out_img = best_fit(img, scale, values['-SCALE_DEVICE-'])
				write_img(filepath, values['-OUT_NAME-'], out_img)
			case '-IN_PATH-':
				if os.path.exists(filepath):
					savepath = Path(filepath)
					window['-OUT_NAME-'].update(savepath.stem + "-scaled")
					window['-SCALE-'].update(values = get_available_scales(filepath), value = 1)
					update_scale(filepath, 1)
			case '-SCALE-':
				update_scale(filepath, scale)

		prev_filepath = filepath

	window.close()

if __name__ == '__main__':
	window = gui.Window('Best Fit Downscale', layout) # Create the window
	main()

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