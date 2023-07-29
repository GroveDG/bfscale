
# std library
import os
from pathlib import Path
from math import floor, sqrt, log2, ceil
from multiprocessing import Pool, shared_memory
from enum import *

from numba import njit, typed, types 			# Optimization through compiling to C
import numpy as np
import matplotlib.pyplot as plt 				# Debug img showing
from imageio.v3 import imread, imwrite, improps # Image IO
from scipy.optimize import curve_fit			# CPU fitting
import pygpufit.gpufit as gf 					# GPU fitting (a fun compile-it-yourself project for the whole family!)
import PySimpleGUI as gui 						# GUI

class ResizeDevice(Enum):
	CPU_SINGLE = auto()
	CPU_MULTI = auto()
	GPU = auto()

# GUI Window Layout
layout = [[gui.Text("Enter the image you wish to process")],
		  [gui.Input(key='-IN_PATH-', enable_events=True), gui.FileBrowse()],
		  [gui.Frame("Save As...", [
			[gui.Text("Name:", pad=((6,4),(1,5))), gui.Input(key='-OUT_NAME-', size=(10,1), expand_x=True, pad=((0,6),(1,5)))],
			], expand_x=True)],
		  [gui.Frame("Scaling", [
			[gui.Text("Image Downscale:", pad=((6,4),(1,5))), gui.Combo([1], default_value = 1, key='-SCALE-', enable_events=True, pad=((0,6),(1,5))), gui.Text("Final Size:", key='-SIZE-', pad=((6,4),(1,5)))],
			[gui.Text("Scaling Device:", pad=((6,4),(1,5))), gui.Combo([device.name for device in ResizeDevice], key='-SCALE_DEVICE-', pad=((6,4),(1,5)))]
			], expand_x=True)],
		  [gui.Text(size=(40,1), key='-OUTPUT-')],
		  [gui.Button('Ok'), gui.Button('Quit')]]

def bilinear(x, c1, c2, c3, c4):
	return c1*(1-x[1])*(1-x[0]) + c2*x[1]*(1-x[0]) + c3*(1-x[1])*x[0] + c4*x[1]*x[0]

def fit_section(args):
	yx_indices, section, initial_parameters = args
	params, _ = curve_fit(bilinear, yx_indices, section, p0=initial_parameters, bounds=(0,255))
	return params

def RESIZE_CPU_MULTI(img, scale):
	out_img_shape = (img.shape[0]//scale, img.shape[1]//scale, img.shape[2], 4)

	number_points = scale ** 2
	number_fits = (out_img_shape[0]-1) * (out_img_shape[1]-1) * out_img_shape[2]
	number_parameters = 4

	data = np.zeros((number_fits, number_points), dtype=np.float32)
	initial_parameters = np.zeros((number_fits, number_parameters), dtype=np.float32)
	for y in range(out_img_shape[0]-1):
		y_ind = y * (out_img_shape[1]-1) * out_img_shape[2]
		for x in range(out_img_shape[1]-1):
			x_ind = x * out_img_shape[2]
			for c in range(out_img_shape[2]):
				ind = y_ind + x_ind + c
				img_section = img[int((y+0.5)*scale):int((y+1.5)*scale), int((x+0.5)*scale):int((x+1.5)*scale), c]
				initial_parameters[ind] = np.asarray(
						[
						img_section[0][0], img_section[0][-1],
						img_section[-1][0], img_section[-1][-1]
						]
					)
				data[ind] = img_section.flatten()

	indices = np.linspace(0, 1-1/scale, scale)
	X, Y = np.meshgrid(indices, indices)
	size = X.shape
	x_1d = X.reshape((1, np.prod(size)))
	y_1d = Y.reshape((1, np.prod(size)))

	yx_indices = np.vstack((y_1d, x_1d))


	out_img = np.full((img.shape[0]//scale, img.shape[1]//scale, img.shape[2], 4), -1, dtype=np.float64)

	print("Process Count:" + str(os.cpu_count()))
	with Pool() as pool: # Pool() uses os.cpu_count() i.e. the max amount possible
		out_img = pool.map(fit_section, [ (yx_indices, data[ind], initial_parameters[ind]) for ind in range(len(data))], data.shape[0]//os.cpu_count())
	
	out_img = np.concatenate(out_img)
	out_img = out_img.reshape((out_img_shape[0]-1, out_img_shape[1]-1, out_img_shape[2], out_img_shape[3]))
	out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values = -1)

	for y in range(out_img.shape[0]-1, -1, -1):
		for x in range(out_img.shape[1]-1, -1, -1):
			for c in range(out_img.shape[2]-1, -1, -1):
				if x-1 > 0:
					out_img[y][x][c][1] = out_img[y][x-1][c][1]
				else:
					out_img[y][x][c][1] = -1

				if y-1 > 0:
					out_img[y][x][c][2] = out_img[y-1][x][c][2]
				else:
					out_img[y][x][c][2] = -1

				if x-1 > 0 and y-1 > 0:
					out_img[y][x][c][3] = out_img[y-1][x-1][c][3]
				else:
					out_img[y][x][c][3] = -1

	return out_img # Not a real image yet

def RESIZE_GPU(img, scale):
	out_img_shape = (img.shape[0]//scale, img.shape[1]//scale, img.shape[2], 4)

	number_points = scale ** 2
	number_fits = (out_img_shape[0]-1) * (out_img_shape[1]-1) * out_img_shape[2]
	number_parameters = 4

	data = np.zeros((number_fits, number_points), dtype=np.float32)
	initial_parameters = np.zeros((number_fits, number_parameters), dtype=np.float32)
	for y in range(out_img_shape[0]-1):
		y_ind = y * (out_img_shape[1]-1) * out_img_shape[2]
		for x in range(out_img_shape[1]-1):
			x_ind = x * out_img_shape[2]
			for c in range(out_img_shape[2]):
				ind = y_ind + x_ind + c
				img_section = img[int((y+0.5)*scale):int((y+1.5)*scale), int((x+0.5)*scale):int((x+1.5)*scale), c]
				initial_parameters[ind] = np.asarray(
						[
						img_section[0][0], img_section[0][-1],
						img_section[-1][0], img_section[-1][-1]
						]
					)
				data[ind] = img_section.flatten()

	constraints = np.tile([0, 255], (number_fits, number_parameters))
	constraints = np.float32(constraints)
	constraint_types = np.array([gf.ConstraintType.LOWER_UPPER]*4, dtype=np.int32)

	tolerance = 0.01

	max_number_iterations = 20

	estimator_id = gf.EstimatorID.LSE

	model_id = gf.ModelID.BILINEAR

	out_img, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, model_id,
																					initial_parameters,
																					constraints, constraint_types,
																					tolerance, max_number_iterations, None,
																					estimator_id, None)

	# print fit results
	converged = states == 0

	# print summary
	print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
	print('iterations:      {:.2f}'.format(np.mean(number_iterations[converged])))
	print('time:            {:.2f} s'.format(execution_time))

	# get fit states
	number_converged = np.sum(converged)
	print('\nratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
	print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))

	# mean, std of fitted parameters
	converged_parameters = out_img[converged, :]
	converged_parameters_mean = np.mean(converged_parameters, axis=0)
	converged_parameters_std = np.std(converged_parameters, axis=0)

	out_img = out_img.reshape((out_img_shape[0]-1, out_img_shape[1]-1, out_img_shape[2], out_img_shape[3]))
	out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values = -1)

	for y in range(out_img.shape[0]-1, -1, -1):
		for x in range(out_img.shape[1]-1, -1, -1):
			for c in range(out_img.shape[2]-1, -1, -1):
				if x-1 > 0:
					out_img[y][x][c][1] = out_img[y][x-1][c][1]
				else:
					out_img[y][x][c][1] = -1

				if y-1 > 0:
					out_img[y][x][c][2] = out_img[y-1][x][c][2]
				else:
					out_img[y][x][c][2] = -1

				if x-1 > 0 and y-1 > 0:
					out_img[y][x][c][3] = out_img[y-1][x-1][c][3]
				else:
					out_img[y][x][c][3] = -1

	return out_img # Not a real image yet

def best_fit(img, scale, SCALE_DEVICE):
	out_img = None

	match ResizeDevice[SCALE_DEVICE]:
		case ResizeDevice.CPU_MULTI:
			out_img = RESIZE_CPU_MULTI(img, scale)
		case ResizeDevice.GPU:
			out_img = RESIZE_GPU(img, scale)

	out_img = np.ma.median(np.ma.masked_values(out_img, -1), axis=-1)
	out_img = np.uint8(np.round(np.ma.getdata(out_img)))
	return out_img

@njit
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