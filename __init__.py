
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
import symfit as fit 							# CPU fitting
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

def best_fit_section(model, img_section, xy_indices, z):
	z[0].value = img_section[0][0]
	z[1].value = img_section[0][-1]
	z[2].value = img_section[-1][0]
	z[3].value = img_section[-1][-1]

	data_values = np.ravel(img_section)

	to_fit = fit.Fit(model, X=xy_indices[1], Y=xy_indices[0], Z=data_values)
	
	fit_result = to_fit.execute(tol=0.5)

	return fit_result.value(z[0]), fit_result.value(z[1]), fit_result.value(z[2]), fit_result.value(z[3])

def best_fit_column(args):
	y, xy_indices, scale, out_shape, img_shape = args

	X, Y, Z = fit.variables('X, Y, Z') 

	z = (
			fit.Parameter('c1', min=0, max=255),
			fit.Parameter('c2', min=0, max=255),
			fit.Parameter('c3', min=0, max=255),
			fit.Parameter('c4', min=0, max=255)
		)

	model = { Z: z[0]*(1-X)*(1-Y) + z[1]*X*(1-Y) + z[2]*(1-X)*Y + z[3]*X*Y }

	shared_mem_out = shared_memory.SharedMemory(name="shared_out_img")
	out_img = np.ndarray(out_shape, dtype=np.float64, buffer=shared_mem_out.buf)

	shared_mem_img = shared_memory.SharedMemory(name="shared_img")
	img = np.ndarray(img_shape, dtype=np.uint8, buffer=shared_mem_img.buf)

	for x in range(out_shape[1]-1):
		for c in range(img_shape[2]):
			c1, c2, c3, c4 = best_fit_section(model, img[int((y+0.5)*scale):int((y+1.5)*scale), int((x+0.5)*scale):int((x+1.5)*scale), c], xy_indices, z)
			out_img[y][x][c][0] = c1
			out_img[y][x+1][c][1] = c2
			out_img[y+1][x][c][2] = c3
			out_img[y+1][x+1][c][3] = c4
		print(x, y, out_img[y][x])

def RESIZE_CPU_MULTI(img, scale):

	indices = np.float64(np.indices((scale, scale))) / scale
	xy_indices = [indices[0].ravel(), indices[1].ravel()]

	out_img = np.full((img.shape[0]//scale, img.shape[1]//scale, img.shape[2], 4), -1, dtype=np.float64)

	print(os.cpu_count())
	with Pool() as pool: # Pool() uses os.cpu_count() i.e. the max amount possible
		shared_mem_out = shared_memory.SharedMemory(name="shared_out_img", create=True, size=out_img.nbytes)
		shared_out_img = np.ndarray(out_img.shape, dtype=out_img.dtype, buffer=shared_mem_out.buf)
		shared_out_img[:] = out_img[:]

		shared_mem_img = shared_memory.SharedMemory(name="shared_img", create=True, size=img.nbytes)
		shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shared_mem_img.buf)
		shared_img[:] = img[:]

		pool.map(best_fit_column, [ (y, xy_indices, scale, out_img.shape, img.shape) for y in range(out_img.shape[0]-1) ], (out_img.shape[0]-1)//os.cpu_count())
		out_img[:] = shared_out_img[:]
		shared_mem_img.unlink()
		shared_mem_out.unlink()
	
	out_img = np.uint8(np.squeeze(np.mean(out_img, axis=-1, keepdims=True, where=out_img != -1)[:,:,:,0]))
	return out_img

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

	tolerance = 0.5

	max_number_iterations = 10

	estimator_id = gf.EstimatorID.MLE

	model_id = gf.ModelID.BILINEAR

	out_img, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, model_id,
																					initial_parameters,
																					constraints, constraint_types,
																					tolerance, max_number_iterations, None,
																					estimator_id, None)

	out_img = out_img.reshape((out_img_shape[0]-1, out_img_shape[1]-1, out_img_shape[2], out_img_shape[3]))
	out_img = np.pad(out_img, ((0,1), (0,1), (0,0), (0,0)), 'constant', constant_values = -1)

	for y in range(out_img.shape[0]-1, -1, -1):
		for x in range(out_img.shape[0]-1, -1, -1):
			for c in range(2, -1, -1):
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

	return np.uint8(np.squeeze(np.mean(out_img, axis=-1, keepdims=True, where=out_img != -1)[:,:,:,0]))

def best_fit(img, scale, SCALE_DEVICE):
	out_img = None

	match ResizeDevice[SCALE_DEVICE]:
		case ResizeDevice.CPU_MULTI:
			out_img = RESIZE_CPU_MULTI(img, scale)
		case ResizeDevice.GPU:
			out_img = RESIZE_GPU(img, scale)
	print(out_img, SCALE_DEVICE)
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