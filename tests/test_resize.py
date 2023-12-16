import bfscale
from imageio.v3 import imread
from pytest import fixture
from distutils import dir_util
import numpy as np
import os

@fixture
def datadir(tmpdir, request):
	'''
	https://stackoverflow.com/a/29631801
	Fixture responsible for searching a folder with the same name of test
	module and, if available, moving all contents to a temporary directory so
	tests can use them freely.
	'''
	filename = request.module.__file__
	test_dir, _ = os.path.splitext(filename)

	if os.path.isdir(test_dir):
		dir_util.copy_tree(test_dir, str(tmpdir))

	return tmpdir

def test_resize(datadir: str):
	img_test = imread(os.path.realpath(datadir.join("billiards.tif")))
	img_test = bfscale.scale(img_test, (2, 2))
	img_sample = imread(os.path.realpath(datadir.join("billiards-scaled2.tif")))
	diff = np.intc(img_sample)-np.intc(img_test)
	assert np.all(abs(diff) <= 1)

	img_test = imread(os.path.realpath(datadir.join("billiards.tif")))
	img_test = bfscale.scale(img_test, (240, 240))
	img_sample = imread(os.path.realpath(datadir.join("billiards-scaled240.tif")))
	diff = np.intc(img_sample)-np.intc(img_test)
	assert np.all(abs(diff) <= 1)
	
	img_test = imread(os.path.realpath(datadir.join("billiards.tif")))
	img_test = bfscale.scale(img_test, (1199, 1199))
	img_sample = imread(os.path.realpath(datadir.join("billiards-scaled1199.tif")))
	diff = np.intc(img_sample)-np.intc(img_test)
	assert np.all(abs(diff) <= 1)