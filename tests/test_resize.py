import bfscale as bf
from imageio.v3 import imread
from pytest import fixture
from distutils import dir_util
import os

@fixture
def datadir(tmpdir, request):
	'''
	Fixture responsible for searching a folder with the same name of test
	module and, if available, moving all contents to a temporary directory so
	tests can use them freely.
	'''
	filename = request.module.__file__
	test_dir, _ = os.path.splitext(filename)

	if os.path.isdir(test_dir):
		dir_util.copy_tree(test_dir, str(tmpdir))

	return tmpdir

def test_resize(datadir):
	img_test = imread(os.path.realpath(datadir.join("billiards.png")))
	img_test = bf.resize(img_test)
	img_sample = imread(os.path.realpath(datadir.join("billiards-scaled.png")))
	assert img_sample == img_test