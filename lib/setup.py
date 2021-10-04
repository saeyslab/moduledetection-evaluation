from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'ebcubed',
  ext_modules = cythonize("ebcubed.pyx"),
  include_dirs=[numpy.get_include()]
)

setup(
  name = 'jaccard',
  ext_modules = cythonize("jaccard.pyx"),
  include_dirs=[numpy.get_include()]
)

setup(
  name = 'cfisher',
  ext_modules = cythonize("cfisher.pyx"),
  include_dirs=[numpy.get_include()]
)
