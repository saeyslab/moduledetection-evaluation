from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'ebcubed',
  ext_modules = cythonize("ebcubed.pyx"),
)

setup(
  name = 'jaccard',
  ext_modules = cythonize("jaccard.pyx"),
)

setup(
  name = 'cfisher',
  ext_modules = cythonize("cfisher.pyx"),
)
