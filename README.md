# Dependencies

Data can be downloaded from zenodo: https://zenodo.org/record/5532578 (Note: we added a new version since 2021/10/04). Unzip the files and you should get a data and results folder in the root directory of the repo.

Several scientific python packages are required to run the code. The easiest way to do this is to use the python 3.X Anaconda distribution (https://www.continuum.io/downloads). Alternatively you can install the manually using pip:
* Jupyter
* scipy
* Pandas
* numpy
* statsmodels
* scitkit-learn
* matplotlib
* cython
* git+https://github.com/jfrelinger/cython-munkres-wrapper
* fisher

The Cython code for the evaluation metrics should also be compiled. For this, go to /lib and run `python setup.py build_ext --inplace`

The Cython munkres library should be installed using e.g. pip install git+https://github.com/jfrelinger/cython-munkres-wrapper

After installation, run jupyter notebook in the notebooks/ folder. This folder contains several Jupter notebooks, each pertaining to a different part of the evaluation study. The notebooks starting with 0- are optional if you downloaded the results folder.

Code was tested on python 3.7 to 3.9
