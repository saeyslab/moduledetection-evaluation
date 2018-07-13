# Dependencies

Data can be downloaded from zenodo: https://zenodo.org/record/1157938 These files should be put into the data folder

Several scientific python packages are required to run the code. The easiest way to do this is to use the python 3.X Anaconda distribution (https://www.continuum.io/downloads). Alternatively you can install the manually using pip:
* Jupyter
* scipy
* Pandas
* numpy
* statsmodels
* scitkit-learn

The Cython code for the evaluation metrics should also be compiled. For this, go to /lib and run `python3 setup.py build_ext --inplace`

The Cython munkres library should be installed:
`sudo pip3 install git+https://github.com/jfrelinger/cython-munkres-wrapper`

After installation, run jupyter notebook in the notebooks/ folder. This folder contains several Jupter notebooks, each pertaining to a different part of the evaluation study.
