from setuptools import setup, find_packages, Extension

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

CommandList={'build_ext': build_ext}


setup(
    name='RSTK',
    ext_modules=cythonize(Extension('RSTK.Model.SVD', ['RSTK/Model/SVD.pyx'], include_dirs=[np.get_include()])),
    cmdclass=CommandList
)
