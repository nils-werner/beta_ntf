import sys
import numpy
import warnings
import setuptools
from setuptools import setup

# See if we can use Cython
USE_CYTHON = False

if "--no-cython" not in sys.argv:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
        USE_CYTHON = True
    except ImportError as e:
        warnings.warn(e.message)
else:
    sys.argv.remove("--no-cython")

ext = '.pyx' if USE_CYTHON else '.c'

extension = [
    setuptools.Extension("beta_ntf.cython_methods",
                         ["beta_ntf/cython_methods" + ext],
                         include_dirs=[".", numpy.get_include()]),
]

if USE_CYTHON:
    extension = cythonize(extension)

setup(
    name='beta_ntf',
    version='0.2',
    description='Super beta-NTF 2000',
    url='https://code.google.com/p/beta-ntf/',
    author='Antoine Liutkus',
    author_email='antoine@liutkus.net',
    install_requires=['NumPy>=1.6.0'],
    license='LGPL',
    packages=['beta_ntf'],
    zip_safe=True,
    ext_modules=extension
)
