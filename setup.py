from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

# Define the Cython extension module
cython_extension = Extension(
    "src.datatypes.graphops.graphops_c",
    ["src/datatypes/graphops/graphops_c.pyx"]
)

# Define the C++ extension module
cpp_extension = Extension(
    "src.metrics.utils.orca.orca",
    sources=["src/metrics/utils/orca/orca.cpp"],
    extra_compile_args=["-O2", "-std=c++11"]
)

# Specify both extension modules
ext_modules = [cython_extension, cpp_extension]

# Setup the package
setup(
    name='graph-generation',
    version="1.0.0",
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)