#!/usr/bin/env python3

from setuptools import Extension, setup
import pybind11


ext_modules = [
    Extension(
        "mcn_dynamics",
        ["mcn_dynamics.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(
    name="mcn_dynamics",
    version="0.1.0",
    ext_modules=ext_modules,
)
