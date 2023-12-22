#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "mixin"
DESCRIPTION = "Music Xtraction with Nonstationary Gabor Transforms"
URL = "https://github.com/sevagh/MiXiN"
EMAIL = "sevagh@protonmail.com"
AUTHOR = "Sevag Hanssian"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.0.0-rc0"

here = os.path.abspath(os.path.dirname(__file__))

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    py_modules=["mixin"],
    include_package_data=True,
    license="MIT",
)
