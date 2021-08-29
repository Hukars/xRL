#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("offlinerl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name='offlinerl',
    description="An base algorithm library for offline reinforcement learning",
    version=get_version(),
    package=['offlinerl'],
    author="NanHu",
    author_email="nukarshu@gmail.com",
    python_requires=">=3.6",
)
