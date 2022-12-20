import os
import platform
import subprocess

from setuptools import setup
from setuptools import find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext

from src.DSCollection.version import __version__

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    # name what to pip install, different from module name to import from code
    name="dscollection",
    author="Chris LIU, Shiboy",
    author_email="chris.lq@hotmail.com",
    url="https://github.com/cvamateur/DSCollection",
    version=__version__,  # 0.0.x imply is unstable
    description="A collection of tools ease dataset manipulation",
    scripts=["bin/dsc"],
    packages=find_packages("src"),  # a list of actual python packages
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "tqdm",
        "pycocotools",
        "pandas",
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "gst": ["pycairo >= 1.16.3", "PyGObject >= 3.32.0"],
    }
)
