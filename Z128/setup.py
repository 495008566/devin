#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='z128',
    version='0.1.0',
    description='Weakly Supervised Rotation Object Detection',
    author='Devin',
    author_email='devin@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'mmcv',
        'numpy',
        'opencv-python',
        'matplotlib',
        'shapely',
    ],
)
