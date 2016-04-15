#!/usr/bin/python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'icgauge',
    'description': 'Integrative Complexity Gauge for Measuring Complexity of Language',
    'author': 'Pamela Toman',
    'author_email': 'ptoman@stanford.edu',
    'packages': ['icgauge'],
    'install_requires': [
        'numpy', 
        'sklearn', 
        'nltk'
    ]
}

setup(**config)