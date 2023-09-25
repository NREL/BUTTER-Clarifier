"""
BUTTER-Clarifier/setup.py
Copyright (c) 2023 Alliance for Sustainable Energy, LLC
License: MIT
Author: Jordan Perr-Sauer
Description: Keras API callback to integrate interpretability module into Keras training routines.
"""

from setuptools import setup, find_packages

setup(name='interpretability',
      version='0.0.1',
      packages=find_packages(
          include=['interpretability']
      ),
      install_requires=[
        'tensorflow',
        'scikit-learn',
        'numpy',
        'scikeras'
      ],
      extras_require={
        "test":[
          'pytest'
        ]
      }
      )