#!/usr/bin/env python

import os
import setuptools


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements_filename():
    if 'READTHEDOCS' in os.environ:
        return "REQUIREMENTS-RTD.txt"
    elif 'DOCKER' in os.environ:
        return "REQUIREMENTS-DOCKER.txt"
    else:
        return "REQUIREMENTS.txt"


install_requires = [
    line.rstrip() for line in open(os.path.join(os.path.dirname(__file__), get_requirements_filename()))
]

setuptools.setup(
    name='mighty_codes',
    version='0.1.0',
    description='A software package for constructing optimal short codes '
                'for ultra low bandwidth systems',
    long_description=readme(),
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research'
      'License :: OSI Approved :: BSD License',
      'Programming Language :: Python :: 3.7',
      'Topic :: Scientific/Engineering',
    ],
    url='http://github.com/broadinstitute/MightyCodes',
    author='Mehrtash Babadi',
    license='BSD (3-Clause)',
    packages=['mighty_codes'],
    install_requires=install_requires,
    entry_points={
        'console_scripts': ['mighty-codes=mighty_codes.cli.base_cli:main'],
    },
    include_package_data=True,
    zip_safe=False
)