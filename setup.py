# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='wbic_bml',
    version='0.0.1',
    description='WBIC based model selection in Bayesian Mixed LiNGAM',
    long_description=readme,
    author='Akimitsu INOUE',
    author_email='akimitsu.inoue@gmail.com',
    url='https://github.com/inoueakimitsu/wbic_bml',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['theano', 'numpy', 'pymc3<=3.2'],
    dependency_links=['git+ssh://git@github.com/taku-y/bmlingam/develop.git#egg=bmlingam'],
    test_suite='tests'
)

