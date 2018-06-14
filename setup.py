from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='wbic_bml',
    version='0.0.1',
    description='WBIC based model selection in Bayesian Mixed LiNGAM',
    long_description=readme,
    author='Akimitsu INOUE and Shohei SHIMIZU',
    author_email='akimitsu.inoue@gmail.com',
    url='https://github.com/inoueakimitsu/wbic_bml',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['theano', 'numpy', 'pymc3<=3.2'],
    dependency_links=['git+ssh://git@github.com/taku-y/bmlingam/develop.git#egg=bmlingam'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
