# wbic_bml

**WBIC model selection in Bayesian Mixed LiNGAM**  
This program is under development.

## Summary
Python Library for statistical causal inference based on the
Bayesian Mixed LiNGAM (linear non-gaussian acyclic model) and
WBIC (Widely applicable Bayesian Information Criterion).


## Installation
Please install the developed version of [`bmlingam`][4670f282] before installation of `wbic_bml`.

  [4670f282]: https://github.com/taku-y/bmlingam

```bash
git clone -b develop https://github.com/taku-y/bmlingam
cd bmlingam
python setup.py install
```

Then install `wbic_bml`.

```bash
git clone https://github.com/inoueakimitsu/wbic_bml.git
cd wbic_bml
python setup.py install
```

## Usage
Please refer to the test code in `tests` directory.

## Dependencies
`wbic_bml` is tested on `Python 3.5` and depends on `PyMC3 3.2`.

## References
- [Shimizu, S., & Bollen, K. (2014). Bayesian estimation of causal direction in acyclic structural equation models with individual-specific confounder variables and non-Gaussian distributions. Journal of Machine Learning Research, 15(1), 2629-2652.](http://jmlr.org/papers/volume15/shimizu14a/shimizu14a.pdf)
