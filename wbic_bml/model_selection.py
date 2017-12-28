# -*- coding: utf-8 -*-

"""WBIC based model selection of Bayesian mixed LiNGAM models.

Copyright 2017 Akimitsu Inoue

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the 
"Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of the Software, and to 
permit persons to whom the Software is furnished to do so, subject to 
the following conditions:

The above copyright notice and this permission notice shall be 
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The major design pattern of this module was abstracted from
Taku Yoshioka's PyMC3 probability model of mixture of Bayesian
mixed LiNGAM models, which is subject to the same license.
Here is the original copyright notice for Mixture of Bayesian mixed
LiNGAM models:

Author: Taku Yoshioka
License: MIT
"""

import numpy as np
from pymc3 import Normal, StudentT, HalfNormal, DensityDist, \
                  Deterministic, Lognormal, Uniform, Beta
import pymc3 as pm
import theano
import theano.tensor as tt
from theano.tensor.slinalg import cholesky
from bmlingam.utils import standardize_samples
floatX = theano.config.floatX


class FullBayesInferParams:
    u"""Parameters for causal inference with full bayes BMLiNGAM.

    The fields of this class are grouped into two sets. The one is the set
    of hyperparameters.

    :param float df_indvdl: Degrees of freedom of T-distribution on
        individual specific effects. This is used when dist_indvdl is :code:`'t'`.
        Default is 8.0.
    :param str dist_l_cov_21: Distribution of correlation coefficients of
        individual specific effects. Default is :code:`'uniform, -0.9, 0.9'`.
    :param str dist_scale_indvdl: Distribution of the scale of individual
        specific effects. Default is :code:`'uniform, 0.01, 1.0'`.
    :param str dist_beta_noise: Distribution of the shape parameter of the
        generalized Gaussian distribution on observation noise. Default is
        :code:`'uniform, 0.5, 6.0'`.
    :param str dist_std_noise: Distribution of the standard deviation of
        observation noise. The possible values are :code:`'tr_normal'`,
        :code:`'log_normal'` and :code:`'uniform'`. Default is :code:`'log_normal'`.
    :param bool standardize: If :code:`True`, observed samples are
        standardize before inference. Thus samples of RVs from variational
        posterior should be appropriately scaled. Default is :code:`True`.
    :param bool subtract_mu_reg: If :code:`True`, in regression, the common
        interception and the means of individual specific effects are subtracted
        from independent variables. Default is :code:`False`.
    :param bool fix_mu_zero: If :code:`True`, the common interception is fixed
        to 0. Default is :code:`True`.
    :param str prior_var_mu: How to set the prior variance of common
        interceptions. Default is :code:`'auto'`.

    The other includes parameters for model selection phase.

    :param int vb_seed: Seed of random number generator for variational inference. The default value is None.
    :param int ic_seed: Seed of random number generator for calculation of information criteria. The default value is None.
    :param bool standardize: If true (default), samples are standardized.
    :param bool subtract_mu_reg: Specify regression variable.
        The default value is :code:`False`.
    :param bool fix_mu_zero: If :code:`True` (default), the common interception
        is fixed to 0.
    :param prior_var_mu: Set prior variance of common interceptions.
        The default value is :code:`auto`.
    :type prior_var_mu: 'auto' (str literal) or float
    :param int n_mc_samples: Number of Monte Carlo samples.
        The default value is :code:`10000`.
    :param int n_vb_iteration: Number of variational bayes iteration.
        The default value is :code:`100000`.
    :param str metric: Metric for model selection from posterior sample.
        The possible values are :code:`'waic'`, :code:`'loo'` and :code:`'wbic'`.
        The default value is :code:`'waic'`.
    """

    def __init__(self, df_indvdl=8.0, dist_l_cov_21='uniform, -0.9, 0.9',
                 dist_scale_indvdl='uniform, 0.01, 1.0',
                 dist_beta_noise='uniform, 0.5, 6.0',
                 dist_std_noise='log_normal',
                 vb_seed=None,
                 ic_seed=None,
                 standardize=True, subtract_mu_reg=False,
                 fix_mu_zero=True, prior_var_mu='auto',
                 n_mc_samples=10000,
                 n_vb_iteration=100000,
                 metric='wbic'
                 ):
        self.df_indvdl = df_indvdl
        self.dist_l_cov_21 = dist_l_cov_21
        self.dist_scale_indvdl = dist_scale_indvdl
        self.dist_beta_noise = dist_beta_noise
        self.dist_std_noise = dist_std_noise
        self.vb_seed = vb_seed
        self.ic_seed = ic_seed
        self.standardize = standardize
        self.subtract_mu_reg = subtract_mu_reg
        self.fix_mu_zero = fix_mu_zero
        self.prior_var_mu = prior_var_mu
        self.n_mc_samples = n_mc_samples
        self.n_vb_iteration = n_vb_iteration
        self.metric = metric


    def get_full_bayes_bmlingam_params(self, forward):
        """Get FullBayesBMLiNGAMParams.

        :param bool forward: forward model or reverse model.
        :return: FullBayesBMLiNGAMParams
        """

        return FullBayesBMLiNGAMParams(
            df_indvdl=self.df_indvdl,
            dist_l_cov_21=self.dist_l_cov_21,
            dist_scale_indvdl=self.dist_scale_indvdl,
            dist_beta_noise=self.dist_beta_noise,
            dist_std_noise=self.dist_std_noise,
            standardize=self.standardize,
            subtract_mu_reg=self.subtract_mu_reg,
            fix_mu_zero=self.fix_mu_zero,
            prior_var_mu=self.prior_var_mu,
            forward=forward)


class FullBayesBMLiNGAMParams:
    u"""Hyperparameters of full bayesian BMLiNGAM model.
    
    :param float df_indvdl: Degrees of freedom of T-distribution on
        individual specific effects. This is used when dist_indvdl is :code:`'t'`.
        Default is 8.0.
    :param str dist_l_cov_21: Distribution of correlation coefficients of
        individual specific effects. Default is :code:`'uniform, -0.9, 0.9'`.
    :param str dist_scale_indvdl: Distribution of the scale of individual
        specific effects. Default is :code:`'uniform, 0.01, 1.0'`.
    :param str dist_beta_noise: Distribution of the shape parameter of the
        generalized Gaussian distribution on observation noise. Default is
        :code:`'uniform, 0.5, 6.0'`.
    :param str dist_std_noise: Distribution of the standard deviation of
        observation noise. The possible values are :code:`'tr_normal'`,
        :code:`'log_normal'` and :code:`'uniform'`. Default is :code:`'log_normal'`.
    :param bool standardize: If :code:`True`, observed samples are
        standardize before inference. Thus samples of RVs from variational
        posterior should be appropriately scaled. Default is :code:`True`.
    :param bool subtract_mu_reg: If :code:`True`, in regression, the common
        interception and the means of individual specific effects are subtracted
        from independent variables. Default is :code:`False`.
    :param bool fix_mu_zero: If :code:`True`, the common interception is fixed
        to 0. Default is :code:`True`.
    :param str prior_var_mu: How to set the prior variance of common
        interceptions. Default is :code:`'auto'`.
    :param bool forward: If the model assumes causality from 1st to 2nd,
        set :code:`True`.
    """
    def __init__(self, df_indvdl=8.0, dist_l_cov_21='uniform, -0.9, 0.9',
                 dist_scale_indvdl='uniform, 0.01, 1.0',
                 dist_beta_noise='uniform, 0.5, 6.0',
                 dist_std_noise='log_normal',
                 standardize=True, subtract_mu_reg=False,
                 fix_mu_zero=True, prior_var_mu='auto', forward=True):
        self.df_indvdl = df_indvdl
        self.dist_l_cov_21 = dist_l_cov_21
        self.dist_scale_indvdl = dist_scale_indvdl
        self.dist_beta_noise = dist_beta_noise
        self.dist_std_noise = dist_std_noise
        self.standardize = standardize
        self.subtract_mu_reg = subtract_mu_reg
        self.fix_mu_zero = fix_mu_zero
        self.prior_var_mu = prior_var_mu
        self.forward = forward

    def as_dict(self):
        d = dict()
        d.update({'df_indvdl': self.df_indvdl})
        d.update({'dist_l_cov_21': self.dist_l_cov_21})
        d.update({'dist_scale_indvdl': self.dist_scale_indvdl})
        d.update({'dist_beta_noise': self.dist_beta_noise})
        d.update({'dist_std_noise': self.dist_std_noise})
        d.update({'standardize': self.standardize})
        d.update({'subtract_mu_reg': self.subtract_mu_reg})
        d.update({'fix_mu_zero': self.fix_mu_zero})
        d.update({'prior_var_mu': self.prior_var_mu})
        d.update({'forward': self.forward})
        return d


def _dist_from_str(name, dist_params_):
    if type(dist_params_) is str:
        dist_params = dist_params_.split(',')

        if dist_params[0].strip(' ') == 'uniform':
            rv = Uniform(name, lower=float(dist_params[1]),
                         upper=float(dist_params[2]))
        else:
            raise ValueError("Invalid value of dist_params: %s" % dist_params_)

    elif type(dist_params_) is float:
        rv = dist_params_

    else:
        raise ValueError("Invalid value of dist_params: %s" % dist_params_)

    return rv


def _get_L_cov(hparams):
    dist_l_cov_21 = hparams.dist_l_cov_21
    l_cov_21 = _dist_from_str('l_cov_21', dist_l_cov_21)
    l_cov = tt.stack([1.0, l_cov_21, l_cov_21, 1.0]).reshape((2, 2))

    return l_cov


def _gg_loglike(mu, beta, std):
    u"""Returns 1-dimensional likelihood function of generalized Gaussian.

    :param mu: Mean.
    :param beta: Shape parameter.
    :param std: Standard deviation.
    """
    def likelihood(xs):
        return tt.sum(
            tt.log(beta) - tt.log(2.0 * std * tt.sqrt(tt.gamma(1. / beta) / tt.gamma(3. / beta))) - tt.gammaln(
                1.0 / beta) + - tt.power(tt.abs_(xs - mu) / std * tt.sqrt(tt.gamma(1. / beta) / tt.gamma(3. / beta)),
                                         beta))
    return likelihood


def _noise_model(hparams, h1, h2):
    u"""Distribution of observation noise.
    """
    dist_beta_noise = hparams.dist_beta_noise
    beta_noise = _dist_from_str('beta_noise', dist_beta_noise)

    def obs1(mu):
        return _gg_loglike(mu=mu, beta=beta_noise, std=h1)

    def obs2(mu):
        return _gg_loglike(mu=mu, beta=beta_noise, std=h2)

    return obs1, obs2


def _indvdl_t(hparams, std_x, n_samples, L_cov, verbose=0):
    df_L = hparams.df_indvdl
    dist_scale_indvdl = hparams.dist_scale_indvdl
    scale1 = std_x[0] * _dist_from_str('scale_mu1s', dist_scale_indvdl)
    scale2 = std_x[1] * _dist_from_str('scale_mu2s', dist_scale_indvdl)

    scale1 = scale1 / np.sqrt(df_L / (df_L - 2))
    scale2 = scale2 / np.sqrt(df_L / (df_L - 2))

    u1s = StudentT('u1s', nu=np.float32(df_L), shape=(n_samples,),
                   dtype=floatX)
    u2s = StudentT('u2s', nu=np.float32(df_L), shape=(n_samples,),
                   dtype=floatX)

    L_cov_ = cholesky(L_cov).astype(floatX)

    mu1s_ = Deterministic('mu1s_',
                          L_cov_[0, 0] * u1s * scale1 + L_cov_[1, 0] * u2s * scale1)
    mu2s_ = Deterministic('mu2s_',
                          L_cov_[1, 0] * u1s * scale2 + L_cov_[1, 1] * u2s * scale2)  # [1, 0] is ... 0?

    if 10 <= verbose:
        print('StudentT for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_


def _noise_variance(hparams, tau_cmmn, verbose=0):
    dist_std_noise = hparams.dist_std_noise

    if dist_std_noise == 'tr_normal':
        h1 = HalfNormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = HalfNormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Truncated normal for prior scales')

    elif dist_std_noise == 'log_normal':
        h1 = Lognormal('h1', tau=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Lognormal('h2', tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Log normal for prior scales')

    elif dist_std_noise == 'uniform':
        h1 = Uniform('h1', upper=np.float32(1 / tau_cmmn[0]), dtype=floatX)
        h2 = Uniform('h2', upper=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        if 10 <= verbose:
            print('Uniform for prior scales')

    else:
        raise ValueError(
            "Invalid value of dist_std_noise: %s" % dist_std_noise
        )

    return h1, h2


def _common_interceptions(hparams, tau_cmmn, verbose=0):
    prior_var_mu = hparams.prior_var_mu
    fix_mu_zero = hparams.fix_mu_zero

    if fix_mu_zero:
        mu1 = np.float32(0.0)
        mu2 = np.float32(0.0)

        if 10 <= verbose:
            print('Fix bias parameters to 0.0')

    else:
        if prior_var_mu == 'auto':
            tau1 = np.float32(1. / tau_cmmn[0])
            tau2 = np.float32(1. / tau_cmmn[1])
        else:
            v = prior_var_mu
            tau1 = np.float32(1. / v)
            tau2 = np.float32(1. / v)
        mu1 = Normal('mu1', mu=np.float32(0.), tau=np.float32(tau1),
                     dtype=floatX)
        mu2 = Normal('mu2', mu=np.float32(0.), tau=np.float32(tau2),
                     dtype=floatX)

        if 10 <= verbose:
            print('mu1.dtype = {}'.format(mu1.dtype))
            print('mu2.dtype = {}'.format(mu2.dtype))

    return mu1, mu2


def _causal_model(hparams, w1_params, w2_params, tau_cmmn, d):
    subtract_mu_reg = hparams.subtract_mu_reg
    mu1, mu1s, obs1 = w1_params
    mu2, mu2s, obs2 = w2_params

    def likelihood(xs):
        # Independent variable
        obs1_ = obs1(mu=mu1 + mu1s)

        # Regression coefficient
        b = Normal('b%s' % d, mu=np.float32(0.),
                   tau=np.float32(1 / tau_cmmn[1]), dtype=floatX)

        # Dependent variable
        obs2_ = obs2(mu=mu2 + mu2s + b * (xs[:, 0] - mu1 - mu1s)) \
                if subtract_mu_reg else \
                obs2(mu=mu2 + mu2s + b * xs[:, 0])

        return obs1_(xs[:, 0]) + obs2_(xs[:, 1])

    return likelihood


def get_full_bayes_bml_model(xs, hparams, use_wbic=False):
    u"""Returns a PyMC3 probablistic model of full bayes version of BMLiNGAM model.

    This function should be invoked within a Model context of PyMC3.

    :param xs: Observation data.
    :type xs: numpy.ndarray, shape=(n_sample, 2), dtype=float
    :param hparams: Hyperparameters for inference
    :param bool use_wbic: When you use WBIC as an information criteria,
        set this flag :code:`True`. Estimation of WBIC needs posterior sample
        on special inverse temperature setting.
    :return: Probablistic model
    :rtype: pymc3.Model
    """
    # Standardize samples
    floatX = 'float32'
    n_samples = xs.shape[0]
    xs = xs.astype(floatX)
    xs = standardize_samples(xs, True) if hparams.standardize else xs

    # Common scaling parameters
    std_x = np.std(xs, axis=0).astype(floatX)
    max_c = 1.0
    tau_cmmn = np.array(
        [(std_x[0] * max_c)**2, (std_x[1] * max_c)**2]).astype(floatX)

    # Prior of individual specific effects (\tilde{\mu}_{l}^{(i)})
    L_cov = _get_L_cov(hparams)
    mu1s, mu2s = _indvdl_t(hparams, std_x, n_samples, L_cov)

    # Noise variance
    h1, h2 = _noise_variance(hparams, tau_cmmn)

    # Common interceptions
    mu1, mu2 = _common_interceptions(hparams, tau_cmmn)

    # Noise model
    obs1, obs2 = _noise_model(hparams, h1, h2)

    # Pair of causal models
    v1_params = [mu1, mu1s, obs1]
    v2_params = [mu2, mu2s, obs2]

    lp_m1 = _causal_model(hparams, v1_params, v2_params, tau_cmmn, '21')
    lp_m2 = _causal_model(hparams, v2_params, v1_params, tau_cmmn, '12')

    def lp_m2_flipped(xs):
        def flip(xs):
            # Filp 1st and 2nd features
            return tt.stack([xs[:, 1], xs[:, 0]], axis=0).T
        return lp_m2(flip(xs))

    # lp_m1: x1 -> x2 (b_21 is non-zero)
    # lp_m2_flipped: x2 -> x1 (b_12 is non-zero)
    if hparams.forward:
        lp = _inverse_temperature(lp_m1, n_samples) if use_wbic else lp_m1
        DensityDist('dist', lp, observed=xs)
    else:
        lp = _inverse_temperature(lp_m2_flipped, n_samples) if use_wbic else lp_m2_flipped
        DensityDist('dist', lp, observed=xs)


def _inverse_temperature(likelihood, n):
    log_n = np.log(n)

    def likelihood_on_inverse_temperature(xs):
        return likelihood(xs) * 1.0 / log_n

    return likelihood_on_inverse_temperature


def _infer_causality_with_posterior(xs, full_bayes_infer_params, varnames=None):
    """Infer causality based on samples given pair of columns in data.

    :param xs: sample vectors.
    :type xs: numpy.ndarray, shape=(n_samples, 2)
    :param full_bayes_infer_params:  Inference parameters.
    :type full_bayes_infer_params: FullBayesInferParams
    :param varnames: List of variable names.
    :type varnames: list of str, len(varnames)=2

    The return value is dictionary containing the following values:

    .. code:: python

        {
            # Variable names given as input arguments
            # str
            'x1_name': x1_name,
            'x2_name': x2_name,

            # Sample vectors
            # numpy.ndarray, shape=(n_samples, 2)
            'xs': xs,

            # Inferred causality
            # [1, 2] (list of int): (var1 -> var2)
            # [2, 1] (list of int): (var2 -> var1)
            'causality': causality,

            # Inferred causality as a string
            # str
            'causality_str': ('%s -> %s' % (src, dst)),

            # Model selection metric of the selected model
            # float
            'metric': metric,

            # Model selection metric of the reverse model
            # float
            'metric_rev': metric_rev,

        }
    """
    assert(type(full_bayes_infer_params) == FullBayesInferParams)

    if varnames is None:
        varnames = ['var1', 'var2']

    forward_hparams = full_bayes_infer_params.get_full_bayes_bmlingam_params(forward=True)
    reverse_hparams = full_bayes_infer_params.get_full_bayes_bmlingam_params(forward=False)

    if full_bayes_infer_params.metric != 'wbic':
        raise NotImplementedError("Sorry, only WBIC is available as an information criteria currently.")

    with pm.Model() as forward_model:
        get_full_bayes_bml_model(xs, forward_hparams, use_wbic=True)
        forward_fit = pm.fit(method='advi', n=full_bayes_infer_params.n_vb_iteration,
                             random_seed=full_bayes_infer_params.vb_seed)
        forward_fit.seed(full_bayes_infer_params.ic_seed)
        forward_trace = forward_fit.sample(full_bayes_infer_params.n_mc_samples, include_transformed=True)

    with pm.Model() as forward_no_inv_temp_model:
        get_full_bayes_bml_model(xs, forward_hparams, use_wbic=False)
        forward_metric = _get_metric_value(forward_trace, "wbic", forward_no_inv_temp_model)

    with pm.Model() as reverse_model:
        get_full_bayes_bml_model(xs, reverse_hparams, use_wbic=True)
        reverse_fit = pm.fit(method='advi', n=full_bayes_infer_params.n_vb_iteration,
                             random_seed=full_bayes_infer_params.vb_seed)
        reverse_fit.seed(full_bayes_infer_params.ic_seed)
        reverse_trace = reverse_fit.sample(full_bayes_infer_params.n_mc_samples, include_transformed=True)

    with pm.Model() as reverse_no_inv_temp_model:
        get_full_bayes_bml_model(xs, reverse_hparams, use_wbic=False)
        reverse_metric = _get_metric_value(reverse_trace, "wbic", reverse_no_inv_temp_model)

    causality = [1, 2] if forward_metric < reverse_metric else [2, 1]

    x1_name = varnames[0]
    x2_name = varnames[1]
    if causality == [1, 2]:
        src, dst = x1_name, x2_name
    else:
        src, dst = x2_name, x1_name

    return {
        'x1_name': x1_name,
        'x2_name': x2_name,
        'xs': xs,
        'causality': causality,
        'causality_str': ('{} -> {}'.format(src, dst)),
        'metric': forward_metric if causality == [1, 2] else reverse_metric,
        'metric_rev': reverse_metric if causality == [1, 2] else forward_metric
    }


def _get_metric_value(trace, metric, model):
    if metric == "waic":
        return pm.stats.waic(trace, model)[0]
    elif metric == "loo":
        return pm.stats.loo(trace, model)[0]
    elif metric == "wbic":
        return wbic(trace, model)


def wbic(trace, model=None):
    """Calculate the Widely Applicable Bayesian Infromation Criterion
    of the samples in trace from the model. Read more theory here - in a paper
    Watanabe, Sumio. "A widely applicable Bayesian information criterion."
    Journal of Machine Learning Research 14.Mar (2013): 867-897.

    :param trace: result of MCMC run
    :param model: PyMC3 Model
        Optional. Default is None, taken from contex.
    :return: wbic value
    """
    model = pm.model.modelcontext(model)
    log_posterior_density = pm.stats._log_post_trace(trace, model)
    value = np.mean(- np.mean(log_posterior_density, axis=1))

    return value


def get_mixbml_model(xs, hparams, verbose=0):
    u"""Returns a PyMC3 probabilistic model of mixture BML.

    This function should be invoked within a Model session of PyMC3.

    :param xs: Observation data.
    :type xs: ndarray, shape=(n_samples, 2), dtype=float
    :param hparams: Hyperparameters for inference.
    :type hparams: MixBMLParams
    :return: Probabilistic model
    :rtype: pymc3.Model
    """
    # Standardize samples
    floatX = 'float32' # TODO: remove literal
    n_samples = xs.shape[0]
    xs = xs.astype(floatX)
    xs = standardize_samples(xs, True) if hparams.standardize else xs

    # Common scaling parameters
    std_x = np.std(xs, axis=0).astype(floatX)
    max_c = 1.0 # TODO: remove literal
    tau_cmmn = np.array(
        [(std_x[0] * max_c)**2, (std_x[1] * max_c)**2]).astype(floatX)

    # Prior of individual specific effects (\tilde{\mu}_{l}^{(i)})
    L_cov = _get_L_cov(hparams)
    mu1s, mu2s = _indvdl_t(hparams, std_x, n_samples, L_cov)

    # Noise variance
    h1, h2 = _noise_variance(hparams, tau_cmmn)

    # Common interceptions
    mu1, mu2 = _common_interceptions(hparams, tau_cmmn)

    # Noise model
    # obs1 (obs2) is a log likelihood function, not RV
    obs1, obs2 = _noise_model(hparams, h1, h2)

    # Pair of causal models
    v1_params = [mu1, mu1s, obs1]
    v2_params = [mu2, mu2s, obs2]

    # lp_m1: x1 -> x2 (b_21 is non-zero)
    # lp_m2: x2 -> x1 (b_12 is non-zero)
    lp_m1 = _causal_model(hparams, v1_params, v2_params, tau_cmmn, '21')
    lp_m2 = _causal_model(hparams, v2_params, v1_params, tau_cmmn, '12')

    # Prior of mixing proportions for causal models
    z = Beta('z', alpha=1, beta=1)

    # Mixture of potentials of causal models
    def lp_mix(xs):
        def flip(xs):
            # Filp 1st and 2nd features
            return tt.stack([xs[:, 1], xs[:, 0]], axis=0).T

        return pm.logsumexp(tt.stack([tt.log(z) + lp_m1(xs),
                                   tt.log(1 - z) + lp_m2(flip(xs))], axis=0))

    DensityDist('dist', lp_mix, observed=xs)
