# -*- coding: utf-8 -*-

from .context import wbic_bml
from bmlingam.utils.gendata import gen_artificial_data, GenDataParams

from wbic_bml.model_selection import FullBayesInferParams, _infer_causality_with_posterior

import pymc3 as pm
import theano
import numpy as np

import unittest


class TestModelSelection(unittest.TestCase):
    """Advanced test cases."""

    def test_full_estimation(self):

        n_confs = 1
        total_noise = 0.5
        noise_scale = total_noise / np.sqrt(n_confs)
        flip = False

        gen_data_params = GenDataParams(
            n_samples=100,
            e1_dist=['laplace'],
            e1_std=3.0,
            e2_dist=['laplace'],
            e2_std=3.0,
            f1_coef=[noise_scale for _ in range(n_confs)],
            f2_coef=[noise_scale for _ in range(n_confs)],
            conf_dist=[['all'] for _ in range(n_confs)],
            fix_causality=True,  # x1 -> x2 (b21 is non-zero)
            seed=0
        )

        data = gen_artificial_data(gen_data_params)
        print('b_true = {}'.format(data['b']))
        xs = data['xs']

        print(xs)

        # Flip causal direction
        if flip:
            xs = np.vstack((xs[:, 1], xs[:, 0])).T
            print('{} (flipped)'.format(data['causality_true']))
        else:
            print('{} (not flipped)'.format(data['causality_true']))


        wbic_infer_params = FullBayesInferParams(metric='wbic', n_mc_samples=10000, vb_seed=1, ic_seed=1)
        wbic_infer_result = _infer_causality_with_posterior(xs, wbic_infer_params)

        print("inferered causality: {}".format(wbic_infer_result['causality']))
        print(
        "WBIC of estimated causality model {} : {}".format(wbic_infer_result['causality'], wbic_infer_result['metric']))
        print("WBIC of reverese model: {}".format(wbic_infer_result['metric_rev']))

        def _infer_causality_wbic_bml(seed):
            wbic_infer_params = FullBayesInferParams(metric='wbic', n_mc_samples=10000, vb_seed=1, ic_seed=3)

            gen_data_params.seed = seed
            data = gen_artificial_data(gen_data_params)
            xs = data['xs']

            result = _infer_causality_with_posterior(xs, wbic_infer_params)
            return result['causality']

        for seed in range(1):
            print(_infer_causality_wbic_bml(seed))



if __name__ == '__main__':
    unittest.main()
