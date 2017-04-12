#!/usr/bin/env python

# authors: J.L. Lanfranchi, P.Eller, and S. Wren
# email:   jll1062+pisa@phys.psu.edu
# date:    March 20, 2016
"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module will load the HypoTesting class from hypo_testing.py and
use it to do an Asimov test across the space of one of the injected parameters.
The user will define the parameter and pass a numpy-interpretable string to 
set the range of values. For example, one could scan over the space of theta23 
by using a string such as `numpy.linspace(0.35,0.65,31)` which will then be 
evaluated to figure out a space of theta23 to inject and run Asimov tests.

"""


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import numpy as np

from pisa import ureg
from pisa.analysis.hypo_testing import HypoTesting, Labels, parse_args
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.prior import Prior
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.stats import ALL_METRICS


def main():
    init_args_d = parse_args(injparamscan=True)

    # Normalize and convert `*_pipeline` filenames; store to `*_maker`
    # (which is argument naming convention that HypoTesting init accepts).
    for maker in ['h0', 'h1', 'data']:
        try:
            filenames = init_args_d.pop(maker + '_pipeline')
        except:
            filenames = None
        if filenames is not None:
            filenames = sorted(
                [normcheckpath(fname) for fname in filenames]
            )
        init_args_d[maker + '_maker'] = filenames

        ps_name = maker + '_param_selections'
        try:
            ps_str = init_args_d[ps_name]
        except:
            ps_str = None
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

    init_args_d['data_maker'] = init_args_d['h0_maker']
    init_args_d['h1_maker'] = init_args_d['h0_maker']

    init_args_d['h0_maker'] = DistributionMaker(init_args_d['h0_maker'])
    init_args_d['h1_maker'] = DistributionMaker(init_args_d['h1_maker'])
    init_args_d['h1_maker'].select_params(init_args_d['h1_param_selections'])
    init_args_d['data_maker'] = DistributionMaker(init_args_d['data_maker'])
    if init_args_d['data_param_selections'] is None:
        init_args_d['data_param_selections'] = \
            init_args_d['h0_param_selections']
        init_args_d['data_name'] = init_args_d['h0_name']
    init_args_d['data_maker'].select_params(
        init_args_d['data_param_selections']
    )

    # Remove final parameters that don't want to be passed to HypoTesting
    param_name = init_args_d.pop('param_name')
    inj_vals = eval(init_args_d.pop('inj_vals'))
    inj_units = init_args_d.pop('inj_units')
    force_prior = init_args_d.pop('use_inj_prior')

    # Instantiate the analysis object
    hypo_testing = HypoTesting(**init_args_d)
    
    logging.info(
        'Scanning over %s between %.4f and %.4f with %i vals'
        %(param_name, min(inj_vals), max(inj_vals), len(inj_vals))
    )
    # Modify parameters if necessary
    if param_name == 'sin2theta23':
        requested_vals = inj_vals
        inj_vals = np.arcsin(np.sqrt(inj_vals))
        logging.info(
            'Converting to theta23 values. Equivalent range is %.4f to %.4f '
            'radians, or %.4f to %.4f degrees'
            %(min(inj_vals), max(inj_vals),
              min(inj_vals)*180/np.pi, max(inj_vals)*180/np.pi)
        )
        test_name = 'theta23'
        inj_units = 'radians'
    elif param_name == 'deltam31':
        raise ValueError('Need to implement a test where it ensures the sign '
                         'of the requested values matches those in truth and '
                         'the hypo makers (else it makes no sense). For now, '
                         'please select deltam3l instead.')
    elif param_name == 'deltam3l':
        # Ensure all values are the same sign, else it doesn't make any sense
        if not np.alltrue(np.sign(inj_vals)):
            raise ValueError("Not all requested values to inject are the same "
                             "sign. This doesn't make any sense given that you"
                             " have requested to inject different values of "
                             "deltam3l.")
        logging.info('Parameter requested was deltam3l - will convert assuming'
                     ' that this is always the largest of the two splittings '
                     'i.e. deltam3l = deltam31 for deltam3l > 0 and deltam3l '
                     '= deltam32 for deltam3l < 0.')
        inj_sign = np.sign(inj_vals)[0]
        requested_vals = inj_vals
        test_name = 'deltam31'
        deltam21_val = hypo_testing.data_maker.params['deltam21'].value.to(
            inj_units
        ).magnitude
        if inj_sign == 1:
            no_inj_vals = requested_vals
            io_inj_vals = (requested_vals - deltam21_val) * -1.0
        else:
            io_inj_vals = requested_vals
            no_inj_vals = (requested_vals * -1.0) + deltam21_val
        inj_vals = []
        for no_inj_val, io_inj_val in zip(no_inj_vals, io_inj_vals):
            o_vals = {}
            o_vals['no'] = no_inj_val
            o_vals['io'] = io_inj_val
            inj_vals.append(o_vals)
    else:
        test_name = param_name
        requested_vals = inj_vals

    unit_inj_vals = []
    for inj_val in inj_vals:
        if isinstance(inj_val, dict):
            o_vals = {}
            for ivkey in inj_val.keys():
                o_vals[ivkey] = inj_val[ivkey]*ureg(inj_units)
            unit_inj_vals.append(o_vals)
        else:
            unit_inj_vals.append(inj_val*ureg(inj_units))
    inj_vals = unit_inj_vals

    # Extend the ranges of the distribution makers so that they reflect the
    # range of the scan. This is a pain if there are different values depending
    # on the ordering. Need to extend the ranges of both values in the
    # hypothesis maker since the hypotheses may minimise over the ordering,
    # and could then go out of range.

    # Also, some parameters CANNOT go negative or else things won't work.
    # To account for this, check if parameters lower value was positive and,
    # if so, enforce that it is positive now.
    if isinstance(inj_vals[0], dict):
        # Calculate ranges for both parameters
        norangediff = max(no_inj_vals) - max(no_inj_vals)
        norangediff = norangediff*ureg(inj_units)
        norangetuple = (min(no_inj_vals)*ureg(inj_units) - 0.5*norangediff,
                        max(no_inj_vals)*ureg(inj_units) + 0.5*norangediff)
        iorangediff = max(io_inj_vals) - max(io_inj_vals)
        iorangediff = iorangediff*ureg(inj_units)
        iorangetuple = (min(io_inj_vals)*ureg(inj_units) - 0.5*iorangediff,
                        max(io_inj_vals)*ureg(inj_units) + 0.5*iorangediff)
        # Select the NO (or nh) parameters in the config file
        hypo_testing.h0_maker.select_params(['nh'])
        hypo_testing.h1_maker.select_params(['nh'])
        if hypo_testing.h0_maker.params[test_name].range is not None:
            if hypo_testing.h0_maker.params[test_name].range[0] > 0:
                enforce_positive = True
            else:
                enforce_positive = False
        else:
            enforce_positive = False
        if hypo_testing.h0_maker.params[test_name].units != inj_units:
            newminrangeval = norangetuple[0].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            newmaxrangeval = norangetuple[1].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            if enforce_positive and newminrangeval < 0:
                newminrangeval = 0.0 * ureg(inj_units)
                newminrangeval = newminrangeval.to(
                    hypo_testing.h0_maker.params[test_name].units
                )
            newrangetuple = (newminrangeval, newmaxrangeval)
            hypo_testing.h0_maker.params[test_name].range = newrangetuple
            hypo_testing.h1_maker.params[test_name].range = newrangetuple
        else:
            if enforce_positive and norangetuple[0] < 0:
                norangetuple = (0.0 * ureg(inj_units),
                                max(no_inj_vals)*ureg(inj_units) + \
                                0.5*norangediff)
            hypo_testing.h0_maker.params[test_name].range = norangetuple
            hypo_testing.h1_maker.params[test_name].range = norangetuple
        # Select the IO (or ih) parameters in the cofig file
        hypo_testing.h0_maker.select_params(['ih'])
        hypo_testing.h1_maker.select_params(['ih'])
        if hypo_testing.h0_maker.params[test_name].range is not None:
            if hypo_testing.h0_maker.params[test_name].range[0] > 0:
                enforce_positive = True
            else:
                enforce_positive = False
        else:
            enforce_positive = False
        if hypo_testing.h0_maker.params[test_name].units != inj_units:
            newminrangeval = iorangetuple[0].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            newmaxrangeval = iorangetuple[1].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            if enforce_positive and newminrangeval < 0:
                newminrangeval = 0.0 * ureg(inj_units)
                newminrangeval = newminrangeval.to(
                    hypo_testing.h0_maker.params[test_name].units
                )
            newrangetuple = (newminrangeval, newmaxrangeval)
            hypo_testing.h0_maker.params[test_name].range = newrangetuple
            hypo_testing.h1_maker.params[test_name].range = newrangetuple
        else:
            if enforce_positive and iorangetuple[0] < 0:
                iorangetuple = (0.0 * ureg(inj_units),
                                max(io_inj_vals)*ureg(inj_units) + \
                                0.5*iorangediff)
            hypo_testing.h0_maker.params[test_name].range = iorangetuple
            hypo_testing.h1_maker.params[test_name].range = iorangetuple
        # BE SURE TO SELECT THE PROPER ONES AGAIN AT THE END
        hypo_testing.h0_maker.select_params(init_args_d['h0_param_selections'])
        hypo_testing.h1_maker.select_params(init_args_d['h1_param_selections'])
        # Now for the data maker we must test the sign of the value
        # However, enforcing positive if the original range was positive is NOT
        # necessary here because their is no minimisation using the data maker.
        # Thus, if the value falls in to an unphysical range here it will be
        # because the user requested an unphysical value and in that case it
        # should be allowed to go there and raise the appropriate error.
        if np.sign(
                hypo_testing.h1_maker.params[test_name].value.magnitude) == 1:
            if hypo_testing.h1_maker.params[test_name].units != inj_units:
                newminrangeval = norangetuple[0].to(
                    hypo_testing.h1_maker.params[test_name].units
                )
                newmaxrangeval = norangetuple[1].to(
                    hypo_testing.h1_maker.params[test_name].units
                )
                newrangetuple = (newminrangeval, newmaxrangeval)
                hypo_testing.h1_maker.params[test_name].range = newrangetuple
            else:
                hypo_testing.h1_maker.params[test_name].range = norangetuple
        else:
            if hypo_testing.h1_maker.params[test_name].units != inj_units:
                newminrangeval = iorangetuple[0].to(
                    hypo_testing.h1_maker.params[test_name].units
                )
                newmaxrangeval = iorangetuple[1].to(
                    hypo_testing.h1_maker.params[test_name].units
                )
                newrangetuple = (newminrangeval, newmaxrangeval)
                hypo_testing.h1_maker.params[test_name].range = newrangetuple
            else:
                hypo_testing.h1_maker.params[test_name].range = iorangetuple
        if np.sign(
                hypo_testing.data_maker.params[test_name].value.magnitude) == 1:
            if hypo_testing.data_maker.params[test_name].units != inj_units:
                newminrangeval = norangetuple[0].to(
                    hypo_testing.data_maker.params[test_name].units
                )
                newmaxrangeval = norangetuple[1].to(
                    hypo_testing.data_maker.params[test_name].units
                )
                newrangetuple = (newminrangeval, newmaxrangeval)
                hypo_testing.data_maker.params[test_name].range = newrangetuple
            else:
                hypo_testing.data_maker.params[test_name].range = norangetuple
        else:
            if hypo_testing.data_maker.params[test_name].units != inj_units:
                newminrangeval = iorangetuple[0].to(
                    hypo_testing.data_maker.params[test_name].units
                )
                newmaxrangeval = iorangetuple[1].to(
                    hypo_testing.data_maker.params[test_name].units
                )
                newrangetuple = (newminrangeval, newmaxrangeval)
                hypo_testing.data_maker.params[test_name].range = newrangetuple
            else:
                hypo_testing.data_maker.params[test_name].range = iorangetuple
    # Otherwise it's way simpler...
    else:
        if hypo_testing.h0_maker.params[test_name].range is not None:
            if hypo_testing.h0_maker.params[test_name].range[0] > 0:
                enforce_positive = True
            else:
                enforce_positive = False
        else:
            enforce_positive = False
        rangediff = max(inj_vals) - min(inj_vals)
        rangetuple = (min(inj_vals) - 0.5*rangediff,
                      max(inj_vals) + 0.5*rangediff)
        if enforce_positive and rangetuple[0] < 0:
            rangetuple = (0.0 * ureg(inj_units),
                          max(inj_vals) + 0.5*rangediff)
        # Ensure that the units match, if not change them
        if hypo_testing.h0_maker.params[test_name].units != inj_units:
            minrangeval = rangetuple[0].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            maxrangeval = rangetuple[1].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            if enforce_positive and minrangeval < 0:
                newminrangeval = 0.0 * ureg(inj_units)
                newminrangeval = newminrangeval.to(
                    hypo_testing.h0_maker.params[test_name].units
                )
            rangetuple = (minrangeval, maxrangeval)
        hypo_testing.h0_maker.params[test_name].range\
            = rangetuple
        hypo_testing.h1_maker.params[test_name].range\
            = rangetuple
        hypo_testing.data_maker.params[test_name].range\
            = rangetuple

    if hypo_testing.data_maker.params[test_name].prior is not None:
        if hypo_testing.data_maker.params[test_name].prior.kind != 'uniform':
            if force_prior:
                logging.warn("Parameter to be scanned, %s, has a %s prior that"
                             " you have requested to be left on. This will "
                             "likely make the results wrong."%(test_name,
                                hypo_testing.data_maker.params[
                                   test_name].prior.kind))
            else:
                logging.info("Parameter to be scanned, %s, has a %s prior. "
                             "This will be changed to a uniform prior (i.e. "
                             "no prior) for this test."%(test_name,
                                hypo_testing.data_maker.params[
                                   test_name].prior.kind))
                uniformprior = Prior(kind='uniform')
                hypo_testing.h0_maker.params[test_name].prior = uniformprior
                hypo_testing.h1_maker.params[test_name].prior = uniformprior
    else:
        if force_prior:
            raise ValueError("Parameter to be scanned, %s, does not have a "
                             "prior but you have requested to force one to be"
                             " left on. Something is potentially wrong."
                             %test_name)
        else:
            logging.info("Parameter to be scanned, %s, does not have a prior. "
                         "So nothing needs to be done."%test_name)

    # Scan over the injected values. We also loop over the requested vals here
    # in case they are different so that value can be put in the labels
    for inj_val, requested_val in zip(inj_vals, requested_vals):
        # Be sure to inject the right value!
        if isinstance(inj_val, dict):
            hypo_testing.h0_maker.select_params(['nh'])
            hypo_testing.h1_maker.select_params(['nh'])
            inj_val['no'] = inj_val['no'].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            hypo_testing.h0_maker.params[test_name].value = inj_val['no']
            hypo_testing.h1_maker.params[test_name].value = inj_val['no']
            hypo_testing.h0_maker.select_params(['ih'])
            hypo_testing.h1_maker.select_params(['ih'])
            inj_val['io'] = inj_val['io'].to(
                hypo_testing.h0_maker.params[test_name].units
            )
            hypo_testing.h0_maker.params[test_name].value = inj_val['io']
            hypo_testing.h1_maker.params[test_name].value = inj_val['io']
            hypo_testing.h0_maker.select_params(
                init_args_d['h0_param_selections']
            )
            hypo_testing.h1_maker.select_params(
                init_args_d['h1_param_selections']
            )
            if np.sign(hypo_testing.data_maker.params[
                    test_name].value.magnitude) == 1:
                hypo_testing.data_maker.params[test_name].value = inj_val['no']
            else:
                hypo_testing.data_maker.params[test_name].value = inj_val['io']
        # This is easy if there's just one of them
        else:
            # Make sure the units are right
            inj_val = inj_val.to(hypo_testing.h0_maker.params[test_name].units)
            # Then set the value in all of the makers
            hypo_testing.h0_maker.params[test_name].value = inj_val
            hypo_testing.h1_maker.params[test_name].value = inj_val
            hypo_testing.data_maker.params[test_name].value = inj_val
        # Make names reflect parameter value
        if param_name == 'deltam3l':
            hypo_testing.labels = Labels(
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']+'_%s_%.4f'
                %(param_name,requested_val*1000.0),
                data_is_data=init_args_d['data_is_data'],
                fluctuate_data=init_args_d['fluctuate_data'],
                fluctuate_fid=init_args_d['fluctuate_fid']
            )
        else:
            hypo_testing.labels = Labels(
                h0_name=init_args_d['h0_name'],
                h1_name=init_args_d['h1_name'],
                data_name=init_args_d['data_name']+'_%s_%.4f'
                %(param_name,requested_val),
                data_is_data=init_args_d['data_is_data'],
                fluctuate_data=init_args_d['fluctuate_data'],
                fluctuate_fid=init_args_d['fluctuate_fid']
            )
        # Setup logging and things.
        hypo_testing.setup_logging(reset_params=False)
        hypo_testing.write_config_summary(reset_params=False)
        hypo_testing.write_minimizer_settings()
        hypo_testing.write_run_info()
        # Now do the fits
        hypo_testing.generate_data()
        hypo_testing.fit_hypos_to_data()
        hypo_testing.produce_fid_data()
        hypo_testing.fit_hypos_to_fid()
        # At the end, reset the parameters in the maker
        hypo_testing.data_maker.params.reset_free()
        hypo_testing.h0_maker.params.reset_free()
        hypo_testing.h1_maker.params.reset_free()
        # Also be sure to remove the data_dist and tor_data_asimov_dist
        # so that it is regenerated next time
        hypo_testing.data_dist = None
        hypo_testing.toy_data_asimov_dist = None


if __name__ == '__main__':
    main()
