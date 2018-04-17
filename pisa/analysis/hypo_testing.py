#!/usr/bin/env python

"""
Hypothesis testing: How do two hypotheses compare for describing MC or data?

This script/module can run either Asimov or LLR analyses. See
`hypo_testing_postprocess.py` to derive significances, etc. from the files
logged by this script.
"""


from __future__ import absolute_import, division


from collections import Counter, Mapping, OrderedDict, Sequence
from copy import copy
import getpass
from itertools import chain, product
import os
import random
import re
import socket
import string
import sys
import time
from traceback import format_exception

import numpy as np

from pisa import ureg, _version, __version__
from pisa.analysis.analysis import Analysis
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import MapSet
from pisa.core.prior import get_prior_bounds
from pisa.utils.comparisons import normQuant
from pisa.utils.config_parser import parse_fit_config, parse_minimizer_config
from pisa.utils.fileio import from_file, get_valid_filename, mkdir, to_file
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging
from pisa.utils.random_numbers import get_random_state
from pisa.utils.resources import find_resource
from pisa.utils.scripting import normcheckpath
from pisa.utils.stats import ALL_METRICS
from pisa.utils.format import timediff, timestamp


__all__ = ['Labels', 'HypoTesting']

__author__ = 'J.L. Lanfranchi, P.Eller, S. Wren, T. Ehrhardt'

__license__ = '''Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''

ALL_MAKER_NAMES = ('h0', 'h1', 'data')


def validate_maker_names(maker_names, require_all=False):
    """Ensure that maker_names contains only names in ALL_MAKER_NAMES and has
    no duplicates.
    """
    if isinstance(maker_names, basestring):
        maker_names = [maker_names]
    # count entry has to disappear exactly
    test = Counter(maker_names)
    ref = Counter(ALL_MAKER_NAMES)
    found_extra = test - ref
    missing_some = ref - test
    if found_extra:
        raise ValueError(
            'Only allowed distribution maker names are: %s. Found'
            ' extra entries: %s.' % (ALL_MAKER_NAMES, found_extra.keys())
        )
    if require_all and missing_some:
        raise ValueError(
            ' Missing distribution maker names: %s.' % missing_some.keys()
        )
    return

def setup_makers_from_pipelines(init_args_d, ref_maker_names):
    """Either setup all distribution makers from the pipeline settings for
    a single one (reference) or set them all up individually.
    """
    validate_maker_names(ref_maker_names, require_all=False)
    if len(ref_maker_names) == 1:
        # just assign the "pipeline" to the reference maker and all other makers
        ref_ind = ALL_MAKER_NAMES.index(ref_maker_names[0])
        ref_maker_name = ALL_MAKER_NAMES[ref_ind]
        other_maker_names = [maker_name for maker_name in ALL_MAKER_NAMES
                             if maker_name != ref_maker_name]
        filenames = init_args_d.pop('pipeline')
        filenames = sorted(
            [normcheckpath(fname) for fname in filenames]
        )
        init_args_d[ref_maker_name+'_maker'] = DistributionMaker(filenames)
        for maker_name in other_maker_names:
            init_args_d[maker_name+'_maker'] = DistributionMaker(filenames)
    else:
        # here we have several makers, so check whether there are any
        # filenames for a given one
        for maker_name in ref_maker_names:
            try:
                mp = maker_name + '_pipeline'
                filenames = init_args_d.pop(mp)
            except KeyError:
                raise KeyError(
                    'Could not find a pipeline entry "%s" in the dict.' % mp
                )
            except:
                raise
            if filenames is not None:
                filenames = sorted(
                    [normcheckpath(fname) for fname in filenames]
                )
                init_args_d[maker_name + '_maker'] = DistributionMaker(filenames)
            else:
                logging.warn(
                    'No filenames for "%s". Setting it up with `None`.'
                    % (maker_name + '_maker')
                )
                init_args_d[maker_name + '_maker'] = None
                """
                raise ValueError(
                    'Only found "None" pipeline settings for "%s",'
                    ' so do not know how to setup a corresponding'
                    ' DistributionMaker.' % maker_name
                )
                """

def collect_maker_selections(init_args_d, maker_names):
    """Collect and process selections from all of the given maker names. An
    entry needs to exist for each of the maker names (but it can be None).
    If 'data' is among the names and there's no selection, the selection from
    'h0' will be applied.
    """
    validate_maker_names(maker_names, require_all=False)
    for maker_name in maker_names:
        ps_name = maker_name + '_param_selections'
        try:
            ps_str = init_args_d[ps_name]
        except KeyError, e:
            raise KeyError('Could not find the param selection entry "%s" in'
                           ' the init args dict.' % ps_name)
        except:
            raise
        if ps_str is None:
            ps_list = None
        else:
            ps_list = [x.strip().lower() for x in ps_str.split(',')]
        init_args_d[ps_name] = ps_list

def select_maker_params(init_args_d, maker_names):
    """Call `select_params` on DistributionMaker objects in `init_args_d`
    with prefixes listed in `maker_names`. Fails if a corresponding param
    selection entry is not found (after trying to fall back to 'h0' selection
    in case of 'data'). Needs to be called after `collect_maker_selections`.
    """
    validate_maker_names(maker_names, require_all=False)
    for maker_name in maker_names:
        ps_name = maker_name + '_param_selections'
        try:
            ps = init_args_d[ps_name]
        except KeyError:
            # cannot tolerate the missing ps_name entry
            raise KeyError('Could not find the param selection entry for the'
                           ' distribution maker "%s" in the dict, so'
                           ' `select_params` cannot be called.' % maker_name)
        except:
            raise
        if isinstance(init_args_d[maker_name+'_maker'], DistributionMaker):
            init_args_d[maker_name+'_maker'].select_params(ps)
        elif init_args_d[maker_name+'_maker'] is None:
            continue

class Labels(object):
    """Derive file labels and naming scheme for data and directories produced
    by the HypoTesting class.

    """
    def __init__(self, h0_name, h1_name, data_name, data_is_data,
                 fluctuate_data, fluctuate_fid,
                 fluctuate_data_method=None, fluctuate_fid_method=None):
        self.h0_name = get_valid_filename(h0_name).lower()
        self.h1_name = get_valid_filename(h1_name).lower()
        self.data_name = get_valid_filename(data_name).lower()
        self.data_is_data = data_is_data
        self.fluctuate_data = fluctuate_data
        self.fluctuate_data_method = fluctuate_data_method
        self.fluctuate_fid = fluctuate_fid
        self.fluctuate_fid_method = fluctuate_fid_method
        self._construct_names()

    def _construct_names(self):
        self.hypo_prefix = 'hypo'

        if self.h0_name == '':
            self.h0 = self.hypo_prefix # pylint: disable=invalid-name
        else:
            self.h0 = '%s_%s' %(self.hypo_prefix, self.h0_name)

        if self.h1_name == '':
            self.h1 = self.hypo_prefix # pylint: disable=invalid-name
        else:
            self.h1 = '%s_%s' %(self.hypo_prefix, self.h1_name)

        if self.data_is_data:
            self.data_prefix = 'data'
            self.data_suffix = ''
        else:
            self.data_prefix = 'toy'
            if self.fluctuate_data:
                self.data_suffix = 'pseudodata_' + self.fluctuate_data_method
            else:
                self.data_suffix = 'asimov'

        self.generic_data = self.data_prefix
        self.data = self.data_prefix
        self.data_disp = self.data_prefix
        if self.data_name != '':
            self.data += '_' + self.data_name
            self.generic_data += '_' + self.data_name
            self.data_disp += ' ' + self.data_name
        if self.data_suffix != '':
            self.data += '_' + self.data_suffix
            self.data_disp += ' ' + self.data_suffix

        if self.fluctuate_fid:
            self.fid_disp = 'fiducial pseudodata'
            self.fid = 'fid_pseudodata_' + self.fluctuate_fid_method
        else:
            self.fid_disp = 'fiducial Asimov'
            self.fid = 'fid_asimov'

        # Fits to data
        self.h0_fit_to_data = '{h0}_fit_to_{data}'.format(**self.dict)
        self.h1_fit_to_data = '{h1}_fit_to_{data}'.format(**self.dict)

        for x, y in product(*[['0', '1']]*2):
            varname = 'h{x}_fit_to_h{y}_fid_'.format(x=x, y=y)
            basestr = ('{h%s}_fit_to_{h%s}_{fid}'%(x, y)).format(**self.dict)
            self.dict[varname+'base'] = basestr
            if self.fluctuate_fid:
                self.dict[varname+'re'] = re.compile(basestr + '_' +
                                                     r'(?P<fid_ind>[0-9]+)')
            else:
                self.dict[varname+'re'] = re.compile(basestr)

            # There're *always* fits performed to fid asimov
            #self.

        # Directory naming pattern
        if self.fluctuate_data:
            self.subdir_re = re.compile(self.data + '_(?P<data_ind>[0-9]+)')
        else:
            self.subdir_re = re.compile(self.data)

    def derive_fid_fits_names(self, fid_ind=None):
        """Fiducial fit names"""
        # Define file name labels
        if self.fluctuate_fid:
            ind_sfx = '_%d' %fid_ind
        else:
            ind_sfx = ''

        for x, y in product(*[['0', '1']]*2):
            dst_varname = 'h{x}_fit_to_h{y}_fid'.format(x=x, y=y)
            src_varname = dst_varname + '_base'
            self.dict[dst_varname] = self.dict[src_varname]  + ind_sfx

    @property
    def dict(self):
        return self.__dict__


class HypoTesting(Analysis):
    """Tools for testing two hypotheses against one another.

    Note that duplicated `*_maker` specifications are _not_ instantiated
    separately, but instead are re-used for all duplicate definitions.
    `*_param_selections` allows for this reuse, whereby sets of parameters
    infixed with the corresponding param_selectors can be switched among to
    simulate different physics using the same DistributionMaker (e.g.,
    switching between h0 and h1 hypotheses).


    Parameters
    ----------
    logdir : string
        Base directory in which to create the directory that stores all
        results. Note that `logdir` will be (recursively) generated if it does
        not exist.

    data_is_data : bool
        Whether the distributions fit to are from real data or not

    fluctuate_data : bool
        Whether to apply random fluctuations to expected toy data from MC

    fluctuate_fid : bool
        Whether to apply random fluctuations to fiducial distributions
        (if applicable)

    fluctuate_data_method : str
        Method according to which toy data distributions are to be fluctuated.
        Choice needs to be recognized by `Map`'s `fluctuate` method.

    fluctuate_fid_method : str
        Method according to which fiducial distributions are to be fluctuated.
        Choice needs to be recognized by `Map`'s `fluctuate` method.

    minimizer_settings : string
        Minimizer settings file or resource path. These will be processed
        internally and passed to the appropriate `scipy.optimize` backend.

    data_maker : None, DistributionMaker or instantiable thereto
        Data maker specification, or None (specify an already-generated data
        distribution with `data_dist`).

    data_param_selections : None, string, or sequence of strings
        Param selections to use for data, or None to accept any param
        selections already made in `data_maker`.

    data_name : None or string
        Name for data distribution. If None, a name is auto-generated.

    data_dist : None, MapSet or instantiable thereto
        Specify an existing distribution as the data distribution instead of a
        generating a new one. Use this instead of `data_maker`.

    h0_name : None or string
        Name for hypothesis 0. If None, a name is auto-generated.

    h0_maker : DistributionMaker or instantiable thereto
        Hypothesis-0-maker specification.

    h0_param_selections : None, string, or sequence of strings
        Param selections to use for hypothesis 0, or None to accept any param
        selections already made in `h0_maker`.

    h0_fid_asimov_dist : None, MapSet or instantiable thereto
        TODO: this parameter is NOT currently used, but is intended to remove
        requirement to re-generate this distribution if it's already been
        generated in a previous run

    h1_name : None or string
        Name for hypothesis 1. If None, a name is auto-generated.

    h1_maker : None, DistributionMaker or instantiable thereto
        Hypothesis-1-maker specification. If None, `h0_maker` is used also for
        hypothesis 1 (in this case, only specifying
        `h1_param_selections` will let h0 and h1 come out to be different).

    h1_param_selections : None, string, or sequence of strings
        Param selections to use for hypothesis 1, or None to accept any param
        selections already made in `h1_maker`.

    h1_fid_asimov_dist : None, MapSet or instantiable thereto
        TODO: this parameter is NOT currently used, but is intended to remove
        requirement to re-generate this distribution if it's already been
        generated in a previous run

    extra_param_selections : None, or sequence of strings
        Extra parameter selections to optimize over in each fit. May not be
        part of any of the regular (h0, h1) selections

    force_fits : bool
        Force performing of all fits (even if data and hypo maker the same)

    num_data_trials : int >= 1
        Number of (pseudo)data trials to run. For each trial, a new pseudodata
        distribution is generated, and then all subsequent fits are preformed.
        Note that data trials recorded to disk are not duplicated in subsequent
        runs (assuming the same `logdir` is specified for each run).

    num_fid_trials : int >= 1
        Number of fiducial-fit trials to run. For each trial, a new fluctuated
        fiducial distribution is generated, and then fits are preformed to
        that. Note that fiducial trials recorded to disk are not duplicated in
        subsequent runs (assuming the same `logdir` is specified for each run).

    data_start_ind : int >= 0 but < 2**12
        Start data trials at this index. Valid indexes begin with 0. The final
        data trial index is (data_start_ind + num_data_trials - 1). Any data
        trials already recorded to disk will be skipped.

    fid_start_ind : int >= 0 but < 2**12
        Start fiducial trials at this index. Valid indexes begin with 0. The
        final fiducial trial index is (fid_start_ind + num_fid_trials - 1). Any
        fiducial trials already recorded to disk will be skipped.

    check_octant : bool
        If True and theta23 is a free parameter, minimization is performed once
        starting wtih theta23 = theta23.nominal_value and then a second and
        possibly a third time starting with theta23 in the octant opposite from
        its current/nominal value (depending on setting of `reset_free`). The
        best overall fit from all of these is recorded.

    metric : string
        Metric for minimizer to use for comparing distributions. Valid metrics
        are defined by `pisa.utils.stats.ALL_METRICS`.

    other_metrics : None, string, or sequence of strings
        Other metric to record to compare distributions. These are not used by
        the minimizer. Valid metrics are defined by
        `pisa.utils.stats.ALL_METRICS`.

    blind : bool
        Set to True to run a blind analysis, whereby free parameter values are
        hidden from terminal display and are removed before storing to log
        files.

    allow_dirty : bool
        !USE WITH CAUTION! Allow for running code despite a "dirty" git
        repository (i.e., files in the repository have been changed but not
        committed). Setting to True is dangerous since this might result in
        irreproducible results.

    allow_no_git_info : bool
        !USE WITH CAUTION! Allow for running without knowing git version
        information. Setting to True is dangerous since this might result in
        irreproducible results.

    pprint : bool
        If True, display fit information as a single line on the terminal that
        updates in-place as the fit proceeds. If False, this information is
        output as a separate line for each iteration.

    reset_free : bool
        If True, before each new fit the free hypothesis parameters are reset
        to their nominal values. Otherwise, each fit will start off from
        whatever state the parameters in the hypothesis maker are in.


    Notes
    -----
    LLR analysis is a very thorough (and computationally expensive) method to
    compare discrete hypotheses. In general, a total of

        num_data_trials * (2 + 4*num_fid_trials)

    fits must be performed (and note that for each fit, many distributions
    (typically dozens or even hundreds) must be generated).

    If the "data" used in the analysis is pseudodata (i.e., `data_maker` uses
    Monte Carlo to produce its distributions, and these are then
    fluctuated--`fluctuate_data` is True and `fluctuate_data_method` is set to
    a valid method), then `num_data_trials` should be as large as is
    computationally feasible.

    Likewise, if the fiducial-fit data is to be pseudodata (i.e.,
    `fluctuate_fid` is True and `fluctuate_fid_method` is set to a valid method
    and regardless if `data_maker` uses Monte Carlo), `num_fid_trials` should
    be as large as computationally feasible.

    Typical analyses include the following:
        * Asimov analysis of data: `data_maker` uses (actual, measured) data
          and both `fluctuate_data` and `fluctuate_fid` are False.
        * Pseudodata analysis of data: `data_maker` uses (actual, measured)
          data, `fluctuate_data` is False, and `fluctuate_fid` is True.
        * Asimov analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid` are False.
        * Pseudodata analysis of Monte Carlo: `data_maker` uses Monte Carlo to
          produce its distributions and both `fluctuate_data` and
          `fluctuate_fid` are False.


    References
    ----------
    TODO


    Examples
    --------
    TODO

    """
    def __init__(self, logdir,
                 data_is_data, fluctuate_data, fluctuate_fid, metric,
                 fluctuate_data_method=None, fluctuate_fid_method=None,
                 other_metrics=None, fit_settings=None, minimizer_settings=None,
                 h0_name=None, h0_maker=None, h0_param_selections=None, h0_fid_asimov_dist=None,
                 h1_name=None, h1_maker=None, h1_param_selections=None, h1_fid_asimov_dist=None,
                 data_name=None, data_maker=None, data_param_selections=None, data_dist=None,
                 extra_param_selections=None, force_fits=False,
                 num_data_trials=1, num_fid_trials=1,
                 data_start_ind=0, fid_start_ind=0,
                 check_octant=True,
                 allow_dirty=False, allow_no_git_info=False,
                 blind=False, store_minimizer_history=True, pprint=False,
                 reset_free=True):
        super(HypoTesting, self).__init__()

        assert num_data_trials >= 0
        assert num_fid_trials >= 0
        assert data_start_ind >= 0
        assert fid_start_ind >= 0
        assert metric in ALL_METRICS

        # Make it possible to seed minimiser off truth
        self.reset_free = reset_free
        self.force_fits = force_fits

        # Instantiate h0 distribution maker to ensure it is a valid spec
        if h0_maker is None:
            raise ValueError('`h0_maker` must be specified (and not None)')
        if not isinstance(h0_maker, DistributionMaker):
            h0_maker = DistributionMaker(h0_maker)
        # Convert empty strings to None, otherwise put the string in a list
        if isinstance(h0_param_selections, basestring):
            h0_param_selections = h0_param_selections.strip().lower()
            if h0_param_selections == '':
                h0_param_selections = None
            else:
                h0_param_selections = [h0_param_selections]
        if isinstance(h1_param_selections, basestring):
            h1_param_selections = h1_param_selections.strip().lower()
            if h1_param_selections == '':
                h1_param_selections = None
            else:
                h1_param_selections = [h1_param_selections]
        if isinstance(extra_param_selections, basestring):
            extra_param_selections = extra_param_selections.strip().lower()
            if extra_param_selections == '':
                extra_param_selections = None
            else:
                extra_param_selections = [extra_param_selections]
        if isinstance(data_param_selections, basestring):
            data_param_selections = data_param_selections.strip().lower()
            if data_param_selections == '':
                data_param_selections = None
            else:
                data_param_selections = [h0_param_selections]

        # empty lists become None also
        if (isinstance(h0_param_selections, Sequence)
                and not h0_param_selections):
            h0_param_selections = None
        if (isinstance(h1_param_selections, Sequence)
                and not h1_param_selections):
            h1_param_selections = None
        if (isinstance(extra_param_selections, Sequence)
                and not extra_param_selections):
            extra_param_selections = None
        if (isinstance(data_param_selections, Sequence)
                and not data_param_selections):
            data_param_selections = None

        # Cannot specify either of `data_maker` or `data_param_selections` if
        # `data_dist` is supplied.
        if data_dist is not None:
            assert data_maker is None
            assert data_param_selections is None
            assert num_data_trials == 1
            if isinstance(data_dist, basestring):
                data_dist = from_file(data_dist)
            if not isinstance(data_dist, MapSet):
                data_dist = MapSet(data_dist)

        # If analyzing actual data, fluctuations should not be applied to the
        # data_dist (fluctuating fiducial-fits Asimov dist is still fine,
        # though).
        if data_is_data and fluctuate_data:
            raise ValueError('Adding fluctuations to actual data distribution'
                             ' is invalid.')

        if fluctuate_data and num_data_trials == 0:
            raise ValueError(
                '`fluctuate_data` is True but `num_data_trials` is set to 0!'
            )
        # Ensure num_{fid_}data_trials is one if fluctuate_{fid_}data is False
        if not fluctuate_data and num_data_trials > 1:
            logging.warn(
                'More than one data trial is unnecessary because'
                ' `fluctuate_data` is False (i.e., all `num_data_trials` data'
                ' distributions will be identical). Forcing `num_data_trials`'
                ' to 1.'
            )
            num_data_trials = 1

        if fluctuate_fid and num_fid_trials == 0:
            raise ValueError(
                '`fluctuate_fid` is True but `num_fid_trials` is set to 0!'
            )
        if not fluctuate_fid and num_fid_trials > 1:
            logging.warn(
                'More than one fid trial is unnecessary because'
                ' `fluctuate_fid` is False (i.e., all'
                ' `num_fid_trials` data distributions will be identical).'
                ' Forcing `num_fid_trials` to 1.'
            )
            num_fid_trials = 1

        if fluctuate_data:
            if fluctuate_data_method is None:
                raise ValueError(
                    '`fluctuate_data` is True but no `fluctuate_data_method`'
                    ' is specified!'
                )
            if not isinstance(fluctuate_data_method, basestring):
                raise TypeError('`fluctuate_data_method` must be a string!')

        if fluctuate_fid:
            if fluctuate_fid_method is None:
                raise ValueError(
                    '`fluctuate_fid` is True but no `fluctuate_fid_method`'
                    ' is specified!'
                )
            if not isinstance(fluctuate_fid_method, basestring):
                raise TypeError('`fluctuate_fid_method` must be a string!')

        # Identify duplicate `*_maker` specifications
        self.h1_maker_is_h0_maker = False
        if h1_maker is None or h1_maker == h0_maker:
            logging.info('h1_maker is identical to h0_maker.')
            self.h1_maker_is_h0_maker = True

        self.data_maker_is_h0_maker = False
        if data_maker is None or data_maker == h0_maker:
            self.data_maker_is_h0_maker = True

        self.data_maker_is_h1_maker = False
        if data_maker == h1_maker:
            self.data_maker_is_h1_maker = True

        # If no data maker settings AND no data param selections are provided,
        # assume that data param selections are to come from hypo h0
        if data_maker is None and data_param_selections is None:
            data_param_selections = h0_param_selections

        # If no h1 maker settings AND no h1 param selections are
        # provided, then we really can't proceed since h0 and h1 will be
        # identical in every way and there's nothing of substance to be done.
        # TE: I disagree
        """
        if h1_maker is None and h1_param_selections is None:
            raise ValueError(
                'Hypotheses h0 and h1 to be generated will use the same'
                ' distribution maker configured the same way, leading to'
                ' trivial behavior. If you wish for this behavior, you'
                ' must explicitly specify `h1_maker` and/or'
                ' `h1_param_selections`.'
            )
        """
        # Instantiate distribution makers only where necessary (otherwise copy)
        if not isinstance(h1_maker, DistributionMaker):
            if self.h1_maker_is_h0_maker:
                h1_maker = h0_maker
            else:
                h1_maker = DistributionMaker(h1_maker)

        # Cannot know if data came from same dist maker if we're given the data
        # distribution directly
        if data_dist is not None:
            self.data_maker_is_h0_maker = False
            self.data_maker_is_h1_maker = False
        # Otherwise instantiate or copy the data dist maker
        else:
            if not isinstance(data_maker, DistributionMaker):
                if self.data_maker_is_h0_maker:
                    data_maker = h0_maker
                elif self.data_maker_is_h1_maker:
                    data_maker = h1_maker
                else:
                    data_maker = DistributionMaker(data_maker)

        # Only allow for either minimizer settings or fit settings to be
        # provided, but not both.
        if minimizer_settings and fit_settings:
            raise ValueError('Please provide either a minimizer settings'
                             ' or a fit settings argument, but not both!')

        # Read in minimizer settings
        if isinstance(minimizer_settings, basestring):
            minimizer_settings = parse_minimizer_config(minimizer_settings)
        assert isinstance(minimizer_settings, Mapping) or minimizer_settings is None

        # Read in fit settings
        if isinstance(fit_settings, basestring):
            fit_settings = parse_fit_config(fit_settings)
        assert isinstance(fit_settings, Mapping) or fit_settings is None

        # Read in and validate extra param selections
        if extra_param_selections is not None:
            assert isinstance(extra_param_selections, Sequence)
            for selection in extra_param_selections:
                for regular_param_selections in\
                    (self.h0_param_selections, self.h1_param_selections):
                    if isinstance(regular_param_selections, Sequence):
                        if selection in regular_param_selections:
                            raise ValueError(
                                'Extra parameter selection "%s" also found in'
                                ' regular parameter selections "%s"!'
                                % (selection, regular_param_selections)
                            )

        # Store variables to `self` for later access
        self.logdir = logdir
        self.minimizer_settings = minimizer_settings
        self.fit_settings = fit_settings
        self.check_octant = check_octant
        self.extra_param_selections = extra_param_selections

        self.h0_maker = h0_maker
        self.h0_param_selections = h0_param_selections

        self.h1_maker = h1_maker
        self.h1_param_selections = h1_param_selections

        self.is_single_hypo_test = (
            self.h1_maker_is_h0_maker and
            self.h0_param_selections == self.h1_param_selections
        )

        self.data_is_data = data_is_data
        self.data_maker = data_maker
        self.data_param_selections = data_param_selections

        self.metric = metric
        self.other_metrics = other_metrics
        self.fluctuate_data = fluctuate_data
        self.fluctuate_data_method = fluctuate_data_method
        self.fluctuate_fid = fluctuate_fid
        self.fluctuate_fid_method = fluctuate_fid_method

        self.num_data_trials = num_data_trials
        self.num_fid_trials = num_fid_trials
        self.data_start_ind = data_start_ind
        self.fid_start_ind = fid_start_ind

        self.data_ind = self.data_start_ind
        self.fid_ind = self.fid_start_ind

        self.allow_dirty = allow_dirty
        self.allow_no_git_info = allow_no_git_info

        self.blind = blind
        self.store_minimizer_history = store_minimizer_history
        self.pprint = pprint

        # Storage for most recent Asimov (un-fluctuated) distributions
        self.toy_data_asimov_dist = None
        self.h0_fid_asimov_dist = h0_fid_asimov_dist
        self.h1_fid_asimov_dist = h1_fid_asimov_dist

        # Storage for most recent "data" (either un-fluctuated--if Asimov
        # analysis being run or if actual data is being used--or fluctuated--if
        # pseudodata is being generated) data
        self.data_dist = data_dist
        self.h0_fid_dist = None
        self.h1_fid_dist = None

        # Storage for most recent fiducial fit parameters
        self.h0_fit_to_h0_fid = None
        self.h0_fit_to_h1_fid = None
        self.h1_fit_to_h0_fid = None
        self.h1_fit_to_h1_fid = None

        # Populate names with user-specified strings or try to make intelligent
        # guesses based on configuration
        if h0_name is None:
            if self.h0_param_selections is not None:
                h0_name = ','.join(self.h0_param_selections)
            else:
                h0_name = 'h0'

        if h1_name is None:
            if (self.h1_maker == self.h0_maker
                    and self.h1_param_selections == self.h0_param_selections):
                h1_name = h0_name
            elif self.h1_param_selections is not None:
                h1_name = ','.join(self.h1_param_selections)
            else:
                h1_name = 'h1'

        if data_name is None:
            if (self.data_maker == self.h0_maker
                    and self.data_param_selections == self.h0_param_selections):
                data_name = h0_name
            elif (self.data_maker == self.h1_maker
                  and self.data_param_selections == self.h1_param_selections):
                data_name = h1_name
            elif self.data_param_selections is not None:
                data_name = ','.join(self.data_param_selections)
            else:
                data_name = ''

        self.labels = Labels(
            h0_name=h0_name, h1_name=h1_name,
            data_name=data_name, data_is_data=self.data_is_data,
            fluctuate_data=self.fluctuate_data,
            fluctuate_fid=self.fluctuate_fid,
            fluctuate_data_method=self.fluctuate_data_method,
            fluctuate_fid_method=self.fluctuate_fid_method
        )

    def run_analysis(self):
        """Run the defined analysis.

        Progress and estimated time remaining is written to stdout/stderr, and
        results are logged to an appropriate directory within `self.logdir`.

        """
        logging.info('Running LLR analysis.')
        self.analysis_start_time = time.time()

        self.setup_logging()
        self.write_config_summary()
        self.write_minimizer_settings()
        self.write_fit_settings()
        self.write_run_info()

        t0 = time.time()
        try:
            # Loop for multiple (if fluctuated) data distributions
            for self.data_ind in xrange(self.data_start_ind,
                                        self.data_start_ind
                                        + self.num_data_trials):
                data_trials_complete = self.data_ind-self.data_start_ind
                pct_data_complete = (
                    100.*(data_trials_complete)/self.num_data_trials
                )
                logging.info(
                    'Working on %s set ID %d (will stop after ID %d).'
                    ' %0.2f%s of %s sets completed.',
                    self.labels.data_disp,
                    self.data_ind,
                    self.data_start_ind+self.num_data_trials-1,
                    pct_data_complete,
                    '%',
                    self.labels.data_disp
                )

                self.generate_data()
                self.fit_hypos_to_data()

                # Loop for multiple (if fluctuated) fiducial data distributions
                for self.fid_ind in xrange(self.fid_start_ind,
                                           self.fid_start_ind
                                           + self.num_fid_trials):
                    fid_trials_complete = self.fid_ind-self.fid_start_ind
                    pct_fid_dist_complete = (
                        100*(fid_trials_complete)/self.num_fid_trials
                    )

                    dt = time.time() - t0
                    total_complete = (self.num_fid_trials*data_trials_complete
                                      + fid_trials_complete)
                    trials_to_go = (self.num_data_trials*self.num_fid_trials
                                    - total_complete)

                    ts_remaining = '???'
                    if total_complete > 0:
                        sec_per_fid = dt / total_complete
                        time_to_go = sec_per_fid * trials_to_go
                        ts_remaining = timediff(time_to_go, sec_decimals=0,
                                                hms_always=True)

                    logging.info(
                        ('Working on {data_disp} set ID %d / {fid_disp} set ID'
                         ' %d. %d trials to go, est time remaining: %s'
                         %(self.data_ind, self.fid_ind, trials_to_go,
                           ts_remaining)).format(**self.labels.dict)
                    )

                    self.produce_fid_data()
                    self.fit_hypos_to_fid()
        except: # pylint: disable=bare-except
            exc = sys.exc_info()
        else:
            exc = (None, None, None)
        finally:
            if exc[0] is not None:
                logging.error('`run_analysis` body failed with exception:')
                for line in format_exception(*exc):
                    for sl in line.splitlines():
                        logging.error(' '*4 + sl)

            try:
                self.write_run_stop_info(exc=exc)
            except: # pylint: disable=bare-except
                exc_l = sys.exc_info()
            else:
                exc_l = (None, None, None)

            if exc_l[0] is not None:
                logging.error('`write_run_stop_info` failed with exception:')
                for line in format_exception(*exc_l):
                    for sl in line.splitlines():
                        logging.error(' '*4 + sl)
                raise exc_l[0], exc_l[1], exc_l[2]

            if exc[0] is not None:
                raise exc[0], exc[1], exc[2]

    def generate_data(self):
        """Generate "data" distribution"""
        logging.info('Generating %s distributions.', self.labels.data_disp)
        # Ambiguous whether we're dealing with Asimov or regular data if the
        # data set is provided for us, so just return it.
        if self.num_data_trials == 1 and self.data_dist is not None:
            return self.data_dist

        # Dealing with data: No such thing as Asimov
        if self.data_is_data:
            if self.data_dist is None:
                self.data_maker.select_params(self.data_param_selections)
                self.data_dist = self.data_maker.get_outputs(return_sum=True)
                self.h0_fit_to_data = None
                self.h1_fit_to_data = None
            return self.data_dist

        # Otherwise: Toy data (MC)...

        # Produce Asimov dist if we don't already have it
        if self.toy_data_asimov_dist is None:
            self.data_maker.select_params(self.data_param_selections)
            self.toy_data_asimov_dist = (
                self.data_maker.get_outputs(return_sum=True)
            )
            self.h0_fit_to_data = None
            self.h1_fit_to_data = None

        if self.fluctuate_data:
            assert self.data_ind is not None
            # Random state for data trials is defined by:
            #   * data vs fid-dist = 0  : data part (outer loop)
            #   * data trial = data_ind : data trial number (use same for data
            #                             and and fid data trials, since on the
            #                             same data trial)
            #   * fid trial = 0         : always 0 since data stays the same
            #                             for all fid trials in this data trial
            data_random_state = get_random_state([0, self.data_ind, 0])

            self.data_dist = self.toy_data_asimov_dist.fluctuate(
                method=self.fluctuate_data_method, random_state=data_random_state
            )

        else:
            self.data_dist = self.toy_data_asimov_dist

        return self.data_dist

    # TODO: use hashes to ensure fits aren't repeated that don't have to be?
    def fit_hypos_to_data(self):
        """Fit both hypotheses to "data" to produce fiducial Asimov
        distributions from *each* of the hypotheses. (i.e., two fits are
        performed unless redundancies are detected).

        """
        # Setup directory for logging results
        self.thisdata_dirpath = self.data_dirpath
        if self.fluctuate_data:
            self.thisdata_dirpath += '_' + format(self.data_ind, 'd')
        mkdir(self.thisdata_dirpath)

        # If h0 maker is same as data maker, we know the fit will end up with
        # the data maker's params. Set these param values and record them.
        if (not self.data_is_data and self.data_maker_is_h0_maker
                and self.h0_param_selections == self.data_param_selections
                and not self.fluctuate_data and not self.force_fits):
            logging.info('Hypo %s will reproduce exactly %s distributions; not'
                         ' running corresponding fit.',
                         self.labels.h0_name, self.labels.data_disp)

            self.data_maker.select_params(self.data_param_selections)
            self.data_maker.reset_free()
            # FIXME:
            self.h0_fit_to_data = self.nofit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.data_maker,
                hypo_param_selections=self.data_param_selections,
                hypo_asimov_dist=self.toy_data_asimov_dist,
                metric=self.metric, other_metrics=self.other_metrics,
                blind=self.blind
            )

        # Otherwise, we do have to do the fit.
        else:
            logging.info('Fitting hypo %s to %s distributions.',
                         self.labels.h0_name, self.labels.data_disp)
            self.h0_fit_to_data = self.optimize_discrete_selections(
                data_dist=self.data_dist,
                hypo_maker=self.h0_maker,
                hypo_param_selections=self.h0_param_selections,
                extra_param_selections=self.extra_param_selections,
                metric=self.metric,
                other_metrics=self.other_metrics,
                fit_settings=self.fit_settings,
                minimizer_settings=self.minimizer_settings,
                check_octant=self.check_octant,
                pprint=self.pprint,
                blind=self.blind,
                reset_free=self.reset_free,
                return_full_scan=False, #FIXME
            )[0]
        self.h0_fid_asimov_dist = self.h0_fit_to_data['hypo_asimov_dist']

        self.log_fit(fit_info=self.h0_fit_to_data,
                     dirpath=self.thisdata_dirpath,
                     label=self.labels.h0_fit_to_data)

        if (not self.data_is_data and self.data_maker_is_h1_maker
                and self.h1_param_selections == self.data_param_selections
                and not self.fluctuate_data and not self.force_fits):
            logging.info('Hypo %s will reproduce exactly %s distributions; not'
                         ' running corresponding fit.',
                         self.labels.h1_name, self.labels.data_disp)

            self.data_maker.select_params(self.data_param_selections)
            self.data_maker.reset_free()
            # FIXME:
            self.h1_fit_to_data = self.nofit_hypo(
                data_dist=self.data_dist,
                hypo_maker=self.h1_maker,
                hypo_param_selections=self.h1_param_selections,
                hypo_asimov_dist=self.toy_data_asimov_dist,
                metric=self.metric, other_metrics=self.other_metrics,
                blind=self.blind
            )
        elif self.is_single_hypo_test:
            logging.info('No fit of h1 to data needs to be done (h1=h0).')
            self.h1_fit_to_data = copy(self.h0_fit_to_data)
        else:
            logging.info('Fitting hypo %s to %s distributions.',
                         self.labels.h1_name, self.labels.data_disp)
            self.h1_fit_to_data = self.optimize_discrete_selections(
                data_dist=self.data_dist,
                hypo_maker=self.h1_maker,
                hypo_param_selections=self.h1_param_selections,
                extra_param_selections=self.extra_param_selections,
                metric=self.metric,
                other_metrics=self.other_metrics,
                fit_settings=self.fit_settings,
                minimizer_settings=self.minimizer_settings,
                check_octant=self.check_octant,
                pprint=self.pprint,
                blind=self.blind,
                reset_free=self.reset_free,
                return_full_scan=False, #FIXME
                )[0]
            self.h1_fid_asimov_dist = self.h1_fit_to_data['hypo_asimov_dist']

            self.log_fit(fit_info=self.h1_fit_to_data,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h1_fit_to_data)

    def produce_fid_data(self):
        """Generate fiducial distribution"""
        logging.info('Generating %s distributions.', self.labels.fid_disp)
        # Retrieve event-rate maps for best fit to data with each hypo

        if self.fluctuate_fid:
            # Random state for data trials is defined by:
            #   * data vs fid-dist = 1     : fid data part (inner loop)
            #   * data trial = data_ind    : data trial number (use same for
            #                                data and and fid data trials,
            #                                since on the same data trial)
            #   * fid trial = fid_ind      : always 0 since data stays the same
            #                                for all fid trials in this data
            #                                trial
            fid_random_state = get_random_state([1, self.data_ind,
                                                 self.fid_ind])

            # Fluctuate h0 fid Asimov
            self.h0_fid_dist = self.h0_fid_asimov_dist.fluctuate(
                method=self.fluctuate_fid_method,
                random_state=fid_random_state
            )
            # The state of `random_state` will be moved forward now as compared
            # to what it was upon definition above. This is the desired
            # behavior, so the *exact* same random state isn't used to
            # fluctuate h1 as was used to fluctuate h0.
            self.h1_fid_dist = self.h1_fid_asimov_dist.fluctuate(
                method=self.fluctuate_fid_method,
                random_state=fid_random_state
            )
        else:
            self.h0_fid_dist = self.h0_fid_asimov_dist
            self.h1_fid_dist = self.h1_fid_asimov_dist

        return self.h1_fid_dist, self.h0_fid_dist

    def fit_hypos_to_fid(self):
        """Fit hypotheses to fiducial distribution"""
        self.labels.derive_fid_fits_names(fid_ind=self.fid_ind)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h0_fit_to_h0_fid + '.json.bz2')
        if not os.path.isfile(fpath):
            # If fid isn't fluctuated, it's redundant to fit a hypo to a dist
            # it generated
            self.h0_maker.select_params(self.h0_param_selections)
            self.h0_maker.reset_free()
            if not self.fluctuate_fid and not self.force_fits:
                logging.info(
                    'Hypo %s %s is not fluctuated; fitting this hypo to its'
                    ' own %s distributions is unnecessary.',
                    self.labels.h0_name, self.labels.fid_disp,
                    self.labels.fid_disp
                )
                self.h0_fit_to_h0_fid = self.nofit_hypo(
                    data_dist=self.h0_fid_dist,
                    hypo_params=self.h0_maker.params,
                    hypo_asimov_dist=self.h0_fid_asimov_dist,
                    metric=self.metric, other_metrics=self.other_metrics,
                    blind=self.blind
                )
            else:
                logging.info('Fitting hypo %s to its own %s distributions.',
                             self.labels.h0_name, self.labels.fid_disp)
                self.h0_fit_to_h0_fid = self.optimize_discrete_selections(
                    data_dist=self.h0_fid_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    extra_param_selections=self.extra_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    fit_settings=self.fit_settings,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind,
                    reset_free=self.reset_free,
                    return_full_scan=False, #FIXME
                )[0]
            self.log_fit(fit_info=self.h0_fit_to_h0_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h0_fit_to_h0_fid)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h1_fit_to_h1_fid + '.json.bz2')
        if not os.path.isfile(fpath):
            self.h1_maker.select_params(self.h1_param_selections)
            self.h1_maker.reset_free()
            if not self.fluctuate_fid and not self.force_fits:
                logging.info(
                    'Hypo %s %s is not fluctuated; fitting this hypo to its'
                    ' own %s distributions is unnecessary.',
                    self.labels.h1_name, self.labels.fid_disp,
                    self.labels.fid_disp
                )
                self.h1_fit_to_h1_fid = self.nofit_hypo(
                    data_dist=self.h1_fid_dist,
                    hypo_params=self.h1_maker.params,
                    hypo_asimov_dist=self.h1_fid_asimov_dist,
                    metric=self.metric, other_metrics=self.other_metrics,
                    blind=self.blind
                )
            else:
                logging.info('Fitting hypo %s to its own %s distributions.',
                             self.labels.h1_name, self.labels.fid_disp)
                self.h1_fit_to_h1_fid = self.optimize_discrete_selections(
                    data_dist=self.h1_fid_dist,
                    hypo_maker=self.h1_maker,
                    hypo_param_selections=self.h1_param_selections,
                    extra_param_selections=self.extra_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    fit_settings=self.fit_settings,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind,
                    reset_free=self.reset_free,
                    return_full_scan=False, #FIXME
                    )[0]

            self.log_fit(fit_info=self.h1_fit_to_h1_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h1_fit_to_h1_fid)

        # TODO: remove redundancy if h0 and h1 are identical
        # Perform fits of one hypo to fid dist produced by other hypo

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h1_fit_to_h0_fid + '.json.bz2')
        if not os.path.isfile(fpath):
            if ((not self.fluctuate_data) and (not self.fluctuate_fid)
                    and self.data_maker_is_h0_maker and not self.force_fits
                    and self.h0_param_selections == self.data_param_selections):
                logging.info(
                    'Fitting hypo %s to hypo %s %s distributions is'
                    ' unnecessary since former was already fit to %s'
                    ' distributions, which are identical distributions.',
                    self.labels.h1_name, self.labels.h0_name,
                    self.labels.fid_disp, self.labels.data_disp
                )
                self.h1_fit_to_h0_fid = copy(self.h1_fit_to_data)
            else:
                logging.info('Fitting hypo %s to hypo %s %s distributions.',
                             self.labels.h1_name, self.labels.h0_name,
                             self.labels.fid_disp)
                self.h1_maker.select_params(self.h1_param_selections)
                self.h1_maker.reset_free()
                self.h1_fit_to_h0_fid = self.optimize_discrete_selections(
                    data_dist=self.h0_fid_dist,
                    hypo_maker=self.h1_maker,
                    hypo_param_selections=self.h1_param_selections,
                    extra_param_selections=self.extra_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    fit_settings=self.fit_settings,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind,
                    reset_free=self.reset_free,
                    return_full_scan=False, #FIXME
                )[0]

            self.log_fit(fit_info=self.h1_fit_to_h0_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h1_fit_to_h0_fid)

        fpath = os.path.join(self.thisdata_dirpath,
                             self.labels.h0_fit_to_h1_fid + '.json.bz2')
        if not os.path.isfile(fpath):
            if ((not self.fluctuate_data) and (not self.fluctuate_fid)
                    and self.data_maker_is_h1_maker and not self.force_fits
                    and self.h1_param_selections == self.data_param_selections):
                logging.info(
                    'Fitting hypo %s to hypo %s %s distributions is'
                    ' unnecessary since former was already fit to %s'
                    ' distributions, which are identical distributions.',
                    self.labels.h0_name, self.labels.h1_name,
                    self.labels.fid_disp, self.labels.data_disp
                )
                self.h0_fit_to_h1_fid = copy(self.h0_fit_to_data)
            else:
                logging.info('Fitting hypo %s to hypo %s %s distributions.',
                             self.labels.h0_name, self.labels.h1_name,
                             self.labels.fid_disp)
                self.h0_maker.select_params(self.h0_param_selections)
                self.h0_maker.reset_free()
                self.h0_fit_to_h1_fid = self.optimize_discrete_selections(
                    data_dist=self.h1_fid_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    extra_param_selections=self.extra_param_selections,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    fit_settings=self.fit_settings,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind,
                    reset_free=self.reset_free,
                    return_full_scan=False,
                )[0]

            self.log_fit(fit_info=self.h0_fit_to_h1_fid,
                         dirpath=self.thisdata_dirpath,
                         label=self.labels.h0_fit_to_h1_fid)

    def setup_logging(self, reset_params=True):
        """
        Should store enough information for the following two purposes:
            1. Be able to completely reproduce the results, assuming access to
               the same git repository.
            2. Be able to easily identify (as a human) the important / salient
               features of this config that might make it different from
               another.

        `config_hash` is generated by creating a list of the following and
        hashing that list:
            * git sha256 for latest commit (will not run if this info isn't
              present or cannot be ascertained or if code is updated since last
              commit and `unsafe_run` is True)
            * hash of instantiated `minimizer_settings` object (sent through
              normQuant)
            * `check_octant`
            * pipelines info for each used for each hypo:
                - stage name, service name, service source code hash
            * name of metric used for minimization

        config_summary.info : Human-readable metadata used to construct hash:
            * config_hash : str
            * source_provenance : dict
                - git_commit_sha256 : str
                - git_repo (?) : str
                - git_branch : str
                - git_remote_url : str
                - git_tag : str
            * minimizer_info : dict
                - minimizer_config_hash : str
                - minimizer_name : str
                - metric_minimized : str
                - check_octant : bool
            * data_is_data : bool
            * data_pipelines : list
                - p0 : list
                    - s0 : dict
                        - stage name : str
                        - service name : str
                        - service source code hash : str
                    - s1 : dict
                    ...
                - p1 : list
                ...
            * data_param_selections
            * h0_pipelines (similarly to data pipelines)
            * h0_param_selections
            * h1_pipelines (list containing list per pipeline)
            * h1_param_selections

        minimizer_settings.cfg : copy of the minimizer settings used

        run_info_<datetime in microseconds, UTC>_<hostname>.info
            * fluctuate_data : bool
            * fluctuate_data_method : str
            * fluctuate_fid : bool
            * fluctuate_fid_method : str
            * data_start_ind (if toy pseudodata)
            * num_data_trials (if toy pseudodata)
            * fid_start_ind (if fid fits to pseudodata)
            * num_fid_trials (if fid fits to pseudodata)

        h0_pipeline0.cfg
        h0_pipeline1.cfg
        ...
        h1_pipeline0.cfg
        h1_pipeline1.cfg
        ...

        Directory Structure
        -------------------
        The base directory for storing data unique to this configuration is

            basedir = logdir/hypo_<h0_name>__hypo_<h1_name>_<config_hash>

        where `config_hash` is derived from the full configuration and is
        independent of `h0_name` and `h1_name` since the latter two entities
        are user-provided and can vary while yielding the same configuration.

        Within the base directory, if we're actually working with data
        (`data_is_data` is True), the following directory is created:

            <basedir>/data_fits

        If "data" actually comes from MC (i.e., `data_is_data` is False), a
        directory

            <basedir>/toy_pseudodata_fits<data_ind>

        is created for each `data_ind` if fluctuations are applied to produce
        pseudodata for fitting to. Otherwise if no fluctuations are applied to
        the toy data distribtuion for fitting to, the directory

            <basedir>/toy_asimov_fits

        is created.

        Files
        -----
        In order to record the full configuration

            /fid<fid_ind>/
            {toy_}data_fits{<data_ind>}/fid<fid_ind>/

        Create or update the files:
            logdir/h0_<h0_name>__h1_<h1_name>/reservations.sqlite
            logdir/h0_<h0_name>__h1_<h1_name>/run_info_<datetime>_<hostname>.info

        run_id comes from (??? hostname and microsecond timestamp??? settings???)

        Notes
        -----
        I have added an optional reset_params option here since I want to be
        able to use this function when doing systematic tests. Here, the
        parameters in the makers may be temporarily changed and this should be
        reflected in the logging. Thus, this option allows the parameters to
        NOT be reset here. This option should be used with caution.

        """
        self.h0_maker.select_params(self.h0_param_selections)
        if reset_params:
            self.h0_maker.reset_free()
        self.h0_hash = self.h0_maker.hash

        self.h1_maker.select_params(self.h1_param_selections)
        if reset_params:
            self.h1_maker.reset_free()
        self.h1_hash = self.h1_maker.hash

        self.data_maker.select_params(self.data_param_selections)
        if reset_params:
            self.data_maker.reset_free()
        self.data_hash = self.data_maker.hash

        # Single unique hash for hypotheses and data configurations
        self.config_hash = hash_obj([self.h0_hash, self.h1_hash,
                                     self.data_hash], hash_to='x')

        # Unique id string for settings related to minimization
        self.minimizer_settings_hash = hash_obj(
            normQuant(self.minimizer_settings), hash_to='x'
        )
        co = 'co1' if self.check_octant else 'co0'
        self.minsettings_flabel = (
            'min_' + '_'.join([self.minimizer_settings_hash, co, self.metric])
        )

        self.fit_settings_hash = hash_obj(
            normQuant(self.fit_settings), hash_to='x'
        )
        self.fitsettings_flabel = (
            'fit_' + '_'.join([self.fit_settings_hash, self.metric])
        )

        # Code versioning
        self.__version__ = __version__
        self.version_info = _version.get_versions()

        no_git_info = self.version_info['error'] is not None
        if no_git_info:
            msg = 'No info about git repo. Version info: %s' %self.version_info
            if self.allow_no_git_info:
                logging.warn(msg)
            else:
                raise Exception(msg)

        dirty_git_repo = self.version_info['dirty']
        if dirty_git_repo:
            msg = 'Dirty git repo. Version info: %s' %self.version_info
            if self.allow_dirty:
                logging.warn(msg)
            else:
                raise Exception(msg)

        logging.debug('Code version: %s', self.__version__)

        # Construct root dir name and create dir if necessary
        dirname = '__'.join([self.labels.h0, self.labels.h1,
                             self.labels.generic_data,
                             self.config_hash, self.minsettings_flabel,
                             self.fitsettings_flabel,
                             'pisa' + self.__version__])
        dirpath = os.path.join(self.logdir, dirname)
        # keep your fingers crossed this path doesn't become too long...
        mkdir(dirpath)
        normpath = find_resource(dirpath)
        self.logroot = normpath
        logging.info('Output will be saved to dir "%s"', self.logroot)

        self.data_dirpath = os.path.join(self.logroot, self.labels.data)

        # Filenames and paths
        self.config_summary_fname = 'config_summary.json'
        self.config_summary_fpath = os.path.join(self.logroot,
                                                 self.config_summary_fname)
        self.invocation_datetime = timestamp(utc=True, winsafe=True)
        self.hostname = socket.gethostname()
        chars = string.ascii_lowercase + string.digits
        self.random_suffix = ''.join([random.choice(chars) for _ in range(8)])
        self.pid = os.getpid()
        self.user = getpass.getuser()
        self.minimizer_settings_fpath = os.path.join(
            self.logroot, 'minimizer_settings.json'
        )
        self.fit_settings_fpath = os.path.join(
            self.logroot, 'fit_settings.json'
        )
        self.run_info_fname = (
            'run_%s_%s_%s.info'
            %(self.invocation_datetime, self.hostname, self.random_suffix)
        )
        self.run_info_fpath = os.path.join(self.logroot, self.run_info_fname)

    def write_config_summary(self, reset_params=True):
        if os.path.isfile(self.config_summary_fpath):
            return
        summary = OrderedDict()
        d = OrderedDict()
        d['version'] = self.version_info['version']
        d['git_revision_sha256'] = self.version_info['full-revisionid']
        d['git_dirty'] = self.version_info['dirty']
        d['git_error'] = self.version_info['error']
        summary['source_provenance'] = d

        d = OrderedDict()
        # TODO
        #d['minimizer_name'] = self.minimizer_settings['method']['value']
        d['minimizer_settings_hash'] = self.minimizer_settings_hash
        d['check_octant'] = self.check_octant
        d['metric_optimized'] = self.metric
        summary['minimizer_info'] = d
        d = OrderedDict()
        d['fit_settings_hash'] = self.fit_settings_hash
        #d['metric_optimized'] = self.metric
        summary['fit_settings'] = d

        summary['data_is_data'] = self.data_is_data

        self.data_maker.select_params(self.data_param_selections)
        if reset_params:
            self.data_maker.reset_free()
        summary['data_name'] = self.labels.data_name
        summary['data_is_data'] = self.data_is_data
        summary['data_hash'] = self.data_hash
        if not self.data_is_data:
            #TODO: ugly but necessary fix
            if self.data_param_selections is None:
                data_param_selections = ['None']
            else:
                data_param_selections = self.data_param_selections
            summary['data_param_selections'] = ','.join(
                data_param_selections)
        summary['data_params_hash'] = self.data_maker.params.hash
        summary['data_params'] = [str(p) for p in self.data_maker.params]
        summary['data_pipelines'] = self.summarize_dist_maker(self.data_maker)

        self.h0_maker.select_params(self.h0_param_selections)
        if reset_params:
            self.h0_maker.reset_free()
        summary['h0_name'] = self.labels.h0_name
        summary['h0_hash'] = self.h0_hash
        #TODO: ugly but necessary fix
        if self.h0_param_selections is None:
            h0_param_selections = ['None']
        else:
            h0_param_selections = self.h0_param_selections
        summary['h0_param_selections'] = ','.join(h0_param_selections)
        summary['h0_params_hash'] = self.h0_maker.params.hash
        summary['h0_params'] = [str(p) for p in self.h0_maker.params]
        summary['h0_pipelines'] = self.summarize_dist_maker(self.h0_maker)
        summary['is_single_hypo_test'] = self.is_single_hypo_test

        self.h1_maker.select_params(self.h1_param_selections)
        if reset_params:
            self.h1_maker.reset_free()
        summary['h1_name'] = self.labels.h1_name
        summary['h1_hash'] = self.h1_hash
        #TODO: ugly but necessary fix
        if self.h1_param_selections is None:
            h1_param_selections = ['None']
        else:
            h1_param_selections = self.h1_param_selections
        summary['h1_param_selections'] = ','.join(h1_param_selections)
        summary['h1_params_hash'] = self.h1_maker.params.hash
        summary['h1_params'] = [str(p) for p in self.h1_maker.params]
        summary['h1_pipelines'] = self.summarize_dist_maker(self.h1_maker)
        if self.extra_param_selections is None:
            extra_param_selections = ['None']
        else:
            extra_param_selections = self.extra_param_selections
        summary['extra_param_selections'] = ','.join(extra_param_selections)

        # Reverse the order so it serializes to a file as intended
        # (want top-to-bottom file convention vs. fifo streaming data
        # convention)
        od = OrderedDict()
        for ok, ov in summary.items():
            if isinstance(ov, OrderedDict):
                od1 = OrderedDict()
                for ik, iv in ov.items():
                    od1[ik] = iv
                ov = od1
            od[ok] = ov

        to_file(od, self.config_summary_fpath, sort_keys=False)

    @staticmethod
    def summarize_dist_maker(dist_maker):
        pipeline_info = []
        for pipeline in dist_maker:
            stage_info = OrderedDict()
            for stage in pipeline:
                k = ':'.join([stage.stage_name, stage.service_name,
                              str(stage.hash)])
                d = OrderedDict()
                for attr in ['input_binning', 'output_binning']:
                    if (hasattr(stage, attr)
                            and getattr(stage, attr) is not None):
                        d[attr] = str(getattr(stage, attr))
                stage_info[k] = d
            pipeline_info.append(stage_info)
        return pipeline_info

    def write_run_info(self):
        run_info = []
        run_info.append('invocation_datetime = %s' %self.invocation_datetime)
        run_info.append('hostname = %s' %self.hostname)
        run_info.append('random_suffix = %s' %self.random_suffix)
        run_info.append('pid = %s' %self.pid)
        run_info.append('user = %s' %self.user)

        run_info.append('logdir = %s' %self.logdir)

        run_info.append('fluctuate_data = %s' %self.fluctuate_data)
        run_info.append('fluctuate_fid = %s' %self.fluctuate_fid)
        if self.fluctuate_data:
            run_info.append('data_start_ind = %d' %self.data_start_ind)
            run_info.append('num_data_trials = %d' %self.num_data_trials)
            run_info.append('fluctuate_data_method = %s' %self.fluctuate_data_method)
        if self.fluctuate_fid:
            run_info.append('fid_start_ind = %d' %self.fid_start_ind)
            run_info.append('num_fid_trials = %d' %self.num_fid_trials)
            run_info.append('fluctuate_fid_method = %s' %self.fluctuate_fid_method)
        run_info.append('metric = %s' %self.metric)
        run_info.append('other_metrics = %s' %self.other_metrics)
        run_info.append('blind = %s' %self.blind)
        run_info.append('allow_dirty = %s' %self.allow_dirty)
        run_info.append('allow_no_git_info = %s' %self.allow_no_git_info)
        run_info.append('store_minimizer_history = %s'
                        %self.store_minimizer_history)
        run_info.append('pprint = %s' %self.pprint)
        for env_var in ['PISA_FTYPE', 'PISA_RESOURCES',
                        'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                        'CUDA_VISIBLE_DEVICES',
                        'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']:
            if env_var in os.environ:
                val = os.environ[env_var]
            else:
                val = ''
            run_info.append('%s = %s' %(env_var, val))

        for prefix in ['PBS_', 'SLURM_']:
            for env_var, val in os.environ.iteritems():
                if env_var.startswith(prefix):
                    run_info.append('%s = %s' %(env_var, val))

        with file(self.run_info_fpath, 'w') as f:
            f.write('\n'.join(run_info) + '\n')
        logging.info('Run info written to: ' + self.run_info_fpath)

    def write_minimizer_settings(self):
        if not os.path.isfile(self.minimizer_settings_fpath):
            to_file(self.minimizer_settings, self.minimizer_settings_fpath)
        return

    def write_fit_settings(self):
        if not os.path.isfile(self.fit_settings_fpath):
            to_file(self.fit_settings, self.fit_settings_fpath)
        return

    def write_run_stop_info(self, exc=None):
        if isinstance(exc, Sequence):
            if exc[0] is None:
                exc = None

        self.stop_datetime = timestamp(utc=True, winsafe=True)
        self.stop_time = time.time()
        self.analysis_runtime = self.stop_time - self.analysis_start_time
        dt_stamp = timediff(self.analysis_runtime, hms_always=True,
                            sec_decimals=0)

        run_info = []
        run_info.append('stop_datetime = %s' %self.stop_datetime)
        run_info.append('analysis_runtime = %s' %dt_stamp)
        if self.fluctuate_data:
            run_info.append('data_stop_ind = %d' %self.data_ind)
        if self.fluctuate_fid:
            run_info.append('fid_stop_ind = %d' %self.fid_ind)
        if exc is None:
            run_info.append('completed = True')
            run_info.append('exception = None')
            run_info.append('traceback = None')
        else:
            run_info.append('completed = False')
            run_info.append('exception = %s: %s' % (exc[0], exc[1]))
            tb = format_exception(*exc)
            formatted_tb = ('\n' + ' '*2).join(
                chain.from_iterable((l.splitlines() for l in tb))
            )
            run_info.append('traceback = %s' %formatted_tb)

        with file(self.run_info_fpath, 'a') as f:
            f.write('\n'.join(run_info) + '\n')

        logging.info('Run stop info written to: ' + self.run_info_fpath)
        logging.info('Total analysis run time: ' + dt_stamp)

    def log_fit(self, fit_info, dirpath, label):
        serialize = ['metric', 'metric_val', 'params', 'fit_time',
                     'detailed_metric_info', #'fit_metadata',
                     'num_distributions_generated', 'hypo_asimov_dist',
                     ]
        if self.store_minimizer_history:
            serialize.append('fit_history')

        info = OrderedDict()
        for k, v in fit_info.iteritems():
            if k not in serialize:
                continue
            if k == 'params':
                d = OrderedDict()
                for param in v: # record *all* hypo parameter values
                    d[param.name] = str(param.value)
                v = d
            if k == 'fit_metadata':
                if 'hess_inv' in v:
                    try:
                        v['hess_inv'] = v['hess_inv'].todense()
                    except AttributeError:
                        v['hess_inv'] = v['hess_inv']
            if isinstance(v, ureg.Quantity):
                v = str(v)
            info[k] = v
        to_file(info, os.path.join(dirpath, label + '.json.bz2'),
                sort_keys=False)

    def set_param_ranges(self, selection, test_name, rangetuple, inj_units):
        """Give the parameter in hypo_testing selected by selection
        (if not None) with name test_name a range defined by
        rangetuple. This should have the correct units even if the
        rangetuple units did not match those of the original parameter
        and also will stay positive if the original range did.

        Parameters
        ----------
        selection : string or None
            Parameter selection e.g. nh or ih.

        test_name : string
            Parameter name e.g. theta23.

        rangetuple : tuple
            Tuple for the parameter range.

        inj_units : string
            Units for this parameter as defined in the config file, so
            the tuple can be converted if needed.

        """
        if selection is not None:
            self.h0_maker.select_params([selection])
            self.h1_maker.select_params([selection])
        if self.h0_maker.params[test_name].range is not None:
            enforce_positive = self.h0_maker.params[test_name].range[0].m >= 0
        else:
            enforce_positive = False
        # Convert the units if necessary
        if self.h0_maker.params[test_name].units != inj_units:
            newminrangeval = rangetuple[0].to(
                self.h0_maker.params[test_name].units
            )
            newmaxrangeval = rangetuple[1].to(
                self.h0_maker.params[test_name].units
            )
            rangetuple = (newminrangeval, newmaxrangeval)
        # Make the lower end equal to zero if it needs to be
        if enforce_positive and rangetuple[0].m < 0:
            newminrangeval = 0.0 * ureg(inj_units)
            newminrangeval = newminrangeval.to(
                self.h0_maker.params[test_name].units
            )
            rangetuple = (newminrangeval, rangetuple[1])
        # no need to artificially decrease the ranges
        # (might make the nominal value fall outside of the new range,
        # meaning any call to `reset_free` will fail)
        h0_range = self.h0_maker.params[test_name].range
        self.h0_maker.params[test_name].range = \
            (min(rangetuple[0], h0_range[0]), max(rangetuple[1], h0_range[1]))
        h1_range = self.h1_maker.params[test_name].range
        self.h1_maker.params[test_name].range = \
            (min(rangetuple[0], h1_range[0]), max(rangetuple[1], h1_range[1]))
        data_range = self.data_maker.params[test_name].range
        # TODO: what is the logic behind this?
        if (selection is not None and
            (np.sign(self.data_maker.params[test_name].value.magnitude)
               == np.sign(rangetuple[1].magnitude))
            or selection is None):
                self.data_maker.params[test_name].range = \
                    (min(rangetuple[0], data_range[0]), max(rangetuple[1], data_range[1]))

    def do_asimov_fits(self):
        """Set up the logging and does an Asimov analysis. Used in the
        injected parameter scans and the systematic tests."""
        # Setup logging and things.
        self.setup_logging(reset_params=False)
        self.write_config_summary(reset_params=False)
        self.write_minimizer_settings()
        self.write_fit_settings()
        self.write_run_info()
        # Now do the fits
        self.generate_data()
        self.fit_hypos_to_data()
        self.produce_fid_data()
        self.fit_hypos_to_fid()

    def reset_makers(self, data=True, h0=True, h1=True):
        """Reset the makers. Set the booleans to false if you don't
        want to reset one or more of the makers. Used in the injected
        parameter scans and the systematic tests."""
        if data:
            self.data_maker.params.reset_free()
        if h0:
            self.h0_maker.params.reset_free()
        if h1:
            self.h1_maker.params.reset_free()

    def clear_data(self):
        """Clear the data distributions so that they are regenerated. This is
        needed for making multiple different data distributions (parameter
        scans, systematics tests) with the same hypo_testing object."""
        self.data_dist = None
        self.toy_data_asimov_dist = None

    def asimov_inj_param_scan(self, param_name, test_name, inj_vals,
                              requested_vals):
        """Perform the Asimov hypo testing analysis over some injected data
        parameter. This will be the parameter specified by test_name and the
        injected values are in inj_vals. The requested vals from the command
        line are also given for making labels for all of the output
        directories.

        Parameters
        ----------
        param_name : string
            The name of the parameter to do the scan over.

        test_name : string
            The name of the parameter as it is defined in the config files.
            This is used, for example, when the scan is over sin2theta23,
            but therefore it's a scan over theta23 in the config file.

        inj_vals : list
            The list of scan values to actually be used in the makers.

        requested_vals : list
            The list of scan values passed by the user. This may not be
            the same as inj_vals in cases where, for example, the units
            had to be changed or the scan is over some special variable
            such as sin2theta23.

        """
        # record so we don't keep appending to it
        data_name = self.labels.data_name
        # Scan over the injected values. We also loop over the requested vals
        # here in case they are different so that value can be put in labels
        for inj_val, requested_val in zip(inj_vals, requested_vals):
            # Be sure to inject the right value!
            if isinstance(inj_val, dict):
                for hierarchy in ['nh', 'ih']:
                    self.h0_maker.select_params([hierarchy])
                    self.h1_maker.select_params([hierarchy])
                    inj_val[hierarchy] = inj_val[hierarchy].to(
                        self.h0_maker.params[test_name].units
                    )
                    self.h0_maker.params[test_name].value = inj_val[hierarchy]
                    self.h1_maker.params[test_name].value = inj_val[hierarchy]
                    if np.sign(self.data_maker.params[
                            test_name].value.magnitude) == 1:
                        self.data_maker.params[test_name].value = inj_val['nh']
                    else:
                        self.data_maker.params[test_name].value = inj_val['ih']
            # This is easy if there's just one of them
            else:
                # Make sure the units are right
                inj_val = inj_val.to(self.h0_maker.params[test_name].units)
                # Then set the value in all of the makers
                self.h0_maker.params[test_name].value = inj_val
                self.h1_maker.params[test_name].value = inj_val
                self.data_maker.params[test_name].value = inj_val
            # Make names reflect parameter value
            if param_name == 'deltam3l':
                self.labels = Labels(
                    h0_name=self.labels.h0_name,
                    h1_name=self.labels.h1_name,
                    data_name=data_name+'_%s_%.4f'
                    %(param_name, requested_val*1000.0),
                    data_is_data=False,
                    fluctuate_data=False,
                    fluctuate_fid=False
                )
            else:
                self.labels = Labels(
                    h0_name=self.labels.h0_name,
                    h1_name=self.labels.h1_name,
                    data_name=data_name+'_%s_%.4f'
                    %(param_name, requested_val),
                    data_is_data=False,
                    fluctuate_data=False,
                    fluctuate_fid=False
                )
            # Setup logging and do the fits
            self.do_asimov_fits()
            # At the end, reset the parameters in the maker
            self.reset_makers()
            # Also be sure to remove the data_dist and toy_data_asimov_dist
            # so that they are regenerated next time
            self.clear_data()

    def asimov_nminusone_test(self, data_param, h0_name, h1_name, data_name):
        """This function will perform the standard N-1 test. This
        function expects h0_name, h1_name and data_name so that the
        labels can be redefined to make everything unique. It is also
        expected that this is used inside of a loop where data_param
        is one of the data params.

        Parameters
        ----------
        data_param : Param
            The param to be fixed in the test.

        *_name : string
            Same as they in HypoTesting.

        """
        self.labels = Labels(
            h0_name=h0_name + '_fixed_%s_baseline'%data_param.name,
            h1_name=h1_name + '_fixed_%s_baseline'%data_param.name,
            data_name=data_name,
            data_is_data=False,
            fluctuate_data=False,
            fluctuate_fid=False
        )
        # This is a standard N-1 test, so fix the parameter in the hypo makers.
        for h0_param in self.h0_maker.params.free:
            if h0_param.name == data_param.name:
                h0_param.is_fixed = True
        for h1_param in self.h1_maker.params.free:
            if h1_param.name == data_param.name:
                h1_param.is_fixed = True
        # Setup logging and do the fits
        self.do_asimov_fits()

    def sys_wrong_asimov_analysis(self, data_param, fit_wrong, direction,
                                  h0_name, h1_name, data_name):
        """This function will perform a modified version of the N-1 test. This
        differs in that here we do not assume the systematics take their
        baseline values but check what happens with something
        systematically wrong. So, the data_param is shifted by 1 sigma or 10%
        off baseline. The direction of this shift should be specified pve or
        nve in the direction argument (meaning positive or negative). Then one
        can allow the minimiser to correct for this by specifying fit_wrong. If
        this is false then the hypothesis maker will be fixed to the baseline
        in this parameter i.e. a systematically wrong hypothesis to what is
        injected. As with the N-1 test below it is assumed that this function
        exists inside of a loop over the parameters in the data_maker and this
        is for the systematic defined in data_param. This function also expects
        h0_name, h1_name and data_name so that the labels can be redefined to
        make everything unique.

        Parameters
        ----------
        data_param : Param
            The param for which a systematically wrong value will be injected.

        fit_wrong : bool
            Whether or not this param will be fitted for or fixed to
            the baseline (wrong) value.

        direction : string
            Either positive (pve) or negative (nve) and defines whether the
            systematically wrong value is higher or lower than the baseline.

        *_name : string
            Same as for HypoTesting.

        """
        def vary_param(data_param, tgt, direction):
            '''Helper function for setting the value of a parameter taking
            into account its allowed range.

            Parameters
            ----------
            data_param : param
            tgt : pint quantity
            direction : string

            '''
            if direction == 'pve':
                if tgt > data_param.range[1]:
                    logging.warn(
                        'Setting %s = %s not possible since it would exceed its'
                        ' allowed range. Will set it to its upper range limit.'
                        % (data_param.name, tgt)
                    )
                data_param.value = min(tgt, data_param.range[1])
            else:
                if tgt < data_param.range[0]:
                    logging.warn(
                        'Setting %s = %s not possible since it would fall below its'
                        ' allowed range. Will set it to its lower range limit.'
                        % (data_param.name, tgt)
                    )
                data_param.value = max(tgt, data_param.range[0])
            logging.debug(
                'Set parameter %s = %s.' % (data_param.name, data_param.value)
            )

        if direction not in ['pve', 'nve']:
            raise ValueError('Direction to shift systematic value must be'
                             ' specified either as "pve" or "nve" for'
                             ' positive and negative respectively')

        # set some default target parameter values - will be refined based on
        # the type of prior detected
        # when the param has a nominal value of 0, tentatively vary it by +/- 1.0
        if data_param.value == 0.0:
            if direction == 'pve':
                tgt = 1.0
            else:
                tgt = -1.0
        # else do 10%
        else:
            if direction == 'pve':
                tgt = 1.1 * data_param.value
            else:
                tgt = 0.9 * data_param.value
        # now check whether the prior allows us to calculate a +/- 1 sigma value
        if hasattr(data_param, 'prior') and data_param.prior is not None:
            # Gaussian priors are easy - just do 1 sigma
            if data_param.prior.kind == 'gaussian':
                if direction == 'pve':
                    tgt = data_param.value + data_param.prior.stddev
                else:
                    tgt = data_param.value - data_param.prior.stddev
            elif data_param.prior.kind == 'spline':
                # 1 sigma variation is also doable in this case
                bounds = get_prior_bounds(data_param.prior)
                if direction == 'pve':
                    if not len(bounds) > 1:
                        # ok, this is fishy, fall back to the 10% case
                        tgt = 1.1 * data_param.value
                    else:
                        tgt = bounds[-1]
                else:
                    if not len(bounds) > 1:
                        tgt = 0.9 * data_param.value
                    else:
                        tgt = bounds[0]
        # now try to set the parameter's value based on the target
        vary_param(
            data_param=data_param,
            tgt=tgt,
            direction=direction
        )
        # If we are not allowing the fit to correct for this, it must be
        # fixed in the hypo makers.
        if not fit_wrong:
            for h0_param in self.h0_maker.params.free:
                if h0_param.name == data_param.name:
                    h0_param.is_fixed = True
            for h1_param in self.h1_maker.params.free:
                if h1_param.name == data_param.name:
                    h1_param.is_fixed = True
        # Set up labels so that each file comes out unique
        if fit_wrong:
            self.labels = Labels(
                h0_name=h0_name,
                h1_name=h1_name,
                data_name='%s_inj_%s_%s_wrong'%(
                    data_name, data_param.name, direction),
                data_is_data=False,
                fluctuate_data=False,
                fluctuate_fid=False
            )
        else:
            self.labels = Labels(
                h0_name='%s_fixed_%s_baseline'%(h0_name, data_param.name),
                h1_name='%s_fixed_%s_baseline'%(h1_name, data_param.name),
                data_name='%s_inj_%s_%s_wrong'%(
                    data_name, data_param.name, direction),
                data_is_data=False,
                fluctuate_data=False,
                fluctuate_fid=False
            )
        # Setup logging and do the fits
        self.do_asimov_fits()

    def asimov_syst_tests(self, inject_wrong, fit_wrong, only_syst,
                          do_baseline, h0_name, h1_name, data_name):
        """The function which actually does the syst tests. The one that will
        actually be performed will be depending on whether inject_wrong is
        true or not.

        Parameters
        ----------
        inject_wrong : bool
            Whether the test is to inject a systematically wrong hypothesis
            or stick on the baseline.

        fit_wrong : bool
            Whether the test will allow this wrong value to be fitted.

        only_syst : list of strings
            Allows for only certain systematic tests to be done if the name
            is specified here. Useful if you need to quickly re-do just some
            of the tests.

        do_baseline : bool
            Whether to get the baseline significance or not. In general
            you want this to compare against since the impact of the
            systematics is only quantifiable relative to the baseline.
            However, this can be skipped to save time if you already
            have this value.

        *_name : string
            Same as for HypoTesting

        """
        if do_baseline:
            # Perform the baseline analysis so that the other results can
            # have a comparison line.
            self.labels = Labels(
                h0_name=h0_name,
                h1_name=h1_name,
                data_name=data_name + '_full_syst_baseline',
                data_is_data=False,
                fluctuate_data=False,
                fluctuate_fid=False
            )
            # Setup logging and do the fits
            self.do_asimov_fits()
            # Reset the makers
            self.reset_makers()
            # Also be sure to remove the data_dist and toy_data_asimov_dist
            # so that they are regenerated next time
            self.clear_data()
        else:
            logging.info('Baseline systematic fit will be skipped.')
        for data_param in self.data_maker.params.free:
            if only_syst is not None:
                do_test = data_param.name in only_syst
            else:
                do_test = True
            if do_test:
                if inject_wrong:
                    # First inject this wrong up by one sigma
                    logging.info(
                        'Injecting shifted up value for %s.' % data_param.name
                    )
                    self.sys_wrong_asimov_analysis(
                        data_param=data_param,
                        fit_wrong=fit_wrong,
                        direction='pve',
                        h0_name=h0_name,
                        h1_name=h1_name,
                        data_name=data_name
                    )
                    # At the end, reset the parameters in the maker
                    self.reset_makers()
                    # Data must be cleared or else it won't be regenerated
                    self.clear_data()
                    # Then inject this wrong down by one sigma
                    logging.info(
                        'Injecting shifted down value for %s.' % data_param.name
                    )
                    self.sys_wrong_asimov_analysis(
                        data_param=data_param,
                        fit_wrong=fit_wrong,
                        direction='nve',
                        h0_name=h0_name,
                        h1_name=h1_name,
                        data_name=data_name
                    )
                else:
                    # Just do the standard N-1 test
                    self.asimov_nminusone_test(
                        data_param=data_param,
                        h0_name=h0_name,
                        h1_name=h1_name,
                        data_name=data_name
                    )
                # At the end, reset the parameters in the maker
                self.reset_makers()
                # Also be sure to remove the data_dist and
                # toy_data_asimov_dist so they are regenerated next time
                self.clear_data()
                # Also unfix the hypo maker parameters
                for h0_param in self.h0_maker.params:
                    if h0_param.name == data_param.name:
                        h0_param.is_fixed = False
                for h1_param in self.h1_maker.params:
                    if h1_param.name == data_param.name:
                        h1_param.is_fixed = False

    def hypo_scan(self, param_names, scan_vals, profile, nuisance_params=None,
                  fix_params=None):
        assert not self.blind # deal with blindess
        if nuisance_params and not profile:
            raise ValueError(
                'Nuisance parameters specified, but "profile" is not set!'
            )
        if nuisance_params and fix_params:
            raise ValueError(
                'Both nuisance parameters and parameters-to-fix specified, but'
                ' these are mutually exclusive!'
            )
        if isinstance(param_names, basestring):
            param_names = [param_names]
        if isinstance(nuisance_params, basestring):
            nuisance_params = [nuisance_params]
        if isinstance(fix_params, basestring):
            fix_params = [fix_params]

        data_name = self.labels.data_name

        # cannot allow for any of the params to be scanned to be part of
        # the fit settings
        if self.fit_settings is not None:
            for method in self.fit_settings:
                for fit_pname in self.fit_settings[method]['params']:
                    if fit_pname in param_names:
                        raise ValueError(
                            'Parameter "%s" found in fit settings even though it'
                            ' is supposed to be scanned over. Please remove it'
                            ' from fit settings.' % fit_pname
                        )

        nparams = len(param_names)
        params = self.h0_maker.params
        # fix the parameters to be scanned - also make the parameter range
        # correspond to the (min, max) of scan values (these take
        # precedence over the preset ranges)
        params.fix(param_names)
        if nparams > 1:
            steplist = []
            for (i, pname) in enumerate(param_names):
                steplist.append([(pname, val) for val in scan_vals[i]])
                rangetuple = min(scan_vals[i]), max(scan_vals[i])
                params[pname].range = rangetuple
        else:
            pname = param_names[0]
            steplist = [[(pname, val) for val in scan_vals[0]]]
            params[pname].range = min(scan_vals[0]), max(scan_vals[0])

        if not profile:
            if fix_params:
                raise ValueError(
                    'Not specifying "profile" automatically fixes all'
                    ' parameters, but parameters to fix have been specified in'
                    ' addition. This seems like an undesired configuration.'
                )
            # explicitly fix all parameters
            for param in self.h0_maker.params.free:
                param.is_fixed = True

        # check which other parameters have to be fixed depending on arguments
        # passed in
        if nuisance_params:
            for nuis_param in nuisance_params:
                if nuis_param in param_names:
                    raise ValueError(
                        'Parameter "%s" cannot be both nuisance parameter and'
                        ' scan parameter simultaneously!' % nuis_param
                    )
                if not nuis_param in params.names:
                    raise ValueError(
                        'Parameter "%s" specified as nuisance parameter is not'
                        ' part of the set of parameters of the hypothesis'
                        ' maker!' % nuis_param
                    )
            # now unfix the selected nuisance parameters, since they might not
            # be among free ones
            self.h0_maker.params.unfix(nuisance_params)
            # now fix all free parameters not among nuisance parameters
            for param in self.h0_maker.params.free:
                if param.name not in nuisance_params:
                    params.fix(param.name)
        if fix_params and profile:
            for param_to_fix in fix_params:
                if not param_to_fix in params.names:
                    raise ValueError(
                        'Parameter "%s" specified as to be fixed is not'
                        ' part of the set of parameters of the hypothesis'
                        ' maker!' % param_to_fix
                    )
                params.fix(param_to_fix)

        t0 = time.time()
        results = {'scan_vals': {pname: [] for pname in param_names}, 'trials': []}

        # Setup logging and things.
        self.setup_logging()
        # doesn't seem to work if stages don't define attributes to hash?
        self.write_config_summary()
        self.write_minimizer_settings()
        self.write_fit_settings()
        self.write_run_info()

        # Loop for multiple (if fluctuated) data distributions
        for self.data_ind in xrange(self.data_start_ind,
                                    self.data_start_ind
                                    + self.num_data_trials):
            data_trials_complete = self.data_ind-self.data_start_ind
            pct_data_complete = (
                100.*(data_trials_complete)/self.num_data_trials
            )
            logging.info('Working on %s set ID %d (will stop after ID %d).'
                ' %0.2f%s of %s sets completed.',
                self.labels.data_disp,
                self.data_ind,
                self.data_start_ind+self.num_data_trials-1,
                pct_data_complete,
                '%',
                self.labels.data_disp
            )

            # Setup directory for logging results
            self.thisdata_dirpath = self.data_dirpath
            if self.fluctuate_data:
                self.thisdata_dirpath += '_' + format(self.data_ind, 'd')
            mkdir(self.thisdata_dirpath)

            self.generate_data()
            trial_results = {'data_dist': self.data_dist, 'results': []}
            for i, pos in enumerate(product(*steplist)):
                pos_msg = ''
                sep = ', '
                for (pname, val) in pos:
                    params[pname].value = val
                    if self.data_ind == self.data_start_ind:
                        results['scan_vals'][pname].append(val)
                    if isinstance(val, float) or isinstance(val, ureg.Quantity):
                        if pos_msg:
                            pos_msg += sep
                        pos_msg += '%s = %s'%(pname, val)
                    else:
                        # no need to raise an error in the case of blindness
                        raise TypeError(
                            "Value is of type %s which I don't know "
                            "how to deal with in the output "
                            "messages."% type(val)
                        )
                logging.info('Working on hypo point ' + pos_msg)
                self.h0_maker.update_params(params)

                self.labels = Labels(
                    h0_name=self.labels.h0_name,
                    h1_name='',
                    data_name=data_name+'_%s'
                    %(pos_msg),
                    data_is_data=self.data_is_data,
                    fluctuate_data=self.fluctuate_data,
                    fluctuate_fid=self.fluctuate_fid,
                    fluctuate_data_method=self.fluctuate_data_method,
                    fluctuate_fid_method=self.fluctuate_fid_method
                )

                # the no-profile case is handled internally
                self.h0_fit_to_data = self.optimize_discrete_selections(
                    data_dist=self.data_dist,
                    hypo_maker=self.h0_maker,
                    hypo_param_selections=self.h0_param_selections,
                    extra_param_selections=self.extra_param_selections,
                    fit_settings=self.fit_settings,
                    metric=self.metric,
                    other_metrics=self.other_metrics,
                    minimizer_settings=self.minimizer_settings,
                    check_octant=self.check_octant,
                    pprint=self.pprint,
                    blind=self.blind,
                    reset_free=self.reset_free,
                    # TODO: allow this
                    return_full_scan=False,
                )[0]

                self.log_fit(fit_info=self.h0_fit_to_data,
                             dirpath=self.thisdata_dirpath,
                             label=self.labels.h0_fit_to_data)
                trial_results['results'].append(self.h0_fit_to_data)
            results['trials'].append(trial_results)
            # At the end, reset the parameters in the maker
            self.reset_makers()
            # Also be sure to remove the data_dist and toy_data_asimov_dist
            # so that they are regenerated next time
            self.clear_data()
        return results
