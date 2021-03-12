"""
Common tools for performing an analysis collected into a single class
`Analysis` that can be subclassed by specific analyses.
"""


from collections.abc import Sequence
from collections import OrderedDict, Mapping
from copy import deepcopy
from operator import setitem
from itertools import product
import re
import sys
import time
import warnings

import numpy as np
import scipy.optimize as optimize
# this is needed for the take_step option in basinhopping
from scipy._lib._util import check_random_state

import pisa
from pisa import EPSILON, FTYPE, ureg
from pisa.core.map import Map, MapSet
from pisa.core.param import ParamSet
from pisa.utils.comparisons import recursiveEquality, FTYPE_PREC
from pisa.utils.log import logging
from pisa.utils.fileio import to_file
from pisa.utils.stats import METRICS_TO_MAXIMIZE, METRICS_TO_MINIMIZE


__all__ = ['MINIMIZERS_USING_SYMM_GRAD', 'MINIMIZERS_USING_CONSTRAINTS',
           'set_minimizer_defaults', 'validate_minimizer_settings',
           'Counter', 'Analysis']

__author__ = 'J.L. Lanfranchi, P. Eller, S. Wren, E. Bourbeau'

__license__ = '''Copyright (c) 2014-2020, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


MINIMIZERS_USING_SYMM_GRAD = ('l-bfgs-b', 'slsqp')
"""Minimizers that use symmetrical steps on either side of a point to compute
gradients. See https://github.com/scipy/scipy/issues/4916"""

MINIMIZERS_USING_CONSTRAINTS = ('cobyla')
"""Minimizers that cannot use the 'bounds' argument and instead need bounds to
be formulated in terms of constraints."""

def merge_mapsets_together(mapset_list=None):
    '''Handle merging of multiple MapSets, when they come in
    the shape of a dict

    '''

    if isinstance(mapset_list[0], Mapping):    
        new_dict = OrderedDict()
        for S in mapset_list:
            for k,v in S.items():

                if k not in new_dict.keys():
                    new_dict[k] = [m for m in v.maps]
                else:
                    new_dict[k] += [m for m in v.maps]

        for k,v in new_dict.items():
            new_dict[k] = MapSet(v)

    else:
        raise TypeError('This function only works when mapsets are provided as dicts')

    return new_dict


# TODO: add Nelder-Mead, as it was used previously...
def set_minimizer_defaults(minimizer_settings):
    """Fill in default values for minimizer settings.

    Parameters
    ----------
    minimizer_settings : dict

    Returns
    -------
    new_minimizer_settings : dict

    """
    new_minimizer_settings = dict(
        method=dict(value='', desc=''),
        options=dict(value=dict(), desc=dict())
    )
    new_minimizer_settings.update(minimizer_settings)

    sqrt_ftype_eps = np.sqrt(np.finfo(FTYPE).eps)
    opt_defaults = {}
    method = minimizer_settings['method']['value'].lower()

    if method == 'l-bfgs-b' and FTYPE == np.float64:
        # From `scipy.optimize.lbfgsb._minimize_lbfgsb`
        opt_defaults.update(dict(
            maxcor=10, ftol=2.2204460492503131e-09, gtol=1e-5, eps=1e-8,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'l-bfgs-b' and FTYPE == np.float32:
        # Adapted to lower precision
        opt_defaults.update(dict(
            maxcor=10, ftol=sqrt_ftype_eps, gtol=1e-3, eps=1e-5,
            maxfun=15000, maxiter=15000, iprint=-1, maxls=20
        ))
    elif method == 'slsqp' and FTYPE == np.float64:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-6, iprint=0, eps=sqrt_ftype_eps,
        ))
    elif method == 'slsqp' and FTYPE == np.float32:
        opt_defaults.update(dict(
            maxiter=100, ftol=1e-4, iprint=0, eps=sqrt_ftype_eps
        ))
    elif method == 'cobyla':
        opt_defaults.update(dict(
            rhobeg=0.1, maxiter=1000, tol=1e-4,
        ))
    else:
        raise ValueError('Unhandled minimizer "%s" / FTYPE=%s'
                         % (method, FTYPE))

    opt_defaults.update(new_minimizer_settings['options']['value'])

    new_minimizer_settings['options']['value'] = opt_defaults

    # Populate the descriptions with something
    for opt_name in new_minimizer_settings['options']['value']:
        if opt_name not in new_minimizer_settings['options']['desc']:
            new_minimizer_settings['options']['desc'] = 'no desc'

    return new_minimizer_settings


# TODO: add Nelder-Mead, as it was used previously...
def validate_minimizer_settings(minimizer_settings):
    """Validate minimizer settings.

    See source for specific thresholds set.

    Parameters
    ----------
    minimizer_settings : dict

    Raises
    ------
    ValueError
        If any minimizer settings are deemed to be invalid.

    """
    ftype_eps = np.finfo(FTYPE).eps
    method = minimizer_settings['method']['value'].lower()
    options = minimizer_settings['options']['value']
    if method == 'l-bfgs-b':
        must_have = ('maxcor', 'ftol', 'gtol', 'eps', 'maxfun', 'maxiter',
                     'maxls')
        may_have = must_have + ('args', 'jac', 'bounds', 'disp', 'iprint',
                                'callback')
    elif method == 'slsqp':
        must_have = ('maxiter', 'ftol', 'eps')
        may_have = must_have + ('args', 'jac', 'bounds', 'constraints',
                                'iprint', 'disp', 'callback')
    elif method == 'cobyla':
        must_have = ('maxiter', 'rhobeg', 'tol')
        may_have = must_have + ('disp', 'catol')

    missing = set(must_have).difference(set(options))
    excess = set(options).difference(set(may_have))
    if missing:
        raise ValueError('Missing the following options for %s minimizer: %s'
                         % (method, missing))
    if excess:
        raise ValueError('Excess options for %s minimizer: %s'
                         % (method, excess))

    eps_msg = '%s minimizer option %s(=%e) is < %d * %s_EPS(=%e)'
    eps_gt_msg = '%s minimizer option %s(=%e) is > %e'
    fp64_eps = np.finfo(np.float64).eps

    if method == 'l-bfgs-b':
        err_lim, warn_lim = 2, 10
        for s in ['ftol', 'gtol']:
            val = options[s]
            if val < err_lim * ftype_eps:
                raise ValueError(eps_msg % (method, s, val, err_lim, 'FTYPE',
                                            ftype_eps))
            if val < warn_lim * ftype_eps:
                logging.warning(eps_msg, method, s, val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, err_lim, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FTYPE', ftype_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)

    if method == 'slsqp':
        err_lim, warn_lim = 2, 10
        val = options['ftol']
        if val < err_lim * ftype_eps:
            raise ValueError(eps_msg % (method, 'ftol', val, err_lim, 'FTYPE',
                                        ftype_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'ftol', val, warn_lim, 'FTYPE', ftype_eps)

        val = options['eps']
        err_lim, warn_lim = 1, 10
        if val < err_lim * fp64_eps:
            raise ValueError(eps_msg % (method, 'eps', val, 1, 'FP64',
                                        fp64_eps))
        if val < warn_lim * ftype_eps:
            logging.warning(eps_msg, method, 'eps', val, warn_lim, 'FP64', fp64_eps)

        err_lim, warn_lim = 0.25, 0.1
        if val > err_lim:
            raise ValueError(eps_gt_msg % (method, 'eps', val, err_lim))
        if val > warn_lim:
            logging.warning(eps_gt_msg, method, 'eps', val, warn_lim)
    
    if method == 'cobyla':
        if options['rhobeg'] > 0.5:
            raise ValueError('starting step-size > 0.5 will overstep boundary')
        if options['rhobeg'] < 1e-2:
            logging.warning('starting step-size is very low, convergence will be slow')


def check_t23_octant(fit_info):
    """Check that theta23 is in the first or second octant.

    Parameters
    ----------
    fit_info

    Returns
    -------
    octant_index : int

    Raises
    ------
    ValueError
        Raised if the theta23 value is not in first (`octant_index`=0) or
        second octant (`octant_index`=1)

    """
    valid_octant_indices = (0, 1)

    theta23 = fit_info.metric_val.theta23.value
    octant_index = int(
        ((theta23 % (360 * ureg.deg)) // (45 * ureg.deg)).magnitude
    )
    if octant_index not in valid_octant_indices:
        raise ValueError('Fitted theta23 value is not in the'
                         ' first or second octant.')
    return octant_index



def get_separate_octant_params(hypo_maker, angle_name, inflection_point) :
    '''
    This function creates versions of the angle param that are confined to 
    a single octant. It does this for both octant cases. This is used to allow 
    fits to be done where only one of the octants is allowed. The fit can then 
    be done for the two octant cases and compared to find the best fit. 

    Parameters
    ----------
    hypo_maker : DistributionMaker or Detector
        The hypothesis maker being used by the fitter
    angle_name : string
        Name of the angle for which to create separate octant params.
    inflection_point : quantity
        Point distinguishing between the two octants, e.g. 45 degrees

    Returns
    -------
    angle_orig : Param
        angle param as it was before applying the octant separation
    angle_case1 : Param
        angle param confined to first octant
    angle_case2 : Param
        angle param confined to second octant
    '''

    # Reset angle before starting
    angle = hypo_maker.params[angle_name]
    angle.reset()

    # Store the original theat23 param before we mess with it
    angle_orig = deepcopy(angle)

    # Get the octant definition
    octants = (
        (angle.range[0], inflection_point) ,
        (inflection_point, angle.range[1])
        )

    # If angle is maximal (e.g. the transition between octants) or very close 
    # to it, offset it slightly to be clearly in one octant (note that fit can 
    # still move the value back to maximal). The reason for this is that 
    # otherwise checks on the parameter bounds (which include a margin for 
    # minimizer tolerance) can an throw exception.
    tolerance = 0.1 * ureg.degree
    dist_from_inflection = angle.value - inflection_point 
    if np.abs(dist_from_inflection) < tolerance :
        sign = -1. if dist_from_inflection < 0. else +1. # Note this creates +ve shift also for theta == 45 (arbitary)
        angle.value = inflection_point + (sign * tolerance)

    # Store the cases
    angle_case1 = deepcopy(angle)
    angle_case2 = deepcopy(angle)

    # Get case 1, e.g. the current octant
    case1_octant_index = 0 if angle_case1.value < inflection_point else 1
    angle_case1.range = octants[case1_octant_index]
    angle_case1.nominal_value = angle_case1.value

    # Also get case 2, e.g. the other octant
    case2_octant_index = 0 if case1_octant_index == 1 else 1
    angle_case2.value = 2*inflection_point - angle_case2.value
    # Also setting nominal value so that `reset_free` won't try to set it out of bounds
    angle_case2.nominal_value = angle_case2.value
    angle_case2.range = octants[case2_octant_index]

    return angle_orig, angle_case1, angle_case2


# TODO: move this to a central location prob. in utils
class Counter(object):
    """Simple counter object for use as a minimizer callback."""
    def __init__(self, i=0):
        self._count = i

    def __str__(self):
        return str(self._count)

    def __repr__(self):
        return str(self)

    def __iadd__(self, inc):
        self._count += inc

    def reset(self):
        """Reset counter"""
        self._count = 0

    @property
    def count(self):
        """int : Current count"""
        return self._count

class BoundedRandomDisplacement(object):
    """
    Add a bounded random displacement of maximum size `stepsize` to each coordinate
    Calling this updates `x` in-place.

    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    bounds : pair of float or sequence of pairs of float
        Bounds on x
    random_gen : {None, `np.random.RandomState`, `np.random.Generator`}
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize=0.5, bounds=(0, 1), random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)
        self.bounds = np.array(bounds).T

    def __call__(self, x):
        x += self.random_gen.uniform(-self.stepsize, self.stepsize,
                                     np.shape(x))
        x = np.clip(x, *self.bounds)  # bounds are automatically broadcast
        return x

class HypoFitResult(object):
    """Holds all relevant information about a fit result."""
    
    _state_attrs = ["metric", "metric_val", "params", "hypo_asimov_dist",
                    "detailed_metric_info", "minimizer_time",
                    "num_distributions_generated", "minimizer_metadata", "fit_history"]
    
    # TODO: initialize from serialized state
    def __init__(
        self,
        metric,
        metric_val,
        data_dist,
        hypo_maker,
        minimizer_time=None,
        num_distributions_generated=None,
        minimizer_metadata=None,
        fit_history=None,
        other_metrics=None,
        counter=None,
        include_detailed_metric_info=True,
    ):
        self.metric = metric
        self.metric_val = metric_val
        self.params = deepcopy(hypo_maker.params)
        # Record the distribution with the optimal param values
        self.hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
        self.detailed_metric_info = None
        if minimizer_time is not None:
            self.minimizer_time = minimizer_time * ureg.sec
        self.num_distributions_generated = num_distributions_generated
        self.minimizer_metadata = minimizer_metadata
        self.fit_history = fit_history
        
        if include_detailed_metric_info:
            if hypo_maker.__class__.__name__ == "Detectors":
                self.detailed_metric_info = [self.get_detailed_metric_info(
                    data_dist=data_dist[i], hypo_asimov_dist=self.hypo_asimov_dist[i],
                    params=hypo_maker._distribution_makers[i].params, metric=metric[i],
                    other_metrics=other_metrics, detector_name=hypo_maker.det_names[i]
                ) for i in range(len(data_dist))]
            else: # DistributionMaker object

                if 'generalized_poisson_llh' == metric[0]:
                    generalized_poisson_dist = hypo_maker.get_outputs(return_sum=False, force_standard_output=False)
                    generalized_poisson_dist = merge_mapsets_together(mapset_list=generalized_poisson_dist)
                else:
                    generalized_poisson_dist = None

                self.detailed_metric_info = self.get_detailed_metric_info(
                    data_dist=data_dist, hypo_asimov_dist=self.hypo_asimov_dist, generalized_poisson_hypo=generalized_poisson_dist,
                    params=hypo_maker.params, metric=metric[0], other_metrics=other_metrics,
                    detector_name=hypo_maker.detector_name
                )
    
    def __getitem__(self, i):
        if i in self._state_attrs:
            return getattr(self, i)
        else:
            raise ValueError(f"Unknown property {i}")
    # TODO: Load from serialized state
    @property
    def state(self):
        state = OrderedDict()
        for attr in self._state_attrs:
            val = getattr(self, attr)
            if hasattr(val, 'state'):
                val = val.state
            setitem(state, attr, val)
        return state

    @property
    def serializable_state(self):
        return self.state

    @staticmethod
    def get_detailed_metric_info(data_dist, hypo_asimov_dist, params, metric,generalized_poisson_hypo=None,
                                 other_metrics=None, detector_name=None):
        """Get detailed fit information, including e.g. maps that yielded the
        metric.

        Parameters
        ----------
        data_dist
        hypo_asimov_dist
        params
        metric
        other_metrics

        Returns
        -------
        detailed_metric_info : OrderedDict

        """
        if other_metrics is None:
            other_metrics = []
        elif isinstance(other_metrics, str):
            other_metrics = [other_metrics]
        all_metrics = sorted(set([metric] + other_metrics))
        detailed_metric_info = OrderedDict()
        if detector_name is not None:
            detailed_metric_info['detector_name'] = detector_name
        for m in all_metrics:
            name_vals_d = OrderedDict()

            # if the metric is not generalized poisson, but the distribution is a dict,
            # retrieve the 'weights' mapset from the distribution output
            if m == 'generalized_poisson_llh':
                name_vals_d['maps'] = data_dist.maps[0].generalized_poisson_llh(expected_values=generalized_poisson_hypo)
                llh_binned = data_dist.maps[0].generalized_poisson_llh(expected_values=generalized_poisson_hypo, binned=True)
                map_binned = Map(name=metric,
                                hist=np.reshape(llh_binned, data_dist.maps[0].shape),
                                binning=data_dist.maps[0].binning
                    )
                name_vals_d['maps_binned'] = MapSet(map_binned)
                name_vals_d['priors'] = params.priors_penalties(metric=metric)
                detailed_metric_info[m] = name_vals_d

            else:
                if isinstance(hypo_asimov_dist,OrderedDict):
                    hypo_asimov_dist = hypo_asimov_dist['weights']

                name_vals_d['maps'] = data_dist.metric_per_map(
                    expected_values=hypo_asimov_dist, metric=m
                )
                metric_hists = data_dist.metric_per_map(
                    expected_values=hypo_asimov_dist, metric='binned_'+m
                )
            
                maps_binned = []
                for asimov_map, metric_hist in zip(hypo_asimov_dist, metric_hists):
                    map_binned = Map(
                        name=asimov_map.name,
                        hist=np.reshape(metric_hists[metric_hist],
                                        asimov_map.shape),
                        binning=asimov_map.binning
                    )
                    maps_binned.append(map_binned)
                name_vals_d['maps_binned'] = MapSet(maps_binned)
                name_vals_d['priors'] = params.priors_penalties(metric=metric)
                detailed_metric_info[m] = name_vals_d
        return detailed_metric_info

class BasicAnalysis(object):
    """A bare-bones analysis that only fits a hypothesis to data.
    
    Full analyses with functionality beyond just fitting (doing scans, for example)
    should sub-class this class.
    
    This basic analysis can be configured to arbitrarily stack global search strategies
    with local fitting schemes. A sub-class then merely needs to produce the dictionary
    with the desired fit strategy definition.
    
    Examples
    --------
    
    A canonical standard oscillation fit fits octants in `theta23` separately and then 
    runs a scipy minimizer to optimize locally in each octant. The arguments that would
    produce that result when passed to `fit_recursively` are:
    
    ```
    method = "fit_octants"
    method_kwargs = {
        "angle": "theta23"
        "inflection_point": 45 * ureg.deg
    }
    local_fit_kwargs = {
        "method": "scipy",
        "method_kwargs": minimizer_settings,
        "local_fit_kwargs": None
    }
    ```

    Let's say we also have a CP violating phase `deltacp24` that we want to fit
    separately per quadrant split at 90 degrees. We want this done within each 
    quadrant fit for `theta23`, making 4 fits in total. Then we would nest the
    quadrant fit for `deltacp24` inside the octant fit like so:
    
    ```
    method = "fit_octants"
    method_kwargs = {
        "angle": "theta23"
        "inflection_point": 45 * ureg.deg
    }
    local_fit_kwargs = {
        "method": "fit_octants",
        "method_kwargs": {
            "angle": "deltacp24",
            "inflection_point": 90 * ureg.deg,
        }
        "local_fit_kwargs": {
            "method": "scipy",
            "method_kwargs": minimizer_settings,
            "local_fit_kwargs": None
        }
    }
    ```
    
    Finally, let's suppose we want to apply a grid-scan global fit method
    to sterile mixing parameters `theta24` and `deltam41`, but we want to marginalize
    over all other parameters with a usual 3-flavor fit configuration. That could be
    achieved as follows:
    
    ```
    method = "grid_scan"
    method_kwargs = {
        "grid": {
            "theta24": np.geomspace(1, 20, 3) * ureg.deg,
            "deltam41": np.geomspace(0.01, 0.5, 4) * ureg["eV^2"],
        },
        "fix_grid_params": False,
    }
    local_fit_kwargs = {
        "method": "fit_octants",
        "method_kwargs": {
            "angle": "theta23",
            "inflection_point": 45 * ureg.deg,
        }
        "local_fit_kwargs": {
            "method": "scipy",
            "method_kwargs": minimizer_settings,
            "local_fit_kwargs": None
        }
    }
    ```
    """

    def __init__(self):
        self._nit = 0
        self.pprint = True
        self.blindness = False
    
    # TODO: Defer sub-fits to cluster
    def fit_recursively(
            self, data_dist, hypo_maker, metric, external_priors_penalty,
            method, method_kwargs, local_fit_kwargs
        ):
        """Recursively apply global search strategies with local sub-fits."""
        
        if method in self.__class__._fit_methods.keys():
            fit_function = self.__class__._fit_methods[method]
        elif method in self.__class__._additional_fit_methods.keys():
            fit_function =  self.__class__._additional_fit_methods[method]
        else:
            raise ValueError(f"Unknown fit method: {method}")
        
        return fit_function(self, data_dist, hypo_maker, metric, external_priors_penalty,
                            method_kwargs, local_fit_kwargs)
    
    def _fit_octants(self, data_dist, hypo_maker, metric, external_priors_penalty,
                     method_kwargs, local_fit_kwargs):
        """
        A simple global optimization scheme that searches mixing angle octants.
        """
        
        angle_name = method_kwargs["angle"]
        if angle_name not in hypo_maker.params.free.names:
            logging.info(f"{angle_name} is not a free parameter, skipping octant check")
            return self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                local_fit_kwargs["local_fit_kwargs"]
            )

        inflection_point = method_kwargs["inflection_point"]
        
        logging.info(f"Entering octant fit for angle {angle_name} with inflection "
                      f"point at {inflection_point}")
        #### Removed, fitting always separately.
        #    Is there a reason not to fit separately, ever?
        # fit_octants_separately = True
        # if "fit_octants_separately" in method_kwargs.keys():
        #     fit_octants_separately = method_kwargs["fit_octants_separately"]

        reset_free = True
        if "reset_free" in method_kwargs.keys():
            reset_free = method_kwargs["reset_free"]
        
        if not reset_free:
            # store so we can reset to the values we currently have rather than
            # resetting free parameters back to their nominal value after the octant
            # check
            minimizer_start_params = hypo_maker.params

        ang_orig, ang_case1, ang_case2 = get_separate_octant_params(hypo_maker, angle_name, inflection_point)
        hypo_maker.update_params(ang_case1)
        
        # Perform the fit
        best_fit_info = self.fit_recursively(
            data_dist, hypo_maker, metric, external_priors_penalty,
            local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
            local_fit_kwargs["local_fit_kwargs"]
        )
        if not self.blindness:
            logging.info(f"found best fit at angle {best_fit_info.params[angle_name].value}")
        logging.info(f'checking other octant of {angle_name}')
        if reset_free:
            hypo_maker.reset_free()
        else:
            for param in minimizer_start_params:
                hypo_maker.params[param.name].value = param.value

        hypo_maker.update_params(ang_case2)

        # Re-run minimizer starting at new point
        new_fit_info = self.fit_recursively(
            data_dist, hypo_maker, metric, external_priors_penalty,
            local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
            local_fit_kwargs["local_fit_kwargs"]
        )
        if not self.blindness:
            logging.info(f"found best fit at angle {new_fit_info.params[angle_name].value}")
        # record the correct range for the angle 
        # If we are at the strictest blindness level 2, no parameters are stored and the 
        # dict only contains an empty dict. Attempting to set a range would cause an eror.
        if self.blindness < 2:
            best_fit_info.params[angle_name].range = deepcopy(ang_orig.range)
            new_fit_info.params[angle_name].range = deepcopy(ang_orig.range)

        # Take the one with the best fit
        if metric[0] in METRICS_TO_MAXIMIZE:
            it_got_better = (
                new_fit_info.metric_val > best_fit_info.metric_val
            )
        else:
            it_got_better = (
                new_fit_info.metric_val < best_fit_info.metric_val
            )

        # TODO: Pass alternative fits up the chain
        if it_got_better:
            # alternate_fits.append(best_fit_info)
            best_fit_info = new_fit_info
            if not self.blindness:
                logging.info('Accepting other-octant fit')
        else:
            # alternate_fits.append(new_fit_info)
            if not self.blindness:
                logging.info('Accepting initial-octant fit')

        # Reset the range of the parameter but keep the fit value
        ang_orig.value = hypo_maker.params[angle_name].value
        hypo_maker.update_params(ang_orig)
        return best_fit_info
    
    def _fit_best_of(self, data_dist, hypo_maker, metric,
                     external_priors_penalty, method_kwargs, local_fit_kwargs):
        """Run several manually configured fits and take the best one.
        
        The specialty here is that `local_fit_kwargs` is a list, where each element
        defines one fit.
        """

        logging.info(f"running several manually configured fits to choose optimum")

        reset_free = True
        if method_kwargs is not None and "reset_free" in method_kwargs.keys():
            reset_free = method_kwargs["reset_free"]
        
        all_fit_results = []
        for i, fit_kwargs in enumerate(local_fit_kwargs):
            if reset_free:
                hypo_maker.reset_free()
            logging.info(f"Beginning fit {i+1} / {len(local_fit_kwargs)}")
            new_fit_info = self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                fit_kwargs["method"], fit_kwargs["method_kwargs"],
                fit_kwargs["local_fit_kwargs"]
            )
            all_fit_results.append(new_fit_info)
        
        all_fit_metric_vals = [fit_info.metric_val for fit_info in all_fit_results]
        # Take the one with the best fit
        if metric[0] in METRICS_TO_MAXIMIZE:
            best_idx = np.argmax(all_fit_metric_vals)
        else:
            best_idx = np.argmin(all_fit_metric_vals)
        
        logging.info(f"Found best fit being index {best_idx} with metric "
                     f"{all_fit_metric_vals[best_idx]}")
        return all_fit_results[best_idx]
    
    def _fit_conditionally(self, data_dist, hypo_maker, metric,
                           external_priors_penalty, method_kwargs, local_fit_kwargs):
        """Run one fit strategy or the other depending on a condition being true.
        
        As in the constrained fit, the condition can be a callable or a string that 
        can be evaluated to a callable via `eval()`.
        
        `local_fit_kwargs` has to be a list of length 2. The first fit runs if the
        condition is true, the second one runs if the condition is false.
        """
        
        assert "condition_func" in method_kwargs.keys()
        assert len(local_fit_kwargs) == 2, ("need to fit specs, first runs if True, "
                                            "second runs if false")
        if type(method_kwargs["condition_func"]) is str:
            logging.warn(
                "Using eval() is potentially dangerous as it can execute "
                "arbitrary code! Do not store your config file in a place"
                "where others have writing access!"
            )
            condition_func = eval(method_kwargs["condition_func"])
            assert callable(condition_func), "evaluated object is not a valid function"
        elif callable(method_kwargs["condition_func"]):
            condition_func = method_kwargs["condition_func"]
        else:
            raise ValueError("Condition function is neither a callable nor a "
                             "string that can be evaluated to a callable.")
        
        if condition_func(hypo_maker):
            logging.info("condition was TRUE, running first fit")
            fit_kwargs = local_fit_kwargs[0]
        else:
            logging.info("condition was FALSE, running second fit")
            fit_kwargs = local_fit_kwargs[1]
        return self.fit_recursively(
            data_dist, hypo_maker, metric, external_priors_penalty,
            fit_kwargs["method"], fit_kwargs["method_kwargs"],
            fit_kwargs["local_fit_kwargs"]
        )

    def _fit_grid_scan(self, data_dist, hypo_maker, metric,
                       external_priors_penalty, method_kwargs, local_fit_kwargs):
        """
        Do a grid scan over starting positions and choose the best fit from the grid.
        
        Alternatively, the parameters used for the grid can be fixed in the fit at each 
        grid point, and only the very best fit is then freed up to be refined.
        """
        
        assert "grid" in method_kwargs.keys()
        reset_free = True
        if "reset_free" in method_kwargs.keys():
            reset_free = method_kwargs["reset_free"]
        fix_grid_params = False
        if "fix_grid_params" in method_kwargs.keys():
            fix_grid_params = method_kwargs["fix_grid_params"]
        grid_params = list(method_kwargs["grid"].keys())
        logging.info(f"Starting grid scan over parameters {grid_params}")
        grid_1d_arrs = []
        grid_units = []
        for p in grid_params:
            d_spec = method_kwargs["grid"][p]
            logging.info(f"{p}: {d_spec}")
            grid_1d_arrs.append(d_spec.m)
            grid_units.append(d_spec.u)
        scan_mesh = np.meshgrid(*grid_1d_arrs)
        scan_mesh = [m * u for m, u in zip(scan_mesh, grid_units)]
        
        if not fix_grid_params:
            logging.info("This grid only defines the starting value of each fit, "
                         "all parameters that are free will stay free.")
        else:
            logging.info("The grid parameters will be fixed at each grid point.")
        
        do_refined_fit = False
        if ("refined_fit" in method_kwargs.keys()
            and method_kwargs["refined_fit"] is not None):
            do_refined_fit = True
            logging.info("The best fit on the grid will be refined using "
                         f"{method_kwargs['refined_fit']['method']}")
        
        if reset_free:
            hypo_maker.reset_free()
        # when we return from the scan, we want to set all parameters free again that
        # were free to begin with
        originally_free = hypo_maker.params.free.names
        all_fit_results = []
        grid_shape = scan_mesh[0].shape
        for grid_idx in np.ndindex(grid_shape):
            point = {name: mesh[grid_idx] for name, mesh in zip(grid_params, scan_mesh)}
            logging.info(f"working on grid point {point}")
            if reset_free:
                hypo_maker.reset_free()
            for param, value in point.items():
                orig_param = deepcopy(hypo_maker.params[param])
                orig_param.value = value
                if fix_grid_params:
                    # it is possible to do a scan over fixed parameters as well as
                    # free ones; fixed ones always stay fixed, free ones are fixed
                    # if requested
                    orig_param.is_fixed = True
                hypo_maker.update_params(orig_param)
            new_fit_info = self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                local_fit_kwargs["local_fit_kwargs"]
            )
            all_fit_results.append(new_fit_info)
        for param in originally_free:
            hypo_maker.params[param].is_fixed = False

        all_fit_metric_vals = np.array([fit_info.metric_val for fit_info in all_fit_results])
        all_fit_metric_vals = all_fit_metric_vals.reshape(grid_shape)
        if not self.blindness:
            logging.info(f"Grid scan metrics:\n{all_fit_metric_vals}")
        # Take the one with the best fit
        if metric[0] in METRICS_TO_MAXIMIZE:
            best_idx = np.argmax(all_fit_metric_vals)
            best_idx_grid = np.unravel_index(best_idx, all_fit_metric_vals.shape)
        else:
            best_idx = np.argmin(all_fit_metric_vals)
            best_idx_grid = np.unravel_index(best_idx, all_fit_metric_vals.shape)
        
        logging.info(f"Found best fit being index {best_idx_grid} with metric "
                     f"{all_fit_metric_vals[best_idx_grid]}")

        best_fit_result = all_fit_results[best_idx]

        if do_refined_fit:
            hypo_maker.update_params(best_fit_result.params)
            # the params stored in the best fit may come from a grid point where 
            # parameters were fixed, so we free them up again
            for param in originally_free:
                hypo_maker.params[param].is_fixed = False
            logging.info("Refining best fit result...")
            # definitely don't want to reset the parameters here, that would defeate
            # the entire purpose...
            method_kwargs["refined_fit"]["reset_free"] = False
            best_fit_result = self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                method_kwargs["refined_fit"]["method"],
                method_kwargs["refined_fit"]["method_kwargs"],
                method_kwargs["refined_fit"]["local_fit_kwargs"]
            )
        
        return best_fit_result
    
    def _fit_constrained(self, data_dist, hypo_maker, metric,
                         external_priors_penalty, method_kwargs, local_fit_kwargs):
            """Run a fit subject to an arbitrary inequality constraint.
            
            The constraint is given as a function that must stay positive. The value of
            this function is scaled by a pre-factor and applied as a penalty to the test
            statistic, where the initial scaling factor is not too large to avoid
            minimizer problems. Should the fit converge to a point violating the
            constraint, the penalty scale is doubled.
            
            The constraining function should calculate the distance of the constraint
            over-stepping in *rescaled* parameter space to make the over-all scale
            uniform.
            """

            assert "ineq_func" in method_kwargs.keys()
            # If certain parameters aren't free, it will be impossible to satisfy the 
            # constraint and we would end up in an infinite loop! If we detect that 
            # these parameters aren't free, we just pass through the inner fit without
            # adding a constraining penalty.
            assert "necessary_free_params" in method_kwargs.keys()
            if not set(method_kwargs["necessary_free_params"]).issubset(
                set(hypo_maker.params.free.names)):
                logging.info("Necessary parameters to satisfy the constraints aren't "
                             "free, running inner fit without constraint...")
                return self.fit_recursively(
                    data_dist, hypo_maker, metric, external_priors_penalty,
                    local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                    local_fit_kwargs["local_fit_kwargs"]
                )

            logging.info("entering constrained fit...")
            if type(method_kwargs["ineq_func"]) is str:
                logging.warn(
                    "Using eval() is potentially dangerous as it can execute "
                    "arbitrary code! Do not store your config file in a place"
                    "where others have writing access!"
                )
                ineq_func = eval(method_kwargs["ineq_func"])
                assert callable(ineq_func), "evaluated object is not a valid function"
            elif callable(method_kwargs["ineq_func"]):
                ineq_func = method_kwargs["ineq_func"]
            else:
                raise ValueError("Inequality function is neither a callable nor a "
                                 "string that can be evaluated to a callable.")

            def constraint_func(params):
                value = ineq_func(params)
                # inequality function must stay positive, so there is no penalty if
                # it is positive, but otherwise we want to return a *positive* penalty
                return 0. if value > 0. else -value
            
            penalty = 1000.
            if "minimum_penalty" in method_kwargs.keys():
                penalty = method_kwargs["minimum_penalty"]
            tol = 1e-4
            if "constraint_tol" in method_kwargs.keys():
                tol = method_kwargs["constraint_tol"]
            penalty_sign = -1 if metric in METRICS_TO_MAXIMIZE else 1
            # resetting after each doubling of the penalty is probably an inefficient
            # thing to do...
            reset_free = False
            if "reset_free" in method_kwargs.keys():
                reset_free = method_kwargs["reset_free"]
            
            if external_priors_penalty is None:
                penalty_func = lambda hypo_maker, metric: (
                    penalty_sign * penalty * constraint_func(params=hypo_maker.params)
                )
            else:
                penalty_func = lambda hypo_maker, metric: (
                    penalty_sign * penalty * constraint_func(params=hypo_maker.params)
                    + external_priors_penalty(hypo_maker=hypo_maker, metric=metric)
                )
            # emulating do-while loop
            while True:
                if reset_free:
                    hypo_maker.reset_free()
                fit_result = self.fit_recursively(
                    data_dist, hypo_maker, metric, penalty_func,
                    local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                    local_fit_kwargs["local_fit_kwargs"]
                )
                penalty *= 2
                if constraint_func(fit_result.params) <= tol:
                    break
                elif not self.blindness:
                    logging.info("Fit result violates constraint condition, re-running "
                        f"with new penalty multiplier: {penalty}")
            return fit_result
    
    def _fit_ranges(self, data_dist, hypo_maker, metric,
                    external_priors_penalty, method_kwargs, local_fit_kwargs):
        """Fit given ranges of a parameter separately."""

        assert "param_name" in method_kwargs.keys()
        assert "ranges" in method_kwargs.keys()
        if not method_kwargs["param_name"] in hypo_maker.params.free.names:
            logging.info(f"parameter {method_kwargs['param_name']} not free, "
                          "skipping fit over ranges...")
            return self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                local_fit_kwargs["local_fit_kwargs"]
            )

        logging.info(f"entering fit over separate ranges in {method_kwargs['param_name']}")
        
        reset_free = False
        if "reset_free" in method_kwargs.keys():
            reset_free = method_kwargs["reset_free"]
        
        # this one we store for later so we can reset all ranges when we are done
        param = deepcopy(hypo_maker.params[method_kwargs["param_name"]])
        logging.info(f"original parameter:\n{param}")
        original_range = deepcopy(param.range)
        # this is the param we play around with
        mod_param = deepcopy(param)
        # The way this works is that we change the range and the set the rescaled
        # value of the parameter to the same number it originally had. This means
        # that, if the parameter was originally set at the lower end of the original
        # range, it will now always start at the lower end of each interval to be
        # fit separately. If it was in the middle, it will start in the middle of
        # each interval.
        original_rescaled_value = param._rescaled_value
        all_fit_results = []
        for i, interval in enumerate(method_kwargs["ranges"]):
            mod_param.range = interval
            mod_param._rescaled_value = original_rescaled_value
            # to make sure that a `reset_free` command will not try to reset the
            # parameter to a place outside of the modified range
            mod_param.nominal_value = mod_param.value
            logging.info(f"now fitting on interval {i+1}/{len(method_kwargs['ranges'])}")
            logging.info(f"parameter with modified range:\n{mod_param}")
            hypo_maker.update_params(mod_param)
            
            fit_result = self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                local_fit_kwargs["method"], local_fit_kwargs["method_kwargs"],
                local_fit_kwargs["local_fit_kwargs"]
            )
            all_fit_results.append(fit_result)
        all_fit_metric_vals = [fit_info.metric_val for fit_info in all_fit_results]
        # Take the one with the best fit
        if metric[0] in METRICS_TO_MAXIMIZE:
            best_idx = np.argmax(all_fit_metric_vals)
        else:
            best_idx = np.argmin(all_fit_metric_vals)
        
        logging.info(f"Found best fit being in interval {best_idx+1} with metric "
                     f"{all_fit_metric_vals[best_idx]}")
        best_fit_result = all_fit_results[best_idx]
        # resetting all parameter properties for the param we played with, but setting
        # the value to the best fit value
        param.value = best_fit_result.params[param.name].value
        hypo_maker.update_params(param)
        best_fit_result.params = hypo_maker.params

        return best_fit_result

    def _fit_scipy(self, data_dist, hypo_maker, metric,
                   external_priors_penalty, method_kwargs, local_fit_kwargs):
        """Run an arbitrary scipy minimizer to modify hypo dist maker's free params
        until the data_dist is most likely to have come from this hypothesis.

        This function uses only local optimization and does not attempt to find
        a global optimum among several local optima.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s)

        hypo_maker : Detectors, DistributionMaker or convertible thereto

        metric : string or iterable of strings

        minimizer_settings : dict

        external_priors_penalty : func
            User defined prior penalty function


        Returns
        -------
        fit_info : HypoFitResult
        """
        
        global_scipy_methods = ["differential_evolution", "basinhopping",
                                "dual_annealing", "shgo"]
        methods_using_local_fits = ["basinhopping", "dual_annealing", "shgo"]
        
        global_method = None
        if "global_method" in method_kwargs.keys():
            global_method = method_kwargs["global_method"]

        if local_fit_kwargs is not None and global_method not in methods_using_local_fits:
            logging.warn(f"local_fit_kwargs are ignored by global method {global_method}")
        
        if global_method is None:
            logging.info(f"entering local scipy fit using {method_kwargs['method']['value']}")
        else:
            assert global_method in global_scipy_methods, "unsupported global fit method"
            logging.info(f"entering global scipy fit using {global_method}")
        if not self.blindness:
            logging.debug("free parameters:")
            logging.debug(hypo_maker.params.free)
        
        if global_method in methods_using_local_fits:
            minimizer_settings = set_minimizer_defaults(local_fit_kwargs)
            validate_minimizer_settings(minimizer_settings)
        elif global_method == "differential_evolution":
            # Unfortunately we are not allowed to pass these, DE with polish=True always
            # uses L-BFGS-B with default settings.
            if ("polish" in method_kwargs["options"].keys()
                and method_kwargs["options"]["polish"]):
                logging.info("Differential Evolution result will be polished using L-BFGS-B")
                # We need to put the method here so that the bounds will be adjusted
                # below, otherwise the polishing fit can cause crashes if it hits the
                # bounds.
                minimizer_settings = {
                    "method": {"value": "L-BFGS-B"},
                    "options": {"value": {"eps": 1e-8}}
                }
            else:
                # We put this here such that the checks farther down don't crash
                minimizer_settings = {"method": {"value": "None"}}
        elif global_method == "dual_annealing":
            minimizer_settings = {
                "method": {"value": "L-BFGS-B"},
                "options": {"value": {"eps": 1e-8}}
            }
            logging.info("Due to a scipy bug, local minimization can only use default"
                         "L-BFGS-B settings. The given settings are ignored.")
        else:
            minimizer_settings = set_minimizer_defaults(method_kwargs)
            validate_minimizer_settings(minimizer_settings)

        if isinstance(metric, str):
            metric = [metric]
        sign = 0
        for m in metric:
            if m in METRICS_TO_MAXIMIZE and sign != +1:
                sign = -1
            elif m in METRICS_TO_MINIMIZE and sign != -1:
                sign = +1
            else:
                raise ValueError("Defined metrics are not compatible")

        # Get starting free parameter values
        x0 = np.array(hypo_maker.params.free._rescaled_values) # pylint: disable=protected-access
        
        # Indicate indices where x0 should be reflected around the mid-point at 0.5.
        # This is only used for the COBYLA minimizer.
        flip_x0 = np.zeros(len(x0), dtype=bool)
        # Guard against over-stepping of boundaries by clipping values during
        # minimization. This is only used for COBYLA because it tolerates some
        # violation of the constraints. Because COBYLA is gradient-free, the clipping
        # should not impact the minimization in a detrimental way.
        guard_bounds = False
        minimizer_method = minimizer_settings["method"]["value"].lower()
        cons = ()
        if minimizer_method in MINIMIZERS_USING_SYMM_GRAD:
            logging.warning(
                'Minimizer %s requires artificial boundaries SMALLER than the'
                ' user-specified boundaries (so that numerical gradients do'
                ' not exceed the user-specified boundaries).',
                minimizer_method
            )
            step_size = minimizer_settings['options']['value']['eps']
            bounds = [(0 + step_size, 1 - step_size)]*len(x0)
            x0[(0 <= x0) & (x0 < step_size)] = step_size
            x0[(1 - step_size < x0) & (x0 <= 1)] = 1 - step_size

        elif minimizer_method in MINIMIZERS_USING_CONSTRAINTS:
            logging.warning(
                'Minimizer %s requires bounds to be formulated in terms of constraints.'
                ' Constraining functions are auto-generated now.',
                minimizer_method
            )
            cons = []
            for idx in range(len(x0)):
                l = {'type': 'ineq',
                     'fun': lambda x, i=idx: x[i] - FTYPE_PREC}  # lower bound at zero
                u = {'type': 'ineq',
                     'fun': lambda x, i=idx: 1. - x[i]}  # upper bound at 1
                cons.append(l)
                cons.append(u)
            # The minimizer begins with a step of size `rhobeg` in the positive
            # direction. Flipping around 0.5 ensures that this initial step will not
            # overstep boundaries if `rhobeg` is 0.5. 
            flip_x0 = np.array(x0) > 0.5
            # The minimizer can't handle bounds, but they are still used below to clip
            # x0 into the valid range if that is necessary.
            guard_bounds = True
            bounds = [(0, 1)]*len(x0)
        else:
            bounds = [(0, 1)]*len(x0)

        clipped_x0 = []
        for param, x0_val, bds in zip(hypo_maker.params.free, x0, bounds):
            if x0_val < bds[0] - EPSILON:
                raise ValueError(
                    'Param %s, initial scaled value %.17e is below lower bound'
                    ' %.17e.' % (param.name, x0_val, bds[0])
                )
            if x0_val > bds[1] + EPSILON:
                raise ValueError(
                    'Param %s, initial scaled value %.17e exceeds upper bound'
                    ' %.17e.' % (param.name, x0_val, bds[1])
                )

            clipped_x0_val = np.clip(x0_val, a_min=bds[0], a_max=bds[1])
            clipped_x0.append(clipped_x0_val)

            if recursiveEquality(clipped_x0_val, bds[0]):
                logging.warning(
                    'Param %s, initial scaled value %e is at the lower bound;'
                    ' minimization may fail as a result.',
                    param.name, clipped_x0_val
                )
            if recursiveEquality(clipped_x0_val, bds[1]):
                logging.warning(
                    'Param %s, initial scaled value %e is at the upper bound;'
                    ' minimization may fail as a result.',
                    param.name, clipped_x0_val
                )

        x0 = np.array(clipped_x0)
        x0 = np.where(flip_x0, 1 - x0, x0)
        
        if global_method is None:
            logging.debug('Running the %s minimizer...', minimizer_method)
        else:
            logging.debug(f"Running the {global_method} global fit method...")
        # Using scipy.optimize.minimize allows a whole host of minimizers to be
        # used.
        counter = Counter()
        
        fit_history = []
        fit_history.append(list(metric) + [v.name for v in hypo_maker.params.free])

        start_t = time.time()

        if self.pprint and not self.blindness:
            free_p = hypo_maker.params.free

            # Display any units on top
            r = re.compile(r'(^[+0-9.eE-]* )|(^[+0-9.eE-]*$)')
            hdr = ' '*(6+1+10+1+12+3)
            unt = []
            for p in free_p:
                u = r.sub('', format(p.value, '~')).replace(' ', '')[0:10]
                if u:
                    u = '(' + u + ')'
                unt.append(u.center(12))
            hdr += ' '.join(unt)
            hdr += '\n'

            # Header names
            hdr += ('iter'.center(6) + ' ' + 'funcalls'.center(10) + ' ' +
                    metric[0][0:12].center(12) + ' | ')
            hdr += ' '.join([p.name[0:12].center(12) for p in free_p])
            if external_priors_penalty is not None:
                hdr += " |   penalty  "
            hdr += '\n'

            # Underscores
            hdr += ' '.join(['-'*6, '-'*10, '-'*12, '+'] + ['-'*12]*len(free_p))
            if external_priors_penalty is not None:
                hdr += " + -----------"
            hdr += '\n'

            sys.stdout.write(hdr)

        # reset number of iterations before each minimization
        self._nit = 0

        #
        # From that point on, optimize starts using the metric and 
        # iterates, no matter what you do 
        #
        if global_method is None:
            optimize_result = optimize.minimize(
                fun=self._minimizer_callable,
                x0=x0,
                args=(hypo_maker, data_dist, metric, counter, fit_history,
                      flip_x0, guard_bounds, external_priors_penalty),
                bounds=bounds,
                constraints=cons,
                method=minimizer_settings['method']['value'],
                options=minimizer_settings['options']['value'],
                callback=self._minimizer_callback
            )
        elif global_method == "differential_evolution":
            optimize_result = optimize.differential_evolution(
                func=self._minimizer_callable,
                bounds=bounds,
                args=(hypo_maker, data_dist, metric, counter, fit_history,
                      flip_x0, guard_bounds, external_priors_penalty),
                callback=self._minimizer_callback,
                **method_kwargs["options"]
            )
        elif global_method == "basinhopping":
            if "seed" in method_kwargs["options"]:
                seed = method_kwargs["options"]["seed"]
            else:
                seed = None
            rng = check_random_state(seed)

            if "step_size" in method_kwargs["options"]:
                step_size = method_kwargs["options"]["step_size"]
            else:
                step_size = 0.5
            
            take_step = BoundedRandomDisplacement(step_size, bounds, rng)
            minimizer_kwargs = deepcopy(local_fit_kwargs)
            minimizer_kwargs["args"] = (
                hypo_maker, data_dist, metric, counter, fit_history,
                flip_x0, guard_bounds, external_priors_penalty
            )
            if "reset_free" in minimizer_kwargs:
                del minimizer_kwargs["reset_free"]
            minimizer_kwargs["method"] = local_fit_kwargs["method"]["value"]
            minimizer_kwargs["options"] = local_fit_kwargs["options"]["value"]
            minimizer_kwargs["bounds"] = bounds
            def basinhopping_callback(x, f, accept):
                self._nit += 1
            optimize_result = optimize.basinhopping(
                func=self._minimizer_callable,
                x0=x0,
                take_step=take_step,
                callback=basinhopping_callback,
                minimizer_kwargs=minimizer_kwargs,
                **method_kwargs["options"]
            )
            optimize_result.success = True  # basinhopping doesn't set this property
        elif global_method == "dual_annealing":
            def annealing_callback(x, f, context):
                self._nit += 1
            # TODO: Enable use of custom minimization if scipy is fixed
            # The scipy implementation is buggy insofar as it doesn't apply bounds to 
            # the inner minimization and there is no way to pass bounds through that 
            # doesn't crash. This leads to evaluations outside of the bounds.
            optimize_result = optimize.dual_annealing(
                func=self._minimizer_callable,
                bounds=bounds,
                x0=x0,
                args=(hypo_maker, data_dist, metric, counter, fit_history,
                      flip_x0, guard_bounds, external_priors_penalty),
                callback=annealing_callback,
                **method_kwargs["options"]
            )
        elif global_method == "shgo":
            minimizer_kwargs = deepcopy(local_fit_kwargs)
            minimizer_kwargs["args"] = (
                hypo_maker, data_dist, metric, counter, fit_history,
                flip_x0, guard_bounds, external_priors_penalty
            )
            if "reset_free" in minimizer_kwargs:
                del minimizer_kwargs["reset_free"]
            minimizer_kwargs["method"] = local_fit_kwargs["method"]["value"]
            minimizer_kwargs["options"] = local_fit_kwargs["options"]["value"]
            optimize_result = optimize.shgo(
                func=self._minimizer_callable,
                bounds=bounds,
                args=(hypo_maker, data_dist, metric, counter, fit_history,
                      flip_x0, guard_bounds, external_priors_penalty),
                callback=self._minimizer_callback,
                minimizer_kwargs=minimizer_kwargs,
                **method_kwargs["options"]
            )
        else:
            raise ValueError("Unsupported global fit method")
        if not optimize_result.success:
            if self.blindness:
                msg = ''
            else:
                msg = ' ' + str(optimize_result.message)
            raise ValueError('Optimization failed.' + msg)

        end_t = time.time()
        if self.pprint:
            # clear the line
            sys.stdout.write('\n\n')
            sys.stdout.flush()

        minimizer_time = end_t - start_t

        logging.info(
            'Total time to optimize: %8.4f s; # of dists generated: %6d;'
            ' avg dist gen time: %10.4f ms',
            minimizer_time, counter.count, minimizer_time*1000./counter.count
        )

        # Will not assume that the minimizer left the hypo maker in the
        # minimized state, so set the values now (also does conversion of
        # values from [0,1] back to physical range)
        rescaled_pvals = optimize_result.pop('x')
        rescaled_pvals = np.where(flip_x0, 1 - rescaled_pvals, rescaled_pvals)
        hypo_maker._set_rescaled_free_params(rescaled_pvals) # pylint: disable=protected-access

        # Get the best-fit metric value
        metric_val = sign * optimize_result.pop('fun')

        # Record minimizer metadata (all info besides 'x' and 'fun'; also do
        # not record some attributes if performing blinded analysis)
        metadata = OrderedDict()
        for k in sorted(optimize_result.keys()):
            if self.blindness and k in ['jac', 'hess', 'hess_inv']:
                continue
            if k=='hess_inv':
                continue
            metadata[k] = optimize_result[k]

        if self.blindness > 1:  # only at stricter blindness level
            # undo flip
            x0 = np.where(flip_x0, 1 - x0, x0)
            # Reset to starting value of the fit, rather than nominal values because
            # the nominal value might be out of range if this is inside an octant check.
            hypo_maker._set_rescaled_free_params(x0)
        
        # TODO: other metrics
        fit_info = HypoFitResult(
            metric,
            metric_val,
            data_dist,
            hypo_maker,
            minimizer_time=minimizer_time,
            minimizer_metadata=metadata,
            fit_history=fit_history,
            other_metrics=None,
            num_distributions_generated=counter.count,
            include_detailed_metric_info=True,
        )
        
        if not self.blindness:
            logging.info(f"found best fit: {fit_info.params.free}")
        return fit_info

    def _minimizer_callable(self, scaled_param_vals, hypo_maker, data_dist,
                            metric, counter, fit_history, flip_x0,
                            guard_bounds, external_priors_penalty=None):
        """Simple callback for use by scipy.optimize minimizers.

        This should *not* in general be called by users, as `scaled_param_vals`
        are stripped of their units and scaled to the range [0, 1], and hence
        some validation of inputs is bypassed by this method.

        Parameters
        ----------
        scaled_param_vals : sequence of floats
            If called from a scipy.optimize minimizer, this sequence is
            provieded by the minimizer itself. These values are all expected to
            be in the range [0, 1] and be simple floats (no units or
            uncertainties attached, etc.). Rescaling the parameter values to
            their original (physical) ranges (including units) is handled
            within this method.

        hypo_maker : Detectors or DistributionMaker
            Creates the per-bin expectation values per map
            based on its param values. Free params in the
            `hypo_maker` are modified by the minimizer to achieve a "best" fit.

        data_dist : Sequence of MapSets or MapSet
            Data distribution to be fit. Can be an actual-, Asimov-, or
            pseudo-data distribution (where the latter two are derived from
            simulation and so aren't technically "data").

        metric : string or iterable of strings
            Metric by which to evaluate the fit. See Map

        counter : Counter
            Mutable object to keep track--outside this method--of the number of
            times this method is called.
        
        flip_x0 : ndarray of type bool
            Indicates which indices of x0 should be flipped around 0.5.
        
        guard_bounds : bool
            Guard against over-stepping of boundaries by clipping x0.

        external_priors_penalty : func
            User defined prior penalty function, which takes `hypo_maker` and 
            `metric` as arguments and returns numerical value of penalty to 
            the metric value. It is expected sign of the penalty is correctly 
            specified inside the `external_priors_penalty` (e.g. negative for 
            llh or positive for chi2).

        """
        # Want to *maximize* e.g. log-likelihood but we're using a minimizer,
        # so flip sign of metric in those cases.
        if isinstance(metric, str):
            metric = [metric]
        sign = 0
        for m in metric:
            if m in METRICS_TO_MAXIMIZE and sign != +1:
                sign = -1
            elif m in METRICS_TO_MINIMIZE and sign != -1:
                sign = +1
            else:
                raise ValueError('Defined metrics are not compatible')

        scaled_param_vals = np.where(flip_x0, 1 - scaled_param_vals, scaled_param_vals)
        if guard_bounds:
            dist_lb = scaled_param_vals
            dist_ub = 1 - scaled_param_vals
            if np.any(dist_lb < 0):
                logging.warn(f"minimizer stepped under lower bound by {-np.min(dist_lb)}")
            if np.any(dist_ub < 0):
                logging.warn(f"minimizer stepped over upper bound by {-np.min(dist_ub)}")
            scaled_param_vals = np.clip(scaled_param_vals, 0, 1)
        # Set param values from the scaled versions the minimizer works with
        hypo_maker._set_rescaled_free_params(scaled_param_vals) # pylint: disable=protected-access

        # Get the map set
        try:
            if metric[0] == 'generalized_poisson_llh':
                hypo_asimov_dist = hypo_maker.get_outputs(return_sum=False, output_mode='binned', force_standard_output=False)
                hypo_asimov_dist = merge_mapsets_together(mapset_list=hypo_asimov_dist)
                data_dist = data_dist.maps[0] # Extract the map from the MapSet
                metric_kwargs = {'empty_bins':hypo_maker.empty_bin_indices}
            else:
                hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
                if isinstance(hypo_asimov_dist, OrderedDict):
                    hypo_asimov_dist = hypo_asimov_dist['weights']
                metric_kwargs = {}

        except Exception as e:
            if self.blindness:
                logging.error('Minimizer failed')
            else:
                logging.error(
                    'Failed to generate distribution with free'
                    ' params %s', hypo_maker.params.free
                )
                logging.error(str(e))
            raise

        # Check number of used metrics
        if hypo_maker.__class__.__name__ == "Detectors":
            if len(metric) == 1: # One metric for all detectors
                metric = list(metric) * len(hypo_maker._distribution_makers)
            elif len(metric) != len(hypo_maker._distribution_makers):
                raise IndexError('Number of defined metrics does not match with number of detectors.')
        else: # DistributionMaker object
            assert len(metric) == 1

        #
        # Assess the fit: whether the data came from the hypo_asimov_dist
        #
        try:
            if hypo_maker.__class__.__name__ == "Detectors":
                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)):
                    data = data_dist[i].metric_total(expected_values=hypo_asimov_dist[i],
                                                  metric=metric[i], metric_kwargs=metric_kwargs)
                    metric_val += data
                priors = hypo_maker.params.priors_penalty(metric=metric[0]) # uses just the "first" metric for prior
                metric_val += priors
            else: # DistributionMaker object
                metric_val = (
                    data_dist.metric_total(expected_values=hypo_asimov_dist,
                                               metric=metric[0], metric_kwargs=metric_kwargs)
                        + hypo_maker.params.priors_penalty(metric=metric[0])
                    )
        except Exception as e:
            if self.blindness:
                logging.error('Minimizer failed')
            else :
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
                )
                logging.error(str(e))
            raise
        
        penalty = 0
        if external_priors_penalty is not None:
            penalty = external_priors_penalty(hypo_maker=hypo_maker,metric=metric)

        # Report status of metric & params (except if blinded)
        if self.blindness:
            msg = ('minimizer iteration: #%6d | function call: #%6d'
                   %(self._nit, counter.count))
        else:
            #msg = '%s=%.6e | %s' %(metric, metric_val, hypo_maker.params.free)
            msg = '%s %s %s | ' %(('%d'%self._nit).center(6),
                                  ('%d'%counter.count).center(10),
                                  format(metric_val, '0.5e').rjust(12))
            msg += ' '.join([('%0.5e'%p.value.m).rjust(12)
                             for p in hypo_maker.params.free])
            if external_priors_penalty is not None:
                msg += f" | {penalty:11.4e}"

        if self.pprint:
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()
        else:
            logging.trace(msg)

        counter += 1

        if not self.blindness:
            fit_history.append(
                [metric_val] + [v.value.m for v in hypo_maker.params.free]
            )

        if external_priors_penalty is not None:
            metric_val += external_priors_penalty(hypo_maker=hypo_maker,metric=metric)
            
        return sign*metric_val

    def _minimizer_callback(self, xk, **unused_kwargs): # pylint: disable=unused-argument
        """Passed as `callback` parameter to `optimize.minimize`, and is called
        after each iteration. Keeps track of number of iterations.

        Parameters
        ----------
        xk : list
            Parameter vector

        """
        self._nit += 1

    _fit_methods = {
        "scipy": _fit_scipy,
        "fit_octants": _fit_octants,
        "best_of": _fit_best_of,
        "grid_scan": _fit_grid_scan,
        "constrained": _fit_constrained,
        "fit_ranges": _fit_ranges,
        "condition": _fit_conditionally,
    }
    _additional_fit_methods = {}


class Analysis(BasicAnalysis):
    """Analysis class for "canonical" IceCube/DeepCore/PINGU analyses.

    * "Data" distribution creation (via passed `data_maker` object)
    * "Expected" distribution creation (via passed `distribution_maker` object)
    * Minimizer Interface (via method `_minimizer_callable`)
        Interfaces to a minimizer for modifying the free parameters of the
        `distribution_maker` to fit its output (as closely as possible) to the
        data distribution is provided. See [minimizer_settings] for

    """
    
    def __init__(self):
        self._nit = 0
        self.pprint = True
        self.blindness = False

    def fit_hypo(self, data_dist, hypo_maker, metric, minimizer_settings, 
                 hypo_param_selections=None, reset_free=True, 
                 check_octant=True, fit_octants_separately=None,
                 check_ordering=False, other_metrics=None,
                 blind=False, pprint=True, external_priors_penalty=None):
        """Fitter "outer" loop: If `check_octant` is True, run
        `fit_hypo_inner` starting in each octant of theta23 (assuming that
        is a param in the `hypo_maker`). Otherwise, just run the inner
        method once.

        Note that prior to running the fit, the `hypo_maker` has
        `hypo_param_selections` applied and its free parameters are reset to
        their nominal values.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization process.

        hypo_maker : Detectors, DistributionMaker or instantiable thereto
            Generates the expectation distribution under a particular
            hypothesis. This typically has (but is not required to have) some
            free parameters which can be modified by the minimizer to optimize
            the `metric`.

        hypo_param_selections : None, string, or sequence of strings
            A pipeline configuration can have param selectors that allow
            switching a parameter among two or more values by specifying the
            corresponding param selector(s) here. This also allows for a single
            instance of a DistributionMaker to generate distributions from
            different hypotheses.

        metric : string or iterable of strings
            The metric to use for optimization. Valid metrics are found in
            `VALID_METRICS`. Note that the optimized hypothesis also has this
            metric evaluated and reported for each of its output maps.

        minimizer_settings : string or dict

        check_octant : bool
            If theta23 is a parameter to be used in the optimization (i.e.,
            free), the fit will be re-run in the second (first) octant if
            theta23 is initialized in the first (second) octant.

        reset_free : bool
            Resets all free parameters to values defined in stages when starting a fit

        fit_octants_separately : bool
            If 'check_octant' is set so that the two octants of theta23 are
            individually checked, this flag enforces that each theta23 can
            only vary within the octant currently being checked (e.g. the 
            minimizer cannot swap octants). Deprecated.

        check_ordering : bool
            If the ordering is not in the hypotheses already being tested, the
            fit will be run in both orderings.

        other_metrics : None, string, or list of strings
            After finding the best fit, these other metrics will be evaluated
            for each output that contributes to the overall fit. All strings
            must be valid metrics, as per `VALID_METRICS`, or the
            special string 'all' can be specified to evaluate all
            VALID_METRICS..

        pprint : bool
            Whether to show live-update of minimizer progress.

        blind : bool or int
            Whether to carry out a blind analysis. If True or 1, this hides actual
            parameter values from display and disallows these (as well as Jacobian,
            Hessian, etc.) from ending up in logfiles. If given an integer > 1, the
            fitted parameters are also prevented from being stored in fit info
            dictionaries.

        external_priors_penalty : func
            User defined prior penalty function. Adds an extra penalty
            to the metric that is minimized, depending on the input function.


        Returns
        -------
        best_fit_info : HypoFitResult (see fit_hypo_inner method for details of
            `fit_info` dict)
        alternate_fits : list of `fit_info` from other fits run

        """
        
        if fit_octants_separately is not None:
            warnings.warn("fit_octants_separately is deprecated and will be ignored, "
                          "octants are always fit separately now.", DeprecationWarning)
        
        self.blindness = blind
        self.pprint = pprint

        if hypo_param_selections is None:
            hypo_param_selections = hypo_maker.param_selections

        if isinstance(metric, str):
            metric = [metric]

        if check_ordering:
            if 'nh' in hypo_param_selections or 'ih' in hypo_param_selections:
                raise ValueError('One of the orderings has already been '
                                 'specified as one of the hypotheses but the '
                                 'fit has been requested to check both. These '
                                 'are incompatible.')

            logging.info('Performing fits in both orderings.')
            extra_param_selections = ['nh', 'ih']
        else:
            extra_param_selections = [None]

        alternate_fits = []
        # TODO: Pass alternative fits up the chain
        for extra_param_selection in extra_param_selections:
            if extra_param_selection is not None:
                full_param_selections = hypo_param_selections
                full_param_selections.append(extra_param_selection)
            else:
                full_param_selections = hypo_param_selections
            # Select the version of the parameters used for this hypothesis
            hypo_maker.select_params(full_param_selections)

            # Reset free parameters to nominal values
            if reset_free:
                hypo_maker.reset_free()
            
            if check_octant:
                method = "fit_octants"
                method_kwargs = {
                    "angle": "theta23",
                    "inflection_point": 45 * ureg.deg,
                    "reset_free": reset_free,
                }
                local_fit_kwargs = {
                    "method": "scipy",
                    "method_kwargs": minimizer_settings,
                    "local_fit_kwargs": None,
                }
            else:
                method = "scipy"
                method_kwargs = minimizer_settings
                local_fit_kwargs = None

            # Perform the fit
            best_fit_info = self.fit_recursively(
                data_dist, hypo_maker, metric, external_priors_penalty,
                method, method_kwargs, local_fit_kwargs
            )

        return best_fit_info, alternate_fits

    def nofit_hypo(self, data_dist, hypo_maker, hypo_param_selections,
                   hypo_asimov_dist, metric, other_metrics=None, blind=False, external_priors_penalty=None):
        """Fitting a hypo to a distribution generated by its own
        distribution maker is unnecessary. In such a case, use this method
        (instead of `fit_hypo`) to still retrieve meaningful information for
        e.g. the match metrics.

        Parameters
        ----------
        data_dist : MapSet or List of MapSets
        hypo_maker : Detectors or DistributionMaker
        hypo_param_selections : None, string, or sequence of strings
        hypo_asimov_dist : MapSet or List of MapSets
        metric : string or iterable of strings
        other_metrics : None, string, or sequence of strings
        blind : bool
        external_priors_penalty : func


        """
        fit_info = HypoFitResult()
        if isinstance(metric, str):
            metric = [metric]
        fit_info.metric = metric

        # NOTE: Select params but *do not* reset to nominal values to record
        # the current (presumably already optimal) param values
        hypo_maker.select_params(hypo_param_selections)

        # Check number of used metrics
        if hypo_maker.__class__.__name__ == "Detectors":
            if len(metric) == 1: # One metric for all detectors
                metric = list(metric) * len(hypo_maker._distribution_makers)
            elif len(metric) != len(hypo_maker._distribution_makers):
                raise IndexError('Number of defined metrics does not match with number of detectors.')
        else: # DistributionMaker object
            assert len(metric) == 1

        # Assess the fit: whether the data came from the hypo_asimov_dist
        try:
            if hypo_maker.__class__.__name__ == "Detectors":
                metric_val = 0
                for i in range(len(hypo_maker._distribution_makers)):
                    data = data_dist[i].metric_total(expected_values=hypo_asimov_dist[i],metric=metric[i])
                    metric_val += data
                priors = hypo_maker.params.priors_penalty(metric=metric[0]) # uses just the "first" metric for prior
                metric_val += priors
            else: # DistributionMaker object

                if 'generalized_poisson_llh' == metric[0]:

                    hypo_asimov_dist = hypo_maker.get_outputs(return_sum=False, output_mode='binned', force_standard_output=False)
                    hypo_asimov_dist = merge_mapsets_together(mapset_list=hypo_asimov_dist)
                    data_dist = data_dist.maps[0] # Extract the map from the MapSet
                    metric_kwargs = {'empty_bins':hypo_maker.empty_bin_indices}
                else:
                    hypo_asimov_dist = hypo_maker.get_outputs(return_sum=True)
                    if isinstance(hypo_asimov_dist, HypoFitResult):
                        hypo_asimov_dist = hypo_asimov_dist['weights']
                    metric_kwargs = {}

                metric_val = (
                    data_dist.metric_total(expected_values=hypo_asimov_dist,
                                           metric=metric[0], metric_kwargs=metric_kwargs)
                    + hypo_maker.params.priors_penalty(metric=metric[0])
                )
                if external_priors_penalty is not None:
                    metric_val += external_priors_penalty(hypo_maker=hypo_maker,metric=metric[0])
                    
        except Exception as e:
            if self.blindness:
                logging.error('Minimizer failed')
            else :
                logging.error(
                    'Failed when computing metric with free params %s',
                    hypo_maker.params.free
                )
                logging.error(str(e))
            raise

        fit_info.metric_val = metric_val

        if self.blindness:
            # Okay, if blind analysis is being performed, reset the values so
            # the user can't find them in the object
            hypo_maker.reset_free()
            fit_info.metric_val = ParamSet()
        else:
            fit_info.metric_val = deepcopy(hypo_maker.params)
        if hypo_maker.__class__.__name__ == "Detectors":
            fit_info.detailed_metric_info = [self.get_detailed_metric_info(
                data_dist=data_dist[i], hypo_asimov_dist=hypo_asimov_dist[i],
                params=hypo_maker._distribution_makers[i].params, metric=metric[i],
                other_metrics=other_metrics, detector_name=hypo_maker.det_names[i]
            ) for i in range(len(data_dist))]
        else: # DistributionMaker object

            if 'generalized_poisson_llh' == metric[0]:
                generalized_poisson_dist = hypo_maker.get_outputs(return_sum=False, force_standard_output=False)
                generalized_poisson_dist = merge_mapsets_together(mapset_list=generalized_poisson_dist)
            else:
                generalized_poisson_dist = None

            fit_info.detailed_metric_info = self.get_detailed_metric_info(
                data_dist=data_dist, hypo_asimov_dist=hypo_asimov_dist, generalized_poisson_hypo=generalized_poisson_dist,
                params=hypo_maker.params, metric=metric[0], other_metrics=other_metrics,
                detector_name=hypo_maker.detector_name
            )

        fit_info.minimizer_time = 0 * ureg.sec
        fit_info.num_distributions_generated = 0
        fit_info.minimizer_metadata = OrderedDict()
        fit_info.hypo_asimov_dist = hypo_asimov_dist
        return fit_info



    # TODO: move the complexity of defining a scan into a class with various
    # factory methods, and just pass that class to the scan method; we will
    # surely want to use scanning over parameters in more general ways, too:
    # * set (some) fixed params, then run (minimizer, scan, etc.) on free
    #   params
    # * set (some free or fixed) params, then check metric
    # where the setting of the params is done for some number of values.
    def scan(self, data_dist, hypo_maker, metric, hypo_param_selections=None,
             param_names=None, steps=None, values=None, only_points=None,
             outer=True, profile=True, minimizer_settings=None, outfile=None,
             debug_mode=1, **kwargs):
        """Set hypo maker parameters named by `param_names` according to
        either values specified by `values` or number of steps specified by
        `steps`, and return the `metric` indicating how well the data
        distribution is described by each distribution.

        Some flexibility in how the user can specify `values` is allowed, based
        upon the shapes of `param_names` and `values` and how the `outer` flag
        is set.

        Either `values` or `steps` must be specified, but not both.

        Parameters
        ----------
        data_dist : Sequence of MapSets or MapSet
            Data distribution(s). These are what the hypothesis is tasked to
            best describe during the optimization/comparison process.

        hypo_maker : Detectors, DistributionMaker or instantiable thereto
            Generates the expectation distribution under a particular
            hypothesis. This typically has (but is not required to have) some
            free parameters which will be modified by the minimizer to optimize
            the `metric` in case `profile` is set to True.

        hypo_param_selections : None, string, or sequence of strings
            A pipeline configuration can have param selectors that allow
            switching a parameter among two or more values by specifying the
            corresponding param selector(s) here. This also allows for a single
            instance of a DistributionMaker to generate distributions from
            different hypotheses.

        metric : string or iterable of strings
            The metric to use for optimization/comparison. Note that the
            optimized hypothesis also has this metric evaluated and reported for
            each of its output maps. Confer `pisa.core.map` for valid metrics.

        param_names : None, string, or sequence of strings
            If None, assume all parameters are to be scanned; otherwise,
            specifies only the name or names of parameters to be scanned.

        steps : None, integer, or sequence of integers
            Number of steps to take within the allowed range of the parameter
            (or parameters). Value(s) specified for `steps` must be >= 2. Note
            that the endpoints of the range are always included, and numbers of
            steps higher than 2 fill in between the endpoints.

            * If integer...
                  Take this many steps for each specified parameter.
            * If sequence of integers...
                  Take the coresponding number of steps within the allowed range
                  for each specified parameter.

        values : None, scalar, sequence of scalars, or sequence-of-sequences
          * If scalar...
                Set this value for the (one) param name in `param_names`.
          * If sequence of scalars...
              * if len(param_names) is 1, set its value to each number in the
                sequence.
              * otherwise, set each param in param_names to the corresponding
                value in `values`. There must be the same number of param names
                as values.
          * If sequence of (sequences or iterables)...
              * Each param name corresponds to one of the inner sequences, in
                the order that the param names are specified.
              * If `outer` is False, all inner sequences must have the same
                length, and there will be one distribution generated for
                each set of values across the inner sequences. In other words,
                there will be a total of len(inner sequence) distribution generated.
              * If `outer` is True, the lengths of inner sequences needn't be
                the same. This takes the outer product of the passed sequences
                to arrive at the permutations of the parameter values that will
                be used to produce the distributions (essentially nested
                loops over each parameter). E.g., if two params are scanned,
                for each value of the first param's inner sequence, a
                distribution is produced for every value of the second param's
                inner sequence. In total, there will be
                ``len(inner seq0) * len(inner seq1) * ...``
                distributions produced.

        only_points : None, integer, or even-length sequence of integers
            Only select subset of points to be analysed by specifying their
            range of positions within the whole set (0-indexed, incremental).
            For the lazy amongst us...

        outer : bool
            If set to True and a sequence of sequences is passed for `values`,
            the points scanned are the *outer product* of the inner sequences.
            See `values` for a more detailed explanation.

        profile : bool
            If set to True, minimizes specified metric over all free parameters
            at each scanned point. Otherwise keeps them at their nominal values
            and only performs grid scan of the parameters specified in
            `param_names`.

        minimizer_settings : dict
            Dictionary containing the settings for minimization, which are
            only needed if `profile` is set to True. Hint: it has proven useful
            to sprinkle with a healthy dose of scepticism.

        outfile : string
            Outfile to store results to. Will be updated at each scan step to
            write out intermediate results to prevent loss of data in case
            the apocalypse strikes after all.

        debug_mode : int, either one of [0, 1, 2]
            If set to 2, will add a wealth of minimisation history and physics
            information to the output file. Otherwise, the output will contain
            the essentials to perform an analysis (0), or will hopefully be
            detailed enough for some simple debugging (1). Any other value for
            `debug_mode` will be set to 2.

        """

        if debug_mode not in (0, 1, 2):
            debug_mode = 2

        # Either `steps` or `values` must be specified, but not both (xor)
        assert (steps is None) != (values is None)

        if isinstance(param_names, str):
            param_names = [param_names]

        nparams = len(param_names)
        hypo_maker.select_params(hypo_param_selections)

        if values is not None:
            if np.isscalar(values):
                values = np.array([values])
                assert nparams == 1
            for i, val in enumerate(values):
                if not np.isscalar(val):
                    # no scalar here, need a corresponding parameter name
                    assert nparams >= i+1
                else:
                    # a scalar, can either have only one parameter or at least
                    # this many
                    assert nparams == 1 or nparams >= i+1
                    if nparams > 1:
                        values[i] = np.array([val])

        else:
            ranges = [hypo_maker.params[pname].range for pname in param_names]
            if np.issubdtype(type(steps), int):
                assert steps >= 2
                values = [np.linspace(r[0], r[1], steps)*r[0].units
                          for r in ranges]
            else:
                assert len(steps) == nparams
                assert np.all(np.array(steps) >= 2)
                values = [np.linspace(r[0], r[1], steps[i])*r[0].units
                          for i, r in enumerate(ranges)]

        if nparams > 1:
            steplist = [[(pname, val) for val in values[i]]
                        for (i, pname) in enumerate(param_names)]
        else:
            steplist = [[(param_names[0], val) for val in values[0]]]

        #Number of steps must be > 0
        assert len(steplist) > 0

        points_acc = []
        if only_points is not None:
            assert len(only_points) == 1 or len(only_points) % 2 == 0
            if len(only_points) == 1:
                points_acc = only_points
            for i in range(0, len(only_points)-1, 2):
                points_acc.extend(
                    list(range(only_points[i], 1 + only_points[i + 1]))
                )

        # Instead of introducing another multitude of tests above, check here
        # whether the lists of steps all have the same length in case `outer`
        # is set to False
        if nparams > 1 and not outer:
            assert np.all(len(steps) == len(steplist[0]) for steps in steplist)
            loopfunc = zip
        else:
            # With single parameter, can use either `zip` or `product`
            loopfunc = product

        params = hypo_maker.params

        # Fix the parameters to be scanned if `profile` is set to True
        params.fix(param_names)

        results = {'steps': {}, 'results': []}
        results['steps'] = {pname: [] for pname in param_names}
        for i, pos in enumerate(loopfunc(*steplist)):
            if points_acc and i not in points_acc:
                continue

            msg = ''
            for (pname, val) in pos:
                params[pname].value = val
                results['steps'][pname].append(val)
                if isinstance(val, float):
                    msg += '%s = %.2f '%(pname, val)
                elif isinstance(val, ureg.Quantity):
                    msg += '%s = %.2f '%(pname, val.magnitude)
                else:
                    raise TypeError("val is of type %s which I don't know "
                                    "how to deal with in the output "
                                    "messages."% type(val))
            logging.info('Working on point ' + msg)
            hypo_maker.update_params(params)

            # TODO: consistent treatment of hypo_param_selections and scanning
            if not profile or not hypo_maker.params.free:
                logging.info('Not optimizing since `profile` set to False or'
                             ' no free parameters found...')
                best_fit = self.nofit_hypo(
                    data_dist=data_dist,
                    hypo_maker=hypo_maker,
                    hypo_param_selections=hypo_param_selections,
                    hypo_asimov_dist=hypo_maker.get_outputs(return_sum=True),
                    metric=metric,
                    **{k: v for k,v in kwargs.items() if k not in ["pprint","reset_free","check_octant"]}
                )
            else:
                logging.info('Starting optimization since `profile` requested.')
                best_fit, _ = self.fit_hypo(
                    data_dist=data_dist,
                    hypo_maker=hypo_maker,
                    hypo_param_selections=hypo_param_selections,
                    metric=metric,
                    minimizer_settings=minimizer_settings,
                    **kwargs
                )
                # TODO: serialisation!
                for k in best_fit.minimizer_metadata:
                    if k in ['hess', 'hess_inv']:
                        logging.debug("deleting %s", k)
                        del best_fit.minimizer_metadata[k]

            best_fit.metric_val = deepcopy(
                best_fit.metric_val.serializable_state
            )
            if isinstance(best_fit.hypo_asimov_dist, Sequence):
                best_fit.hypo_asimov_dist = [deepcopy(
                    best_fit.hypo_asimov_dist[i].serializable_state
                ) for i in range(len(best_fit.hypo_asimov_dist))]
            else:
                best_fit.hypo_asimov_dist = deepcopy(
                    best_fit.hypo_asimov_dist.serializable_state
                )

            # decide which information to retain based on chosen debug mode
            if debug_mode == 0 or debug_mode == 1:
                try:
                    del best_fit['fit_history']
                    del best_fit.hypo_asimov_dist
                except KeyError:
                    pass

            if debug_mode == 0:
                # torch the woods!
                try:
                    del best_fit.minimizer_metadata
                    del best_fit.minimizer_time
                except KeyError:
                    pass

            results['results'].append(best_fit)
            if outfile is not None:
                # store intermediate results
                to_file(results, outfile)

        return results

