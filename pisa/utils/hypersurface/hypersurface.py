"""
Tools for working with hypersurfaces, which are continuous functions in N-D
with arbitrary functional forms.

Hypersurfaces can be used to model systematic uncertainties derived from discrete
simulation datasets, for example for detedctor uncertainties.
"""

__all__ = ['Hypersurface', 'HypersurfaceParam', 'fit_hypersurfaces',
           'load_hypersurfaces']

__author__ = 'T. Stuttard, A. Trettin'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


import os
import collections
import copy

import numpy as np
from iminuit import Minuit
from iminuit.iminuit_warnings import HesseFailedWarning

from pisa import FTYPE, ureg
from pisa.utils.jsons import from_json, to_json
from pisa.core.pipeline import Pipeline
from pisa.core.binning import OneDimBinning, MultiDimBinning, is_binning
from pisa.core.map import Map
from pisa.core.param import Param, ParamSet
from pisa.utils.resources import find_resource
from pisa.utils.fileio import mkdir
from pisa.utils.log import logging, set_verbosity
from pisa.utils.comparisons import ALLCLOSE_KW
from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp
from .hypersurface_plotting import plot_bin_fits, plot_bin_fits_2d

'''
Hypersurface functional forms

   Define functional forms for HypersurfaceParam instances here.

   Functions defined here MUST:
     - Support numba guvectorization.
     - Function arguments must observed this convention:
         `p`, `<coefficient 0>`, ..., `<coefficient N>`, `out` where `p` is the
         systematic parameter, `out is the array to write the results to, and there are
         N coefficients of the parameterisation.

   The format of these arguments depends on the use case, of which there are two:
     - When fitting the function coefficients. This is done bin-wise using multiple
     datasets.
       - Params are then: `p` is array (one value per dataset), coefficients and `out`
         are scalar (representing a single bin).
     - Evaluating a fitted hypersurface. This is done for all bins simultaneously, using
       a single value for p.
       - Params are then: `p` is scalar (current value of systematic parameter,
         coefficients and `out` are arrays representing the hypersurfaces of all bins
         per bin.
'''


class linear_hypersurface_func(object):
    '''
    Linear hypersurface functional form

    f(p) = m * p
    '''

    def __init__(self):
        self.nargs = 1

    def __call__(self, p, m, out):
        result = m * p
        np.copyto(src=result, dst=out)

    def grad(self, p, m, out):
        # because m itself is not in the actual calculation, we have to broadcast
        # manually to yield the same shape as if we had done m*p and added one axis
        foo = m*p
        result = np.broadcast_to(p, foo.shape)[..., np.newaxis]
        np.copyto(src=result, dst=out)


class quadratic_hypersurface_func(object):
    '''
    Quadratic hypersurface functional form

    f(p) = m1*p + m2*p**2
    '''

    def __init__(self):
        self.nargs = 2

    def __call__(self, p, m1, m2, out):
        result = m1*p + m2*p**2
        np.copyto(src=result, dst=out)
    # the gradient *must* have all these arguments, even if they are un-used!

    def grad(self, p, m1, m2, out):
        # because m itself is not in the actual calculation, we have to broadcast
        # manually to yield the same shape as if we had done m*p and stacked on the last
        # axis
        foo = m1*p
        result = np.stack([np.broadcast_to(p, foo.shape),
                           np.broadcast_to(p**2, foo.shape)],
                          axis=-1
                          )
        np.copyto(src=result, dst=out)

class exponential_hypersurface_func(object):
    '''
    Exponential hypersurface functional form

    f(p) = exp(b*p) - 1

    The functional form ensures that it is zero at the nominal point.
    '''

    def __init__(self):
        self.nargs = 1

    def __call__(self, p, b, out):
        result = np.exp(b*p) - 1.
        np.copyto(src=result, dst=out)

    def grad(self, p, b, out):
        # because parameters and coefficients both appear, everything is broadcast
        # automatically
        result = np.array([p*np.exp(b*p)])[..., np.newaxis]
        np.copyto(src=result, dst=out)

class scaled_exponential_hypersurface_func(object):
    '''
    Exponential hypersurface functional form

    f(p) = (a + 1) * (exp(b*p) - 1)

    The functional form is chosen such that it is zero at the nominal point.
    If a strong prior is imposed on a, it becomes equivalent to the un-scaled
    exponential hypersurface function.
    '''

    def __init__(self):
        self.nargs = 2

    def __call__(self, p, a, b, out):
        result = (a + 1.) * (np.exp(b*p) - 1.)
        np.copyto(src=result, dst=out)

    def grad(self, p, a, b, out):
        # because parameters and coefficients both appear, everything is broadcast
        # automatically
        result = np.stack([np.exp(b*p) - 1., (a + 1.)*p*np.exp(b*p)], axis=-1)
        np.copyto(src=result, dst=out)

class logarithmic_hypersurface_func(object):
    '''
    Logarithmic hypersurface functional form

    f(p) = log(1 + mp)

    Allows the fit of an effectively linear multiplicative
    function while in logmode, since:
    exp(log(1 + mp) + h) = (1 + mp) exp(h)
    '''

    def __init__(self):
        self.nargs = 1

    def __call__(self, p, m, out):
        result = np.log(1 + m*p)
        np.copyto(src=result, dst=out)

    def grad(self, p, m, out):
        # because parameters and coefficients both appear, everything is broadcast
        # automatically
        result = np.array(p/(1 + m*p))[..., np.newaxis]
        np.copyto(src=result, dst=out)


# Container holding all possible functions
HYPERSURFACE_PARAM_FUNCTIONS = collections.OrderedDict()
HYPERSURFACE_PARAM_FUNCTIONS["linear"] = linear_hypersurface_func
HYPERSURFACE_PARAM_FUNCTIONS["quadratic"] = quadratic_hypersurface_func
HYPERSURFACE_PARAM_FUNCTIONS["exponential"] = exponential_hypersurface_func
HYPERSURFACE_PARAM_FUNCTIONS["exponential_scaled"] = scaled_exponential_hypersurface_func
HYPERSURFACE_PARAM_FUNCTIONS["logarithmic"] = logarithmic_hypersurface_func

class Hypersurface(object):
    '''
    A class defining the hypersurface

    Contains :
      - A single common intercept
      - N systematic parameters, inside which the functional form is defined

    This class can be configured to hold both the functional form of the hypersurface
    and values (likely fitted from simulation datasets) for the free parameters of this
    functional form.

    Fitting functionality is provided to fit these free parameters.

    This class can simultaneously hold hypersurfaces for every bin in a histogram (Map).

    The functional form of the systematic parameters can be arbitrarily complex.

    The class has a fit method for fitting the hypersurface to some data (e.g.
    discrete systematics sets).

    Serialization functionality is included to allow fitted hypersurfaces to be stored
    to a file and re-loaded later (e.g. to be used in analysis).

    The main use cases are:
        1) Fit hypersurfaces
             - Define the desired HypersurfaceParams (functional form, intial coefficient guesses).
             - Instantiate the `Hypersurface` class, providing the hypersurface params and initial intercept guess.
             - Use `Hypersurface.fit` function (or more likely the `fit_hypersurfaces` helper function provided below),
               to fit the hypersurface coefficients to some provided datasets.
             - Store to file
        2) Evaluate an existing hypersurface
             - Load existing fitted Hypersurface from a file (`load_hypersurfaces` helper function)
             - Get the resulting hypersurface value for each bin for a given set of systemaic param
               values using the `Hypersurface.evaluate` method.
             - Use the hypersurface value for each bin to re-weight events

    The class stores information about the datasets used to fit the hypersurfaces, including the Maps
    used and nominal and systematic parameter values.

    Parameters
    ----------
    params : list
        A list of HypersurfaceParam instances defining the hypersurface.
        The `initial_fit_coeffts` values in this instances will be used as the starting
        point for any fits.

    initial_intercept : float
        Starting point for the hypersurface intercept in any fits

    log : bool, optional
        Set hypersurface to log mode. The surface is fit to the log of the bin counts.
        The fitted surface is exponentiated during evaluation. Default: False
    '''

    def __init__(self, params, initial_intercept=None, log=False):

        # Store args
        self.initial_intercept = initial_intercept
        # Store params as dict for ease of lookup
        self.params = collections.OrderedDict()
        for param in params:
            assert param.name not in self.params, "Duplicate param name found : %s" % param.name
            self.params[param.name] = param

        self.log = log
        # Internal state
        self._initialized = False

        # Containers for storing fitting information
        self.fit_complete = False
        self.fit_info_stored = False
        self.fit_maps_norm = None
        self.fit_maps_raw = None
        self.fit_chi2 = None
        self.fit_cov_mat = None
        self.fit_method = None

        # Serialization
        self._serializable_state = None

        # Legacy handling
        self.using_legacy_data = False

    def _init(self, binning, nominal_param_values):
        '''
        Actually initialise the hypersurface.

        Internal function, not to be called by a user.
        '''

        #
        # Binning
        #

        # Store the binning
        self.binning = binning

        # Set a default initial intercept value if none provided
        if self.initial_intercept is None:
            self.initial_intercept = 0. if self.log else 1.

        # Create the fit coefficient arrays
        # Have one fit per bin
        self.intercept = np.full(
            self.binning.shape, self.initial_intercept, dtype=FTYPE)
        self.intercept_sigma = np.full_like(self.intercept, np.NaN)
        for param in list(self.params.values()):
            param._init_fit_coefft_arrays(self.binning)

        #
        # Nominal values
        #

        # Store the nominal param values
        # TODO better checks, including not already set
        for param in list(self.params.values()):
            param.nominal_value = nominal_param_values[param.name]

        #
        # Done
        #

        self._initialized = True

    @property
    def initialized(self):
        '''
        Return flag indicating if hypersurface has been initialized
        Not giving use direct write-access to the variable as they should nt be setting it themselves
        '''
        return self._initialized

    @property
    def param_names(self):
        '''
        Return the (ordered) names of the systematic parameters
        '''
        return list(self.params.keys())

    def evaluate(self, param_values, bin_idx=None, return_uncertainty=False):
        '''
        Evaluate the hypersurface, using the systematic parameter values provided.
        Uses the current internal values for all functional form coefficients.

        Parameters
        ----------
        param_values : dict
            A dict specifying the values of the systematic parameters to use in the evaluation.
            Format is :
                { sys_param_name_0 : sys_param_0_val, ..., sys_param_name_N : sys_param_N_val }.
                The keys must be string and correspond to the HypersurfaceParam instances.
                The values must be scalars.

        bin_idx : tuple or None
            Optionally can specify a particular bin (using numpy indexing). d
            Othewise will evaluate all bins.

        return_uncertainty : bool, optional
            return the uncertainty on the output (default: False)
        '''

        assert self._initialized, "Cannot evaluate hypersurface, it haas not been initialized"

        #
        # Check inputs
        #

        # Determine number of sys param values (per sys param)
        # This will be >1 when fitting, and == 1 when evaluating the hypersurface within the stage
        num_param_values = np.asarray(list(param_values.values())[0]).size

        # Check same number of values for all sys params
        for k, v in list(param_values.items()):
            n = np.asarray(v).size
            assert n == num_param_values, "All sys params must have the same number of values"

        # Determine whether using single bin or not
        single_bin_mode = bin_idx is not None

        #
        # Prepare output array
        #

        # Determine shape of output array
        # Two possible cases, with limitations on both based on how the sys param functional forms are defined
        if not single_bin_mode:
            # Case 1 : Calculating for all bins simultaneously (e.g. `bin_idx is None`)
            #          Only support a single scalar value for each systematic parameters
            #          Use case is evaluating the hypersurfaces during the hypersurface stage
            assert num_param_values == 1, "Can only provide one value per sys param when evaluating all bins simultaneously"
            for v in list(param_values.values()):
                assert np.isscalar(
                    v), "sys param values must be a scalar when evaluating all bins simultaneously"
            out_shape = self.binning.shape
            bin_idx = Ellipsis

        else:
            # Case 2 : Calculating for multiple sys param values, but only a single bin
            #          Use case is fitting the hypersurfaces fucntional form fit params
            out_shape = (num_param_values,)

        # Create the output array
        out = np.full(out_shape, np.NaN, dtype=FTYPE)

        #
        # Evaluate the hypersurface
        #

        # Start with the intercept
        for i in range(num_param_values):
            if single_bin_mode:
                out[i] = self.intercept[bin_idx]
            else:
                np.copyto(src=self.intercept[bin_idx], dst=out[bin_idx])

        # Evaluate each individual parameter
        for k, p in list(self.params.items()):
            param_val = param_values[k] if self.using_legacy_data else param_values[k] - p.nominal_value
            p.evaluate(param_val, out=out, bin_idx=bin_idx)

        output_factors = np.exp(out) if self.log else out

        if return_uncertainty:
            # create buffer array for the gradients
            n_coeffs = 1  # start with 1 because intercept is an additional coefficient
            for param in list(self.params.values()):
                n_coeffs += param.num_fit_coeffts
            gradient_buffer = np.full(
                out_shape + (n_coeffs,), np.NaN, dtype=FTYPE)
            # Start with the intercept, its gradient is always 1
            gradient_buffer[..., 0] = 1.

            # Evaluate gradient each individual parameter and store in buffer.
            i = 1  # start at one because the intercept was already treated
            for k, p in list(self.params.items()):
                gbuf = np.full(out_shape + (p.num_fit_coeffts,),
                               np.NaN, dtype=FTYPE)
                param_val = param_values[k] if self.using_legacy_data else param_values[k] - p.nominal_value
                p.gradient(param_val, out=gbuf, bin_idx=bin_idx)
                for j in range(p.num_fit_coeffts):
                    gradient_buffer[..., i] = gbuf[..., j]
                    i += 1

            # In log-mode, the output is exponentiated. For the gradient this simply means multiplying
            # with the output itself.
            if self.log:
                gradient_buffer = output_factors[...,
                                                 np.newaxis]*gradient_buffer
            # Calculate uncertainty from gradients and covariance matrix
            transformed_jacobian = np.einsum(
                '...j,...kj->...k', gradient_buffer, self.fit_cov_mat[bin_idx])
            variance = np.einsum(
                '...j,...j', transformed_jacobian, gradient_buffer)
            assert np.all(variance[np.isfinite(variance)] >= 0.), "invalid covariance"

        if return_uncertainty:
            return output_factors, np.sqrt(variance)
        else:
            return output_factors

    def fit(self, nominal_map, nominal_param_values, sys_maps, sys_param_values,
            norm=True, method="L-BFGS-B", fix_intercept=False, intercept_bounds=None,
            intercept_sigma=None, include_empty=False, keep_maps=True):
        '''
        Fit the hypersurface coefficients (in every bin) to best match the provided
        nominal and systematic datasets.

        Writes the results directly into this data structure.

        Parameters
        ----------
        nominal_map : Map
            Map from the nominal dataset

        nominal_param_values : dict
            Value of each systematic param used to generate the nominal dataset
            Format: { param_0_name : param_0_nom_val, ..., param_N_name : param_N_nom_val }

        sys_maps : list of Maps
            List containing the Map from each systematic dataset

        sys_param_values : list of dicts
            List where each element if a dict containing the values of each systematic
            param used to generate the that dataset Each list element specified the
            parameters for the corresponding element in `sys_maps`

        norm : bool
            Normalise the maps to the nominal map. This is what you want to do when
            using the hypersurface to re-weight simulation (which is the main use case).
            In principal the hypersurfaces are more general though and could be used for
            other tasks too, hence this option.

        method : str
            `method` arg to pass to `scipy.optimize.minimiza`

        fix_intercept : bool
            Fix intercept to the initial intercept.

        intercept_bounds : 2-tuple, optional
            Bounds on the intercept. Default is None (no bounds)

        include_empty : bool
            Include empty bins in the fit. If True, empty bins are included with value 0
            and sigma 1.
            Default: False
        
        keep_maps : bool
            Keep maps used to make the fit. If False, maps will be set to None after
            the fit is complete. This helps to reduce the size of JSONS if the 
            Hypersurface is to be stored on disk.
        '''

        #
        # Check inputs
        #

        # Check nominal dataset definition
        assert isinstance(nominal_map, Map)
        assert isinstance(nominal_param_values, collections.Mapping)
        assert set(nominal_param_values.keys()) == set(self.param_names)
        assert all([isinstance(k, str) for k in nominal_param_values.keys()])
        assert all([np.isscalar(v) for v in nominal_param_values.values()])
        # Check systematic dataset definitions
        assert isinstance(sys_maps, collections.Sequence)
        assert isinstance(sys_param_values, collections.Sequence)
        assert len(sys_maps) == len(sys_param_values)
        for sys_map, sys_param_vals in zip(sys_maps, sys_param_values):
            assert isinstance(sys_map, Map)
            assert isinstance(sys_param_vals, collections.Mapping)
            assert set(sys_param_vals.keys()) == set(self.param_names)
            assert all([isinstance(k, str) for k in sys_param_vals.keys()])
            assert all([np.isscalar(v) for v in sys_param_vals.values()])
            assert sys_map.binning == nominal_map.binning

        assert not (
            include_empty and self.log), "empty bins cannot be included in log mode"
        #
        # Format things before getting started
        #

        # Store thr fitting method
        self.fit_method = method

        # Initialise hypersurface using nominal dataset
        self._init(binning=nominal_map.binning,
                   nominal_param_values=nominal_param_values)

        # Combine nominal and sys sets
        maps = [nominal_map] + sys_maps
        param_values = [nominal_param_values] + sys_param_values

        # Store raw maps
        self.fit_maps_raw = maps

        # Convert params values from `list of dicts` to `dict of lists`
        param_values_dict = {name: np.array([p[name] for p in param_values])
                             for name in list(param_values[0].keys())}

        # Save the param values used for fitting in the param objects (useful for plotting later)
        for name, values in list(param_values_dict.items()):
            self.params[name].fit_param_values = values

        # Format the fit `x` values : [ [sys param 0 values], [sys param 1 values], ... ]
        # Order of the params must match the order in `self.params`
        x = np.asarray([param_values_dict[param_name]
                        for param_name in list(self.params.keys())], dtype=FTYPE)
        # Prepare covariance matrix array
        self.fit_cov_mat = np.full(
            list(self.binning.shape)+[self.num_fit_coeffts, self.num_fit_coeffts], np.NaN)

        #
        # Normalisation
        #

        # All map values are finite, but if have empty bins the nominal map will end up
        # with inf bins in the normalised map (divide by zero). Use a mask to handle
        # this.
        finite_mask = nominal_map.nominal_values != 0

        # Normalise bin values, if requested
        if norm:

            # Normalise the maps by dividing the nominal map This means the hypersurface
            # results can be interpretted as a re-weighting factor, relative to the
            # nominal

            # Formalise, handling inf values
            normed_maps = []
            for m in maps:
                norm_m = copy.deepcopy(m)
                norm_m.hist[finite_mask] = norm_m.hist[finite_mask] / \
                    unp.nominal_values(nominal_map.hist[finite_mask])
                norm_m.hist[~finite_mask] = ufloat(np.NaN, np.NaN)
                normed_maps.append(norm_m)

            # Store for plotting later
            self.fit_maps_norm = normed_maps

        # Record that fit info is now stored
        self.fit_info_stored = True

        #
        # Some final checks
        #

        # Not expecting any bins to have negative values (negative counts doesn't make
        # sense)
        # TODO hypersurface in general could consider -ve values (no explicitly
        # tied to histograms), so maybe can relax this constraint
        for m in self.fit_maps:
            assert np.all(m.nominal_values[finite_mask]
                          >= 0.), "Found negative bin counts"

        #
        # Loop over bins
        #

        for bin_idx in np.ndindex(self.binning.shape):  # TODO grab from input map

            #
            # Format this bin's data for fitting
            #

            # Format the fit `y` values : [ bin value 0, bin_value 1, ... ]
            # Also get the corresonding uncertainty
            y = np.asarray([m.nominal_values[bin_idx]
                            for m in self.fit_maps], dtype=FTYPE)
            y_sigma = np.asarray([m.std_devs[bin_idx]
                                  for m in self.fit_maps], dtype=FTYPE)

            # Create a mask for keeping all these points
            # May remove some points before fitting if find issues
            scan_point_mask = np.ones(y.shape, dtype=bool)

            # Cases where we have a y_sigma element = 0 (normally because the
            # corresponding y element = 0) screw up the fits (least squares divides by
            # sigma, so get infs) By default, we ignore empty bins. If the user wishes
            # to include them, it can be done with a value of zero and standard
            # deviation of 1.
            bad_sigma_mask = y_sigma == 0.
            if bad_sigma_mask.sum() > 0:
                if include_empty:
                    y_sigma[bad_sigma_mask] = 1.
                else:
                    scan_point_mask = scan_point_mask & ~bad_sigma_mask

            # Apply the mask to get the values I will actually use
            x_to_use = np.array([xx[scan_point_mask] for xx in x])
            y_to_use = y[scan_point_mask]
            y_sigma_to_use = y_sigma[scan_point_mask]

            # Checks
            assert x_to_use.shape[0] == len(self.params)
            assert x_to_use.shape[1] == y_to_use.size

            # Get flat list of the fit param guesses
            # The param coefficients are ordered as [ param 0 cft 0, ..., param 0 cft N,
            # ..., param M cft 0, ..., param M cft N ]
            p0_intercept = self.intercept[bin_idx]
            p0_param_coeffts = [param.get_fit_coefft(bin_idx=bin_idx, coefft_idx=i_cft)
                                for param in list(self.params.values())
                                for i_cft in range(param.num_fit_coeffts)]
            if fix_intercept:
                p0 = np.array(p0_param_coeffts, dtype=FTYPE)
            else:
                p0 = np.array([p0_intercept] + p0_param_coeffts, dtype=FTYPE)

            #
            # Check if have valid data in this bin
            #

            # If have empty bins, cannot fit In particular, if the nominal map has an
            # empty bin, it cannot be rescaled (x * 0 = 0) If this case, no need to try
            # fitting

            # Check if have NaNs/Infs
            if np.any(~np.isfinite(y_to_use)):  # TODO also handle missing sigma

                # Not fitting, add empty variables
                popt = np.full_like(p0, np.NaN)
                pcov = np.NaN

            # Otherwise, fit...
            else:

                #
                # Fit
                #

                # Must have at least as many sets as free params in fit or else curve_fit will fail
                assert y.size >= p0.size, "Number of datasets used for fitting (%i) must be >= num free params (%i)" % (
                    y.size, p0.size)

                # Define a callback function for use with `curve_fit`
                #   x : sys params
                #   p : func/shape params
                def callback(x, *p):

                    # Note that this is using the dynamic variable `bin_idx`, which
                    # cannot be passed as an arg as `curve_fit` cannot handle fixed
                    # parameters.
                    #
                    # Unflatten list of the func/shape params, and write them to the
                    # hypersurface structure
                    self.intercept[bin_idx] = self.initial_intercept if fix_intercept else p[0]
                    i = 0 if fix_intercept else 1
                    for param in list(self.params.values()):
                        for j in range(param.num_fit_coeffts):
                            bin_fit_idx = tuple(list(bin_idx) + [j])
                            param.fit_coeffts[bin_fit_idx] = p[i]
                            i += 1

                    # Unflatten sys param values
                    params_unflattened = collections.OrderedDict()
                    for i in range(len(self.params)):
                        param_name = list(self.params.keys())[i]
                        params_unflattened[param_name] = x[i]

                    return self.evaluate(params_unflattened, bin_idx=bin_idx)

                inv_param_sigma = []
                if intercept_sigma is not None:
                    inv_param_sigma.append(1./intercept_sigma)
                else:
                    inv_param_sigma.append(0.)
                for param in list(self.params.values()):
                    if param.coeff_prior_sigma is not None:
                        for j in range(param.num_fit_coeffts):
                            inv_param_sigma.append(
                                1./param.coeff_prior_sigma[j])
                    else:
                        for j in range(param.num_fit_coeffts):
                            inv_param_sigma.append(0.)
                inv_param_sigma = np.array(inv_param_sigma)
                assert np.all(np.isfinite(
                    inv_param_sigma)), "invalid values found in prior sigma. They must not be zero."

                # coefficient names to pass to Minuit. Not strictly necessary
                coeff_names = [] if fix_intercept else ['intercept']
                for name, param in self.params.items():
                    for j in range(param.num_fit_coeffts):
                        coeff_names.append(name + '_p{:d}'.format(j))

                def loss(p):
                    '''
                    Loss to be minimized during the fit.
                    '''
                    fvals = callback(x_to_use, *p)
                    return np.sum(((fvals - y_to_use)/y_sigma_to_use)**2) + np.sum((inv_param_sigma*p)**2)

                # Define fit bounds for `minimize`. Bounds are pairs of (min, max)
                # values for each parameter in the fit. Use 'None' in place of min/max
                # if there is
                # no bound in that direction.
                fit_bounds = []
                if intercept_bounds is None:
                    fit_bounds.append(tuple([None, None]))
                else:
                    assert (len(intercept_bounds) == 2) and (
                        np.ndim(intercept_bounds) == 1), "intercept bounds must be given as 2-tuple"
                    fit_bounds.append(intercept_bounds)

                for param in self.params.values():
                    if param.bounds is None:
                        fit_bounds.extend(
                            ((None, None),)*param.num_fit_coeffts)
                    else:
                        if np.ndim(param.bounds) == 1:
                            assert len(
                                param.bounds) == 2, "bounds on single coefficients must be given as 2-tuples"
                            fit_bounds.append(param.bounds)
                        elif np.ndim(param.bounds) == 2:
                            assert np.all([len(t) == 2 for t in param.bounds]
                                          ), "bounds must be given as a tuple of 2-tuples"
                            fit_bounds.extend(param.bounds)

                # Define the EPS (step length) used by the fitter Need to take care with
                # floating type precision, don't want to go smaller than the FTYPE being
                # used by PISA can handle
                eps = np.finfo(FTYPE).eps

                # Debug logging
                test_bin_idx = (0, 0, 0)
                if bin_idx == test_bin_idx:
                    msg = ">>>>>>>>>>>>>>>>>>>>>>>\n"
                    msg += "Curve fit inputs to bin %s :\n" % (bin_idx,)
                    msg += "  x           : \n%s\n" % x
                    msg += "  y           : \n%s\n" % y
                    msg += "  y sigma     : \n%s\n" % y_sigma
                    msg += "  x used      : \n%s\n" % x_to_use
                    msg += "  y used      : \n%s\n" % y_to_use
                    msg += "  y sigma used: \n%s\n" % y_sigma_to_use
                    msg += "  p0          : %s\n" % p0
                    msg += "  bounds      : \n%s\n" % fit_bounds
                    msg += "  inv sigma   : \n%s\n" % inv_param_sigma
                    msg += "  fit method  : %s\n" % self.fit_method
                    msg += "<<<<<<<<<<<<<<<<<<<<<<<"
                    logging.debug(msg)

                # Perform fit
                # errordef =1 for least squares fit and 0.5 for nllh fit
                m = Minuit.from_array_func(loss, p0,
                                           # only initial step size, not very important
                                           error=(0.1)*len(p0),
                                           limit=fit_bounds,
                                           name=coeff_names,
                                           errordef=1)
                m.migrad()
                m.hesse()

                popt = m.np_values()
                try:
                    pcov = m.np_matrix()
                except:
                    logging.warn(f"HESSE call failed for bin {bin_idx}, covariance matrix unavailable")
                    pcov = np.full((len(p0), len(p0)), np.nan)
                if bin_idx == test_bin_idx:
                    logging.debug(m.get_fmin())
                    logging.debug(m.get_param_states())
                    logging.debug(m.covariance)

            #
            # Re-format fit results
            #

            # Use covariance matrix to get uncertainty in fit parameters Using
            # uncertainties.correlated_values, and will extract the std dev (including
            # correlations) shortly Fit may fail to determine covariance matrix
            # (method-dependent), so only do this if have a finite covariance matrix
            corr_vals = correlated_values(popt, pcov) if np.all(
                np.isfinite(pcov)) else None

            # Write the fitted param results (and sigma, if available) back to the
            # hypersurface structure
            i = 0
            if not fix_intercept:
                self.intercept[bin_idx] = popt[i]
                self.intercept_sigma[bin_idx] = np.NaN if corr_vals is None else corr_vals[i].std_dev
                i += 1
            for param in list(self.params.values()):
                for j in range(param.num_fit_coeffts):
                    idx = param.get_fit_coefft_idx(
                        bin_idx=bin_idx, coefft_idx=j)
                    param.fit_coeffts[idx] = popt[i]
                    param.fit_coeffts_sigma[idx] = np.NaN if corr_vals is None else corr_vals[i].std_dev
                    i += 1
            # Store the covariance matrix
            if fix_intercept and np.all(np.isfinite(pcov)):
                self.fit_cov_mat[bin_idx] = np.pad(pcov, ((1, 0), (1, 0)))
            else:
                self.fit_cov_mat[bin_idx] = pcov

        #
        # chi2
        #

        # Compare the result of the fitted hypersurface function with the actual data
        # points used for fitting Compute the resulting chi2 to have an estimate of the
        # fit quality

        self.fit_chi2 = []

        # Loop over datasets
        for i_set in range(self.num_fit_sets):

            # Get expected bin values according tohypersurface value
            predicted = self.evaluate(
                {name: values[i_set] for name, values in list(param_values_dict.items())})

            # Get the observed value
            observed = self.fit_maps[i_set].nominal_values
            sigma = self.fit_maps[i_set].std_devs
            # we have to apply the same condition on which values we include
            # as we did during the fit above
            valid_idx = sigma > 0.
            if include_empty:
                sigma[~valid_idx] = 1.

            # Compute chi2
            with np.errstate(divide='ignore'):
                chi2 = ((predicted - observed) / sigma) ** 2

            # Add to container
            self.fit_chi2.append(chi2)

        # Combine into single array
        self.fit_chi2 = np.stack(self.fit_chi2, axis=-1).astype(FTYPE)
        
        if not keep_maps:
            self.fit_maps_raw = None
            self.fit_maps_norm = None
        # Record some provenance info about the fits
        self.fit_complete = True

    @property
    def nominal_values(self):
        '''
        Return the stored nominal parameter for each dataset
        Returns: { param_0_name : param_0_nom_val, ..., param_N_name : param_N_nom_val }
        '''
        assert self.fit_info_stored, "Cannot get fit dataset nominal values, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")
        return collections.OrderedDict([(name, param.nominal_value) for name, param in list(self.params.items())])

    @property
    def fit_param_values(self):
        '''
        Return the stored systematic parameters from the datasets used for fitting
        Returns: { param_0_name : [ param_0_sys_val_0, ..., param_0_sys_val_M ], ..., param_N_name : [ param_N_sys_val_0, ..., param_N_sys_val_M ] }
        '''
        assert self.fit_info_stored, "Cannot get fit dataset param values, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")
        return collections.OrderedDict([(name, param.fit_param_values) for name, param in list(self.params.items())])

    def get_nominal_mask(self):
        '''
        Return a mask indicating which datasets have nominal values for all parameters
        '''

        assert self.fit_info_stored, "Cannot get nominal mask, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")

        nom_mask = np.ones((self.num_fit_sets,), dtype=bool)

        for param in list(self.params.values()):
            nom_mask = nom_mask & np.isclose(
                param.fit_param_values, param.nominal_value)

        return nom_mask

    def get_on_axis_mask(self, param_name):
        '''
        Return a mask indicating which datasets are "on-axis" for a given parameter.

        "On-axis" means "generated using the nominal value for this parameter". Parameters other
        than the one specified can have non-nominal values.

        Parameters
        ----------
        param_name : str
            The name of systematic parameter for which we want on-axis datasets
        '''

        assert self.fit_info_stored, "Cannot get on-axis mask, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")

        assert param_name in self.param_names

        on_axis_mask = np.ones((self.num_fit_sets,), dtype=bool)

        # Loop over sys params
        for param in list(self.params.values()):

            # Ignore the chosen param
            if param.name != param_name:

                # Define a "nominal" mask
                on_axis_mask = on_axis_mask & np.isclose(
                    param.fit_param_values, param.nominal_value)

        return on_axis_mask

    def report(self, bin_idx=None):
        '''
        Return a string version of the hypersurface contents

        Parameters
        ----------
        bin_idx : tuple of None
            Specify a particular bin (using numpy indexing). In this case only report on that bin.
        '''

        msg = ""

        # Fit results
        msg += ">>>>>> Fit coefficients >>>>>>" + "\n"
        bin_indices = np.ndindex(
            self.binning.shape) if bin_idx is None else [bin_idx]
        for bin_idx in bin_indices:
            msg += "  Bin %s :" % (bin_idx,) + "\n"
            msg += "     Intercept : %0.5g" % (self.intercept[bin_idx],) + "\n"
            for param in list(self.params.values()):
                msg += "     %s : %s" % (param.name, ", ".join(["%0.5g" % param.get_fit_coefft(
                    bin_idx=bin_idx, coefft_idx=cft_idx) for cft_idx in range(param.num_fit_coeffts)])) + "\n"
        msg += "<<<<<< Fit coefficients <<<<<<" + "\n"

        return msg

    def __str__(self):
        return self.report()

    @property
    def fit_maps(self):
        '''
        Return the `Map instances used for fitting
        These will be normalised if the fit was performend to normalised maps.
        '''
        assert self.fit_info_stored, "Cannot get fit maps, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")
        return self.fit_maps_raw if self.fit_maps_norm is None else self.fit_maps_norm

    @property
    def num_fit_sets(self):
        '''
        Return number of datasets used for fitting
        '''
        assert self.fit_info_stored, "Cannot get fit sets, fit info not stored%s" % (
            " (using legacy data)" if self.using_legacy_data else "")
        return len(list(self.fit_param_values.values())[0])

    @property
    def num_fit_coeffts(self):
        '''
        Return the total number of coefficients in the hypersurface fit
        This is the overall intercept, plus the coefficients for each individual param
        '''
        return int(1 + np.sum([param.num_fit_coeffts for param in list(self.params.values())]))

    @property
    def fit_coeffts(self):
        '''
        Return all coefficients, in all bins, as a single array
        This is the overall intercept, plus the coefficients for each individual param
        Dimensions are: [binning ..., fit coeffts]
        '''

        array = [self.intercept]
        for param in list(self.params.values()):
            for i in range(param.num_fit_coeffts):
                array.append(param.get_fit_coefft(coefft_idx=i))
        array = np.stack(array, axis=-1)
        return array

    @fit_coeffts.setter
    def fit_coeffts(self, fit_coeffts):
        '''
        Setter to conveniently set the coefficients in the parameters
        of the hypersurface in the same order in which they are also
        returned by the getter.
        '''
        assert fit_coeffts.shape == self.fit_coeffts.shape, "incorrect shape of coefficients"
        self.intercept = fit_coeffts[..., 0]
        n = 1
        for param in self.params.values():
            for i in range(param.num_fit_coeffts):
                idx = param.get_fit_coefft_idx(coefft_idx=i)
                param.fit_coeffts[idx] = fit_coeffts[..., n]
                n += 1

    @property
    def fit_coefft_labels(self):
        '''
        Return labels for each fit coefficient
        '''
        return ["intercept"] + ["%s p%i" % (param.name, i) for param in list(self.params.values()) for i in range(param.num_fit_coeffts)]

    @property
    def serializable_state(self):
        """
        OrderedDict containing savable state attributes
        """

        if self._serializable_state is None:  # TODO always redo?

            state = collections.OrderedDict()

            state["_initialized"] = self._initialized
            state["binning"] = self.binning.serializable_state
            state["initial_intercept"] = self.initial_intercept
            state["log"] = self.log
            state["intercept"] = self.intercept
            state["intercept_sigma"] = self.intercept_sigma
            state["fit_complete"] = self.fit_complete
            state["fit_info_stored"] = self.fit_info_stored
            state["fit_maps_norm"] = self.fit_maps_norm
            state["fit_maps_raw"] = self.fit_maps_raw
            state["fit_chi2"] = self.fit_chi2
            state["fit_cov_mat"] = self.fit_cov_mat
            state["fit_method"] = self.fit_method
            state["using_legacy_data"] = self.using_legacy_data

            state["params"] = collections.OrderedDict()
            for name, param in list(self.params.items()):
                state["params"][name] = param.serializable_state

            self._serializable_state = state

        return self._serializable_state

    @classmethod
    def from_state(cls, state):
        """
        Instantiate a new object from the contents of a serialized state dict

        Parameters
        ----------
        resource : dict
            A dict

        See Also
        --------
        to_json
        """

        #
        # Get the state
        #

        # If it is not already a a state, alternativey try to load it in case a JSON
        # file was passed
        if not isinstance(state, collections.Mapping):
            state = from_json(state)

        #
        # Create params
        #

        params = []

        # Loop through params in the state
        params_state = state.pop("params")
        for param_name, param_state in list(params_state.items()):
            param = HypersurfaceParam.from_state(param_state)
            params.append(param)

        #
        # Create hypersurface
        #

        # Instantiate
        hypersurface = cls(
            params=params,
            initial_intercept=state.pop("initial_intercept"),
        )

        # Add binning
        hypersurface.binning = MultiDimBinning(**state.pop("binning"))

        # Add maps
        fit_maps_raw = state.pop("fit_maps_raw")
        hypersurface.fit_maps_raw = None if fit_maps_raw is None else [
            Map(**map_state) for map_state in fit_maps_raw]
        fit_maps_norm = state.pop("fit_maps_norm")
        hypersurface.fit_maps_norm = None if fit_maps_norm is None else [
            Map(**map_state) for map_state in fit_maps_norm]

        # Define rest of state
        for k in list(state.keys()):
            setattr(hypersurface, k, state.pop(k))

        return hypersurface


class HypersurfaceParam(object):
    '''
    A class representing one of the parameters (and corresponding functional forms) in
    the hypersurface.

    A user creates the initial instances of thse params, before passing the to the
    Hypersurface instance. Once this has happened, the user typically does not need to
    directly interact woth these HypersurfaceParam instances.

    Parameters
    ----------
    name : str
        Name of the parameter

    func_name : str
        Name of the hypersurface function to use.
        See "Hypersurface functional forms" section for more details, including
        available functions.

    initial_fit_coeffts : array
        Initial values for the coefficients of the functional form
        Number and meaning of coefficients depends on functional form

    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds. Each
        element of the tuple must be either an array with the length equal to the number
        of parameters, or a scalar (in which case the bound is taken to be the same for
        all parameters.) Use ``np.inf`` with an appropriate sign to disable bounds on
        all or some parameters.

    coeff_prior_sigma : array, optional
        Prior sigma values for the coefficients. If None (default), no regularization
        will be applied during the fit.
    '''

    def __init__(self, name, func_name, initial_fit_coeffts=None, bounds=None, coeff_prior_sigma=None):

        # Store basic members
        self.name = name

        # Handle functional form fit parameters
        self.fit_coeffts = None  # Fit params container, not yet populated
        self.fit_coeffts_sigma = None  # Fit param sigma container, not yet populated
        # The initial values for the fit parameters
        self.initial_fit_coeffts = initial_fit_coeffts
        self.bounds = bounds
        self.coeff_prior_sigma = coeff_prior_sigma

        # Record information relating to the fitting
        self.fitted = False  # Flag indicating whether fit has been performed
        # The values of this sys param in each of the fitting datasets
        self.fit_param_values = None

        # Placeholder for nominal value
        self.nominal_value = None

        # Serialization
        self._serializable_state = None
        self.binning_shape = None  # initialized when used in Hypersurface
        #
        # Init the functional form
        #

        # Get the function
        self.func_name = func_name
        self._hypersurface_func = self._get_hypersurface_func(self.func_name)

        # Get the number of functional form parameters
        self.num_fit_coeffts = self._hypersurface_func.nargs
        if self.coeff_prior_sigma is not None:
            assert len(
                self.coeff_prior_sigma) == self.num_fit_coeffts, "number of prior sigma values must equal the number of parameters."
        # Check and init the fit param initial values
        # TODO Add support for "per bin" initial values
        if initial_fit_coeffts is None:
            # No values provided, use 0 for all
            self.initial_fit_coeffts = np.zeros(
                self.num_fit_coeffts, dtype=FTYPE)
        else:
            # Use the provided initial values
            self.initial_fit_coeffts = np.array(self.initial_fit_coeffts)
            assert self.initial_fit_coeffts.size == self.num_fit_coeffts, "'initial_fit_coeffts' should have %i values, found %i" % (
                self.num_fit_coeffts, self.initial_fit_coeffts.size)

    def _get_hypersurface_func(self, func_name):
        '''
        Find the function defining the hypersurface functional form.

        User specifies this by it's string name, which must correspond to a pre-defined
        function in `HYPERSURFACE_PARAM_FUNCTIONS`.

        Internal function, not to be called by a user.
        '''

        assert isinstance(func_name, str), "'func_name' must be a string"

        assert func_name in HYPERSURFACE_PARAM_FUNCTIONS, "Cannot find hypersurface function '%s', choose from %s" % (
            func_name, list(HYPERSURFACE_PARAM_FUNCTIONS.keys()))
        return HYPERSURFACE_PARAM_FUNCTIONS[func_name]()

    def _init_fit_coefft_arrays(self, binning):
        '''
        Create the arrays for storing the fit parameters
        Have one fit per bin, for each parameter
        The shape of the `self.fit_coeffts` arrays is: (binning shape ..., num fit params )

        Internal function, not to be called by a user.
        '''

        arrays = []

        self.binning_shape = binning.shape

        for fit_coefft_initial_value in self.initial_fit_coeffts:

            fit_coefft_array = np.full(
                self.binning_shape, fit_coefft_initial_value, dtype=FTYPE)
            arrays.append(fit_coefft_array)

        self.fit_coeffts = np.stack(arrays, axis=-1)
        self.fit_coeffts_sigma = np.full_like(self.fit_coeffts, np.NaN)

    def evaluate(self, param, out, bin_idx=None):
        '''
        Evaluate the functional form for the given `param` values.
        Uses the current values of the fit coefficients.

        By default evaluates all bins, but optionally can specify a particular bin (used
        when fitting).
        '''

        # Create an array to file with this contorubtion
        this_out = np.full_like(out, np.NaN, dtype=FTYPE)

        # Form the arguments to pass to the functional form
        # Need to be flexible in terms of the number of fit parameters
        args = [param]
        for cft_idx in range(self.num_fit_coeffts):
            args += [self.get_fit_coefft(bin_idx=bin_idx, coefft_idx=cft_idx)]
        args += [this_out]

        # Call the function
        self._hypersurface_func(*args)

        # Add to overall hypersurface result
        out += this_out

    def gradient(self, param, out, bin_idx=None):
        '''
        Evaluate gradient of the functional form for the given `param` values.
        Uses the current values of the fit coefficients.

        By default evaluates all bins, but optionally can specify a particular bin (used when fitting).
        '''
        # Create an array to fill with the gradient
        this_out = np.full_like(out, np.NaN, dtype=FTYPE)

        # Form the arguments to pass to the functional form
        # Need to be flexible in terms of the number of fit parameters
        args = [param]
        for cft_idx in range(self.num_fit_coeffts):
            args += [self.get_fit_coefft(bin_idx=bin_idx, coefft_idx=cft_idx)]
        args += [this_out]

        # Call the function
        self._hypersurface_func.grad(*args)
        # Copy to wherever the gradient is to be stored
        np.copyto(src=this_out, dst=out)

    def get_fit_coefft_idx(self, bin_idx=None, coefft_idx=None):
        '''
        Indexing the fit_coefft matrix is a bit of a pain
        This helper function eases things
        '''

        # TODO can probably do this more cleverly with numpy indexing, but works for now...

        # Indexing based on the bin
        if (bin_idx is Ellipsis) or (bin_idx is None):
            idx = [Ellipsis]
        else:
            idx = list(bin_idx)

        # Indexing based on the coefficent
        if isinstance(coefft_idx, slice):
            idx.append(coefft_idx)
        elif coefft_idx is None:
            idx.append(slice(0, -1))
        else:
            idx.append(coefft_idx)

        # Put it all together
        idx = tuple(idx)
        return idx

    def get_fit_coefft(self, *args, **kwargs):
        '''
        Get a fit coefficient values from the matrix
        Basically just wrapping the indexing function
        '''
        idx = self.get_fit_coefft_idx(*args, **kwargs)
        return self.fit_coeffts[idx]

    @property
    def serializable_state(self):
        """
        OrderedDict containing savable state attributes
        """

        if self._serializable_state is None:  # TODO always redo?

            state = collections.OrderedDict()
            state["name"] = self.name
            state["func_name"] = self.func_name
            state["num_fit_coeffts"] = self.num_fit_coeffts
            state["fit_coeffts"] = self.fit_coeffts
            state["fit_coeffts_sigma"] = self.fit_coeffts_sigma
            state["initial_fit_coeffts"] = self.initial_fit_coeffts
            state["fitted"] = self.fitted
            state["fit_param_values"] = self.fit_param_values
            state["binning_shape"] = self.binning_shape
            state["nominal_value"] = self.nominal_value
            state["bounds"] = self.bounds
            self._serializable_state = state

        return self._serializable_state

    @classmethod
    def from_state(cls, state):
        # Create the param
        param = cls(
            name=state.pop("name"),
            func_name=state.pop("func_name"),
            initial_fit_coeffts=state.pop("initial_fit_coeffts"),
        )

        # Define rest of state
        for k in list(state.keys()):
            setattr(param, k, state.pop(k))
        
        return param
        
'''
Hypersurface fitting and loading helper functions
'''


def get_hypersurface_file_name(hypersurface, tag):
    '''
    Create a descriptive file name
    '''

    num_dims = len(hypersurface.params)
    param_str = "_".join(hypersurface.param_names)
    output_file = "%s__hypersurface_fits__%dd__%s.json" % (
        tag, num_dims, param_str)

    return output_file


def fit_hypersurfaces(nominal_dataset, sys_datasets, params, output_dir, tag, combine_regex=None,
                      log=True, **hypersurface_fit_kw):
    '''
    A helper function that a user can use to fit hypersurfaces to a bunch of simulation
    datasets, and save the results to a file. Basically a wrapper of Hypersurface.fit,
    handling common pre-fitting tasks like producing mapsets from piplelines, merging
    maps from similar specifies, etc.

    Note that this supports fitting multiple hypersurfaces to the datasets, e.g. one per
    simulated species. Returns a dict with format: { map_0_key : map_0_hypersurface,
    ..., map_N_key : map_N_hypersurface, }

    Parameters
    ----------
    nominal_dataset : dict
        Definition of the nominal dataset. Specifies the pipleline with which the maps
        can be created, and the values of all systematic parameters used to produced the
        dataset.
        Format must be:
            nominal_dataset = {
                "pipeline_cfg" = <pipeline cfg file (either cfg file path or dict)>),
                "sys_params" = { param_0_name : param_0_value_in_dataset, ..., param_N_name : param_N_value_in_dataset }
            }
        Sys params must correspond to the provided HypersurfaceParam instances provided
        in the `params` arg.

    sys_datasets : list of dicts
        List of dicts, where each dict defines one of the systematics datasets to be
        fitted. The format of each dict is the same as explained for `nominal_dataset`

    params : list of HypersurfaceParams
        List of HypersurfaceParams instances that define the hypersurface. Note that
        this defined ALL hypersurfaces fitted in this function, e.g. only supports a
        single parameterisation for all maps (this is almost almost what you want).

    output_dir : str
        Path to directly to write results file in

    tag : str
        A string identifier that will be included in the file name to help you make
        sense of the file in the future. Note that additional information on the
        contents will be added to the file name by this function.

    combine_regex : list of str, or None
        List of string regex expressions that will be used for merging maps. Used to
        combine similar species. Must be something that can be passed to the
        `MapSet.combine_re` function (see that functions docs for more details). Choose
        `None` is do not want to perform this merging.

    hypersurface_fit_kw : kwargs
        kwargs will be passed on to the calls to `Hypersurface.fit`
    '''

    # TODO Current yneed to manually ensure consistency between `combine_regex` here and
    # the `links` param in `hypersurface` Need to make `hypersurface` directly use
    # the value of `combine_regex` from the Hypersurface instance

    #
    # Make copies
    #

    # Take (deep) copies of lists/dicts to avoid modifying the originals
    # Useful for cases where this function is called in a loop (e.g. leave-one-out tests)
    nominal_dataset = copy.deepcopy(nominal_dataset)
    sys_datasets = copy.deepcopy(sys_datasets)
    params = copy.deepcopy(params)

    #
    # Check inputs
    #

    # Check types
    assert isinstance(sys_datasets, collections.Sequence)
    assert isinstance(params, collections.Sequence)
    assert isinstance(output_dir, str)
    assert isinstance(tag, str)

    # Check formatting of datasets is as expected
    all_datasets = [nominal_dataset] + sys_datasets
    for dataset in all_datasets:
        assert isinstance(dataset, collections.Mapping)
        assert "pipeline_cfg" in dataset
        assert isinstance(dataset["pipeline_cfg"], (str, collections.Mapping))
        assert "sys_params" in dataset
        assert isinstance(dataset["sys_params"], collections.Mapping)

    # Check params
    assert len(params) >= 1
    for p in params:
        assert isinstance(p, HypersurfaceParam)

    # Report inputs
    msg = "Hypersurface fit details :"
    msg += "  Num params            : %i" % len(params)
    msg += "  Num fit coefficients  : %i" % sum(
        [p.num_fit_coeffts for p in params])
    msg += "  Num datasets          : 1 nominal + %i systematics" % len(
        sys_datasets)
    msg += "  Nominal values        : %s" % nominal_dataset["sys_params"]
    logging.info(msg)

    #
    # Generate MapSets
    #

    # Create and run the nominal and systematics pipelines (using the pipeline configs
    # provided) to get maps
    nominal_dataset["mapset"] = Pipeline(
        nominal_dataset["pipeline_cfg"]).get_outputs()  # return_sum=False)
    for sys_dataset in sys_datasets:
        sys_dataset["mapset"] = Pipeline(
            sys_dataset["pipeline_cfg"]).get_outputs()  # return_sum=False)

    # Merge maps according to the combine regex, is one was provided
    if combine_regex is not None:
        nominal_dataset["mapset"] = nominal_dataset["mapset"].combine_re(
            combine_regex)
        for sys_dataset in sys_datasets:
            sys_dataset["mapset"] = sys_dataset["mapset"].combine_re(
                combine_regex)

    # TODO check every mapset has the same elements

    #
    # Loop over maps
    #

    # Create the container to fill
    hypersurfaces = collections.OrderedDict()

    # Loop over maps
    for map_name in nominal_dataset["mapset"].names:

        #
        # Prepare data for fit
        #

        nominal_map = nominal_dataset["mapset"][map_name]
        nominal_param_values = nominal_dataset["sys_params"]

        sys_maps = [sys_dataset["mapset"][map_name]
                    for sys_dataset in sys_datasets]
        sys_param_values = [sys_dataset["sys_params"]
                            for sys_dataset in sys_datasets]

        #
        # Fit the hypersurface
        #

        # Create the hypersurface
        hypersurface = Hypersurface(
            params=copy.deepcopy(params),
            initial_intercept=0. if log else 1.,  # Initial value for intercept
            log=log
        )

        # Perform fit
        hypersurface.fit(
            nominal_map=nominal_map,
            nominal_param_values=nominal_param_values,
            sys_maps=sys_maps,
            sys_param_values=sys_param_values,
            norm=True,
            **hypersurface_fit_kw
        )

        # Report the results
        logging.debug("\nFitted hypersurface report:\n%s" % hypersurface)

        # Store for later write to disk
        hypersurfaces[map_name] = hypersurface

    #
    # Store results
    #

    # Create a file name
    output_path = os.path.join(output_dir, get_hypersurface_file_name(
        list(hypersurfaces.values())[0], tag))

    # Create the output directory
    mkdir(output_dir)

    # Write to a json file
    to_json(hypersurfaces, output_path)

    logging.info("Fit results written : %s" % output_path)

    return output_dir


def load_hypersurfaces(input_file, expected_binning=None):
    '''
    User function to load file containing hypersurface fits, as written using `fit_hypersurfaces`.
    Can be multiple hypersurfaces assosicated with different maps.

    Returns a dict with the format: { map_0_key : map_0_hypersurface, ..., map_N_key : map_N_hypersurface, }

    Hnadling the following input files cases:
        1) Load files produced using this code (recommended)
        2) Load files producing using older versions of PISA
        3) Load public data releases csv formatted files

    Parameters
    ----------
    input_file : str
        Path to the file contsaining the hypersurface fits.
        For the special case of the datareleases these needs to be the path to all
        relevent CSV fles, e.g. "<path/to/datarelease>/hyperplanes_*.csv".
    expected_binning : One/MultiDimBinning
        (Optional) Expected binning for hypersurface.
        It will checked enforced that this mathes the binning found in the parsed
        hypersurfaces. For certain legacy cases where binning info is not stored, this
        will be assumed to be the actual binning.
    '''

    #
    # Check inputs
    #

    assert isinstance(input_file, str)

    if expected_binning is not None:
        assert is_binning(expected_binning)

    #
    # PISA hypersurface files
    #

    if input_file.endswith("json") or input_file.endswith("json.bz2"):

        # Load file
        input_data = from_json(input_file)
        assert isinstance(input_data, collections.Mapping)

        # Testing various cases to support older files as well as modern ones...
        if "sys_list" in input_data:

            # Legacy case, create a modern hypersurface instance using old hyperplane fits
            hypersurfaces = _load_hypersurfaces_legacy(input_data)
            logging.warn("Old fit files detected, loaded via legacy mode")

        else:

            # Otherwise assume file is using the modern format
            hypersurfaces = collections.OrderedDict()
            for map_name, hypersurface_state in list(input_data.items()):
                hypersurfaces[map_name] = Hypersurface.from_state(
                    hypersurface_state)

    #
    # Public data release file
    #

    elif input_file.endswith("csv") or input_file.endswith("csv.bz2"):

        hypersurfaces = _load_hypersurfaces_data_release(
            input_file, expected_binning)

    #
    # Done
    #

    else:
        raise Exception("Unknown file format")

    # Check binning
    if expected_binning is not None:
        for hypersurface in hypersurfaces.values():
            if not hypersurface.binning.hash == expected_binning.hash:
                for a, b, in zip(hypersurface.binning.dims, expected_binning.dims):
                    assert a == b, "Incompatible binning dimension %s and %s"%(a, b)

    return hypersurfaces


def _load_hypersurfaces_legacy(input_data):
    '''
    Load an old hyperpane (not surface) fit file from older PISA version.

    Put the results into an instance the new `Hypersurface` class so can use the
    resulting hypersurface in modern code.

    User should not use this directly, instead call `load_hypersurfaces`.
    '''

    hypersurfaces = collections.OrderedDict()

    #
    # Loop over map names
    #

    for map_name in input_data["map_names"]:

        #
        # Create the params
        #

        # Get the param names
        param_names = input_data["sys_list"]

        # Create the param instances.
        # Using linear functional forms (legacy files only supported linear forms, e.g.
        # hyperplanes rather than surfaces).
        params = [HypersurfaceParam(
            name=name, func_name="linear", initial_fit_coeffts=None, ) for name in param_names]

        #
        # Get binning
        #

        # This varies depending on how old the file is...
        # Note that the hypersurface class really only needs to know the binning
        # shape (to create the coefficient arrays).

        # If the (serialized version of the) binning is stored, great! Use it
        if "binning" in input_data:
            binning = MultiDimBinning(**input_data["binning"])

        # If no binning is available, can at least get the correct shape (using
        # one of the map arrays) and create a dummy binning instance.
        # Remember that the final dimension is the sys params, not binning
        else:
            # Remove last dimension
            binning_shape = input_data[map_name][..., 0].shape
            binning = MultiDimBinning([OneDimBinning(name="dummy_%i" % i, domain=[
                                      0., 1.], is_lin=True, num_bins=dim) for i, dim in enumerate(binning_shape)])

        #
        # Create the hypersurface instance
        #

        # Create the hypersurface
        hypersurface = Hypersurface(
            params=params,  # Specify the systematic parameters
            initial_intercept=1.,  # Intercept value (or first guess for fit)
        )

        # Set some internal members that would normally be configured during fitting
        # Don't know the nominal values with legacy files, so just stores NaNs
        hypersurface._init(
            binning=binning,
            nominal_param_values={
                name: np.NaN for name in hypersurface.param_names},
        )

        # Indicate this is legacy data (not all functionality will work)
        hypersurface.using_legacy_data = True

        #
        # Get the fit values
        #

        # Handling two different legacy cases here...
        fitted_coefficients = input_data["hyperplanes"][map_name][
            "fit_params"] if "hyperplanes" in input_data else input_data[map_name]

        # Fitted coefficients have following array shape: [ binning dim 0,  ..., binning dim N, sys params (inc. intercept) ]
        intercept_values = fitted_coefficients[..., 0]
        sys_param_gradient_values = {
            n: fitted_coefficients[..., i+1] for i, n in enumerate(param_names)}

        # Write the values to the hypersurface
        np.copyto(src=intercept_values, dst=hypersurface.intercept)
        for param in hypersurface.params.values():
            np.copyto(
                src=sys_param_gradient_values[param.name], dst=param.fit_coeffts[..., 0])

        # Done, store the hypersurface
        hypersurfaces[map_name] = hypersurface

    return hypersurfaces


def _load_hypersurfaces_data_release(input_file_prototype, binning):
    '''
    Load the hypersurface CSV files from an official IceCube data release

    User should not use this directly, instead call `load_hypersurfaces`.
    '''

    # TODO Current only handles DRAGON (analysis B) data release (as was also the case for
    # the older hyperplane code)
    # TODO Would need to add support for muon hypersurface (including non-linear params)
    # as well as a different binning

    import pandas as pd

    hypersurfaces = collections.OrderedDict()

    #
    # Check inputs
    #

    assert binning is not None, "Must provide binning when loading data release hypersurfaces"

    #
    # Load CSV files
    #

    fit_results = {}
    fit_results['nue_cc+nuebar_cc'] = pd.read_csv(find_resource(
        input_file_prototype.replace('*', 'nue_cc')))
    fit_results['numu_cc+numubar_cc'] = pd.read_csv(find_resource(
        input_file_prototype.replace('*', 'numu_cc')))
    fit_results['nutau_cc+nutaubar_cc'] = pd.read_csv(find_resource(
        input_file_prototype.replace('*', 'nutau_cc')))
    fit_results['nu_nc+nubar_nc'] = pd.read_csv(find_resource(
        input_file_prototype.replace('*', 'all_nc')))

    #
    # Get hyperplane info
    #

    param_names = None

    for map_name, map_fit_results in fit_results.items():

        #
        # Get hypersurface params
        #

        # Remove the bin info from the data frame (only want hyperplane params)
        # Check that find the same dimensions as the expected binning
        # TODO Also check bin centers are within expected bins
        for n in binning.names:
            midpoints_found = np.unique(map_fit_results.pop(n).values)
            assert midpoints_found.size == binning[n].num_bins, "Mismatch between expected and actual binning dimensions"

        # Also extract the special case of the offset
        offset = map_fit_results.pop("offset")

        # Get the param names (everything remaining is a hypersurface param)
        if param_names is None:
            param_names = map_fit_results.columns.tolist()
        else:
            assert param_names == map_fit_results.columns.tolist(
            ), "Mismatch between hypersurface params in different files"

        # Create the params
        params = [HypersurfaceParam(
            name=name, func_name="linear", initial_fit_coeffts=None, ) for name in param_names]

        #
        # Create the hypersurface instance
        #

        # Create the hypersurface
        hypersurface = Hypersurface(
            params=params,  # Specify the systematic parameters
            initial_intercept=1.,  # Intercept value (or first guess for fit)
        )

        # Set some internal members that would normally be configured during fitting
        # Don't know the nominal values with legacy files, so just stores NaNs
        hypersurface._init(
            binning=binning,
            nominal_param_values={
                name: np.NaN for name in hypersurface.param_names},
        )

        # Indicate this is legacy data (not all functionality will work)
        hypersurface.using_legacy_data = True

        #
        # Get the fit values
        #

        # Intercept
        intercept_values = offset.values.reshape(binning.shape)
        np.copyto(src=intercept_values, dst=hypersurface.intercept)

        # Param gradients
        for param in hypersurface.params.values():
            sys_param_gradient_values = map_fit_results[param.name].values.reshape(
                binning.shape)
            np.copyto(src=sys_param_gradient_values,
                      dst=param.fit_coeffts[..., 0])

        # Done, store the hypersurface
        hypersurfaces[map_name] = hypersurface

    return hypersurfaces


#
# Test/example
#
def generate_asimov_testdata(binning, parameters, true_param_coeffs,
                             nominal_param_values, sys_param_values,
                             error_scale=0.1, log=False, intercept=2.,
                             ):
    hypersurface = Hypersurface(
        params=parameters,  # Specify the systematic parameters
        # Intercept value (or first guess for fit)
        initial_intercept=intercept,
        log=log,
    )
    assert set(hypersurface.params.keys()) == set(nominal_param_values.keys())
    assert set(hypersurface.params.keys()) == set(true_param_coeffs.keys())

    hypersurface._init(binning=binning, nominal_param_values=nominal_param_values)
    from pisa.core.map import Map, MapSet
    for bin_idx in np.ndindex(binning.shape):
        for name, coeffs in true_param_coeffs.items():
            assert len(coeffs) == hypersurface.params[name].num_fit_coeffts, ("number "
                                                                              "of coefficients in the parameter must match")
            for j, c in enumerate(coeffs):
                idx = hypersurface.params[name].get_fit_coefft_idx(bin_idx=bin_idx,
                                                                   coefft_idx=j,
                                                                   )
                hypersurface.params[name].fit_coeffts[idx] = c
    logging.debug("Truth hypersurface report:\n%s" % str(hypersurface))

    # Only consider one particle type for simplicity
    particle_key = "nue_cc"
    # Create each dataset, e.g. set the systematic parameter values, calculate a bin count
    hist = hypersurface.evaluate(nominal_param_values)
    assert np.all(hist >= 0.), ("nominal map has negative values! "
                                "Choose different true parameters.")
    nom_map = Map(name=particle_key, binning=binning,
                  hist=hist, error_hist=np.sqrt(hist)*error_scale,
                  )
    logging.debug("Nominal hist: \n%s" % str(nom_map.hist))
    sys_maps = []
    for i in range(len(sys_param_values)):
        hist = hypersurface.evaluate(sys_param_values[i])
        assert np.all(hist > 0.), ("a systematic map has negative values! values: "
                                   "%s systematics: %s" % (str(hist), str(sys_param_values[i])))
        sys_maps.append(Map(name=particle_key, binning=binning,
                            hist=hist, error_hist=np.sqrt(hist)*error_scale))
    return nom_map, sys_maps


def test_hypersurface_uncertainty(plot=False):
    '''
    Simple test of hypersurface fits + uncertainty
    1. Creates some Asimov test data matching a true hypersurface and checks the ability
       to fit back the truth.
    2. Fluctuates Asimov test data randomly to check uncertainties of hypersurface
    '''

    # Define systematic parameters in the hypersurface
    params = [
        HypersurfaceParam(name="foo", func_name="linear",
                          initial_fit_coeffts=[1.]),
        HypersurfaceParam(name="bar", func_name="quadratic",
                          initial_fit_coeffts=[1., -1.]),
    ]
    # Create the hypersurface
    hypersurface = Hypersurface(
        params=params,  # Specify the systematic parameters
        initial_intercept=1.,  # Intercept value (or first guess for fit)
        log=False
    )
    # Define binning with one dummy bin
    binning = MultiDimBinning([OneDimBinning(name="reco_energy",
                                             domain=[0., 10.],
                                             num_bins=1,
                                             units=ureg.GeV,
                                             is_lin=True
                                             )])
    # Define true coefficients
    true_coeffs = {'foo': [-0.4], 'bar': [0.5, 1.]}
    true_intercept = 5.
    nominal_param_values = {'foo': 1., 'bar': 0.}
    # making combinations of systematic values
    foo_vals = np.linspace(-2., 2., 6)
    bar_vals = np.linspace(-2, 1.5, 8)
    sys_param_values = []
    for f in foo_vals:
        for b in bar_vals:
            sys_param_values.append({'foo': f, 'bar': b})

    nom_map, sys_maps = generate_asimov_testdata(binning,
                                                 params,
                                                 true_coeffs,
                                                 nominal_param_values,
                                                 sys_param_values,
                                                 intercept=true_intercept,
                                                 log=False,
                                                 error_scale=0.2,
                                                 )
    # Perform fit
    hypersurface.fit(
        nominal_map=nom_map,
        nominal_param_values=nominal_param_values,
        sys_maps=sys_maps,
        sys_param_values=sys_param_values,
        norm=False,
    )
    # Report the results
    logging.debug("Fitted hypersurface report:\n%s" % hypersurface)

    assert np.allclose(hypersurface.intercept, true_intercept,
                       rtol=ALLCLOSE_KW['rtol']*10.)
    for param_name in hypersurface.param_names:
        assert np.allclose(hypersurface.params[param_name].fit_coeffts,
                           true_coeffs[param_name], rtol=ALLCLOSE_KW['rtol']*10.)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_bin_fits(ax, hypersurface, bin_idx=[0], param_name='foo',
                      label='Asimov test map')
        ax.grid()
        plt.savefig('test_hypersurface_foo.pdf')

        fig, ax = plt.subplots()
        plot_bin_fits(ax, hypersurface, bin_idx=[0], param_name='bar',
                      label='Asimov test map')
        ax.grid()
        plt.savefig('test_hypersurface_bar.pdf')

    # Evaluate hypersurface and uncertainties at some points
    # that just happen to be the systematic values (but choice could be different)
    asimov_true_points = []
    asimov_fit_points = []
    asimov_fit_errs = []
    for i in range(len(sys_param_values)):
        hist, errs = hypersurface.evaluate(
            sys_param_values[i], return_uncertainty=True)
        asimov_fit_points.append(hist)
        asimov_fit_errs.append(errs)
        asimov_true_points.append(sys_maps[i].nominal_values)
    asimov_true_points = np.concatenate(asimov_true_points)
    asimov_fit_points = np.concatenate(asimov_fit_points)
    asimov_fit_errs = np.concatenate(asimov_fit_errs)

    logging.debug("Asimov true points:\n%s" % str(asimov_true_points))
    logging.debug("Asimov fit points:\n%s" % str(asimov_fit_points))
    logging.debug("Asimov fit error estimates:\n%s" % str(asimov_fit_errs))
    assert np.allclose(asimov_true_points, asimov_fit_points, rtol=ALLCLOSE_KW['rtol']*10.)
    logging.debug("Fluctuating maps and re-fitting...")
    # do several rounds of fluctuation, re-fit and storage of results
    n_rounds = 100
    fluctuated_fit_points = []
    for i in range(n_rounds):
        #logging.info("Round %d/%d" % (i+1, n_rounds))
        nom_map_fluct = nom_map.fluctuate(method='gauss')
        sys_maps_fluct = []
        for s in sys_maps:
            sys_maps_fluct.append(s.fluctuate(method='gauss'))
        hypersurface.fit(
            nominal_map=nom_map_fluct,
            nominal_param_values=nominal_param_values,
            sys_maps=sys_maps_fluct,
            sys_param_values=sys_param_values,
            norm=False,
        )
        fluctuated_fit_points.append([])
        for j in range(len(sys_param_values)):
            hist = hypersurface.evaluate(
                sys_param_values[j], return_uncertainty=False)
            fluctuated_fit_points[-1].append(hist)
        fluctuated_fit_points[-1] = np.concatenate(fluctuated_fit_points[-1])
        logging.trace("Fluctuated fit points:\n%s" %
                      str(fluctuated_fit_points[-1]))
    # evaluate whether the actual fluctuations match the estimated errors
    fluctuated_fit_points = np.array(fluctuated_fit_points)
    fit_differences = fluctuated_fit_points - asimov_fit_points
    all_pulls = fit_differences / asimov_fit_errs
    avg_fit_differences = np.mean(fit_differences, axis=0)
    std_pulls = np.std(all_pulls, axis=0)
    logging.debug("Average fluctuated fit difference:\n%s" %
                  str(avg_fit_differences))
    logging.debug("Mean pulls per point:\n%s" % str(std_pulls))
    logging.debug("Mean pull: %.3f" % np.mean(std_pulls))
    assert np.abs(np.mean(std_pulls) -
                  1.) < 0.1, "avg. pulls too far from expectation"

    if plot:
        plt.figure()
        plt.hist(all_pulls.flatten(), bins=50,
                 density=True, label='fluctuated fits')
        x_plot = np.linspace(-4, 4, 100)
        plt.plot(x_plot, np.exp(-x_plot**2/2.) /
                 np.sqrt(2.*np.pi), label='expectation')
        plt.title('pull distribution')
        plt.xlabel('pull')
        plt.ylabel('density')
        plt.legend()
        plt.savefig('test_hypersurface_pull.pdf')
    logging.info('<< PASS : test_hypersurface_uncertainty >>')


def test_hypersurface_basics():
    '''
    Test basic fitting, inject/recover, storing and loading
    '''
    import tempfile
    from pisa.core.map import Map

    params = [HypersurfaceParam(name="foo", func_name="linear",
                                initial_fit_coeffts=[1.],
                                ),
              # the exponential HS function did not reliably recover injected true
              # parameters, probably due to the degeneracy with the intercept.
              HypersurfaceParam(name="bar", func_name="quadratic",
                                initial_fit_coeffts=[.1, .1],
                                ),
              ]

    # Create the hypersurface
    hypersurface = Hypersurface(params=params,  # Specify the systematic parameters
                                initial_intercept=1.,  # Intercept first guess for fit)
                                log=False,
                                )

    binning = MultiDimBinning([OneDimBinning(name="reco_energy",
                                             domain=[0., 10.],
                                             num_bins=3,
                                             units=ureg.GeV,
                                             is_lin=True,
                                             )])

    # Define the values for the parameters for each dataset
    nom_param_values = {}
    sys_param_values_dict = {}

    if "foo" in [p.name for p in params]:
        nom_param_values["foo"] = 0.
        sys_param_values_dict["foo"] = [0., 0., 0., -1., +1., 1.]

    if "bar" in [p.name for p in params]:
        nom_param_values["bar"] = 10.
        sys_param_values_dict["bar"] = [20., 30., 0., 10., 10., 15.]

    # Get number of datasets
    num_sys_datasets = len(list(sys_param_values_dict.values())[0])

    # Only consider one particle type for simplicity
    particle_key = "nue_cc"

    # Create a dummy "true" hypersurface that can be used to generate
    # some fake bin values for the dataset
    true_hypersurface = copy.deepcopy(hypersurface)
    true_hypersurface._init(
        binning=binning, nominal_param_values=nom_param_values)
    true_hypersurface.intercept.fill(10.)
    if "foo" in true_hypersurface.params:
        true_hypersurface.params["foo"].fit_coeffts[..., 0].fill(2.)
    if "bar" in true_hypersurface.params:
        true_hypersurface.params["bar"].fit_coeffts[..., 0].fill(-.1)
        true_hypersurface.params["bar"].fit_coeffts[..., 1].fill(0.05)

    logging.debug("Truth hypersurface report:\n%s" % str(true_hypersurface))

    # Create each dataset, e.g. set the systematic parameter values, calculate bin count
    hist = true_hypersurface.evaluate(nom_param_values)
    nom_map = Map(name=particle_key, binning=binning,
                  hist=hist, error_hist=np.sqrt(hist),
                  )
    sys_maps = []
    sys_param_values = []
    for i in range(num_sys_datasets):
        sys_param_values.append({name: sys_param_values_dict[name][i]
                                 for name in list(true_hypersurface.params.keys())
                                 })
        hist = true_hypersurface.evaluate(sys_param_values[-1])
        sys_maps.append(Map(name=particle_key, binning=binning,
                            hist=hist, error_hist=np.sqrt(hist),
                            )
                        )

    # Perform fit
    hypersurface.fit(nominal_map=nom_map,
                     nominal_param_values=nom_param_values,
                     sys_maps=sys_maps,
                     sys_param_values=sys_param_values,
                     norm=False,
                     )

    logging.debug("Fitted hypersurface report:\n%s" % hypersurface)

    # Check the fitted parameter values match the truth
    # This only works if `norm=False` in the `hypersurface.fit` call just above
    logging.debug("Checking fit recovered truth...")
    assert np.allclose(hypersurface.intercept,
                       true_hypersurface.intercept, rtol=ALLCLOSE_KW['rtol']*10.)
    for param_name in hypersurface.param_names:
        assert np.allclose(hypersurface.params[param_name].fit_coeffts,
                           true_hypersurface.params[param_name].fit_coeffts,
                           rtol=ALLCLOSE_KW['rtol']*10.
                           )
    logging.debug("... fit was successful!")

    # testing save/reload
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = "hypersurface.json.bz2"
        file_path = os.path.join(tmpdirname, file_name)
        to_json(hypersurface, file_path)

        reloaded_hypersurface = Hypersurface.from_state(file_path)

        logging.debug(
            "Checking saved and re-loaded hypersurfaces are identical...")
        assert np.allclose(hypersurface.intercept,
                           reloaded_hypersurface.intercept,
                           rtol=ALLCLOSE_KW['rtol']*10.
                           )
        for param_name in hypersurface.param_names:
            assert np.allclose(hypersurface.params[param_name].fit_coeffts,
                               reloaded_hypersurface.params[param_name].fit_coeffts,
                               rtol=ALLCLOSE_KW['rtol']*10.
                               )
        logging.debug("... save+re-load was successful!")

    # test getting and setting coefficients
    coeffts = hypersurface.fit_coeffts
    reloaded_hypersurface.fit_coeffts = coeffts
    logging.debug(
        "Checking hypersurfaces are identical after getting and setting coeffts...")
    assert np.allclose(hypersurface.intercept, reloaded_hypersurface.intercept,
                       rtol=ALLCLOSE_KW['rtol']*10.)
    for param_name in hypersurface.param_names:
        assert np.allclose(hypersurface.params[param_name].fit_coeffts,
                           reloaded_hypersurface.params[param_name].fit_coeffts,
                           rtol=ALLCLOSE_KW['rtol']*10.)
    logging.debug("... setting and getting coefficients was successful!")
    logging.info('<< PASS : test_hypersurface_basics >>')


# Run the examp'es/tests
if __name__ == "__main__":
    set_verbosity(2)
    test_hypersurface_basics()
    test_hypersurface_uncertainty()
