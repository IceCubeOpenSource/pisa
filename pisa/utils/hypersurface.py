"""
Tools for working with hypersurfaces, which are continuous functions in N-D 
with arbitrary functional forms. 

Hypersurfaces can be used to model systematic uncertainties derived from discrete 
simulation datasets, for example for detedctor uncertainties.
"""

__all__ = ['get_num_args', 'Hypersurface', 'HypersurfaceParam', 'fit_hypersurfaces', 'load_hypersurfaces', 'plot_bin_fits', 'plot_bin_fits_2d']

__author__ = 'T. Stuttard'

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


import os, sys, collections, copy, inspect

import numpy as np
from scipy.optimize import curve_fit

from pisa import FTYPE, TARGET, ureg
from pisa.utils import vectorizer
from pisa.utils.jsons import from_json, to_json
from pisa.core.pipeline import Pipeline
from pisa.core.binning import OneDimBinning, MultiDimBinning
from pisa.core.map import Map
from pisa.utils.fileio import mkdir

import numba
from numba import guvectorize, int32, float64

from uncertainties import ufloat, correlated_values
from uncertainties import unumpy as unp


'''
Helper functions
'''

def get_num_args(func) :
    '''
    Function for grabbing the number of arguments to a function

    Parameters
    ----------
    func : function
        Function to determine args for. 
        Can be a standard pythin function or a numpy ufunc

    '''

    #TODO numba funcs

    if isinstance(func, np.ufunc):
        return func.nargs
    else :
        return len(inspect.getargspec(func).args)


'''
Hypersurface functional forms

   Define functional forms for HypersurfaceParam instances here.

   Functions defined here MUST:
     - Be named <something>_hypersurface_func (they are then chosen by the user using `func_name=<something>`).
     - Support numba guvectorization.
     - Function arguments must observed this convention: 
         `p`, `<coefficient 0>`, ..., `<coefficient N>`, `out`
         where `p` is the systematic parameter, `out is the array to write the results to, and there 
         are N coefficients of the parameterisation.

   The format of these arguments depends on the use case, of which there are two:
     - When fitting the function coefficients. This is done bin-wise using multiple datasets.
       - Params are then: `p` is array (one value per dataset), coefficients and `out` 
         are scalar (representing a single bin).
     - Evaluating a fitted hypersurface. This is done for all bins simultaneously, using a single value for p.
       - Params are then: `p` is scalar (current value of systematic parameter, coefficients and `out` are arrays
         representing the hypersurfaces of all bins per bin.
 
   The argument definitions are then:
     - 
'''

#TODO support uncertainty propagation (difficult because `uncertainties` modules not compatible with numba)

def linear_hypersurface_func(p,m,out) :
    '''
    Linear hypersurface functional form

    f(p) = m * p
    '''
    result = m * p
    np.copyto(src=result,dst=out)


def exponential_hypersurface_func(p,a,b,out) :
    '''
    Exponential hypersurface functional form

    f(p) = a * exp(b*p)
    '''
    result = a * np.exp(b*p)
    np.copyto(src=result,dst=out)



'''
Core hypersurface classes
'''

class Hypersurface(object) :
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

    debug : bool
        True -> significantly more print out
    '''

    def __init__(self, params, initial_intercept=None, debug=False ) :

        # Store args
        self.initial_intercept = initial_intercept
        self.debug = debug

        # Store params as dict for ease of lookup
        self.params = collections.OrderedDict()
        for param in params :
            assert param.name not in self.params, "Duplicate param name found : %s" % param.name
            self.params[param.name] = param

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


    def _init(self, binning, nominal_param_values ) :
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
        if self.initial_intercept is None :
            self.initial_intercept = 0.

        # Create the fit coefficient arrays
        # Have one fit per bin
        self.intercept = np.full(self.binning.shape,self.initial_intercept,dtype=FTYPE)
        self.intercept_sigma = np.full_like(self.intercept,np.NaN)
        for param in list(self.params.values()) :
            param._init_fit_coefft_arrays(self.binning)


        #
        # Nominal values
        #

        # Store the nominal param values
        #TODO better checks, including not already set
        for param in list(self.params.values()) :
            param.nominal_value = nominal_param_values[param.name]


        #
        # Done
        #

        self._initialized = True


    @property
    def param_names(self) :
        '''
        Return the (ordered) names of the systematic parameters
        '''
        return list(self.params.keys())


    def evaluate(self, param_values, bin_idx=None) :
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
        '''

        assert self._initialized


        #
        # Check inputs
        #

        # Determine number of sys param values (per sys param)
        # This will be >1 when fitting, and == 1 when evaluating the hypersurface within the stage
        num_param_values = np.asarray(list(param_values.values())[0]).size

        # Check same number of values for all sys params
        for k,v in list(param_values.items()) :
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
            for v in list(param_values.values()) :
                assert np.isscalar(v), "sys param values must be a scalar when evaluating all bins simultaneously"
            out_shape = self.binning.shape
            bin_idx = Ellipsis

        else :
            # Case 2 : Calculating for multiple sys param values, but only a single bin
            #          Use case is fitting the hypersurfaces fucntional form fit params
            out_shape = (num_param_values,)

        # Create the output array
        out = np.full(out_shape,np.NaN,dtype=FTYPE)


        #
        # Evaluate the hypersurface
        #

        # Start with the intercept
        for i in range(num_param_values) :
            if single_bin_mode :
                out[i] = self.intercept[bin_idx]
            else :
                np.copyto( src=self.intercept[bin_idx], dst=out[bin_idx] )

        # Evaluate each individual parameter
        for k,p in list(self.params.items()) :
            p.evaluate(param_values[k],out=out,bin_idx=bin_idx)

        return out



    def fit(self, nominal_map, nominal_param_values, sys_maps, sys_param_values, norm=True, method=None, smooth=False, smooth_kw=None ) :
        '''
        Fit the hypersurface coefficients (in every bin) to best match the provided nominal
        and systematic datasets.

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
            List where each element if a dict containing the values of each systematic param used to generate the that dataset
            Each list element specified the parameters for the corresponding element in `sys_maps`

        norm : bool
            Normalise the maps to the nominal map.
            This is what you want to do when using the hypersurface to re-weight simulation (which is the main use case).
            In principal the hypersurfaces are more general though and could be used for other tasks too, hence this option.

        method : str
            `method` arg to pass to `scipy.optimize.curve_fit`

        smooth : str
            Smoothing method to use. Choose `None` if do not want smoothing.
            Smoothing methods supported:
                - `gaussian_filter`

        smooth_kw : dict
            kwargs to pass to smoothing method underlying function
            Format depends on smoothing method:
              `gaussian_filter`
                 kwargs for `scipy.ndimage.filters.gaussian_filter`
                 MUST include `sigma`, `order`
        '''

        #TODO Add option to exclude bins with too few stats from the fit, leving null hypersurface for them.
        #     This is to avoid issues with bins with tiny stats having crazy gradients from statistical 
        #     fluctuations (if there are very few events in that bin for that speciesi then that bin shouldn't
        #     be significant in the fit).


        #
        # Check inputs
        #

        # Check nominal dataset definition
        assert isinstance(nominal_map, Map)
        assert isinstance(nominal_param_values, collections.Mapping)
        assert set(nominal_param_values.keys()) == set(self.param_names)
        assert all([ isinstance(k, str) for k in nominal_param_values.keys() ])
        assert all([ np.isscalar(v) for v in nominal_param_values.values() ])

        # Check systematic dataset definitions
        assert isinstance(sys_maps, collections.Sequence)
        assert isinstance(sys_param_values, collections.Sequence)
        assert len(sys_maps) == len(sys_param_values)
        for sys_map, sys_param_vals in zip(sys_maps, sys_param_values) :
            assert isinstance(sys_map, Map)
            assert isinstance(sys_param_vals, collections.Mapping)
            assert set(sys_param_vals.keys()) == set(self.param_names)
            assert all([ isinstance(k, str) for k in sys_param_vals.keys() ])
            assert all([ np.isscalar(v) for v in sys_param_vals.values() ])
            assert sys_map.binning == nominal_map.binning


        #
        # Format things before getting started
        #

        # Default fit method
        # Choosing one that produces covariance matrix results reliably
        self.fit_method = method
        if self.fit_method is None :
            self.fit_method = "lm"  # lm, trf, dogbox

        # Initialise hypersurface using nominal dataset
        self._init(binning=nominal_map.binning, nominal_param_values=nominal_param_values)

        # Combine nominal and sys sets
        maps = [nominal_map] + sys_maps
        param_values = [nominal_param_values] + sys_param_values

        # Store raw maps
        self.fit_maps_raw = maps

        # Convert params values from `list of dicts` to `dict of lists`
        param_values_dict = { name:np.array([ p[name] for p in param_values ]) for name in list(param_values[0].keys()) }

        # Save the param values used for fitting in the param objects (useful for plotting later)
        for name,values in list(param_values_dict.items()) :
            self.params[name].fit_param_values = values

        # Format the fit `x` values : [ [sys param 0 values], [sys param 1 values], ... ]
        # Order of the params must match the order in `self.params`
        x = np.asarray( [ param_values_dict[param_name] for param_name in list(self.params.keys()) ], dtype=FTYPE )

        # Prepare covariance matrix array
        self.fit_cov_mat = np.full( list(self.binning.shape)+[self.num_fit_coeffts,self.num_fit_coeffts] ,np.NaN )

 
        #
        # Smoothing
        #

        #TODO Factor out smoothing functions so can use them in other places too
        #TODO Add handling for user to provide their own smoothing function

        # Optionally can apply smoothing to histograms before the fit
        # Can be useful for poorlt populated templates
        if smooth :

            assert isinstance(smooth,str), "`smooth` should be a string, found %s %s" % (smooth,type(smooth)) 

            if smooth_kw is None :
                smooth_kw = {}

            # Use Gaussian filter smoothing (useful for noisy data)
            if smooth.lower() == "gaussian_filter" :

                from scipy.ndimage.filters import gaussian_filter

                assert "sigma" in smooth_kw
                assert "order" in smooth_kw

                for i,m in enumerate(self.fit_maps_raw) :
                    new_map_state = m.serializable_state
                    new_map_state["hist"] = gaussian_filter( m.nominal_values, sigma=smooth_kw["sigma"], order=smooth_kw["order"] )
                    new_map_state["error_hist"] = gaussian_filter( m.std_devs, sigma=smooth_kw["sigma"], order=smooth_kw["order"] ) #TODO Not sure this is a good way to handle sigma?
                    self.fit_maps_raw[i] = Map(**new_map_state) #TODO Store smoothed maps separately to raw version

            #TODO also consider zoom smoothing?


        #
        # Normalisation
        #

        # All map values are finite, but if have empty bins the nominal map will end up with 
        # inf bins in the normalised map (divide by zero). Use a mask to handle this.
        finite_mask = nominal_map.nominal_values != 0

        # Normalise bin values, if requested
        if norm :

            # Normalise the maps by dividing the nominal map
            # This means the hypersurface results can be interpretted as a re-weighting factor, 
            # relative to the nominal

            # Formalise, handling inf values
            normed_maps = []
            for m in maps :
                norm_m = copy.deepcopy(m)
                norm_m.hist[finite_mask] = norm_m.hist[finite_mask] / nominal_map.hist[finite_mask]
                norm_m.hist[~finite_mask] = ufloat(np.NaN, np.NaN)
                normed_maps.append(norm_m)

            # Store for plotting later
            self.fit_maps_norm = normed_maps  
        
        # Record that fit info is now stored
        self.fit_info_stored = True


        #
        # Some final checks
        #

        # Not expecting any bins to have negative values (negative counts doesn't make sense)
        #TODO hypersurface in general could consider -ve values (no explicitly tied to histograms), so maybe can relax this constraint
        for m in self.fit_maps :
            assert np.all( m.nominal_values[finite_mask] >= 0. ), "Found negative bin counts"


        #
        # Loop over bins
        #

        for bin_idx in np.ndindex(self.binning.shape) : #TODO grab from input map


            #
            # Format this bin's data for fitting
            #

            # Format the fit `y` values : [ bin value 0, bin_value 1, ... ]
            # Also get the corresonding uncertainty
            y = np.asarray([ m.nominal_values[bin_idx] for m in self.fit_maps ], dtype=FTYPE)
            y_sigma = np.asarray([ m.std_devs[bin_idx] for m in self.fit_maps ], dtype=FTYPE)

            # Create a mask for keeping all these points
            # May remove some points before fitting if find issues
            scan_point_mask = np.ones( y.shape, dtype=bool) 

            # Cases where we have a y_sigma element = 0 (normally because the corresponding y element = 0) screw up the fits (least squares divides by sigma, so get infs)
            # Need to handle these cases here
            # For now, I assing an new non-zero sigma value instead
            # Could also try masking off the points, but find that I have cases where I then don't have enough sets to fit the number of parameters I need
            #TODO Look into a good solution to this in more detail
            bad_sigma_mask = y_sigma == 0.
            if bad_sigma_mask.sum() > 0 :
                y_sigma[bad_sigma_mask] = 1.

            # Apply the mask to get the values I will actually use
            x_to_use = np.array([ xx[scan_point_mask] for xx in x ])
            y_to_use = y[scan_point_mask]
            y_sigma_to_use = y_sigma[scan_point_mask]

            # Checks
            assert x_to_use.shape[0] == len(self.params)
            assert x_to_use.shape[1] == y_to_use.size

            # Get flat list of the fit param guesses
            p0 = np.array( [self.intercept[bin_idx]] + [ param.get_fit_coefft(bin_idx=bin_idx,coefft_idx=i_cft) for param in list(self.params.values()) for i_cft in range(param.num_fit_coeffts) ], dtype=FTYPE )


            #
            # Check if have valid data in this bin
            #

            # If have empty bins, cannot fit
            # In particular, if the nominal map has an empty bin, it cannot be rescaled (x * 0 = 0)
            # If this case, no need to try fitting

            # Check if have NaNs/Infs
            if np.any(~np.isfinite(y_to_use)) : #TODO also handle missing sigma

                # Not fitting, add empty variables
                popt = np.full_like( p0, np.NaN )
                pcov = np.NaN 

            # Otherwise, fit...
            else :


                #
                # Fit
                #

                # Must have at least as many sets as free params in fit or else curve_fit will fail
                assert y.size >= p0.size, "Number of datasets used for fitting (%i) must be >= num free params (%i)" % (y.size, p0.size)

                # Define a callback function for use with `curve_fit`
                #   x : sys params
                #   p : func/shape params
                def callback(x,*p) :

                    # Note that this is using the dynamic variable `bin_idx`, which cannot be passed as 
                    # an arg as `curve_fit` cannot handle fixed parameters.

                    # Unflatten list of the func/shape params, and write them to the hypersurface structure
                    self.intercept[bin_idx] = p[0]
                    i = 1
                    for param in list(self.params.values()) :
                        for j in range(param.num_fit_coeffts) :
                            bin_fit_idx = tuple( list(bin_idx) + [j] )
                            param.fit_coeffts[bin_fit_idx] = p[i]
                            i += 1

                    # Unflatten sys param values
                    params_unflattened = collections.OrderedDict()
                    for i in range(len(self.params)) :
                        param_name = list(self.params.keys())[i]
                        params_unflattened[param_name] = x[i]

                    return self.evaluate(params_unflattened,bin_idx=bin_idx)


                # Define the EPS (step length) used by the fitter
                # Need to take care with floating type precision, don't want to go smaller than the FTYPE being used by PISA can handle
                eps = np.finfo(FTYPE).eps
 
                # Debug logging
                if self.debug :
                    test_bin_idx = (0,0,0) 
                    if bin_idx == test_bin_idx :
                        print(">>>>>>>>>>>>>>>>>>>>>>>")
                        print("Curve fit inputs to bin %s :" % (bin_idx,) )
                        print("  x           : %s" % x)
                        print("  y           : %s" % y)
                        print("  y sigma     : %s" % y_sigma)
                        print("  p0          : %s" % p0)
                        print("  fit method  : %s" % self.fit_method)
                        print("<<<<<<<<<<<<<<<<<<<<<<<")

                # Define some settings to use with `curve_fit` that vary with fit method
                curve_fit_kw = {}
                if self.fit_method == "lm" :
                    curve_fit_kw["epsfcn"] = eps

                # Perform fit
                #TODO rescale all params to [0,1] as we do for minimizers?
                popt, pcov = curve_fit(
                    callback,
                    x_to_use,
                    y_to_use,
                    p0=p0,
                    sigma=y_sigma_to_use,
                    absolute_sigma=True, #TODO check this is really what we want
                    maxfev=1000000, #TODO arg?
                    method=self.fit_method,
                    **curve_fit_kw
                )

                # Check the fit was successful
                #TODO curve_fit doesn't return anything that use here, so need another method. Check on chi2 could work...


            #
            # Re-format fit results
            #

            # Use covariance matrix to get uncertainty in fit parameters
            # Using uncertainties.correlated_values, and will extract the std dev (including correlations) shortly
            # Fit may fail to determine covariance matrix (method-dependent), so only do this if have a finite covariance matrix
            corr_vals = correlated_values(popt,pcov) if np.all(np.isfinite(pcov)) else None

            # Write the fitted param results (and sigma, if available) back to the hypersurface structure
            i = 0
            self.intercept[bin_idx] = popt[i]
            self.intercept_sigma[bin_idx] = np.NaN if corr_vals is None else corr_vals[i].std_dev
            i += 1
            for param in list(self.params.values()) :
                for j in range(param.num_fit_coeffts) :
                    idx = param.get_fit_coefft_idx(bin_idx=bin_idx,coefft_idx=j)
                    param.fit_coeffts[idx] = popt[i]
                    param.fit_coeffts_sigma[idx] = np.NaN if corr_vals is None else corr_vals[i].std_dev
                    i += 1

            # Store the covariance matrix
            self.fit_cov_mat[bin_idx] = pcov #TODO copyto?


        #
        # chi2
        #

        # Compare the result of the fitted hypersurface function with the actual data points used for fitting
        # Compute the resulting chi2 to have an estimate of the fit quality

        self.fit_chi2 = []

        # Loop over datasets
        for i_set in range(self.num_fit_sets) :

            # Get expected bin values according tohypersurface value
            predicted = self.evaluate({ name:values[i_set] for name,values in list(param_values_dict.items()) })

            # Get the observed value
            observed = self.fit_maps[i_set].nominal_values
            sigma = self.fit_maps[i_set].std_devs

            # Compute chi2
            chi2 = ((predicted - observed) / sigma) ** 2

            # Add to container
            self.fit_chi2.append(chi2)

        # Combine into single array
        self.fit_chi2 = np.stack(self.fit_chi2,axis=-1).astype(FTYPE)


        #
        # Done
        #

        # Record some provenance info about the fits
        self.fit_complete = True



    @property
    def nominal_values(self) :
        '''
        Return the stored nominal parameter for each dataset
        Returns: { param_0_name : param_0_nom_val, ..., param_N_name : param_N_nom_val }
        '''
        assert self.fit_info_stored, "Cannot get fit dataset nominal values, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")
        return collections.OrderedDict([ (name,param.nominal_value) for name,param in list(self.params.items()) ])

    @property
    def fit_param_values(self) :
        '''
        Return the stored systematic parameters from the datasets used for fitting
        Returns: { param_0_name : [ param_0_sys_val_0, ..., param_0_sys_val_M ], ..., param_N_name : [ param_N_sys_val_0, ..., param_N_sys_val_M ] }
        '''
        assert self.fit_info_stored, "Cannot get fit dataset param values, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")
        return collections.OrderedDict([ (name,param.fit_param_values) for name,param in list(self.params.items()) ])


    @property
    def num_fit_sets(self) :
        '''
        Return number of datasets used for fitting
        '''
        assert self.fit_info_stored, "Cannot get fit datasets, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")
        return list(self.params.values())[0].num_fit_sets


    def get_nominal_mask(self) :
        '''
        Return a mask indicating which datasets have nominal values for all parameters
        '''

        assert self.fit_info_stored, "Cannot get nominal mask, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")

        nom_mask = np.ones((self.num_fit_sets,),dtype=bool)

        for param in list(self.params.values()) :
            nom_mask = nom_mask & np.isclose(param.fit_param_values,param.nominal_value) 

        return nom_mask


    def get_on_axis_mask(self, param_name) :
        '''
        Return a mask indicating which datasets are "on-axis" for a given parameter.

        "On-axis" means "generated using the nominal value for this parameter". Parameters other 
        than the one specified can have non-nominal values.

        Parameters
        ----------
        param_name : str
            The name of systematic parameter for which we want on-axis datasets
        '''

        assert self.fit_info_stored, "Cannot get on-axis mask, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")

        assert param_name in self.param_names

        on_axis_mask = np.ones((self.num_fit_sets,),dtype=bool)

        # Loop over sys params
        for param in list(self.params.values()) :

            # Ignore the chosen param
            if param.name  != param_name :

                # Define a "nominal" mask
                on_axis_mask = on_axis_mask & np.isclose(param.fit_param_values,param.nominal_value) 

        return on_axis_mask


    def report(self,bin_idx=None) :
        '''
        Return a string version of the hypersurface contents

        Parameters
        ----------
        bin_idx : tupel of None
            Specify a particular bin (using numpy indexing). In this case only report on that bin. 
        '''

        msg = ""

        # Fit results
        msg += ">>>>>> Fit coefficients >>>>>>" + "\n"
        bin_indices = np.ndindex(self.binning.shape) if bin_idx is None else [bin_idx]
        for bin_idx in bin_indices :
            msg += "  Bin %s :" % (bin_idx,)  + "\n"
            msg += "     Intercept : %0.3g" % (self.intercept[bin_idx],)  + "\n"
            for param in list(self.params.values()) :
                msg += "     %s : %s" % ( param.name, ", ".join([ "%0.3g"%param.get_fit_coefft(bin_idx=bin_idx,coefft_idx=cft_idx) for cft_idx in range(param.num_fit_coeffts) ]))  + "\n"
        msg += "<<<<<< Fit coefficients <<<<<<" + "\n"

        return msg


    def __str__(self) :
        return self.report()


    @property
    def fit_maps(self) :
        '''
        Return the `Map instances used for fitting
        These will be normalised if the fit was performend to normalised maps.
        '''
        assert self.fit_info_stored, "Cannot get fit maps, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")
        return self.fit_maps_raw if self.fit_maps_norm is None else self.fit_maps_norm


    @property
    def num_fit_sets(self) :
        '''
        Return number of datasets used for fitting
        '''
        assert self.fit_info_stored, "Cannot get fit sets, fit info not stored%s" % (" (using legacy data)" if self.using_legacy_data else "")
        return len(list(self.fit_param_values.values())[0])


    @property
    def num_fit_coeffts(self) :
        '''
        Return the total number of coefficients in the hypersurface fit
        This is the overall intercept, plus the coefficients for each individual param
        '''
        return int( 1 + np.sum([ param.num_fit_coeffts for param in list(self.params.values()) ]) )


    @property
    def fit_coeffts(self) :
        '''
        Return all coefficients, in all bins, as a single array
        This is the overall intercept, plus the coefficients for each individual param
        Dimensions are: [binning ..., fit coeffts]
        '''
        
        array = [self.intercept]
        for param in list(self.params.values()) :
            for i in range(param.num_fit_coeffts) :
                array.append( param.get_fit_coefft(coefft_idx=i) )
        array = np.stack(array,axis=-1)
        return array


    @property
    def fit_coefft_labels(self) :
        '''
        Return labels for each fit coefficient
        '''
        return ["intercept"] + [ "%s p%i"%(param.name,i) for param in list(self.params.values()) for i in range(param.num_fit_coeffts) ]


    @property
    def serializable_state(self):
        """
        OrderedDict containing savable state attributes
        """

        if self._serializable_state is None: #TODO always redo?

            state = collections.OrderedDict()

            state["_initialized"] = self._initialized
            state["binning"] = self.binning.serializable_state
            state["initial_intercept"] = self.initial_intercept
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
            for name,param in list(self.params.items()) :
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

        # If it is not already a a state, alternativey try to load it in case a JSON file was passed
        if not isinstance(state,collections.Mapping) :
            try :
                state = from_json(state)
            except:
                raise IOError("Could not load state")


        #
        # Create params
        #

        params = []

        # Loop through params in the state        
        params_state = state.pop("params")
        for param_name,param_state in list(params_state.items()) :

            # Create the param
            param = HypersurfaceParam(
                name=param_state.pop("name"),
                func_name=param_state.pop("func_name"),
                initial_fit_coeffts=param_state.pop("initial_fit_coeffts"),
            )

            # Define rest of state
            for k in list(param_state.keys()) :
                setattr(param,k,param_state.pop(k))
                # print param.name,k,type(getattr(param,k)),getattr(param,k)

            # Store
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
        hypersurface.fit_maps_raw = None if fit_maps_raw is None else [ Map(**map_state) for map_state in fit_maps_raw ]
        fit_maps_norm = state.pop("fit_maps_norm")
        hypersurface.fit_maps_norm = None if fit_maps_norm is None else [ Map(**map_state) for map_state in fit_maps_norm ]

        # Define rest of state
        for k in list(state.keys()) :
            setattr(hypersurface,k,state.pop(k))
            # print k,type(getattr(hypersurface,k)),getattr(hypersurface,k)

        return hypersurface


class HypersurfaceParam(object) :
    '''
    A class representing one of the parameters (and corresponding functional forms) in the hypersurface.

    A user creates the initial instances of thse params, before passing the to the Hypersurface instance.
    Once this has happened, the user typically does not need to directly interact woth these 
    HypersurfaceParam instances.

    Parameters
    ----------
    name : str
        Name of the parameter

    func_name : str
        Name of the hypersurface function to use.
        See "Hypersurface functional forms" section for more details, including available functions.
        Reminder: Functions must be named, `<something>_hypersurface_func`, and then `func_name=<something>`.
        Note that a global search for functions named `<something>_hypersurface_func` is performed, so the 
        user can define new functions externally to this file.

    initial_fit_coeffts : array
        Initial values for the coefficients of the functional form
        Number and meaning of coefficients depends on functional form
    '''


    def __init__(self, name, func_name, initial_fit_coeffts=None ) :

        # Store basic members
        self.name = name

        # Handle functional form fit parameters
        self.fit_coeffts = None # Fit params container, not yet populated
        self.fit_coeffts_sigma = None # Fit param sigma container, not yet populated
        self.initial_fit_coeffts = initial_fit_coeffts # The initial values for the fit parameters

        # Record information relating to the fitting
        self.fitted = False # Flag indicating whether fit has been performed
        self.fit_param_values = None # The values of this sys param in each of the fitting datasets

        # Placeholder for nominal value
        self.nominal_value = None

        # Serialization
        self._serializable_state = None


        #
        # Init the functional form
        #

        # Get the function
        self.__name__ = func_name
        self._hypersurface_func = self._get_hypersurface_func(self.__name__)

        # Get the number of functional form parameters
        # This is the functional form function parameters, excluding the systematic paramater and the output object
        #TODO Not testwd for GPus
        self.num_fit_coeffts = get_num_args(self._hypersurface_func) - 2

        # Check and init the fit param initial values
        #TODO Add support for "per bin" initial values
        if initial_fit_coeffts is None :
            # No values provided, use 0 for all
            self.initial_fit_coeffts = np.zeros(self.num_fit_coeffts,dtype=FTYPE)
        else :
            # Use the provided initial values
            self.initial_fit_coeffts = np.array(self.initial_fit_coeffts)
            assert self.initial_fit_coeffts.size == self.num_fit_coeffts, "'initial_fit_coeffts' should have %i values, found %i" % (self.num_fit_coeffts,self.initial_fit_coeffts.size)


    def _get_hypersurface_func(self,func_name) :
        '''
        Find the function defining the hypersurface functional form.

        User specifies this by it's string name, which must correspond to a pre-defined 
        function with the name `<func_name>_hypersurface_func`.

        Note that a global search for functions named `<something>_hypersurface_func` is 
        performed, so the user can define new functions externally to this file.

        Internal function, not to be called by a user.
        '''

        assert isinstance(func_name,str), "'func_name' must be a string"

        # Form the expected function name
        hypersurface_func_suffix = "_hypersurface_func"
        fullfunc_name = func_name + hypersurface_func_suffix

        # Find all functions
        all_hypersurface_functions = { k:v for k,v in list(globals().items()) if k.endswith(hypersurface_func_suffix) }
        assert fullfunc_name in all_hypersurface_functions, "Cannot find hypersurface function '%s', choose from %s" % (func_name,[f.split(hypersurface_func_suffix)[0] for f in all_hypersurface_functions])
        return all_hypersurface_functions[fullfunc_name]


    def _init_fit_coefft_arrays(self,binning) :
        '''
        Create the arrays for storing the fit parameters
        Have one fit per bin, for each parameter
        The shape of the `self.fit_coeffts` arrays is: (binning shape ..., num fit params )

        Internal function, not to be called by a user.
        '''

        arrays = []

        self.binning_shape = binning.shape

        for fit_coefft_initial_value in self.initial_fit_coeffts :

            fit_coefft_array = np.full(self.binning_shape,fit_coefft_initial_value,dtype=FTYPE)
            arrays.append(fit_coefft_array)

        self.fit_coeffts = np.stack(arrays,axis=-1)
        self.fit_coeffts_sigma = np.full_like(self.fit_coeffts,np.NaN)


    def evaluate(self,param,out,bin_idx=None) :
        '''
        Evaluate the functional form for the given `param` values.
        Uses the current values of the fit coefficients.

        By default evaluates all bins, but optionally can specify a particular bin (used when fitting).
        '''

        #TODO properly use SmartArrays

        # Create an array to file with this contorubtion
        this_out = np.full_like(out,np.NaN,dtype=FTYPE)

        # Form the arguments to pass to the functional form
        # Need to be flexible in terms of the number of fit parameters
        args = [param]
        for cft_idx in range(self.num_fit_coeffts) :
            # idx = tuple(list(bin_idx) + [cft_idx])
            # args += [self.fit_coeffts[idx]]
            args += [self.get_fit_coefft(bin_idx=bin_idx,coefft_idx=cft_idx)]
        args += [this_out]

        # Call the function
        self._hypersurface_func(*args)

        # Add to overall hypersurface result
        out += this_out


    def get_fit_coefft_idx(self,bin_idx=None,coefft_idx=None) :
        '''
        Indexing the fit_coefft matrix is a bit of a pain
        This helper function eases things
        '''

        # TODO can probably do this more cleverly with numpy indexing, but works for now...

        # Indexing based on the bin
        if (bin_idx is Ellipsis) or (bin_idx is None) :
            idx = [Ellipsis]
        else :
            idx = list(bin_idx)

        # Indexing based on the coefficent
        if isinstance(coefft_idx,slice) :
            idx.append(coefft_idx)
        elif coefft_idx is None :
            idx.append(slice(0,-1))
        else :
            idx.append(coefft_idx)

        # Put it all together
        idx = tuple(idx)
        return idx


    def get_fit_coefft(self,*args,**kwargs) :
        '''
        Get a fit coefficient values from the matrix
        Basically just wrapping the indexing function
        '''
        idx = self.get_fit_coefft_idx(*args,**kwargs)
        return self.fit_coeffts[idx]


    @property
    def serializable_state(self):
        """
        OrderedDict containing savable state attributes
        """

        if self._serializable_state is None: #TODO always redo?

            state = collections.OrderedDict()
            state["name"] = self.name
            state["func_name"] = self.__name__
            state["num_fit_coeffts"] = self.num_fit_coeffts
            state["fit_coeffts"] = self.fit_coeffts
            state["fit_coeffts_sigma"] = self.fit_coeffts_sigma
            state["initial_fit_coeffts"] = self.initial_fit_coeffts
            state["fitted"] = self.fitted
            state["fit_param_values"] = self.fit_param_values
            state["binning_shape"] = self.binning_shape
            state["nominal_value"] = self.nominal_value

            self._serializable_state = state

        return self._serializable_state 



'''
Hypersurface fitting and loading helper functions
'''


def get_hypersurface_file_name(hypersurface, tag) :
    '''
    Create a descriptive file name
    '''

    num_dims = len(hypersurface.params)
    param_str = "_".join(hypersurface.param_names)
    output_file = "%s__hypersurface_fits__%dd__%s.json" % (tag, num_dims, param_str)

    return output_file


def fit_hypersurfaces(nominal_dataset, sys_datasets, params, output_dir, tag, combine_regex=None, **hypersurface_fit_kw) :
    '''
    A helper function that a user can use to fit hypersurfaces to a bunch of simulation datasets,
    and save the results to a file. Basically a wrapper of Hypersurface.fit, handling common pre-fitting tasks
    like producing mapsets from piplelines, merging maps from similar specifies, etc.

    Note that this supports fitting multiple hypersurfaces to the datasets, e.g. one per simulated
    species. Returns a dict with format: { map_0_key : map_0_hypersurface, ..., map_N_key : map_N_hypersurface, }

    Parameters
    ----------
    nominal_dataset : dict
        Definition of the nominal dataset. Specifies the pipleline with which the maps can be created, and the 
        values of all systematic parameters used to produced the dataset.
        Format must be: 
            nominal_dataset = {
                "pipeline_cfg" = <pipeline cfg file (either cfg file path or dict)>),
                "sys_params" = { param_0_name : param_0_value_in_dataset, ..., param_N_name : param_N_value_in_dataset }
            }
        Sys params must correspond to the provided HypersurfaceParam instances provided in the `params` arg.

    sys_datasets : list of dicts
        List of dicts, where each dict defines one of the systematics datasets to be fitted.
        The format of each dict is the same as explained for `nominal_dataset`

    params : list of HypersurfaceParams
        List of HypersurfaceParams instances that define the hypersurface.
        Note that this defined ALL hypersurfaces fitted in this function, e.g. only supports a single parameterisation 
        for all maps (this is almost almost what you want).

    output_dir : str
        Path to directly to write results file in

    tag : str
        A string identifier that will be included in the file name to help you make sense of the file in the future.
        Note that additional information on the contents will be added to the file name by this function.

    combine_regex : list of str, or None
        List of string regex expressions that will be used for merging maps.
        Used to combine similar species. 
        Must be something that can be passed to the `MapSet.combine_re` function (see that functions docs for more details).
        Choose `None` is do not want to perform this merging.
        
    hypersurface_fit_kw : kwargs
        kwargs will be passed on to the calls to `Hypersurface.fit`
    '''

    #TODO Current yneed to manually ensure consistency between `combine_regex` here and the `links` param in `pi_hypersurface`
    #     Need to make `pi_hypersurface` directly use the value of `combine_regex` from the Hypersurface instance

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
    all_datasets = [ nominal_dataset ] + sys_datasets
    for dataset in all_datasets :
        assert isinstance(dataset, collections.Mapping)
        assert "pipeline_cfg" in dataset
        assert isinstance(dataset["pipeline_cfg"], (str, collections.Mapping) )
        assert "sys_params" in dataset
        assert isinstance(dataset["sys_params"], collections.Mapping)

    # Check params
    assert len(params) >= 1
    for p in params :
        assert isinstance(p, HypersurfaceParam)

    # Report inputs
    print("Hypersurface fit details :")
    print("  Num params            : %i" % len(params) )
    print("  Num fit coefficients  : %i" % sum([ p.num_fit_coeffts for p in params ]) )
    print("  Num datasets          : 1 nominal + %i systematics" % len(sys_datasets) )
    print("  Nominal values        : %s" % nominal_dataset["sys_params"] )


    #
    # Generate MapSets
    #

    # Create and run the nominal and systematics pipelines (using the pipeline configs provided) to get maps
    nominal_dataset["mapset"] = Pipeline(nominal_dataset["pipeline_cfg"]).get_outputs() #return_sum=False)
    for sys_dataset in sys_datasets :
        sys_dataset["mapset"] = Pipeline(sys_dataset["pipeline_cfg"]).get_outputs() #return_sum=False)

    # Merge maps according to the combine regex, is one was provided
    if combine_regex is not None :
        nominal_dataset["mapset"] = nominal_dataset["mapset"].combine_re(combine_regex)
        for sys_dataset in sys_datasets :
            sys_dataset["mapset"] = sys_dataset["mapset"].combine_re(combine_regex)

    #TODO check every mapset has the same elements


    #
    # Loop over maps
    #

    # Create the container to fill
    hypersurfaces = collections.OrderedDict()

    # Loop over maps
    for map_name in nominal_dataset["mapset"].names :


        #
        # Prepare data for fit
        #

        nominal_map = nominal_dataset["mapset"][map_name]
        nominal_param_values = nominal_dataset["sys_params"]

        sys_maps = [ sys_dataset["mapset"][map_name] for sys_dataset in sys_datasets   ]
        sys_param_values = [ sys_dataset["sys_params"] for sys_dataset in sys_datasets   ]


        #
        # Fit the hypersurface
        #

        # Create the hypersurface
        hypersurface = Hypersurface( 
            params=copy.deepcopy(params),
            initial_intercept=1., # Initial value for intercept
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
        # print("\nFitted hypersurface report:")
        # print(hypersurface)

        # Store for later write to disk
        hypersurfaces[map_name] = hypersurface


    #
    # Store results
    #

    # Create a file name
    output_path = os.path.join( output_dir, get_hypersurface_file_name( list(hypersurfaces.values())[0], tag) )

    # Create the output directory
    mkdir(output_dir)

    # Write to a json file
    to_json(hypersurfaces,output_path)

    print("Fit results written : %s" % output_path)

    return output_dir



def load_hypersurfaces(input_file) :
    '''
    User function to load file containing hypersurface fits, as written using `fit_hypersurfaces`.
    Can be multiple hypersurfaces assosicated with different maps.

    Returns a dict with the format: { map_0_key : map_0_hypersurface, ..., map_N_key : map_N_hypersurface, }

    Parameters
    ----------
    input_file : str
        Path to the file contsaining the hypersurface fits.
    '''

    # Testing various cases to support older files as well as modern ones...
    try :

        #
        # Current files
        #

        # Load file
        hypersurface_states = from_json(input_file)
        assert isinstance(hypersurface_states,collections.Mapping)

        # Load hypersurfaces
        hypersurfaces = collections.OrderedDict()
        for map_name,hypersurface_state in list(hypersurface_states.items()) :
            hypersurfaces[map_name] = Hypersurface.from_state(hypersurface_state)
        return hypersurfaces

                # Loop over hypersurface states and load them

    except :
        pass

    try :

        #
        # Legacy files
        #

        hypersurfaces = load_hypersurfaces_legacy(input_file)
        print("Old fit files detected, loaded via legacy mode")
        return hypersurfaces

    except :
        pass

    #TODO DRAGON/GRECO data release files

    # If made it here, nothing worked
    raise Exception("Could not load fits in `modern` or `legacy` mode, something is wrong with the file")



def load_hypersurfaces_legacy(input_file) :
    '''
    Load an old hyperpane (not surface) fit file from older PISA version.

    Put the results into an instance the new `Hypersurface` class so can use the 
    resulting hypersurface in modern code.
    '''

    hypersurfaces = collections.OrderedDict()


    #
    # Load file
    #

    input_dict = from_json(input_file)
    assert isinstance(input_dict,collections.Mapping)


    #
    # Loop over map names
    #

    for map_name in  input_dict["map_names"] :


        #
        # Create the params
        #

        # Get the param names
        param_names = input_dict["sys_list"]

        # Create the param instances.
        # Using linear functional forms (legacy files only supported linear forms, e.g. 
        # hyperplanes rather than surfaces).
        params = [ HypersurfaceParam( name=name, func_name="linear", initial_fit_coeffts=None, ) for name in param_names ]


        #
        # Get binning
        #

        # This varies depending on how old the file is...
        # Note that the hypersurface class really only needs to know the binning 
        # shape (to create the coefficient arrays).

        # If the (serialized version of the) binning is stored, great! Use it
        if "binning" in input_dict :
            binning = MultiDimBinning(**input_dict["binning"])

        # If no binning is available, can at least get the correct shape (using 
        # one of the map arrays) and create a dummy binning instance.
        # Remember that the final dimension is the sys params, not binning
        else :
            binning_shape = input_dict[map_name][...,0].shape # Remove last dimension
            binning = MultiDimBinning([ OneDimBinning( name="dummy_%i"%i, domain=[0.,1.], is_lin=True, num_bins=dim ) for i,dim in enumerate(binning_shape) ])
            

        #
        # Create the hypersurface instance
        #

        # Create the hypersurface
        hypersurface = Hypersurface( 
            params=params, # Specify the systematic parameters
            initial_intercept=0., # Intercept value (or first guess for fit)
        )

        # Set some internal members that would normally be configured during fitting
        # Don't know the nominal values with legacy files, so just stores NaNs
        hypersurface._init( 
            binning=binning, 
            nominal_param_values={ name:np.NaN for name in hypersurface.param_names },
        )

        # Indicate this is legacy data (not all functionality will work)
        hypersurface.using_legacy_data = True


        #
        # Get the fit values
        #

        # Handling two different legacy cases here...
        fitted_coefficients = input_dict["hyperplanes"][map_name]["fit_params"] if "hyperplanes" in input_dict else input_dict[map_name]

        # Fitted coefficients have following array shape: [ binning dim 0,  ..., binning dim N, sys params (inc. intercept) ]
        intercept_values = fitted_coefficients[...,0]
        sys_param_gradient_values = { n:fitted_coefficients[...,i+1] for i,n in enumerate(param_names) }

        # Write the values to the hypersurface
        np.copyto( src=intercept_values, dst=hypersurface.intercept )
        for param in hypersurface.params.values() :       
            np.copyto( src=sys_param_gradient_values[param.name], dst=param.fit_coeffts[...,0] )

        # Done, store the hypersurface
        hypersurfaces[map_name] = hypersurface

    return hypersurfaces


'''
Plotting
'''

def plot_bin_fits(ax, hypersurface, bin_idx, param_name, color=None, label=None, show_nominal=False) :
    '''
    Plot the hypersurface for a given bin, in 1D w.r.t. to a single specified parameter.
    Plots the following:
      - on-axis data points used in the fit
      - hypersurface w.r.t to the specified parameter (1D)
      - nominal value of the specified parameter

    Parameters
    ----------
    ax : matplotlib.Axes
        matplotlib ax to draw the plot on

    hypersurface : Hypersurface
        Hypersurface to make the plots from

    bin_idx : tuple
        Index (numpy array indexing format) of the bin to plot

    param_name : str
        Name of the parameter of interest

    color : str
        color to use for hypersurface curve

    label : str
        label to use for hypersurface curve

    show_nominal : bool
        Indicate the nominal value of the param on the plot
    '''

    import matplotlib.pyplot as plt

    # Get the param
    param = hypersurface.params[param_name]

    # Check bin index
    assert len(bin_idx) == len(hypersurface.binning.shape)

    # Get bin values for this bin only
    chosen_bin_values = [ m.nominal_values[bin_idx] for m in hypersurface.fit_maps ]
    chosen_bin_sigma = [ m.std_devs[bin_idx] for m in hypersurface.fit_maps ]

    # Define a mask for selecting on-axis points only
    on_axis_mask = hypersurface.get_on_axis_mask(param.name)

    # Plot the points from the datasets used for fitting
    x = np.asarray(param.fit_param_values)[on_axis_mask]
    y = np.asarray(chosen_bin_values)[on_axis_mask]
    yerr = np.asarray(chosen_bin_sigma)[on_axis_mask]
    ax.errorbar( x=x, y=y, yerr=yerr, marker="o", color=("black" if color is None else color), linestyle="None", label=label )

    # Plot the hypersurface
    # Generate as bunch of values along the sys param axis to make the plot
    # Then calculate the hypersurface value at each point, using the nominal values for all other sys params
    x_plot = np.linspace( np.nanmin(param.fit_param_values), np.nanmax(param.fit_param_values), num=100 )
    params_for_plot = { param.name : x_plot, }
    for p in list(hypersurface.params.values()) :
        if p.name != param.name :
            params_for_plot[p.name] = np.full_like(x_plot,hypersurface.nominal_values[p.name])
    y_plot = hypersurface.evaluate(params_for_plot,bin_idx=bin_idx)
    ax.plot( x_plot, y_plot, color=("red" if color is None else color) )

    #TODO Add fit uncertainty. Problem using uarrays with np.exp at the minute, may need to shift to bin-wise calc...
    # ax.fill_between( curve_x[i,:], unp.nominal_values(y_opt)-unp.std_devs(y_opt), unp.nominal_values(y_opt)+unp.std_devs(y_opt), color='red', alpha=0.2 )

    # # Optional : For testing, overlay the uncertainty one would find under the assumption fit parameters are uncorrelated
    #TODO This is removed until hyperplane uncertainty is re-implemented
    # # Typically straight line fit parameters are strongly correlated, so expect this to be a large overestimation
    # if False :
    #     cov_mat = fit_results["hypersurfaces"][map_name]["cov_matrices"][:,:,zind][idx]
    #     fit_params_uncorr = unp.uarray( unp.nominal_values(fit_params) , np.sqrt(np.diag(cov_mat)) )
    #     y_opt_uncorr = hypersurface_fun(curve_x, *fit_params_uncorr)
    #     ax.fill_between( curve_x[i,:], unp.nominal_values(y_opt_uncorr)-unp.std_devs(y_opt_uncorr), unp.nominal_values(y_opt_uncorr)+unp.std_devs(y_opt_uncorr), color='blue', alpha=0.5 )

    # Mark the nominal value
    if show_nominal :
        ax.axvline( x=param.nominal_value, color="blue", alpha=0.7, linestyle="-", label="Nominal", zorder=-1 )

    # Format ax
    ax.set_xlabel(param.name)
    ax.grid(True)
    ax.legend()




def plot_bin_fits_2d(ax, hypersurface, bin_idx, param_names ) :
    '''
    Plot the hypersurface for a given bin, in 2D w.r.t. to a pair of params
    Plots the following:
      - All data points used in the fit
      - hypersurface w.r.t to the specified parameters (2D)
      - nominal value of the specified parameters

    Parameters
    ----------
    ax : matplotlib.Axes
        matplotlib ax to draw the plot on

    hypersurface : Hypersurface
        Hypersurface to make the plots from

    bin_idx : tuple
        Index (numpy array indexing format) of the bin to plot

    param_names : list of str
        List containing the names of the two parameters of interest
    '''

    import matplotlib.pyplot as plt

    assert len(param_names) == 2
    assert len(bin_idx) == len(hypersurface.binning.shape)

    # Get bin values for this bin only
    chosen_bin_values = [ m.nominal_values[bin_idx] for m in hypersurface.fit_maps ]
    chosen_bin_sigma = [ m.std_devs[bin_idx] for m in hypersurface.fit_maps ]

    # Shortcuts to the param values and bin values
    p0 = hypersurface.params[param_names[0]]
    p1 = hypersurface.params[param_names[1]]
    z = np.asarray(chosen_bin_values)
    # zerr = #TODO error bars

    # Choose categories of points to plot
    nominal_mask = hypersurface.get_nominal_mask()
    p0_on_axis_mask = hypersurface.get_on_axis_mask(p0.name) & (~nominal_mask)
    p1_on_axis_mask = hypersurface.get_on_axis_mask(p1.name) & (~nominal_mask)

    off_axis_mask = np.ones_like(p1_on_axis_mask,dtype=bool)
    for p in list(hypersurface.params.values()) : # Ignore points that are off-axis for other params
        if p.name not in param_names :
            off_axis_mask = off_axis_mask & (p.fit_param_values == p.nominal_value)
    off_axis_mask = off_axis_mask & ~(p0_on_axis_mask | p1_on_axis_mask | nominal_mask)

    # Plot data points
    ax.scatter( p0.fit_param_values[p0_on_axis_mask], p1.fit_param_values[p0_on_axis_mask], z[p0_on_axis_mask], marker="o", color="blue", label="%s on-axis"%p0.name )
    ax.scatter( p0.fit_param_values[p1_on_axis_mask], p1.fit_param_values[p1_on_axis_mask], z[p1_on_axis_mask], marker="^", color="red", label="%s on-axis"%p1.name )
    ax.scatter( p0.fit_param_values[off_axis_mask], p1.fit_param_values[off_axis_mask], z[off_axis_mask], marker="s", color="black", label="Off-axis" )
    ax.scatter( p0.fit_param_values[nominal_mask], p1.fit_param_values[nominal_mask], z[nominal_mask], marker="*", color="magenta", label="Nominal" )

    # Plot hypersurface (as a 2D surface)
    x_plot = np.linspace( p0.fit_param_values.min(), p0.fit_param_values.max(), num=100 )
    y_plot = np.linspace( p1.fit_param_values.min(), p1.fit_param_values.max(), num=100 )
    x_grid, y_grid = np.meshgrid(x_plot,y_plot)
    x_grid_flat = x_grid.flatten()
    y_grid_flat = y_grid.flatten()
    params_for_plot = { p0.name : x_grid_flat, p1.name : y_grid_flat, }
    for p in list(hypersurface.params.values()) :
        if p.name not in list(params_for_plot.keys()) :
            params_for_plot[p.name] = np.full_like(x_grid_flat,hypersurface.nominal_values[p.name])
    z_grid_flat = hypersurface.evaluate(params_for_plot,bin_idx=bin_idx)
    z_grid = z_grid_flat.reshape(x_grid.shape)
    surf = ax.plot_surface( x_grid, y_grid, z_grid, cmap="viridis", linewidth=0, antialiased=False, alpha=0.2 )#, label="Hypersurface" )

    # Format
    ax.set_xlabel(p0.name)
    ax.set_ylabel(p1.name)
    ax.legend()



#
# Test/example
#

def hypersurface_example() :
    '''
    Simple hypersurface example covering:
      - Defining the hypersurface
      - Fitting the coefficients (to toy data)
      - Saving and re-loading the hypersurfaces
      - Plotting the results
    '''

    import sys

    #TODO turn this into a PASS/FAIL test, and add more detailed test of specific functions

    #
    # Create hypersurface
    #

    # Define systematic parameters in the hypersurface
    params = [
        HypersurfaceParam( name="foo", func_name="linear", initial_fit_coeffts=[1.], ),
        HypersurfaceParam( name="bar", func_name="exponential", initial_fit_coeffts=[1.,-1.], ),
    ]

    # Create the hypersurface
    hypersurface = Hypersurface( 
        params=params, # Specify the systematic parameters
        initial_intercept=0., # Intercept value (or first guess for fit)
    )


    #
    # Create fake datasets
    #

    from pisa.core.map import Map, MapSet

    # Just doing something quick here for demonstration purposes
    # Here I'm only assigning a single value per dataset, e.g. one bin, for simplicity, but idea extends to realistic binning

    # Define binning
    binning = MultiDimBinning([OneDimBinning(name="reco_energy",domain=[0.,10.],num_bins=3,units=ureg.GeV,is_lin=True)])
    # binning = MultiDimBinning([OneDimBinning(name="reco_energy",domain=[0.,10.],num_bins=2,units=ureg.GeV,is_lin=True),OneDimBinning(name="reco_coszen",domain=[-1.,1.],num_bins=3,is_lin=True)])

    # Define the values for the parameters for each dataset
    nom_param_values = {}
    sys_param_values_dict = {}

    if "foo" in [ p.name for p in params ] :
        nom_param_values["foo"] = 0.
        sys_param_values_dict["foo"] = [ 0., 0., 0.,-1.,+1., 1.]

    if "bar" in [ p.name for p in params ] :
        nom_param_values["bar"] = 10.
        sys_param_values_dict["bar"] = [20.,30.,0.,10.,10., 15.]

    # Get number of datasets
    num_sys_datasets = len(list(sys_param_values_dict.values())[0])

    # Only consider one particle type for simplicity
    particle_key = "nue_cc"

    # Create a dummy "true" hypersurface that can be used to generate some fake bin values for the dataset 
    true_hypersurface = copy.deepcopy(hypersurface)
    true_hypersurface._init(binning=binning,nominal_param_values=nom_param_values)
    true_hypersurface.intercept.fill(3.)
    if "foo" in true_hypersurface.params :
        true_hypersurface.params["foo"].fit_coeffts[...,0].fill(2.)
    if "bar" in true_hypersurface.params :
        # true_hypersurface.params["bar"].fit_coeffts[...,0].fill(2.)
        true_hypersurface.params["bar"].fit_coeffts[...,0].fill(5.)
        true_hypersurface.params["bar"].fit_coeffts[...,1].fill(-0.1)

    print("\nTruth hypersurface report:")
    true_hypersurface.report()

    # Create each dataset, e.g. set the systematic parameter values, calculate a bin count
    hist = true_hypersurface.evaluate(nom_param_values)
    nom_map = Map(name=particle_key,binning=binning,hist=hist,error_hist=np.sqrt(hist))
    sys_maps = []
    sys_param_values = []
    for i in range(num_sys_datasets) :
        sys_param_values.append( { name:sys_param_values_dict[name][i] for name in list(true_hypersurface.params.keys()) } )
        hist = true_hypersurface.evaluate(sys_param_values[-1])
        sys_maps.append( Map(name=particle_key,binning=binning,hist=hist,error_hist=np.sqrt(hist)) )


    #
    # Fit hypersurfaces
    #

    # Perform fit
    hypersurface.fit(
        nominal_map=nom_map,
        nominal_param_values=nom_param_values,
        sys_maps=sys_maps,
        sys_param_values=sys_param_values,
        norm=False,
    )

    # Report the results
    print("\nFitted hypersurface report:")
    print(hypersurface)

    # Check the fitted parameter values match the truth
    # This only works if `norm=False` in the `hypersurface.fit` call just above
    print("\nChecking fit recovered truth...")
    assert np.allclose( hypersurface.intercept, true_hypersurface.intercept )
    for param_name in hypersurface.param_names :
        assert np.allclose( hypersurface.params[param_name].fit_coeffts, true_hypersurface.params[param_name].fit_coeffts )
    print("... fit was successful!\n")


    #
    # Save/load
    #

    # Save
    file_path = "hypersurface.json.bz2"
    to_json(hypersurface,file_path)

    # Re-load
    reloaded_hypersurface = Hypersurface.from_state(file_path)

    # Test the re-loaded hypersurface matches the one we saved
    print("\nChecking saved and re-loaded hypersurfaces are identical...")
    assert np.allclose( hypersurface.intercept, reloaded_hypersurface.intercept )
    for param_name in hypersurface.param_names :
        assert np.allclose( hypersurface.params[param_name].fit_coeffts, reloaded_hypersurface.params[param_name].fit_coeffts )
    print("... fit was successful!\n")

    # Continue with the reloaded version
    hypersurface = reloaded_hypersurface


    #
    # 1D plot
    #

    import matplotlib.pyplot as plt

    # Create the figure
    fig,ax = plt.subplots(1,len(hypersurface.params))

    # Choose an arbitrary bin for plotting
    bin_idx = tuple([ 0 for i in range(hypersurface.binning.num_dims) ])

    # Plot each param
    for i,param in enumerate(hypersurface.params.values()) :

        plot_ax = ax if len(hypersurface.params) == 1 else ax[i]

        plot_bin_fits(
            ax=plot_ax,
            hypersurface=hypersurface,
            bin_idx=bin_idx,
            param_name=param.name,
            show_nominal=True,
        )

    # Format
    fig.tight_layout()


    # Save
    fig_file_path = "hypersurface_1d.pdf"
    fig.savefig(fig_file_path)
    print("Figure saved : %s" % fig_file_path)


    #
    # 2D plot
    #

    if len(hypersurface.params) > 1 :

        from mpl_toolkits.mplot3d import Axes3D

        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot
        plot_bin_fits_2d(
            ax=ax,
            hypersurface=hypersurface,
            bin_idx=bin_idx,
            param_names=["foo","bar"],
        )

        plt.show()

        # Format
        fig.tight_layout()

        # Save
        fig_file_path = "hypersurface_2d.pdf"
        fig.savefig(fig_file_path)
        print("Figure saved : %s" % fig_file_path)


# Run the examp'es/tests
if __name__ == "__main__" : 
    hypersurface_example()