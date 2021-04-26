"""
Classes and methods needed to do hypersurface interpolation over arbitrary parameters.
"""

__all__ = ['HypersurfaceInterpolator', 'run_interpolated_fit', 'prepare_interpolated_fit',
            'assemble_interpolated_fits', 'load_interpolated_hypersurfaces', 'pipeline_cfg_from_states',
            'serialize_pipeline_cfg', 'get_incomplete_job_idx']

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
from scipy import interpolate
from .hypersurface import Hypersurface, HypersurfaceParam
from pisa import FTYPE, ureg
from pisa.utils import matrix
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


class HypersurfaceInterpolator(object):
    """Factory for interpolated hypersurfaces.

    After being initialized with a set of hypersurface fits produced at different
    parameters, it uses interpolation to produce a Hypersurface object
    at a given point in parameter space using scipy's `RegularGridInterpolator`.
    
    The interpolation is piecewise-linear between points. All points must lie on a
    rectilinear ND grid.

    Parameters
    ----------
    interpolation_param_spec : dict
        Specification of interpolation parameter grid of the form::
            interpolation_param_spec = {
                'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                ...
                'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
            }
        where values are given as :obj:`Quantity`.
    hs_fits : list of dict
        list of dicts with hypersurfacesthat were fit at the points of the parameter mesh
        defined by interpolation_param_spec
    ignore_nan : bool
        Ignore empty bins in hypersurfaces. The intercept in those bins is set to 1 and
        all slopes are set to 0.

    Notes
    -----
    Be sure to give a support that covers the entire relevant parameter range and a
    good distance beyond! To prevent minimization failure from NaNs, extrapolation
    is used if hypersurfaces outside the support are requested but needless to say
    these numbers are unreliable.

    See Also
    --------
    scipy.interpolate.RegularGridInterpolator :
        class used for interpolation
    """

    def __init__(self, interpolation_param_spec, hs_fits, ignore_nan=True):
        self.ndim = len(interpolation_param_spec.keys())
        # key ordering is important to guarantee that dimensions stay consistent
        msg = "interpolation params must be specified as a dict with ordered keys"
        assert isinstance(interpolation_param_spec, collections.OrderedDict), msg
        for k, v in interpolation_param_spec.items():
            assert set(v.keys()) == {"values", "scales_log"}
            assert isinstance(v["values"], collections.Sequence)
        self.interp_param_spec = interpolation_param_spec
        reference_hs = hs_fits[0]["hs_fit"]
        # we are going to produce the hypersurface from a state that is the same
        # as the reference, only the coefficients and covariance matrices are
        # injected from the interpolation.
        self._reference_state = copy.deepcopy(reference_hs.serializable_state)
        # for cleanliness we wipe numbers from the original state
        self._reference_state["intercept_sigma"] = np.nan
        self._reference_state["fit_maps_norm"] = None
        self._reference_state["fit_maps_raw"] = None
        self._reference_state["fit_chi2"] = np.nan
        for param in self._reference_state['params'].values():
            param['fit_coeffts_sigma'] = np.full_like(
                param['fit_coeffts_sigma'], np.nan)
        # Instead of holding numbers, these coefficients and covariance matrices are
        # interpolator objects the produce them at the requested point.
        # The shape of fit_coeffts is [binning ..., fit coeffts]
        self.coeff_shape = reference_hs.fit_coeffts.shape
        self.coefficients = None
        # The shape of fit_cov_mat is [binning ..., fit coeffts, fit coeffts]
        self.covars_shape = reference_hs.fit_cov_mat.shape
        self.covars = None
        
        # We now need to massage the fit coefficients into the correct shape
        # for interpolation.
        # The dimensions of the interpolation parameters come first, the dimensions
        # of the hypersurface coefficients comes last.
        self.interp_shape = tuple(len(v["values"]) for v in self.interp_param_spec.values())
        # dimension is [interp_shape, binning..., fit coeffts]
        self._coeff_z = np.zeros(self.interp_shape + self.coeff_shape)
        # dimension is [interp_shape, binning..., fit coeffts, fit coeffts]
        self._covar_z = np.zeros(self.interp_shape + self.covars_shape)
        # Here we use the same indexing as below in `fit_hypersurfaces`
        for i, idx in enumerate(np.ndindex(self.interp_shape)):
            # As an additional safety measure, we check that the parameters are what
            # we expect to find at this index.
            expected_params = dict(
                (n, self.interp_param_spec[n]["values"][idx[j]])
                for j, n in enumerate(self.interp_param_spec.keys())
            )
            param_values = hs_fits[i]["param_values"]
            msg = ("The stored values where hypersurfaces were fit do not match those"
                   "in the interpolation grid.")
            assert np.all([expected_params[n].m == param_values[n].m
                           for n in self.interp_param_spec.keys()]), msg
            self._coeff_z[idx] = hs_fits[i]["hs_fit"].fit_coeffts
            self._covar_z[idx] = hs_fits[i]["hs_fit"].fit_cov_mat
        
        grid_coords = list(
            np.array([val.m for val in val_list["values"]])
            for val_list in self.interp_param_spec.values()
        )
        # If a parameter scales as log, we give the log of the parameter to the 
        # interpolator. We must not forget to do this again when we call the
        # interpolator later!
        for i, param_name in enumerate(self.interpolation_param_names):
            if self.interp_param_spec[param_name]["scales_log"]:
                grid_coords[i] = np.log10(grid_coords[i])
        self.coefficients = interpolate.RegularGridInterpolator(
            grid_coords,
            self._coeff_z,
            # while we do enable extrapolation, it is not at all reliable... we only
            # use it to avoid crashes/nans if the minimizer tests extreme values outside
            # the domain
            bounds_error=False, fill_value=None
        )
        self.covars = interpolate.RegularGridInterpolator(
            grid_coords,
            self._covar_z,
            bounds_error=False, fill_value=None
        )
        # In order not to spam warnings, we only want to warn about non positive
        # semi definite covariance matrices once for each bin. We store the bin
        # indeces for which the warning has already been issued.
        self.covar_bins_warning_issued = []
        self.ignore_nan = ignore_nan
    
    @property
    def interpolation_param_names(self):
        return list(self.interp_param_spec.keys())
    
    @property
    def param_names(self):
        return list(self._reference_state["params"].keys())
    
    @property
    def num_interp_params(self):
        return len(self.interp_param_spec.keys())
    
    def get_hypersurface(self, **param_kw):
        """
        Get a Hypersurface object with interpolated coefficients.

        Parameters
        ----------
        **param_kw
            Parameters are given as keyword arguments, where the names
            of the arguments must match the names of the parameters over
            which the hypersurfaces are interpolated. The values
            are given as :obj:`Quantity` objects with units.
        """
        assert set(param_kw.keys()) == set(self.interp_param_spec.keys()), "invalid parameters"
        # getting param magnitudes in the same units as the parameter specification
        x = np.array([
            param_kw[p].m_as(self.interp_param_spec[p]["values"][0].u)
            # we have checked that this is an OrderedDict so that the order of x is not
            # ambiguous here
            for p in self.interp_param_spec.keys()
        ])
        # if a parameter scales as log, we have to take the log here again
        for i, param_name in enumerate(self.interpolation_param_names):
            if self.interp_param_spec[param_name]["scales_log"]:
                x[i] = np.log10(x[i])
        
        state = copy.deepcopy(self._reference_state)
        # fit covariance matrices are stored directly in the state while fit coeffts
        # must be assigned with the setter method...
        # need squeeze here because the RegularGridInterpolator always puts another 
        # dimension around the output
        state["fit_cov_mat"] = np.squeeze(self.covars(x))
        assert state["fit_cov_mat"].shape == self.covars_shape
        for idx in np.ndindex(state['fit_cov_mat'].shape):
            if self.ignore_nan: continue
            assert np.isfinite(state['fit_cov_mat'][idx]), ("invalid cov matrix "
                f"element encountered at {param_kw} in loc {idx}")
        # check covariance matrices for symmetry, positive semi-definiteness
        for bin_idx in np.ndindex(state['fit_cov_mat'].shape[:-2]):
            m = state['fit_cov_mat'][bin_idx]
            if self.ignore_nan and np.any(~np.isfinite(m)):
                state['fit_cov_mat'][bin_idx] = np.identity(m.shape[0])
                m = state['fit_cov_mat'][bin_idx]
            assert np.allclose(
                m, m.T, rtol=ALLCLOSE_KW['rtol']*10.), f'cov matrix not symmetric in bin {bin_idx}'
            if not matrix.is_psd(m):
                state['fit_cov_mat'][bin_idx] = matrix.fronebius_nearest_psd(m)
                if not bin_idx in self.covar_bins_warning_issued:
                    logging.warn(
                        f'Invalid covariance matrix fixed in bin: {bin_idx}')
                    self.covar_bins_warning_issued.append(bin_idx)
        hypersurface = Hypersurface.from_state(state)
        coeffts = np.squeeze(self.coefficients(x))  # calls interpolator
        assert coeffts.shape == self.coeff_shape
        # check that coefficients exist and if not replace with default values
        for idx in np.ndindex(self.coeff_shape):
            if self.ignore_nan and ~np.isfinite(coeffts[idx]):
                coeffts[idx] = 1 if idx[-1] == 0 else 0  # set intercept to 1, slopes 0
            assert np.isfinite(coeffts[idx]), ("invalid coeff encountered at "
                f"{param_kw} in loc {idx}")
        # the setter method defined in the Hypersurface class takes care of
        # putting the coefficients in the right place in their respective parameters
        hypersurface.fit_coeffts = coeffts
        return hypersurface
    
    def make_slices(self, *xi):
        """Make slices of hypersurfaces for plotting.

        In some covariance matrices, the spline fits are corrected to make
        the matrix positive semi-definite. The slices produced by this function
        include all of those effects.

        Parameters
        ----------
        xi : list of ndarray
            Points at which the hypersurfaces are to be evaluated. The length of the 
            list must equal the number of parameters, each ndarray in the list must have
            the same shape (slice_shape).

        Returns
        -------
        coeff_slices : numpy.ndarray
            slices in fit coefficients. Size: (binning..., number of coeffs) + slice_shape
        covar_slices : numpy.ndarray
            slices in covariance matrix elements.
            Size: (binning..., number of coeffs, number of coeffs) + slice_shape
        """
        slice_shape = xi[0].shape
        for x in xi:
            assert x.shape == slice_shape
        assert len(xi) == self.num_interp_params
        coeff_slices = np.zeros(self.coeff_shape + slice_shape)
        covar_slices = np.zeros(self.covars_shape + slice_shape)
        for idx in np.ndindex(slice_shape):
            pars = collections.OrderedDict()
            for i, name in enumerate(self.interpolation_param_names):
                pars[name] = xi[i][idx]
            hs = self.get_hypersurface(**pars)
            slice_idx = (Ellipsis,) + idx
            coeff_slices[slice_idx] = hs.fit_coeffts
            covar_slices[slice_idx] = hs.fit_cov_mat
        return coeff_slices, covar_slices

    def plot_fits_in_bin(self, bin_idx, ax=None, n_steps=20, **param_kw):
        """
        Plot the coefficients as well as covariance matrix elements as a function
        of the interpolation parameters.

        Parameters
        ----------
            bin_idx : tuple
                index of the bin for which to plot the fits
            ax : 2D array of axes, optional
                axes into which to place the plots. If None (default),
                appropriate axes will be generated. Must have at least
                size (n_coeff, n_coeff + 1).
            n_steps : int, optional
                number of steps to plot between minimum and maximum
            **param_kw :
                Parameters to be fixed when producing slices. If the interpolation 
                is in N-D, then (N-2) parameters need to be fixed to produce 2D plots
                of the remaining 2 parameters and (N-1) need to be fixed to produce a
                1D slice.
        """
        plot_dim = self.ndim - len(param_kw.keys())
        assert plot_dim <= 2, "plotting only supported in 1D or 2D"
        import matplotlib.pyplot as plt
        n_coeff = self.coeff_shape[-1]
        hs_param_names = list(self._reference_state['params'].keys())
        hs_param_labels = ["intercept"] + [f"{p} p{i}" for p in hs_param_names
                                           for i in range(self._reference_state['params'][p]['num_fit_coeffts'])]
        if ax is None:
            fig, ax = plt.subplots(nrows=n_coeff, ncols=n_coeff+1,
                                   squeeze=False, sharex=True,
                                   figsize=(20, 10))
        # remember whether the plots need log scale or not, by default not
        x_is_log = False
        y_is_log = False
        
        # names of the variables we are plotting
        plot_names = set(self.interpolation_param_names) - set(param_kw.keys())
        if plot_dim == 1:
            x_name = list(plot_names)[0]
        else:
            x_name, y_name = list(plot_names)

        # in both 1D and 2D cases, we always plot at least an x-variable
        x_unit = self.interp_param_spec[x_name]["values"][0].u
        # we need the magnitudes here so that units are unambiguous when we make
        # the linspace/geomspace for plotting
        x_mags = [v.m_as(x_unit) for v in self.interp_param_spec[x_name]["values"]]
        if self.interp_param_spec[x_name]["scales_log"]:
            x_plot = np.geomspace(np.min(x_mags), np.max(x_mags), n_steps)
            x_is_log = True
        else:
            x_plot = np.linspace(np.min(x_mags), np.max(x_mags), n_steps)
        # we put the unit back later
        if plot_dim == 1:
            # To make slices, we need to set any variables we do not plot over to the
            # value given in param_kw.
            slice_args = []
            # We need to make sure that we give the values in the correct order!
            for n in self.interpolation_param_names:
                if n == x_name:
                    slice_args.append(x_plot * x_unit)
                elif n in param_kw.keys():
                    # again, insure that the same unit is used that went into the 
                    # interpolation
                    param_unit = self.interp_param_spec[n]["values"][0].u
                    slice_args.append(
                        np.full(x_plot.shape, param_kw[n].m_as(param_unit)) * param_unit
                    )
                else:
                    raise ValueError("parameter neither specified nor plotted")
            coeff_slices, covar_slices = self.make_slices(*slice_args)
        else:
            # if we are in 2D, we need to do the same procedure again for the y-variable
            y_unit = self.interp_param_spec[y_name]["values"][0].u
            y_mags = [v.m_as(y_unit) for v in self.interp_param_spec[y_name]["values"]]
            if self.interp_param_spec[y_name]["scales_log"]:
                # we add one step to the size in y so that transposition is unambiguous
                y_plot = np.geomspace(np.min(y_mags), np.max(y_mags), n_steps + 1)
                y_is_log = True
            else:
                y_plot = np.linspace(np.min(y_mags), np.max(y_mags), n_steps + 1)
            
            x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
            slice_args = []
            for n in self.interpolation_param_names:
                if n == x_name:
                    slice_args.append(x_mesh * x_unit)
                elif n == y_name:
                    slice_args.append(y_mesh * y_unit)
                elif n in param_kw.keys():
                    # again, insure that the same unit is used that went into the 
                    # interpolation
                    param_unit = self.interp_param_spec[n]["values"][0].u
                    slice_args.append(
                        np.full(x_mesh.shape, param_kw[n].m_as(param_unit)) * param_unit
                    )
                else:
                    raise ValueError("parameter neither specified nor plotted")
            coeff_slices, covar_slices = self.make_slices(*slice_args)

        # first column plots fit coefficients
        for i in range(n_coeff):
            z_slice = coeff_slices[bin_idx][i]
            if plot_dim == 1:
                ax[i, 0].plot(x_plot, z_slice, label='interpolation')
                # Plotting the original input points only works if the interpolation
                # is in 1D. If we are plotting a 1D slice from a 2D interpolation, this
                # does not work.
                # The number of fit points is the first dimension in self._coeff_z
                if plot_dim == self.ndim:
                    slice_idx = (Ellipsis,) + bin_idx + (i,)
                    ax[i, 0].scatter(x_mags, self._coeff_z[slice_idx],
                                     color='k', marker='x', label='fit points')
                ax[i, 0].set_ylabel(hs_param_labels[i])
            else:
                pc = ax[i, 0].pcolormesh(x_mesh, y_mesh, z_slice)
                cbar = plt.colorbar(pc, ax=ax[i, 0])
                cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
                ax[i, 0].set_ylabel(y_name)
                ax[i, 0].set_xlabel(x_name)
            
            # later column plots the elements of the covariance matrix
            for j in range(0, n_coeff):
                z_slice = covar_slices[bin_idx][i, j]
                if plot_dim == 1:
                    ax[i, j+1].plot(x_plot, z_slice, label='interpolation')
                    # Same problem as above, only in 1D case can this be shown
                    # the number of points is the first dim in self._covar_z
                    if plot_dim == self.ndim:
                        coeff_idx = (Ellipsis,) + bin_idx + (i, j)
                        ax[i, j+1].scatter(x_mags, self._covar_z[coeff_idx],
                                           color='k', marker='x', label='fit points')
                else:
                    pc = ax[i, j+1].pcolormesh(x_mesh, y_mesh, z_slice)
                    cbar = plt.colorbar(pc, ax=ax[i, j+1])
                    cbar.ax.ticklabel_format(style='sci', scilimits=(0, 0))
                    ax[i, j+1].set_ylabel(y_name)
                    ax[i, j+1].set_xlabel(x_name)
        
        if plot_dim == 1:
            # in the 1D case, labels can be placed on the x and y axes
            for j in range(n_coeff+1):
                ax[-1, j].set_xlabel(x_name)
            ax[0, 0].set_title('coefficient')
            for j in range(n_coeff):
                ax[0, j+1].set_title(f'cov. {hs_param_labels[j]}')
        else:
            # in the 2D case, we need separate annotations
            rows = hs_param_labels
            cols = ["coefficient"] + [f"cov. {hl}" for hl in hs_param_labels]
            pad = 20
            for a, col in zip(ax[0], cols):
                a.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                           xycoords='axes fraction', textcoords='offset points',
                           size='x-large', ha='center', va='baseline')

            for a, row in zip(ax[:, 0], rows):
                a.annotate(row, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - pad, 0),
                           xycoords=a.yaxis.label, textcoords='offset points',
                           size='x-large', ha='right', va='center')
        for i, j in np.ndindex((n_coeff, n_coeff+1)):
            if x_is_log: ax[i, j].set_xscale("log")
            if y_is_log: ax[i, j].set_yscale("log")
            ax[i, j].grid()
            if plot_dim == 1:
                ax[i, j].legend()
            ax[i, j].relim()
            ax[i, j].autoscale_view()
            if not x_is_log:
                ax[i, j].ticklabel_format(style='sci', scilimits=(0, 0), axis="x")
            if not y_is_log:
                ax[i, j].ticklabel_format(style='sci', scilimits=(0, 0), axis="y")
        fig.tight_layout()
        if plot_dim == 2:
            fig.subplots_adjust(left=0.15, top=0.95)
        return fig


def pipeline_cfg_from_states(state_dict):
    """Recover a pipeline cfg containing PISA objects from a raw state.
    
    When a pipeline configuration is stored to JSON, the PISA objects turn into 
    their serialized states. This function looks through the dictionary returned by
    `from_json` and recovers the PISA objects such as `ParamSet` and `MultiDimBinning`.
        
    It should really become part of PISA file I/O functionality to read and write
    PISA objects inside dictionaries/lists into a JSON and be able to recover
    them...
    """
    
    # TODO: Make this a core functionality of PISA
    
    # This is just a mess... some objects have a `from_state` method, some take the
    # unpacked state dict as input, some take the state...
    
    pipeline_cfg = collections.OrderedDict()
    for stage_key in state_dict.keys():
        # need to check all of this manually... no automatic way to do it :(
        if stage_key == "pipeline":
            pipeline_cfg[stage_key] = copy.deepcopy(state_dict[stage_key])
            pipeline_cfg[stage_key]["output_key"] = tuple(
                pipeline_cfg[stage_key]["output_key"])
            binning_state = pipeline_cfg[stage_key]["output_binning"]
            pipeline_cfg[stage_key]["output_binning"] = MultiDimBinning(**binning_state)
            continue
        # undo what we did in `serialize_pipeline_cfg` by splitting the keys into tuples
        tuple_key = tuple(stage_key.split("__"))
        pipeline_cfg[tuple_key] = copy.deepcopy(state_dict[stage_key])
        for k in ["calc_mode", "apply_mode", "node_mode"]:
            if k in pipeline_cfg[tuple_key]:
                if isinstance(pipeline_cfg[tuple_key][k], collections.Mapping):
                    pipeline_cfg[tuple_key][k] = MultiDimBinning(
                        **pipeline_cfg[tuple_key][k])
        if "params" in pipeline_cfg[tuple_key].keys():
            pipeline_cfg[tuple_key]["params"] = ParamSet(
                pipeline_cfg[tuple_key]["params"])
    # if any stage takes any other arguments that we didn't think of here, they
    # won't work
    return pipeline_cfg

def serialize_pipeline_cfg(pipeline_cfg):
    """Turn a pipeline configuration into something we can store to JSON.
    
    It doesn't work by default because tuples are not allowed as keys when storing to
    JSON. All we do is to turn the tuples into strings divided by a double underscore.
    """
    serializable_state = collections.OrderedDict()
    serializable_state["pipeline"] = pipeline_cfg["pipeline"]
    for k in pipeline_cfg.keys():
        if k == "pipeline": continue
        flat_key = "__".join(k)
        serializable_state[flat_key] = pipeline_cfg[k]
    # this isn't _really_ a serializable state, the objects are still PISA objects...
    # bit it will convert correctly when thrown into `to_json`
    return serializable_state
    
    
def assemble_interpolated_fits(fit_directory, output_file):
    """After all of the fits on the cluster are done, assemble the results to one JSON.
    
    The JSON produced by this function is what `load_interpolated_hypersurfaces`
    expects.
    """
    assert os.path.isdir(fit_directory), "fit directory does not exist"
    metadata = from_json(os.path.join(fit_directory, "metadata.json"))
    
    combined_data = collections.OrderedDict()
    combined_data["interpolation_param_spec"] = metadata["interpolation_param_spec"]
    hs_fits = []
    grid_shape = tuple(metadata["grid_shape"])
    for job_idx, grid_idx in enumerate(np.ndindex(grid_shape)):
        gridpoint_json = os.path.join(fit_directory, f"gridpoint_{job_idx:06d}.json.bz2")
        logging.info(f"Reading {gridpoint_json}")
        gridpoint_data = from_json(gridpoint_json)
        assert job_idx == gridpoint_data["job_idx"]
        assert np.all(grid_idx == gridpoint_data["grid_idx"])
        # TODO: Offer to run incomplete fits locally
        assert gridpoint_data["fit_successful"], f"job no. {job_idx} not finished"
        hs_fits.append(collections.OrderedDict(
            param_values=gridpoint_data["param_values"],
            hs_fit=gridpoint_data["hs_fit"]
        ))
    combined_data["hs_fits"] = hs_fits
    to_json(combined_data, output_file)

def get_incomplete_job_idx(fit_directory):
    """Get job indices of fits that are not flagged as successful."""
    
    assert os.path.isdir(fit_directory), "fit directory does not exist"
    
    failed_idx = []
    this_idx = 0
    while True:
        # Is this a too dumb way to do it? I scream, for I do not know...
        try:
            gridpoint_json = os.path.join(fit_directory,
                                          f"gridpoint_{this_idx:06d}.json.bz2")
            logging.info(f"Reading {gridpoint_json}")
            gridpoint_data = from_json(gridpoint_json)
        except:
            break
        if not gridpoint_data["fit_successful"]:
            failed_idx.append(this_idx)
        this_idx += 1
    return failed_idx

def run_interpolated_fit(fit_directory, job_idx, skip_successful=False):
    """Run the hypersurface fit for a grid point.
    
    If `skip_successful` is true, do not run if the `fit_successful` flag is already
    True.
    """
    
    assert os.path.isdir(fit_directory), "fit directory does not exist"
    
    gridpoint_json = os.path.join(fit_directory, f"gridpoint_{job_idx:06d}.json.bz2")
    gridpoint_data = from_json(gridpoint_json)

    if skip_successful and gridpoint_data["fit_successful"]:
        logging.info(f"Fit at job index {job_idx} already successful, skipping...")
        return

    metadata = from_json(os.path.join(fit_directory, "metadata.json"))
    
    interpolation_param_spec = metadata["interpolation_param_spec"]
    
    # this is a pipeline configuration in the form of an OrderedDict
    nominal_dataset = metadata["nominal_dataset"]
    # Why can we still not load PISA objects from JSON that are inside a dict?! Grrr...
    nominal_dataset["pipeline_cfg"] = pipeline_cfg_from_states(
        nominal_dataset["pipeline_cfg"]
    )
    # this is a list of pipeline configurations
    sys_datasets = metadata["sys_datasets"]
    for sys_dataset in sys_datasets:
        sys_dataset["pipeline_cfg"] = pipeline_cfg_from_states(
            sys_dataset["pipeline_cfg"]
        )
    # this is a dict of param_name : value pairs
    param_values = gridpoint_data["param_values"]
    # we do a redundant check to make sure the parameter values at this grid point are
    # correct
    interpolation_param_names = metadata["interpolation_param_names"]
    grid_shape = tuple(metadata["grid_shape"])
    # the grid point index of this job
    grid_idx = list(np.ndindex(grid_shape))[job_idx]
    for i, n in enumerate(interpolation_param_names):
        ms = "Inconsistent parameter values at grid point!"
        assert interpolation_param_spec[n]["values"][grid_idx[i]] == param_values[n], ms
    
    # now we need to adjust the values of the parameter in all pipelines for this point
    logging.info(f"updating pipelines with parameter values: {param_values}")
    for dataset in [nominal_dataset] + sys_datasets:
        for stage_cfg in dataset["pipeline_cfg"].values():
            if "params" not in stage_cfg.keys(): continue
            for param in interpolation_param_names:
                if param in stage_cfg["params"].names:
                    stage_cfg["params"][param].value = param_values[param]
    
    # these are the parameters of the hypersurface, NOT the ones we interpolate them
    # over!
    hypersurface_params = []
    for param_state in metadata["hypersurface_params"]:
        hypersurface_params.append(HypersurfaceParam.from_state(param_state))
    
    # We create Pipeline objects, get their outputs and then forget about the Pipeline
    # object on purpose! The memory requirement to hold all systematic sets at the same
    # time is just too large, especially on the cluster. The way we do it below we
    # only need enough memory for one dataset at a time.
    nominal_dataset["mapset"] = Pipeline(nominal_dataset["pipeline_cfg"]).get_outputs()
    for sys_dataset in sys_datasets:
        sys_dataset["mapset"] = Pipeline(sys_dataset["pipeline_cfg"]).get_outputs()
    
    # Merge maps according to the combine regex, if one was provided
    combine_regex = metadata["combine_regex"]
    if combine_regex is not None:
        nominal_dataset["mapset"] = nominal_dataset["mapset"].combine_re(combine_regex)
        for sys_dataset in sys_datasets:
            sys_dataset["mapset"] = sys_dataset["mapset"].combine_re(combine_regex)

    hypersurface_fit_kw = metadata["hypersurface_fit_kw"]
    hypersurfaces = collections.OrderedDict()
    log = metadata["log"]  # flag determining whether hs fit is run in log-space or not
    for map_name in nominal_dataset["mapset"].names:
        nominal_map = nominal_dataset["mapset"][map_name]
        nominal_param_values = nominal_dataset["sys_params"]

        sys_maps = [sys_dataset["mapset"][map_name] for sys_dataset in sys_datasets]
        sys_param_values = [sys_dataset["sys_params"] for sys_dataset in sys_datasets]

        hypersurface = Hypersurface(
            # Yes, this MUST be a deepcopy! Otherwise weird memory overwrites happen
            # and all the numbers get jumbled across the hypersurfaces of different maps
            params=copy.deepcopy(hypersurface_params),
            initial_intercept=0. if log else 1.,  # Initial value for intercept
            log=log
        )

        hypersurface.fit(
            nominal_map=nominal_map,
            nominal_param_values=nominal_param_values,
            sys_maps=sys_maps,
            sys_param_values=sys_param_values,
            norm=True,
            # Is the space or loading time really a problem?
            # keep_maps=False,  # it would take a lot more space otherwise
            **hypersurface_fit_kw
        )

        logging.debug("\nFitted hypersurface report:\n%s" % hypersurface)
        hypersurfaces[map_name] = hypersurface

    gridpoint_data["hs_fit"] = hypersurfaces
    gridpoint_data["fit_successful"] = True
    
    to_json(gridpoint_data, gridpoint_json)


def prepare_interpolated_fit(
    nominal_dataset, sys_datasets, params, fit_directory, interpolation_param_spec,
    combine_regex=None, log=False, **hypersurface_fit_kw
):
    '''
    Writes steering files for fitting hypersurfaces on a grid of arbitrary parameters.
    The fits can then be run on a cluster with `run_interpolated_fit`.

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
        single parameterisation for all maps (this is almost always what you want).

    output_directory : str
        Directory in which the fits will be run. Steering files for the fits to be run
        will be stored here.

    combine_regex : list of str, or None
        List of string regex expressions that will be used for merging maps. Used to
        combine similar species. Must be something that can be passed to the
        `MapSet.combine_re` function (see that functions docs for more details). Choose
        `None` is do not want to perform this merging.
    
    interpolation_param_spec : collections.OrderedDict
        Specification of parameter grid that hypersurfaces should be interpolated over.
        The dict should have the following form::
            interpolation_param_spec = {
                'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                ...
                'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
            }
        The hypersurfaces will be fit on an N-dimensional rectilinear grid over
        parameters 1 to N. The flag `scales_log` indicates that the interpolation over
        that parameter should happen in log-space.

    hypersurface_fit_kw : kwargs
        kwargs will be passed on to the calls to `Hypersurface.fit`
    '''

    # Take (deep) copies of lists/dicts to avoid modifying the originals
    # Useful for cases where this function is called in a loop (e.g. leave-one-out tests)
    nominal_dataset = copy.deepcopy(nominal_dataset)
    sys_datasets = copy.deepcopy(sys_datasets)
    params = copy.deepcopy(params)

    # Check types
    assert isinstance(sys_datasets, collections.Sequence)
    assert isinstance(params, collections.Sequence)
    assert isinstance(fit_directory, str)
    # there must not be any ambiguity between fitting the hypersurfaces and 
    # interpolating them later
    msg = "interpolation params must be specified as a dict with ordered keys"
    assert isinstance(interpolation_param_spec, collections.OrderedDict), msg
    for k, v in interpolation_param_spec.items():
        assert set(v.keys()) == {"values", "scales_log"}
        assert isinstance(v["values"], collections.Sequence)
    
    # Check output format and path
    assert os.path.isdir(fit_directory), "fit directory does not exist"
    
    # Check formatting of datasets is as expected
    all_datasets = [nominal_dataset] + sys_datasets
    for dataset in all_datasets:
        assert isinstance(dataset, collections.Mapping)
        assert "pipeline_cfg" in dataset
        assert isinstance(dataset["pipeline_cfg"], (str, collections.Mapping))
        assert "sys_params" in dataset
        assert isinstance(dataset["sys_params"], collections.Mapping)
        
        dataset["pipeline_cfg"] = serialize_pipeline_cfg(dataset["pipeline_cfg"])

    # Check params
    assert len(params) >= 1
    for p in params:
        assert isinstance(p, HypersurfaceParam)

    # Report inputs
    msg = "Hypersurface fit details :\n"
    msg += f"  Num params            : {len(params)}\n"
    msg += f"  Num fit coefficients  : {sum([p.num_fit_coeffts for p in params])}\n"
    msg += f"  Num datasets          : 1 nominal + {len(sys_datasets)} systematics\n"
    msg += f"  Nominal values        : {nominal_dataset['sys_params']}\n"
    msg += "Hypersurface fits are prepared on the following grid:\n"
    msg += str(interpolation_param_spec)
    logging.info(msg)

    # because we require this to be an OrderedDict, there is no ambiguity in the
    # construction of the mesh here
    param_names = list(interpolation_param_spec.keys())
    grid_shape = tuple(len(v["values"]) for v in interpolation_param_spec.values())

    # We store all information needed to run a fit in metadata
    metadata = collections.OrderedDict(
        interpolation_param_spec=interpolation_param_spec,
        interpolation_param_names=param_names,  # convenience
        grid_shape=grid_shape,  # convenience
        nominal_dataset=nominal_dataset,
        sys_datasets=sys_datasets,
        hypersurface_params=params,
        combine_regex=combine_regex,
        log=log,
        hypersurface_fit_kw=hypersurface_fit_kw
    )
    
    to_json(metadata, os.path.join(fit_directory, "metadata.json"))
    
    # we write on JSON file for each grid point
    for job_idx, grid_idx in enumerate(np.ndindex(grid_shape)):
        # Although this is technically redundant, we store the parameter values
        # explicitly for each grid point.
        param_values = {}
        for i, n in enumerate(param_names):
            param_values[n] = interpolation_param_spec[n]["values"][grid_idx[i]]

        gridpoint_data = {
            "param_values": param_values,
            "hs_fit": None,
            "job_idx": job_idx,
            "grid_idx": grid_idx,
            "fit_successful": False
        }
        to_json(gridpoint_data, os.path.join(fit_directory,
            f"gridpoint_{job_idx:06d}.json.bz2"))

    logging.info(f"Grid fit preparation complete! Total number of jobs: {job_idx+1}")
    return job_idx+1  # zero-indexing

def load_interpolated_hypersurfaces(input_file):
    '''
    Load a set of interpolated hypersurfaces from a file.

    Analogously to "load_hypersurfaces", this function returns a
    collection with a HypersurfaceInterpolator object for each Map.

    Parameters
    ----------
    input_file : str
        A JSON input file as produced by fit_hypersurfaces if interpolation params
        were given. It has the form::
            {
                interpolation_param_spec = {
                    'param1': {"values": [val1_1, val1_2, ...], "scales_log": True/False}
                    'param2': {"values": [val2_1, val2_2, ...], "scales_log": True/False}
                    ...
                    'paramN': {"values": [valN_1, valN_2, ...], "scales_log": True/False}
                },
                'hs_fits': [
                    <list of dicts where keys are map names such as 'nue_cc' and values
                    are hypersurface states>
                ]
            }

    Returns
    -------
    collections.OrderedDict
        dictionary with a :obj:`HypersurfaceInterpolator` for each map
    '''
    assert isinstance(input_file, str)

    if input_file.endswith("json") or input_file.endswith("json.bz2"):
        logging.info(f"Loading interpolated hypersurfaces from file: {input_file}")
        input_data = from_json(input_file)
        assert set(['interpolation_param_spec', 'hs_fits']).issubset(
            set(input_data.keys())), 'missing keys'
        map_names = None
        # input_data['hs_fits'] is a list of dicts, each dict contains "param_values"
        # and "hs_fit"
        logging.info("Reading file complete, generating hypersurfaces...")
        for hs_fit_dict in input_data['hs_fits']:
            # this is still not the actual Hypersurface, but a dict with the (linked)
            # maps and the HS fit for the map...
            hs_state_maps = hs_fit_dict["hs_fit"]
            if map_names is None:
                map_names = list(hs_state_maps.keys())
            else:
                assert set(map_names) == set(hs_state_maps.keys()), "inconsistent maps"
            # When data is recovered from JSON, the object states are not automatically
            # converted to the corresponding objects, so we need to do it manually here.
            for map_name in map_names:
                hs_state_maps[map_name] = Hypersurface.from_state(hs_state_maps[map_name])

        logging.info(f"Read hypersurface maps: {map_names}")
        
        # Now we have a list of dicts where the map names are on the lower level.
        # We need to convert this into a dict of HypersurfaceInterpolator objects.
        output = collections.OrderedDict()
        for m in map_names:
            hs_fits = [{"param_values": fd["param_values"], "hs_fit": fd['hs_fit'][m]} for fd in input_data['hs_fits']]
            output[m] = HypersurfaceInterpolator(input_data['interpolation_param_spec'], hs_fits)
    else:
        raise Exception("unknown file format")
    return output
