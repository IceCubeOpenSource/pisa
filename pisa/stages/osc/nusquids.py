"""
nuSQuIDS oscillation probabilities with layered Earth model.

This implementation supports several filtering and interpolation techniques designed
to deal with fast oscillations that occur in the presence of eV-scale sterile neutrinos.

It is required that SQuIDS and nuSQuIDS are updated to include the layered Earth model
class `nuSQUIDSLayers` in nuSQuIDS as well as low-pass filtering and range averaging 
methods in SQuIDS.
"""

import math
import numpy as np
from scipy.interpolate import RectBivariateSpline

from pisa import FTYPE, TARGET
from pisa.core.stage import Stage
from pisa.utils.log import logging
from pisa.utils.profiler import profile, line_profile
from pisa.stages.osc.layers import Layers
from pisa.core.binning import MultiDimBinning
from pisa.utils.resources import find_resource
from pisa import ureg

import nuSQUIDSpy as nsq

__all__ = ["nusquids"]

__author__ = "T. Stuttard, T. Ehrhardt, A. Trettin"


class nusquids(Stage):
    """
    PISA Pi stage for weighting events due to the effect of neutrino oscillations, using
    nuSQuIDS as the oscillation probability calculator. One specialty here is that we
    have to specify an additional binning to determine where to place nodes for the
    exact calculation. The points where the actual probability is evaluated is
    determined by calc_mode as usual and may be much finer than node_mode or even
    event-wise since the interpolation step is fast.

    Parameters
    ----------

    Uses the standard parameters as required by a PISA pi stage
    (see `pisa/core/stage.py`)
    
    node_mode : MultiDimBinning
        Binning to determine where to place nodes at which the evaluation of interaction
        states occurs. The nodes are places at the _corners_ of the binning to avoid
        extrapolation.

    use_decoherence : bool
        set to true to include neutrino decoherence in the oscillation probability
        calculation

    num_decoherence_gamma : int
        number of decoherence gamma parameters to be considered in the decoherence model
        must be either 1 or 3

    use_nsi : bool
        set to true to include Non-Standard Interactions (NSI) in the oscillation
        probability calculation

    num_neutrinos : int
        Number of neutrino flavors to include. This stage supports 3 or 4 flavors, but
        nuSQuIDS allows up to 6 such that this stage could easily be expanded.
    
    earth_model : str
        Path to Earth model (PREM) file.
    
    detector_depth : quantity (distance)
    
    prop_height : quantity (distance) or str
        Height at which neutrinos are produced. If a quantity is given, the height is
        assumed to be the same for all neutrinos. An alternative is to pass
        `from_container`. In that case, the stage will search the input container for a
        key `prop_height` and take the height from there on a bin-wise or event-wise
        basis depending on `calc_specs`.
    
    prop_height_range : quantity (distance)
        Production height is averaged around the mean set by `prop_height` assuming
        a uniform distribution in [mean - range/2, mean + range/2]. The production
        heights are projected onto the direction of the neutrino, such that the
        averaging range is longer for shallow angles above the horizon.
    
    apply_lowpass_above_hor : bool
        Whether to apply the low-pass filter for evaluations above the horizon. If
        `True` (default), the low-pass filter is applied everywhere. If `False`, the 
        filter is applied only below the horizon. Because propagation distances are
        very short above the horizon, fast oscillations no longer average out and the
        filter might wash out important features.
    
    apply_height_avg_below_hor : bool
        Whether to apply the production height averaging below the horizon. If `True`
        (default), the production height averaging is applied everywhere if a 
        `prop_height_range` is set. If `False`, the height averaging is only applied
        above the horizon. Since the production height is only a very small fraction
        of the total propagation distance below the horizon, the height averaging is
        no longer important and a little bit of time can be saved by computing the
        slightly cheaper non-averaged probabilities.

    YeI : quantity (dimensionless)
        Inner electron fraction.
    
    YeO : quantity (dimensionless)
        Outer electron fraction.
        
    YeM : quantity (dimensionless)
        Mantle electron fraction.

    rel_err : float
        Relative error of the numerical integration

    abs_err : float
        Absolute error of the numerical integration
    
    prop_lowpass_cutoff : quantity (1/distance)
        Frequency cutoff for fast oscillations applied during numerical integration
        of the interaction state. The frequency is passed as oscillations per distance.
        A reasonable order of magnitude would allow ~100 oscillations over 12000 km.
    
    prop_lowpass_frac : quantity (dimensionless)
        This number determines how harsh the cut-off of the low-pass filter is applied
        during numerical integration is. A value of 0.1 would mean that the filter would
        begin to kick in when 90% of the cutoff frequency is reached and linearly
        decrease oscillation amplitudes until the cutoff is reached.
    
    eval_lowpass_cutoff : quantity (1/distance)
        Same as `prop_lowpass_cutoff`, but applied during evaluation of interpolated 
        states, not during integration.
    
    eval_lowpass_frac : quantity (1/distance)
        Same as `prop_lowpass_frac`, but applied during evaluation of interpolated 
        states, not during integration.
    
    suppress_interpolation_warning : bool
        Suppress warning about negative probabilities that can indicate insufficient
        nodes in a problematic region of energy and coszen. Set this option only at your
        own risk after you optimized nodes and are sure that remaining negative
        probabilities won't be a problem!
    
    exact_mode : bool
        With this turned on, the probabilities are evaluated using the exact calculation
        for constant densities in every layer without numerical integration. This method
        is much faster than the numerical integration for a node, but you lose the
        option to project out probabilities from interaction picture states. In this
        mode, nuSQuIDS behaves essentially like GLoBES with the same speed trade-off.
        You cannot apply filters in this mode either. Its only recommended use is for
        pseudo-data generation, where you may want an exact event-by-event calculation
        that is allowed to take several minutes.

    concurrent_threads : int
        Numer of parallel threads used for state integration.

    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params .. ::
            theta12 : quantity (angle)
            theta13 : quantity (angle)
            theta23 : quantity (angle)
            deltam21 : quantity (mass^2)
            deltam31 : quantity (mass^2)
            deltacp : quantity (angle)
        Additional expected params if `num_neutrinos == 4` .. ::
            theta14 : quantity (angle)
            theta24 : quantity (angle)
            deltam41 : quantity (mass^2)
            deltacp14 : quantity (angle)
            deltacp24 : quantity (angle)

    Additional ParamSet params expected when using the `use_decoherence` argument:
        n_energy : quantity (dimensionless)
        * If using `num_decoherence_gamma` == 1:
            gamma : quantity (energy)
        * If using `num_decoherence_gamma` == 3:
            gamma12 : quantity (energy)
            gamma13 : quantity (energy)
            gamma23 : quantity (energy)

    """
    def __init__(
        self,
        earth_model=None,
        detector_depth=None,
        prop_height=None,
        prop_height_range=None,
        YeI=None,
        YeO=None,
        YeM=None,
        rel_err=None,
        abs_err=None,
        prop_lowpass_cutoff=None,
        prop_lowpass_frac=None,
        eval_lowpass_cutoff=None,
        eval_lowpass_frac=None,
        apply_lowpass_above_hor=True,
        apply_height_avg_below_hor=True,
        suppress_interpolation_warning=False,
        node_mode=None,
        use_decoherence=False,
        num_decoherence_gamma=1,
        use_nsi=False,
        num_neutrinos=3,
        exact_mode=False,
        concurrent_threads=1,
        **std_kwargs,
    ):

        if use_nsi:
            raise NotImplementedError("NSI not implemented")
        if use_decoherence:
            raise NotImplementedError("Decoherence not implemented")
        if type(prop_height) is not ureg.Quantity:
            raise NotImplementedError("Getting propagation heights from containers is "
                "not yet implemented")
        self.num_neutrinos = int(num_neutrinos)
        assert self.num_neutrinos < 5, "currently only supports up to 4 flavor oscillations"
        self.use_nsi = use_nsi
        self.use_decoherence = use_decoherence
        self.num_decoherence_gamma = num_decoherence_gamma
        self.node_mode = node_mode

        self.earth_model = earth_model
        self.YeI = YeI.m_as("dimensionless")
        self.YeO = YeO.m_as("dimensionless")
        self.YeM = YeM.m_as("dimensionless")
        self.detector_depth = detector_depth.m_as("km")
        self.prop_height = prop_height.m_as("km")
        self.avg_height = False
        self.concurrent_threads = int(concurrent_threads)
        self.prop_height_range = None
        self.apply_height_avg_below_hor = apply_height_avg_below_hor
        if prop_height_range is not None:  # this is optional
            self.prop_height_range = prop_height_range.m_as("km")
            self.avg_height = True            
        
        self.layers = None
        
        self.rel_err = rel_err.m_as("dimensionless") if rel_err is not None else 1.0e-10
        self.abs_err = abs_err.m_as("dimensionless") if abs_err is not None else 1.0e-10
        self.prop_lowpass_cutoff = (prop_lowpass_cutoff.m_as("1/km")
                                    if prop_lowpass_cutoff is not None else 0.)
        self.prop_lowpass_frac = (prop_lowpass_frac.m_as("dimensionless")
                                  if prop_lowpass_frac is not None else 0.)
        self.eval_lowpass_cutoff = (eval_lowpass_cutoff.m_as("1/km")
                                    if eval_lowpass_cutoff is not None else 0.)
        self.eval_lowpass_frac = (eval_lowpass_frac.m_as("dimensionless")
                                  if eval_lowpass_frac is not None else 0.)

        if self.prop_lowpass_frac > 1. or self.eval_lowpass_frac > 1.:
            raise ValueError("lowpass filter fraction cannot be greater than one")
        
        if self.prop_lowpass_frac < 0. or self.eval_lowpass_frac < 0.:
            raise ValueError("lowpass filter fraction cannot be smaller than zero")
        
        self.apply_lowpass_above_hor = apply_lowpass_above_hor
        
        self.nus_layer = None
        self.nus_layerbar = None
        
        # Define standard params
        expected_params = [
            "theta12",
            "theta13",
            "theta23",
            "deltam21",
            "deltam31",
            "deltacp",
        ]

        # Add decoherence parameters
        assert self.num_decoherence_gamma in [1, 3], ("Must choose either 1 or 3 "
            "decoherence gamma parameters"
        )
        if self.use_decoherence:
            if self.num_decoherence_gamma == 1:
                expected_params.extend(["gamma"])
            elif self.num_decoherence_gamma == 3:
                expected_params.extend(["gamma21",
                                        "gamma31",
                                        "gamma32"])
            expected_params.extend(["n_energy"])
        
        # We may want to reparametrize this with the difference between deltacp14 and
        # deltacp24, as the absolute value seems to play a small role (see
        # https://arxiv.org/pdf/2010.06321.pdf)
        if self.num_neutrinos == 4:
            expected_params.extend([
                "theta14",
                "theta24",
                "theta34",
                "deltam41",
                "deltacp14",
                "deltacp24",
            ])
        
        # init base class
        super().__init__(
            expected_params=expected_params,
            **std_kwargs,
        )
        
        # This is special: We have an additional "binning" to account for. It is in
        # principle possible to work in event mode even for the nodes, which would mean
        # that the full oscillation problem is solved for all events individually.
        # Together with the constant oscillation mode, this can be used to calculate
        # probabilities in exact mode in a time that is reasonable at least for
        # generating pseudodata.

        assert not (self.use_nsi and self.use_decoherence), ("NSI and decoherence not "
            "suported together, must use one or the other")

        self.exact_mode = exact_mode
            
        if exact_mode:
            # No interpolation is happening in exact mode so any passed node_mode
            # will be ignored. Probabilities are calculated at calc_specs.
            if self.node_mode is not None:
                logging.warn("nuSQuIDS is configured in exact mode, the passed "
                    f"`node_mode`\n({self.node_mode})\n will be ignored!")
            if self.prop_lowpass_cutoff > 0 or self.eval_lowpass_cutoff > 0:
                logging.warn("nuSQuIDS is configured in exact mode, low-pass filters "
                    "will be ignored")
        else:
            if isinstance(self.calc_mode, MultiDimBinning):
                assert isinstance(self.node_mode, MultiDimBinning), ("cannot use "
                    "event-wise nodes with binned calculation")

        self.e_node_mode = None
        self.e_mesh = None
        self.coszen_node_mode = None
        self.cosz_mesh = None
        
        # We don't want to spam the user with repeated warnings about the same issue.
        self.interpolation_warning_issued = suppress_interpolation_warning
    
    def set_osc_parameters(self, nus_layer):
        # nuSQuIDS uses zero-index for mixing angles
        nus_layer.Set_MixingAngle(0, 1, self.params.theta12.value.m_as("rad"))
        nus_layer.Set_MixingAngle(0, 2, self.params.theta13.value.m_as("rad"))
        nus_layer.Set_MixingAngle(1, 2, self.params.theta23.value.m_as("rad"))
        
        # mass differences in nuSQuIDS are always w.r.t. m_1
        nus_layer.Set_SquareMassDifference(1, self.params.deltam21.value.m_as("eV**2"))
        nus_layer.Set_SquareMassDifference(2, self.params.deltam31.value.m_as("eV**2"))
    
        nus_layer.Set_CPPhase(0, 2, self.params.deltacp.value.m_as("rad"))
        
        if self.num_neutrinos == 3: return
        
        nus_layer.Set_MixingAngle(0, 3, self.params.theta14.value.m_as("rad"))
        nus_layer.Set_MixingAngle(1, 3, self.params.theta24.value.m_as("rad"))
        nus_layer.Set_MixingAngle(2, 3, self.params.theta34.value.m_as("rad"))
        nus_layer.Set_SquareMassDifference(3, self.params.deltam41.value.m_as("eV**2"))
        nus_layer.Set_CPPhase(0, 3, self.params.deltacp14.value.m_as("rad"))
        nus_layer.Set_CPPhase(1, 3, self.params.deltacp24.value.m_as("rad"))
        # TODO: Implement NSI, decoherence

    def apply_prop_settings(self, nus_layer):
        nsq_units = nsq.Const()
        nus_layer.Set_rel_error(self.rel_err)
        nus_layer.Set_abs_error(self.abs_err)
        nus_layer.Set_EvolLowPassCutoff(self.prop_lowpass_cutoff / nsq_units.km)
        # The ramp of the low-pass filter starts to drop at (cutoff - scale)
        scale = self.prop_lowpass_frac * self.prop_lowpass_cutoff / nsq_units.km
        nus_layer.Set_EvolLowPassScale(scale)
        nus_layer.Set_AllowConstantDensityOscillationOnlyEvolution(self.exact_mode)
        nus_layer.Set_EvalThreads(self.concurrent_threads)

    def setup_function(self):

        earth_model = find_resource(self.earth_model)
        prop_height = self.prop_height
        detector_depth = self.detector_depth
        self.layers = Layers(earth_model, detector_depth, prop_height)
        # We must treat densities and electron fractions correctly here, so we set them
        # to 1 in the Layers module to get unweighted densities.
        self.layers.setElecFrac(1, 1, 1)
        
        nsq_units = nsq.Const()  # natural units for nusquids
        # Because we don't want to extrapolate, we check that all points at which we
        # want to evaluate probabilities are fully contained within the node specs. This
        # is of course not necessary in events mode.
        if isinstance(self.node_mode, MultiDimBinning) and not self.exact_mode:
            logging.debug("setting up nuSQuIDS nodes in binned mode")
            # we can prepare the calculator like this only in binned mode, see 
            # compute_function for node_mode == "events"
            self.data.representation = self.calc_mode
            for container in self.data:
                for var in ["true_coszen", "true_energy"]:
                    upper_bound = np.max(self.node_mode[var].bin_edges)
                    lower_bound = np.min(self.node_mode[var].bin_edges)
                    err_msg = (
                        "The outer edges of the node_mode must encompass "
                        "the entire range of calc_specs to avoid extrapolation"
                    )
                    if np.any(container[var] > upper_bound):
                        maxval = np.max(container[var])
                        raise ValueError(err_msg + f"\nmax input: {maxval}, upper "
                            f"bound: {upper_bound}")
                    if np.any(container[var] < lower_bound):
                        minval = np.max(container[var])
                        raise ValueError(err_msg + f"\nmin input: {minval}, lower "
                            f"bound: {lower_bound}")

            # Layers in nuSQuIDS are special: We need all the individual distances and
            # densities for the nodes to solve the interaction picture states, but on
            # the final calculation grid (or events) we only need the *total* traversed
            # distance. Because we are placing nodes at the bin edges rather than the
            # bin middle, this doesn't really fit with how containers store data, so we
            # are making arrays as variables that never go into the container.
            
            # These are stored because we need them later during interpolation
            self.coszen_node_mode = self.node_mode["true_coszen"].bin_edges.m_as("dimensionless")
            self.e_node_mode = self.node_mode["true_energy"].bin_edges.m_as("GeV")
            logging.debug(
                f"Setting up nodes at\n"
                f"cos_zen = \n{self.coszen_node_mode}\n"
                f"energy = \n{self.e_node_mode}\n"
            )
            # things are getting a bit meshy from here...
            self.e_mesh, self.cosz_mesh = np.meshgrid(self.e_node_mode, self.coszen_node_mode)
            e_nodes = self.e_mesh.ravel()
            coszen_nodes = self.cosz_mesh.ravel()
        
            # The lines below should not be necessary because we will always get at
            # least two numbers from the bin edges. However, if either energy or coszen
            # somehow was just a scalar, we would need to broadcast it out to the same
            # size. Keeping the code in here in case you want to use the stage in 1D.
            # convert lists to ndarrays and scalars to ndarrays with length 1
            e_nodes = np.atleast_1d(e_nodes)
            coszen_nodes = np.atleast_1d(coszen_nodes)
            # broadcast against each other and make a copy
            # (see https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html)
            e_nodes, coszen_nodes = [np.array(a) for a in np.broadcast_arrays(e_nodes, coszen_nodes)]
        
            assert len(e_nodes) == len(coszen_nodes)
            assert coszen_nodes.ndim == 1
            assert e_nodes.ndim == 1
            
            self.layers.calcLayers(coszen_nodes)
            distances = np.reshape(self.layers.distance,
                                   (len(e_nodes), self.layers.max_layers))
            densities = np.reshape(self.layers.density,
                                   (len(e_nodes), self.layers.max_layers))
            # HACK: We need the correct electron densities for each layer. We can 
            # determine whether we are in the core or mantle based on the density.
            # Needless to say it isn't optimal to have these numbers hard-coded.
            ye = np.zeros_like(densities)
            ye[densities < 10] = self.YeM
            ye[(densities >= 10) & (densities < 13)] = self.YeO
            ye[densities >= 13] = self.YeI
            self.nus_layer = nsq.nuSQUIDSLayers(
                distances * nsq_units.km,
                densities,
                ye,
                e_nodes * nsq_units.GeV,
                self.num_neutrinos,
                nsq.NeutrinoType.both,
            )
            self.apply_prop_settings(self.nus_layer)
        
        # Now that we have our nusquids calculator set up on the node grid, we make 
        # container output space for the probability output which may be on a finer grid
        # than the nodes or even working in events mode.
        self.data.representation = self.calc_mode

        # --- calculate the layers ---
        if isinstance(self.calc_mode, MultiDimBinning):
            # as layers don't care about flavour
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc",
                                             "nuebar_cc", "numubar_cc", "nutaubar_cc",
                                             "nuebar_nc", "numubar_nc", "nutaubar_nc"])

        # calculate the distance difference between minimum and maximum production
        # height, if applicable
        if self.avg_height:
            layers_min = Layers(earth_model, detector_depth,
                                self.prop_height - self.prop_height_range/2.)
            layers_min.setElecFrac(1, 1, 1)
            layers_max = Layers(earth_model, detector_depth,
                                self.prop_height + self.prop_height_range/2.)
            layers_max.setElecFrac(1, 1, 1)
        for container in self.data:
            self.layers.calcLayers(container["true_coszen"])
            distances = self.layers.distance.reshape((container.size, -1))
            tot_distances = np.sum(distances, axis=1)
            if self.avg_height:
                layers_min.calcLayers(container["true_coszen"])
                dists_min = layers_min.distance.reshape((container.size, -1))
                min_tot_dists = np.sum(dists_min, axis=1)
                
                layers_max.calcLayers(container["true_coszen"])
                dists_max = layers_max.distance.reshape((container.size, -1))
                max_tot_dists = np.sum(dists_max, axis=1)
                # nuSQuIDS assumes the original distance is the longest distance and 
                # the averaging range is the difference between the minimum and maximum
                # distance.
                avg_ranges = max_tot_dists - min_tot_dists
                tot_distances = max_tot_dists
                assert np.all(avg_ranges > 0)
            # If the low-pass cutoff is zero, nusquids will not evaluate the filter.
            container["lowpass_cutoff"] = (self.eval_lowpass_cutoff
                                           * np.ones(container.size))
            if not self.apply_lowpass_above_hor:
                container["lowpass_cutoff"] = np.where(
                    container["true_coszen"] >= 0,
                    0,
                    container["lowpass_cutoff"]
                )
            if isinstance(self.node_mode, MultiDimBinning) and not self.exact_mode:
                # To project out probabilities we only need the *total* distance
                container["tot_distances"] = tot_distances
                if self.avg_height:
                    container["avg_ranges"] = avg_ranges
                else:
                    container["avg_ranges"] = np.zeros(container.size, dtype=FTYPE)
                if not self.apply_height_avg_below_hor:
                    container["avg_ranges"] = np.where(
                        container["true_coszen"] >= 0,
                        container["avg_ranges"],
                        0.
                    )
            elif self.node_mode == "events" or self.exact_mode:
                # in any other mode (events or exact) we store all densities and 
                # distances in the container in calc_specs
                densities = self.layers.density.reshape((container.size, -1))
                container["densities"] = densities
                container["distances"] = distances
        
        self.data.unlink_containers()
        
        if isinstance(self.calc_mode, MultiDimBinning):
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])

        # setup more empty arrays
        for container in self.data:
            container["prob_e"] = np.empty((container.size), dtype=FTYPE)
            container["prob_mu"] = np.empty((container.size), dtype=FTYPE)
        self.data.unlink_containers()
        
        if self.exact_mode: return
        
        # --- containers for interpolated states ---
        # This is not needed in exact mode
        if isinstance(self.calc_mode, MultiDimBinning):
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc"])
            self.data.link_containers("nubar", ["nuebar_cc", "numubar_cc", "nutaubar_cc",
                                                "nuebar_nc", "numubar_nc", "nutaubar_nc"])
        for container in self.data:
            container["interp_states_e"] = np.empty(
                (container.size, self.num_neutrinos**2),
                dtype=FTYPE,
            )
            container["interp_states_mu"] = np.empty(
                (container.size, self.num_neutrinos**2),
                dtype=FTYPE,
            )
        self.data.unlink_containers()
        self.interpolation_warning_issued = False
    
    # @line_profile
    def calc_node_probs(self, nus_layer, flav_in, flav_out, n_nodes):
        """
        Evaluate oscillation probabilities at nodes. This does not require any
        interpolation.
        """
        ini_state = np.array([0] * self.num_neutrinos)
        ini_state[flav_in] = 1
        nus_layer.Set_initial_state(ini_state, nsq.Basis.flavor)
        nus_layer.EvolveState()
        prob_nodes = nus_layer.EvalFlavorAtNodes(flav_out)
        return prob_nodes

    def calc_interpolated_states(self, evolved_states, e_out, cosz_out):
        """
        Calculate interpolated states at the energies and zenith angles requested.
        """
        nsq_units = nsq.Const()
        interp_states = np.zeros((e_out.size, evolved_states.shape[1]))
        
        assert np.all(e_out <= np.max(self.e_node_mode * nsq_units.GeV))
        assert np.all(e_out >= np.min(self.e_node_mode * nsq_units.GeV))
        assert np.all(cosz_out <= np.max(self.coszen_node_mode))
        assert np.all(cosz_out >= np.min(self.coszen_node_mode))

        for i in range(evolved_states.shape[1]):
            z = evolved_states[:, i].reshape(self.e_mesh.shape).T
            assert np.all(np.isfinite(z))
            # RectBivariateSpline takes in the 1D node position and assumes that they 
            # are on a mesh.
            f = RectBivariateSpline(
                np.log10(self.e_node_mode * nsq_units.GeV),
                self.coszen_node_mode,
                z,
                kx=2,
                ky=2,
            )
            interp_states[..., i] = f(np.log10(e_out), cosz_out, grid=False)
        return interp_states

    def calc_probs_interp(self, flav_out, nubar, interp_states, out_distances,
                          e_out, avg_ranges=0, lowpass_cutoff=0):
        """
        Project out probabilities from interpolated interaction picture states.
        """
        nsq_units = nsq.Const()

        prob_interp = np.zeros(e_out.size)
        scale = self.eval_lowpass_frac * lowpass_cutoff
        prob_interp = self.nus_layer.EvalWithState(
            flav_out,
            out_distances,
            e_out,
            interp_states,
            avg_cutoff=0.,
            avg_scale=0.,
            rho=int(nubar),
            # Range averaging is only computed in the places where t_range > 0, so
            # we don't need to introduce switches for averaged and non-averaged regions.
            t_range=avg_ranges,
            lowpass_cutoff=lowpass_cutoff,
            lowpass_scale=scale,
        )
        return prob_interp
    
    def compute_function_no_interpolation(self):
        """
        Version of the compute function that does not use any interpolation between
        nodes.
        """
        nsq_units = nsq.Const()
        # it is possible to work in binned calc mode while being in exact mode
        if isinstance(self.calc_mode, MultiDimBinning):
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])
        for container in self.data:
            nubar = container["nubar"] < 0
            flav = container["flav"]
            # HACK: We need the correct electron densities for each layer. We can 
            # determine whether we are in the core or mantle based on the density.
            ye = np.zeros_like(container["densities"])
            ye[container["densities"] < 10] = self.YeM
            ye[(container["densities"] >= 10) & (container["densities"] < 13)] = self.YeO
            ye[container["densities"] >= 13] = self.YeI
            nus_layer = nsq.nuSQUIDSLayers(
                container["distances"] * nsq_units.km,
                container["densities"],
                ye,
                container["true_energy"] * nsq_units.GeV,
                self.num_neutrinos,
                nsq.NeutrinoType.antineutrino if nubar else nsq.NeutrinoType.neutrino,
            )
            self.apply_prop_settings(nus_layer)
            self.set_osc_parameters(nus_layer)
            container["prob_e"] = self.calc_node_probs(nus_layer, 0, flav,
                                                       container.size)
            container["prob_mu"] = self.calc_node_probs(nus_layer, 1, flav,
                                                        container.size)
        
            container.mark_changed("prob_e")
            container.mark_changed("prob_mu")
        self.data.unlink_containers()
    
    @line_profile
    def compute_function_interpolated(self):
        """
        Version of the compute function that does use interpolation between nodes.
        """
        nsq_units = nsq.Const()
        # We need to make two evolutions, one for numu and the other for nue.
        # These produce neutrino and antineutrino states at the same time thanks to
        # the "both" neutrino mode of nuSQuIDS.
        self.apply_prop_settings(self.nus_layer)
        self.set_osc_parameters(self.nus_layer)

        ini_state_nue = np.array([1, 0] + [0] * (self.num_neutrinos - 2))
        ini_state_numu = np.array([0, 1] + [0] * (self.num_neutrinos - 2))
        
        self.nus_layer.Set_initial_state(ini_state_nue, nsq.Basis.flavor)
        self.nus_layer.EvolveState()
        evolved_states_nue = self.nus_layer.GetStates(0)
        evolved_states_nuebar = self.nus_layer.GetStates(1)
        
        self.nus_layer.Set_initial_state(ini_state_numu, nsq.Basis.flavor)
        self.nus_layer.EvolveState()
        evolved_states_numu = self.nus_layer.GetStates(0)
        evolved_states_numubar = self.nus_layer.GetStates(1)

        
        # Now comes the step where we interpolate the interaction picture states
        # and project out oscillation probabilities. This can be done in either events
        # or binned mode.
        if isinstance(self.calc_mode, MultiDimBinning):
            self.data.link_containers("nu", ["nue_cc", "numu_cc", "nutau_cc",
                                             "nue_nc", "numu_nc", "nutau_nc"])
            self.data.link_containers("nubar", ["nuebar_cc", "numubar_cc", "nutaubar_cc",
                                                "nuebar_nc", "numubar_nc", "nutaubar_nc"])
        for container in self.data:
            nubar = container["nubar"] < 0
            container["interp_states_e"] = self.calc_interpolated_states(
                evolved_states_nuebar if nubar else evolved_states_nue,
                container["true_energy"] * nsq_units.GeV,
                container["true_coszen"]
            )
            container["interp_states_mu"] = self.calc_interpolated_states(
                evolved_states_numubar if nubar else evolved_states_numu,
                container["true_energy"] * nsq_units.GeV,
                container["true_coszen"]
            )
        self.data.unlink_containers()
        
        if isinstance(self.calc_mode, MultiDimBinning):
            self.data.link_containers("nue", ["nue_cc", "nue_nc"])
            self.data.link_containers("numu", ["numu_cc", "numu_nc"])
            self.data.link_containers("nutau", ["nutau_cc", "nutau_nc"])
            self.data.link_containers("nuebar", ["nuebar_cc", "nuebar_nc"])
            self.data.link_containers("numubar", ["numubar_cc", "numubar_nc"])
            self.data.link_containers("nutaubar", ["nutaubar_cc", "nutaubar_nc"])
        
        for container in self.data:
            nubar = container["nubar"] < 0
            flav_out = container["flav"]
            for flav_in in ["e", "mu"]:
                container["prob_"+flav_in] = self.calc_probs_interp(
                    flav_out,
                    nubar,
                    container["interp_states_"+flav_in],
                    container["tot_distances"] * nsq_units.km,
                    container["true_energy"] * nsq_units.GeV,
                    container["avg_ranges"] * nsq_units.km,
                    container["lowpass_cutoff"] / nsq_units.km
                )
                # It is possible to get slightly negative probabilities from imperfect
                # state interpolation between nodes.
                # It's impractical to avoid any probability dipping below zero in every
                # conceivable situation because that would require very dense node
                # spacing. We get around this by flooring the probability at zero.
                # However, dipping below zero by more than 1% may indicate that nodes
                # aren't spaced tightly enough to achieve an acceptable accuracy, so we
                # issue a warning.
                if (np.any(container["prob_"+flav_in] < -0.01)
                    and not self.interpolation_warning_issued):
                    mask = container["prob_"+flav_in] < -0.01
                    en_med = np.median(container["true_energy"][mask])
                    cz_med = np.median(container["true_coszen"][mask])
                    logging.warn(
                        f"Some probabilities in nu_{flav_in} -> {container.name} dip "
                        "below zero by more than 1%! This may indicate too few nodes "
                        f"in the problematic region. Median energy: {en_med}, median "
                        f"coszen: {cz_med}. This warning is only issued once."
                    )
                    self.interpolation_warning_issued = True
                container["prob_"+flav_in][container["prob_"+flav_in] < 0] = 0.
            container.mark_changed("prob_e")
            container.mark_changed("prob_mu")
        self.data.unlink_containers()
    

    def compute_function(self):
        if self.node_mode == "events" or self.exact_mode:
            self.compute_function_no_interpolation()
        else:
            self.compute_function_interpolated()

    @profile
    def apply_function(self):
        for container in self.data:
            scales = container['nu_flux'][:, 0] * container['prob_e'] + container['nu_flux'][:, 1] * container['prob_mu']
            container['weights'] = container["weights"] * scales
