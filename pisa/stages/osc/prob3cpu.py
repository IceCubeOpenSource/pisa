# authors: T.Arlen, J.Lanfranchi, P.Eller
# date:   March 20, 2016


from __future__ import division

import numpy as np

from pisa.core.binning import MultiDimBinning
from pisa.core.param import ParamSet, ParamSelector
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet
from pisa.utils.resources import find_resource
from pisa.stages.osc.prob3.BargerPropagator import BargerPropagator
from pisa.utils.comparisons import normQuant
from pisa.utils.profiler import profile
from pisa.utils.log import logging


__all__ = ['prob3cpu']


SIGFIGS = 12
"""Significant figures for determining if numbers and quantities normalised
(using pisa.utils.comparisons.normQuant) are equal. Make sure this is less than
the numerical precision that calculations are being performed in to have the
desired effect that "essentially equal" things evaluate to be equal."""


class prob3cpu(Stage):
    """Neutrino oscillations calculation via Prob3.

    Parameters
    ----------
    params : ParamSet
        All of the following param names (and no more) must be in `params`.
        Earth parameters:
            * earth_model : str (resource location with earth model file)
            * YeI : float (electron fraction, inner core)
            * YeM : float (electron fraction, mantle)
            * YeO : float (electron fraction, outer core)
        Detector parameters:
            * detector_depth : float >= 0
            * prop_height
        Oscillation parameters:
            * deltacp
            * deltam21
            * deltam31
            * theta12
            * theta13
            * theta23
        Nutau (and nutaubar) normalization:
            * nutau_norm


    input_binning : MultiDimBinning
    output_binning : MultiDimBinning
    transforms_cache_depth : int >= 0
    outputs_cache_depth : int >= 0
    debug_mode : bool

    Input Names
    -----------
    The `inputs` container must include objects with `name` attributes:
      * 'nue'
      * 'numu'
      * 'nuebar'
      * 'numubar'

    Output Names
    ------------
    The `outputs` container generated by this service will be objects with the
    following `name` attribute:
      * 'nue'
      * 'numu'
      * 'nutau'
      * 'nuebar'
      * 'numubar'
      * 'nutaubar'

    """
    def __init__(self, params, input_binning, output_binning,
                 memcache_deepcopy, error_method, transforms_cache_depth,
                 outputs_cache_depth, debug_mode=None):

        expected_params = (
            'earth_model', 'YeI', 'YeM', 'YeO',
            'detector_depth', 'prop_height',
            'deltacp', 'deltam21', 'deltam31',
            'theta12', 'theta13', 'theta23',
            'nutau_norm'
        )

        # Define the names of objects that are required by this stage (objects
        # will have the attribute `name`: i.e., obj.name)
        input_names = (
            'nue', 'numu', 'nuebar', 'numubar'
        )

        # Define the names of objects that get produced by this stage
        output_names = (
            'nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar'
        )

        # Invoke the init method from the parent class (Stage), which does a
        # lot of work (caching, providing public interfaces, etc.)
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            outputs_cache_depth=outputs_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            transforms_cache_depth=transforms_cache_depth,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # If no binning provided then we want to use this to calculate
        # probabilities for events instead of transforms for maps.
        # Set up this self.calc_transforms to use as an assert on the
        # appropriate functions.
        self.calc_transforms = (input_binning is not None
                                and output_binning is not None)
        if self.calc_transforms:
            self.compute_binning_constants()

    def compute_binning_constants(self):
        # Only works if energy and coszen are in input_binning
        if 'true_energy' not in self.input_binning \
                or 'true_coszen' not in self.input_binning:
            raise ValueError(
                'Input binning must contain both "true_energy" and'
                ' "true_coszen" dimensions.'
            )

        # Not handling rebinning (or oversampling)
        assert self.input_binning == self.output_binning

        # Get the energy/coszen (ONLY) weighted centers here, since these
        # are actually used in the oscillations computation. All other
        # dimensions are ignored. Since these won't change so long as the
        # binning doesn't change, attache these to self.
        self.ecz_binning = MultiDimBinning([
            self.input_binning.true_energy.to('GeV'),
            self.input_binning.true_coszen.to('dimensionless')
        ])
        e_centers, cz_centers = self.ecz_binning.weighted_centers
        self.e_centers = e_centers.magnitude
        self.cz_centers = cz_centers.magnitude

        self.num_czbins = self.input_binning.true_coszen.num_bins
        self.num_ebins = self.input_binning.true_energy.num_bins

        self.e_dim_num = self.input_binning.names.index('true_energy')
        self.cz_dim_num = self.input_binning.names.index('true_coszen')

        self.extra_dim_nums = range(self.input_binning.num_dims)
        [self.extra_dim_nums.remove(d) for d in (self.e_dim_num,
                                                 self.cz_dim_num)]

    def create_transforms_datastructs(self):
        xform_shape = [3, 2] + list(self.input_binning.shape)
        nu_xform = np.empty(xform_shape)
        antinu_xform = np.empty(xform_shape)
        return nu_xform, antinu_xform

    def setup_barger_propagator(self):
        # If already instantiated with same parameters, don't instantiate again
        if (hasattr(self, 'barger_propagator')
                and hasattr(self, '_barger_earth_model')
                and hasattr(self, '_barger_detector_depth')
                and (normQuant(self._barger_detector_depth, sigfigs=SIGFIGS)
                     == normQuant(self.params.detector_depth.m_as('km'),
                                  sigfigs=SIGFIGS))
                and self.params.earth_model.value == self._barger_earth_model):
            return

        # Some private variables to keep track of the state of the barger
        # propagator that has been instantiated, so if it is requested to be
        # instantiated again with equivalent parameters, this step can be
        # skipped (see checks above).
        self._barger_detector_depth = self.params.detector_depth.m_as('km')
        self._barger_earth_model = self.params.earth_model.value

        # TODO: can we pass kwargs to swig-ed C++ code?
        if self._barger_earth_model is not None:
            self.barger_propagator = BargerPropagator(
                find_resource(self._barger_earth_model),
                self._barger_detector_depth
            )
        else:
            # Initialise with the 12 layer model that should be there. All
            # calculations will use the GetVacuumProb so what we define here
            # doesn't matter.
            self.barger_propagator = BargerPropagator(
                find_resource('osc/PREM_12layer.dat'),
                self._barger_detector_depth
            )
        self.barger_propagator.UseMassEigenstates(False)

    def _derive_nominal_transforms_hash(self):
        """No nominal transforms implemented for this service."""
        return

    @profile
    def _compute_transforms(self):
        """Compute oscillation transforms using Prob3 CPU code."""
        self.setup_barger_propagator()

        # Read parameters in, convert to the units used internally for
        # computation, and then strip the units off. Note that this also
        # enforces compatible units (but does not sanity-check the numbers).
        theta12 = self.params.theta12.m_as('rad')
        theta13 = self.params.theta13.m_as('rad')
        theta23 = self.params.theta23.m_as('rad')
        deltam21 = self.params.deltam21.m_as('eV**2')
        deltam31 = self.params.deltam31.m_as('eV**2')
        deltacp = self.params.deltacp.m_as('rad')
        prop_height = self.params.prop_height.m_as('km')
        nutau_norm = self.params.nutau_norm.m_as('dimensionless')

        # The YeX will not be in params if the Earth model is None
        if self._barger_earth_model is not None:
            YeI = self.params.YeI.m_as('dimensionless')
            YeO = self.params.YeO.m_as('dimensionless')
            YeM = self.params.YeM.m_as('dimensionless')

            total_bins = int(len(self.e_centers)*len(self.cz_centers))
            # We use 18 since we have 3*3 possible oscillations for each of
            # neutrinos and antineutrinos.
            prob_list = np.empty(total_bins*18, dtype='double')
            
            # The 1.0 was energyscale from earlier versions. Perhaps delete this
            # if we no longer want energyscale.
            prob_list, evals, czvals = self.barger_propagator.fill_osc_prob_c(
                self.e_centers, self.cz_centers, 1.0,
                deltam21, deltam31, deltacp,
                prop_height,
                YeI, YeO, YeM,
                total_bins*18, total_bins, total_bins,
                theta12, theta13, theta23
            )
        else:
            # Code copied from BargerPropagator.cc but fill_osc_prob_c but
            # pythonised and modified to use the python binding to
            # GetVacuumProb.
            prob_list = self.get_vacuum_prob_maps(
                deltam21, deltam31, deltacp,
                prop_height,
                theta12, theta13, theta23
            )

        # Slice up the transform arrays into views to populate each transform
        dims = ['true_energy', 'true_coszen']
        xform_dim_indices = [0, 1]
        users_dim_indices = [self.input_binning.index(d) for d in dims]
        xform_shape = [2] + [self.input_binning[d].num_bins for d in dims]

        # TODO: populate explicitly by flavor, don't assume any particular
        # ordering of the outputs names!
        transforms = []
        for out_idx, output_name in enumerate(self.output_names):
            xform = np.empty(xform_shape)
            if out_idx < 3:
                # Neutrinos
                xform[0] = np.array([
                    prob_list[out_idx + 18*i*self.num_czbins
                              : out_idx + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                xform[1] = np.array([
                    prob_list[out_idx+3 + 18*i*self.num_czbins
                              : out_idx+3 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                input_names = self.input_names[0:2]

            else:
                # Antineutrinos
                xform[0] = np.array([
                    prob_list[out_idx+6 + 18*i*self.num_czbins
                              : out_idx+6 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                xform[1] = np.array([
                    prob_list[out_idx+9 + 18*i*self.num_czbins
                              : out_idx+9 + 18*(i+1)*self.num_czbins
                              : 18]
                    for i in range(0, self.num_ebins)
                ])
                input_names = self.input_names[2:4]

            xform = np.moveaxis(
                xform,
                source=[0] + [i+1 for i in xform_dim_indices],
                destination=[0] + [i+1 for i in users_dim_indices]
            )
            if nutau_norm != 1 and output_name in ['nutau', 'nutaubar']:
                xform *= nutau_norm
            transforms.append(
                BinnedTensorTransform(
                    input_names=input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=xform
                )
            )

        return TransformSet(transforms=transforms)

    def get_vacuum_prob_maps(self, deltam21, deltam31, deltacp, prop_height,
                             theta12, theta13, theta23):
        """
        Calculate oscillation probabilities in the case of vacuum oscillations
        Here we use Prob3 but only because it has already implemented the 
        vacuum oscillations and so makes life easier.
        """
        # Set up oscillation parameters needed to initialise MNS matrix
        kSquared = True
        sin2th12Sq = np.sin(theta12)*np.sin(theta12)
        sin2th13Sq = np.sin(theta13)*np.sin(theta13)
        sin2th23Sq = np.sin(theta23)*np.sin(theta23)
        if deltam31 < 0.0:
            mAtm = deltam31
        else:
            mAtm = deltam31 - deltam21
        # Initialise objects to look over for neutrino and antineutrino flavours
        # 1 - nue, 2 - numu, 3 - nutau
        nuflavs = [1,2,3]
        nubarflavs = [-1,-2,-3]
        prob_list = []
        # Set up the distance to the detector. Radius of Earth is 6371km and
        # we then account for the depth of the detector in the Earth.
        depth = self.params.detector_depth.m_as('km')
        rdetector = 6371.0 - depth
        # Probability is separately calculated for each energy and zenith bin
        # center as well as every initial and final neutrno flavour.
        for e_cen in self.e_centers:
            for cz_cen in self.cz_centers:
                # Neutrinos are calculated for first
                kNuBar = 1
                for alpha in nuflavs:
                    for beta in nuflavs:
                        path = self.calc_path(
                            coszen=cz_cen,
                            rdetector=rdetector,
                            prop_height=prop_height,
                            depth=depth
                        )
                        self.barger_propagator.SetMNS(
                            sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                            mAtm,deltacp,e_cen,kSquared,kNuBar
                        )
                        prob_list.append(
                            self.barger_propagator.GetVacuumProb(
                                alpha, beta, e_cen, path
                            )
                        )
                # Then antineutrinos. With this, the layout of this prob_list
                # matches the output of the matter oscillations calculation.
                kNuBar = -1
                for alpha in nubarflavs:
                    for beta in nubarflavs:
                        path = self.calc_path(
                            coszen=cz_cen,
                            rdetector=rdetector,
                            prop_height=prop_height,
                            depth=depth
                        )
                        self.barger_propagator.SetMNS(
                            sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                            mAtm,deltacp,e_cen,kSquared,kNuBar
                        )
                        prob_list.append(
                            self.barger_propagator.GetVacuumProb(
                                alpha, beta, e_cen, path
                            )
                        )
        return prob_list

    def calc_path(self, coszen, rdetector, prop_height, depth):
        """
        Calculates the path through a spherical body of radius rdetector for
        a neutrino coming in with at coszen from prop_height to a detector
        at detph.
        """
        if coszen < 0:
            path = np.sqrt(
                (rdetector + prop_height + depth) * \
                (rdetector + prop_height + depth) - \
                (rdetector*rdetector)*(1 - coszen*coszen)
            ) - rdetector*coszen
        else:
            kappa = (depth + prop_height)/rdetector
            path = rdetector * np.sqrt(
                coszen*coszen - 1 + (1 + kappa)*(1 + kappa)
            ) - rdetector*coszen
        return path
    
    def calc_probs(self, true_e_scale, events_dict):
        """
        Calculate oscillation probabilities in the case of vacuum oscillations
        Here we use Prob3 but only because it has already implemented the 
        vacuum oscillations and so makes life easier. This is for the case of
        event-by-event calculations, nto for PISA maps.
        """
        if self.calc_transforms:
            raise ValueError("You have initialised prob3cpu for the case of "
                             "PISA maps and so this is the wrong function for"
                             " calculating the probabilities.")
        self.setup_barger_propagator()
        # Set up oscillation parameters needed to initialise MNS matrix
        kSquared = True
        theta12 = self.params['theta12'].value.m_as('rad')
        theta13 = self.params['theta13'].value.m_as('rad')
        theta23 = self.params['theta23'].value.m_as('rad')
        deltam21 = self.params['deltam21'].value.m_as('eV**2')
        deltam31 = self.params['deltam31'].value.m_as('eV**2')
        deltacp = self.params['deltacp'].value.m_as('rad')
        sin2th12Sq = np.sin(theta12)*np.sin(theta12)
        sin2th13Sq = np.sin(theta13)*np.sin(theta13)
        sin2th23Sq = np.sin(theta23)*np.sin(theta23)
        if deltam31 < 0.0:
            mAtm = deltam31
        else:
            mAtm = deltam31 - deltam21
        prob_e = []
        prob_mu = []
        if self._barger_earth_model is None:
            logging.info("Calculating vacuum oscillations")
            # Set up the distance to the detector. Radius of Earth is 6371km and
            # we then account for the depth of the detector in the Earth.
            depth = self.params.detector_depth.m_as('km')
            prop_height = self.params.prop_height.m_as('km')
            rdetector = 6371.0 - depth
            # Probability is separately calculated for each event
            for i, (en, cz) in enumerate(zip(events_dict['true_energy'],
                                             events_dict['true_coszen'])):
                en *= true_e_scale
                path = self.calc_path(
                    coszen=cz,
                    rdetector=rdetector,
                    prop_height=prop_height,
                    depth=depth
                )
                self.barger_propagator.SetMNS(
                    sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                    mAtm,deltacp,en,kSquared,events_dict['kNuBar']
                )
                # kFlav is zero-start indexed, whereas Prob3 wants it from 1
                prob_e.append(self.barger_propagator.GetVacuumProb(
                    1, events_dict['kFlav']+1, en, path
                ))
                prob_mu.append(self.barger_propagator.GetVacuumProb(
                    2, events_dict['kFlav']+1, en, path
                ))
        else:
            logging.info("Calculating matter oscillations")
            YeI = self.params.YeI.m_as('dimensionless')
            YeO = self.params.YeO.m_as('dimensionless')
            YeM = self.params.YeM.m_as('dimensionless')
            depth = self.params.detector_depth.m_as('km')
            prop_height = self.params.prop_height.m_as('km')
            # Probability is separately calculated for each event
            for i, (en, cz) in enumerate(zip(events_dict['true_energy'],
                                             events_dict['true_coszen'])):
                en *= true_e_scale
                self.barger_propagator.SetMNS(
                    sin2th12Sq,sin2th13Sq,sin2th23Sq,deltam21,
                    mAtm,deltacp,en,kSquared,events_dict['kNuBar']
                )
                self.barger_propagator.DefinePath(
                    float(cz), prop_height, YeI, YeO, YeM
                )
                self.barger_propagator.propagate(events_dict['kNuBar'])
                prob_e.append(self.barger_propagator.GetProb(
                    0, events_dict['kFlav']
                ))
                prob_mu.append(self.barger_propagator.GetProb(
                    1, events_dict['kFlav']
                ))
        events_dict['prob_e'] = prob_e
        events_dict['prob_mu'] = prob_mu
            
    def validate_params(self, params):
        if params['earth_model'].value is None:
            if params['YeI'].value is not None:
                raise ValueError("A none Earth model has been set but the YeI "
                                 "value is set to %s. Set this to none."
                                 %params['YeI'].value)
            if params['YeO'].value is not None:
                raise ValueError("A none Earth model has been set but the YeO "
                                 "value is set to %s. Set this to none."
                                 %params['YeO'].value)
            if params['YeM'].value is not None:
                raise ValueError("A none Earth model has been set but the YeM "
                                 "value is set to %s. Set this to none."
                                 %params['YeM'].value)
        pass
