import numpy as np
from numba import guvectorize, SmartArray

from pisa import *
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.core.binning import MultiDimBinning
from pisa.core.map import Map, MapSet
from pisa.stages.osc.osc_params import OscParams
from pisa.stages.osc.layers import Layers
from prob3numba.numba_osc import *
from prob3numba.numba_tools import *


class pi_prob3(PiStage):
    """
    prob3 osc PISA Pi class

    Paramaters
    ----------

    None

    Notes
    -----

    """
    def __init__(self,
                 events=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 input_specs=None,
                 calc_specs=None,
                 apply_specs=None,
                 ):

        expected_params = ()
        input_names = ()
        output_names = ()

        # init base class!
        super(pi_prob3, self).__init__(
                                       events=events,
                                       params=params,
                                       expected_params=expected_params,
                                       input_names=input_names,
                                       output_names=output_names,
                                       debug_mode=debug_mode,
                                       input_specs=input_specs,
                                       calc_specs=calc_specs,
                                       apply_specs=apply_specs,
                                       )

        #assert input_specs is not None
        assert calc_specs is not None
        assert apply_specs is not None


    def setup(self):

        # Set up some dumb mixing parameters
        OP = OscParams(7.5e-5, 2.524e-3, np.sqrt(0.306), np.sqrt(0.02166), np.sqrt(0.441), 261/180.*np.pi)
        self.mix = OP.mix_matrix_complex
        self.dm = OP.dm_matrix
        self.nsi_eps = np.zeros_like(self.mix)

        # setup the layers
        earth_model = '/home/peller/cake/pisa/resources/osc/PREM_59layer.dat'
        det_depth = 2
        atm_height = 20
        myLayers = Layers(earth_model, det_depth, atm_height)
        myLayers.setElecFrac(0.4656, 0.4656, 0.4957)

        if self.calc_mode == 'events':
            for name, val in self.events.items():
                # calc layers
                myLayers.calcLayers(val['true_coszen'].get('host'))
                nevts = val['true_coszen'].shape[0]
                numberOfLayers = myLayers.n_layers
                densities = myLayers.density.reshape((nevts,myLayers.max_layers))
                distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
                # empty array to be filled
                probability = np.zeros((nevts,3,3), dtype=FTYPE)
                # put into smart array
                val['densities'] = SmartArray(densities)
                val['distances'] = SmartArray(distances)
                val['probability'] = SmartArray(probability)

        elif self.calc_mode == 'binned':
            # set up the map grid
            self.grid_values = {}
            e = self.calc_specs['true_energy'].weighted_centers.m.astype(FTYPE)
            cz = self.calc_specs['true_coszen'].weighted_centers.m.astype(FTYPE)
            nevts = len(e) * len(cz)
            e_vals, cz_vals = np.meshgrid(e, cz)
            myLayers.calcLayers(self.grid_values['true_coszen'].get('host'))
            numberOfLayers = myLayers.n_layers
            densities = myLayers.density.reshape((nevts,myLayers.max_layers))
            distances = myLayers.distance.reshape((nevts,myLayers.max_layers))
            # empty array to be filled
            probability_nu = np.zeros((nevts,3,3), dtype=FTYPE)
            probability_nubar = np.zeros((nevts,3,3), dtype=FTYPE)
            # put into smart array
            self.grid_values['true_energy'] = SmartArray(e_vals.ravel())
            self.grid_values['true_coszen'] = SmartArray(cz_vals.ravel())
            self.grid_values['densities'] = SmartArray(densities)
            self.grid_values['distances'] = SmartArray(distances)
            self.grid_values['probability_nubar'] = SmartArray(probability_nubar)
            self.grid_values['probability_nu'] = SmartArray(probability_nu)



    def calc_probs(self, nubar, e_array, rho_array, len_array, out):
        ''' wrapper to execute osc. calc '''
        propagate_array(self.dm,
                        self.mix,
                        self.nsi_eps,
                        nubar,
                        e_array.get(WHERE),
                        rho_array.get(WHERE),
                        len_array.get(WHERE),
                        out=out.get(WHERE)
                        )
        out.mark_changed(WHERE)

    def compute(self):

        if self.calc_mode == 'events':
            for name, val in self.events.items():
                self.calc_probs(val['nubar'],
                                val['true_energy'],
                                val['densities'],
                                val['distances'],
                                out=val['probability'],
                                )

        elif self.calc_mode == 'binned':
            for nubar, probs in zip([1, -1], ['probability_nu', 'probability_nubar']):
                self.calc_probs(nubar,
                                self.grid_values['true_energy'],
                                self.grid_values['densities'],
                                self.grid_values['distances'],
                                out=self.grid_values[probs],
                                )


    def apply(self, inputs=None):

        if not self.calc_mode is None:
            self.compute()
        
        if self.apply_mode is None:
            return self.inputs

        if self.calc_mode == 'binned':
            if self.apply_mode == 'binned':
                assert self.calc_specs == self.apply_specs, 'cannot do different binnings yet'
                if self.input_mode is None:
                    maps = []
                    flavs = ['e', 'mu', 'tau']
                    hists = self.grid_values['probability_nu'].get('host')
                    print hists
                    n_e = self.apply_specs['true_energy'].num_bins
                    n_cz = self.apply_specs['true_coszen'].num_bins
                    for i in range(3):
                        for j in range(3):
                            hist = hists[:,i,j]
                            hist = hist.reshape(n_e, n_cz)
                            maps.append(Map(name='prob_%s_to_%s'%(flavs[i],flavs[j]), hist=hist, binning=self.apply_specs))
                    self.outputs = MapSet(maps)
                    return self.outputs

                elif self.input_mode == 'binned':
                    raise NotImplementedError

                elif self.input_mode == 'events':
                    raise NotImplementedError


            elif self.apply_mode == 'events':
                assert self.inputs is None, 'This would ignore the input maps, not working right now'
                # do LUT
                raise NotImplementedError
                if self.input_mode is None:
                    raise NotImplementedError
                elif self.input_mode == 'binned':
                    raise NotImplementedError

                elif self.input_mode == 'events':
                    raise NotImplementedError


        if self.calc_mode == 'events':
            if self.apply_mode == 'events':
                # redirect inputs to outputs
                self.outputs = self.inputs
                print self.events

                for name, evts in self.events.items():
                    # this is shitty, needs to change
                    if 'tau' in name:
                        end_flav = 2
                    elif 'mu' in name:
                        end_flav = 1
                    else:
                        end_flav = 0
                    # calc weight from initial flux * probability it oscillated into event's flavour
                    # we can define a function that does that on CPU or GPU later
                    evts['weight'] = evts['flux_nue'].get('host') * evts['probability'].get('host')[:,0,end_flav] + evts['flux_numu'].get('host') * evts['probability'].get('host')[:,1,end_flav]
                    evts['weight'].mark_changed('host')
                return None

            elif self.apply_mode == 'binned':
                if self.input_mode is None:
                    # histogram event weights
                    binning = self.apply_specs
                    bin_edges = [edges.magnitude for edges in binning.bin_edges]
                    binning_cols = binning.names

                    maps = []
                    for name, evts in self.events.items():
                        if 'tau' in name:
                            end_flav = 2
                        elif 'mu' in name:
                            end_flav = 1
                        else:
                            end_flav = 0
                        hist_weights = evts['flux_nue'].get('host') * evts['probability'].get('host')[:,0,end_flav] + evts['flux_numu'].get('host') * evts['probability'].get('host')[:,1,end_flav]
                        sample = [evts[colname].get('host') for colname in binning_cols]
                        hist, _ = np.histogramdd(sample=sample,
                                                 weights=hist_weights,
                                                 bins=bin_edges,
                                                 )

                        maps.append(Map(name=name, hist=hist, binning=binning))
                    self.outputs = MapSet(maps)

                elif self.input_mode == 'binned':
                    # histogram event weights and apply to maps
                    raise NotImplementedError
        return self.outputs

