
"""
Docstring
"""
from __future__ import absolute_import, print_function, division
import numpy as np

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils import vectorizer

class pi_hist(PiStage):
    """
    stage to histogram events

    Paramaters
    ----------

    None

    Notes
    -----

    """
    def __init__(self,
                 data=None,
                 params=None,
                 input_names=None,
                 output_names=None,
                 debug_mode=None,
                 error_method=None,
                 input_specs=None,
                 calc_specs=None,
                 output_specs=None,
                ):

        expected_params = ()
        input_names = ()
        output_names = ()

        # what are the keys used from the inputs during apply
        input_keys = ('weights',
                     )
        # what are keys added or altered in the calculation used during apply
        assert calc_specs is None
        if error_method in ['sumw2']:
            calc_keys = ('weights_squared',
                        )
            output_keys = ('weights',
                           'error',
                          )
            calc_specs = input_specs
        else:
            calc_keys = ()
            output_keys = ('weights',
                          )


        # init base class
        super(pi_hist, self).__init__(data=data,
                                      params=params,
                                      expected_params=expected_params,
                                      input_names=input_names,
                                      output_names=output_names,
                                      debug_mode=debug_mode,
                                      error_method=error_method,
                                      input_specs=input_specs,
                                      calc_specs=calc_specs,
                                      output_specs=output_specs,
                                      input_keys=input_keys,
                                      calc_keys=calc_keys,
                                      output_keys=output_keys,
                                     )

        assert self.input_mode is not None
        assert self.output_mode == 'binned'

    def setup_function(self):
        if self.error_method in ['sumw2']:
            self.data.data_specs = self.input_specs
            for container in self.data:
                container['weights_squared'] = np.empty((container.size), dtype=FTYPE)
            self.data.data_specs = self.output_specs
            for container in self.data:
                container['error'] = np.empty((container.size), dtype=FTYPE)

    @profile
    def apply(self):
        self.compute()

        # this is special, we want the actual event weights in the histo
        if self.input_mode == 'binned':
            self.data.data_specs = self.output_specs
            for container in self.data:
                container.array_to_binned('event_weights', self.output_specs, averaged=False)
                vectorizer.multiply_and_scale(container['event_weights'], out=container['weights'])
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.square(container['weights'], out=container['weights_squared'])
                    vectorizer.multiply(container['event_weights'], out=container['weights_squared'])
                    vectorizer.sqrt(container['weights_squared'], out=container['error'])

        elif self.input_mode == 'events':
            for container in self.data:
                self.data.data_specs = self.input_specs
                vectorizer.multiply(container['event_weights'],out=container['weights'])
                # calcualte errors
                if self.error_method in ['sumw2']:
                    vectorizer.square(container['weights'], out=container['weights_squared'])
                self.data.data_specs = self.output_specs
                container.array_to_binned('weights', self.output_specs, averaged=False)
                if self.error_method in ['sumw2']:
                    container.array_to_binned('weights_squared', self.output_specs, averaged=False)
                    vectorizer.sqrt(container['weights_squared'], out=container['error'])
