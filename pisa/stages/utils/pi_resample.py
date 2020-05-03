"""
Stage to transform binned data from one binning to another while also dealing with
uncertainty estimates in a reasonable way. In particular, this allows up-sampling from a
more coarse binning to a finer binning.

The implementation is similar to that of the pi_hist stage, hence the over-writing of
the `apply` method.
"""

from __future__ import absolute_import, print_function, division

import numpy as np
from enum import Enum, auto

from pisa import FTYPE
from pisa.core.pi_stage import PiStage
from pisa.utils.profiler import profile
from pisa.utils import vectorizer
from pisa.core import translation
from numba import SmartArray


class ResampleMode(Enum):
    """Enumerates sampling methods of the `pi_resample` stage."""

    UP = auto()
    DOWN = auto()
    ARB = auto()

class pi_resample(PiStage):  # pylint: disable=invalid-name
    """
    Stage to resample weighted MC histograms from one binning to another.
    
    Parameters
    ----------
    
    scale_errors : bool, optional
        If `True` (default), apply scaling to errors.
    """

    def __init__(
        self,
        scale_errors=True,
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
        input_apply_keys = ("weights",)

        # what are keys added or altered in the calculation used during apply
        assert calc_specs is None
        if scale_errors:
            output_apply_keys = ("weights", "errors")
        else:
            output_apply_keys = ("weights",)
        # init base class
        super().__init__(
            data=data,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            debug_mode=debug_mode,
            error_method=error_method,
            input_specs=input_specs,
            calc_specs=calc_specs,
            output_specs=output_specs,
            input_apply_keys=input_apply_keys,
            output_apply_keys=output_apply_keys,
        )

        # This stage only makes sense when going binned to binned.
        assert self.input_mode == "binned", "stage only takes binned input"
        assert self.output_mode == "binned", "stage only produces binned output"
        
        self.scale_errors = scale_errors
        
        # The following tests whether `output_specs` is a strict up-sample
        # from `input_specs`, i.e. the bin edges of `output_specs` are a superset
        # of the bin edges of `input_specs`.
        
        # TODO: Test for ability to resample in two steps
        # TODO: Update to new test nomenclature
        if input_specs.is_compat(output_specs):
            self.rs_mode = ResampleMode.UP
        elif output_specs.is_compat(input_specs):
            self.rs_mode = ResampleMode.DOWN
        else:
            raise ValueError("Binnings are not compatible with each other for resample")

        # TODO: Implement downsampling
        # TODO: Implement arbitrary resampling
        if self.rs_mode == ResampleMode.DOWN:
            raise NotImplementedError("Downsampling not yet implemented.")
        if self.rs_mode == ResampleMode.ARB:
            raise NotImplementedError("Arbitrary resampling not yet implemented.")

    def setup_function(self):
        # create the variables to be filled in `apply`
        if self.scale_errors:
            self.data.data_specs = self.input_specs
            for container in self.data:
                container["variances"] = np.empty((container.size), dtype=FTYPE)

    @profile
    def apply(self):
        # DO NOT USE THIS STAGE AS YOUR TEMPLATE IF YOU ARE NEW TO PISA!
        # --------------------------------------------------------------
        #
        # We are overwriting the `apply` method rather than the `apply_function` method
        # because we are manipulating the data binning in a delicate way that doesn't
        # work with automatic rebinning.
        if self.scale_errors:
            self.data.data_specs = self.input_specs
            for container in self.data:
                vectorizer.pow(
                    vals=container["errors"],
                    pwr=2,
                    out=container["variances"],
                )
        self.data.data_specs = self.output_specs
        for container in self.data:
            # The built-in `binned_to_binned` method behaves as follows:
            # - When several bins are merged into one, the large bin contains the
            #   average of the smaller bins.
            # - When a bin is split into smaller bins, each of the smaller bins gets
            #   the same value as the large bin.
            # This first step is the same whether we sample up or down.
            container.binned_to_binned("weights", self.output_specs)
            if self.scale_errors:
                container.binned_to_binned("variances", self.output_specs)
                container.binned_to_binned("errors", self.output_specs)

            # We now have to scale the weights and squared weights according to the bin
            # volumes depending on the sampling mode.
            if self.rs_mode == ResampleMode.UP:
                # These are the volumes of the bins we sample *into*
                upsampled_binvols = SmartArray(
                    self.output_specs.weighted_bin_volumes(attach_units=False).ravel()
                )
                # These are the volumes of the bins we sample *from*
                coarse_volumes = SmartArray(
                    self.input_specs.weighted_bin_volumes(attach_units=False).ravel()
                )
                # For every upsampled bin, we need to know what the volume of the bin
                # was where it came from. First, we get the position of the midpoint of
                # each fine (output) bin:
                fine_gridpoints = [
                    # The `unroll_binning` function returns the midpoints of the bins
                    # in the dimension `name`.
                    SmartArray(container.unroll_binning(name, self.output_specs))
                    for name in self.output_specs.names
                ]
                # We look up at which bin index of the input binning the midpoints of
                # the output binning can be found, and assign to each the volume of the
                # bin of that index.
                origin_binvols = translation.lookup(
                    fine_gridpoints, coarse_volumes, self.input_specs
                )
                # Finally, we scale the weights and variances by the ratio of the
                # bin volumes in place:
                vectorizer.imul(upsampled_binvols, container["weights"])
                vectorizer.itruediv(origin_binvols, container["weights"])
                container["weights"].mark_changed()
                if self.scale_errors:
                    vectorizer.imul(upsampled_binvols, container["variances"])
                    vectorizer.itruediv(origin_binvols, container["variances"])
                    container["variances"].mark_changed()
            elif self.rs_mode == ResampleMode.DOWN:
                pass  # not yet implemented

            if self.scale_errors:
                vectorizer.sqrt(
                    vals=container["variances"], out=container["errors"]
                )
                container["errors"].mark_changed()
