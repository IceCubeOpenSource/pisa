#
# Creates the pdfs of the reconstructed energy and coszen from the
# true parameters. Provides reco event rate maps using these pdfs.
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   April 9, 2014
#

import sys
import h5py
import numpy as np
from itertools import product
from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.utils.log import logging
from pisa.resources.resources import find_resource

class RecoServiceMC(RecoServiceBase):
    """
    From the simulation file, creates 4D histograms of
    [true_energy][true_coszen][reco_energy][reco_coszen] which act as
    2D pdfs for the probability that an event with (true_energy,
    true_coszen) will be reconstructed as (reco_energy,reco_coszen).

    From these histograms, and the true event rate maps, calculates
    the reconstructed even rate templates.
    """
    def __init__(self, ebins, czbins, reco_mc_wt_file=None, **kwargs):
        """
        Parameters needed to instantiate a MC-based reconstruction service:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * reco_weight_file: HDF5 containing the MC events to construct the kernels
        """
        self.simfile = reco_mc_wt_file
        RecoServiceBase.__init__(self, ebins, czbins, simfile=self.simfile, **kwargs)


    def kernel_from_simfile(self, simfile=None, **kwargs):
        logging.info('Opening file: %s'%(simfile))
        try:
            fh = h5py.File(find_resource(simfile),'r')
        except IOError,e:
            logging.error("Unable to open event data file %s"%simfile)
            logging.error(e)
            sys.exit(1)

        # Create the 4D distribution kernels...
        kernels = {}
        logging.info("Creating kernel dict...")
        kernels['ebins'] = self.ebins
        kernels['czbins'] = self.czbins
        for flavor in ['nue','nue_bar','numu','numu_bar','nutau','nutau_bar']:
            flavor_dict = {}
            logging.debug("Working on %s kernels"%flavor)
            for int_type in ['cc','nc']:
                true_energy = np.array(fh[flavor+'/'+int_type+'/true_energy'])
                true_coszen = np.array(fh[flavor+'/'+int_type+'/true_coszen'])
                reco_energy = np.array(fh[flavor+'/'+int_type+'/reco_energy'])
                reco_coszen = np.array(fh[flavor+'/'+int_type+'/reco_coszen'])

                # True binning, reco binning...
                bins = (self.ebins,self.czbins,self.ebins,self.czbins)
                data = (true_energy,true_coszen,reco_energy,reco_coszen)
                kernel,_ = np.histogramdd(data,bins=bins)

                # Normalize correctly - I THINK this is correct...
                k_shape = np.shape(kernel)
                for true_bin in product(range(k_shape[0]),range(k_shape[1])):
                    kernel_sum = np.sum(kernel[true_bin])
                    if kernel_sum > 0.0: kernel[true_bin] /= kernel_sum

                flavor_dict[int_type] = kernel
            kernels[flavor] = flavor_dict

        return kernels


    def _get_reco_kernels(self, simfile=None, **kwargs):

        for reco_scale in ['e_reco_scale', 'cz_reco_scale']:
            if reco_scale in kwargs:
                if not kwargs[reco_scale]==1:
                    raise ValueError('%s = %.2f not valid for RecoServiceMC!'
                                     %(reco_scale, kwargs[reco_scale]))

        if not simfile in [self.simfile, None]:
            logging.info('Reconstruction from non-default MC file %s!'%simfile)
            return kernel_from_simfile(simfile=simfile)

        if not hasattr(self, 'kernels'):
            logging.info('Using file %s for default reconstruction'%(simfile))
            self.kernels = self.kernel_from_simfile(simfile=simfile)

        return self.kernels
