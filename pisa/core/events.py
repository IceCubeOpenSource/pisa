#! /usr/bin/env python
#
# Events class for working with PISA events files
#
# author: Justin L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   October 24, 2015
#
"""
Events class for working with PISA events files
"""


from collections import Iterable, Sequence

import h5py
import numpy as np
from uncertainties import unumpy as unp

from pisa import ureg, Q_
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils import resources
from pisa.utils.comparisons import normQuant, recursiveEquality
from pisa.utils.flavInt import FlavIntData, NuFlavIntGroup
from pisa.utils.hash import hash_obj
from pisa.utils import hdf
from pisa.utils.log import logging, set_verbosity


# TODO: test hash function (attr)
class Events(FlavIntData):
    """Container for storing events, including metadata about the events.
    
    Examples
    --------
    >>> from pisa.core.binning import OneDimBinning, MultiDimBinning

    >>> # Load events from a PISA HDF5 file
    >>> events = Events('events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5')

    >>> # Apply a simple cut
    >>> events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    >>> np.max(events[fi]['true_coszen']) <= 0.5
    True

    >>> # Apply an "inbounds" cut via a OneDimBinning
    >>> true_e_binning = OneDimBinning(
    ...    name='true_energy', num_bins=80, is_log=True, domain=[10, 60]*ureg.GeV
    ... )
    >>> events.keepInbounds(true_e_binning)
    >>> np.min(events[fi]['true_energy']) >= 10
    True

    >>> print [(k, events.metadata[k]) for k in sorted(events.metadata.keys())]
    [('cuts', ['analysis']),
      ('detector', 'pingu'),
      ('flavints_joined',
         ['nue_cc+nuebar_cc',
             'numu_cc+numubar_cc',
             'nutau_cc+nutaubar_cc',
             'nuall_nc+nuallbar_nc']),
      ('geom', 'v39'),
      ('proc_ver', '5.1'),
      ('runs', [620, 621, 622])]
   
	"""
    def __init__(self, val=None):
        self.metadata = {
            'detector': '',
            'geom': '',
            'runs': [],
            'proc_ver': '',
            'cuts': [],
            'flavints_joined': [],
        }
        meta = {}
        data = FlavIntData()
        if isinstance(val, basestring) or isinstance(val, h5py.Group):
            data, meta = self.__load(val)
        elif isinstance(val, Events):
            self.metadata = val.metadata
            data = val
        elif isinstance(val, dict):
            data = val
        self.metadata.update(meta)
        self.validate(data)
        self.update(data)
        self._hash = hash_obj(normQuant(self.metadata))

    @property
    def hash(self):
        return self._hash

    def meta_eq(self, other):
        """Test whether the metadata for this object matches that of `other`"""
        return recursiveEquality(self.metadata, other.metadata)

    def data_eq(self, other):
        """Test whether the data for this object matche that of `other`"""
        return recursiveEquality(self, other)

    def __eq__(self, other):
        return self.meta_eq(other) and self.data_eq(other)

    def __load(self, fname):
        fpath = resources.find_resource(fname)
        with h5py.File(fpath, 'r') as open_file:
            meta = dict(open_file.attrs)
            for k, v in meta.items():
                if hasattr(v, 'tolist'):
                    meta[k] = v.tolist()
            data = hdf.from_hdf(open_file)
        self.validate(data)
        return data, meta

    def save(self, fname, **kwargs):
        hdf.to_hdf(self, fname, attrs=self.metadata, **kwargs)

    def histogram(self, kinds, binning, binning_cols=None, weights_col=None,
            errors=False):
        """Histogram the events of all `kinds` specified, with `binning` and
        optionally applying `weights`.

        Parameters
        ----------
        kinds : string, sequence of NuFlavInt, or NuFlavIntGroup
        binning : OneDimBinning, MultiDimBinning or sequence of arrays (one array per binning dimension)
        weights_col : string

        Returns
        -------
        hist : numpy ndarray with as many dimensions as specified by `binning`
        argument

        """
        if not isinstance(kinds, NuFlavIntGroup):
            kinds = NuFlavIntGroup(kinds)
        #if not isinstance(binning, (OneDimBinning, MultiDimBinning, Sequence)):
        #    binning = MultiDimBinning(binning)
        if isinstance(binning_cols, basestring):
            binning_cols = [binning_cols]
        assert weights_col is None or isinstance(weights_col, basestring)

        # TODO: units of columns, and convert bin edges if necessary
        if isinstance(binning, OneDimBinning):
            bin_edges = [binning.magnitude]
            if binning_cols is None:
                binning_cols = [binning.name]
            else:
                assert len(binning_cols) == 1 and binning_cols[0] == binning.name
        elif isinstance(binning, MultiDimBinning):
            bin_edges = [edges.magnitude for edges in binning.bin_edges]
            if binning_cols is None:
                binning_cols = binning.names
            else:
                assert set(binning_cols).issubset(set(binning.names))
        elif isinstance(binning, (Sequence, Iterable)):
            assert len(binning_cols) == len(binning)
            bin_edges = binning

        # Extract the columns' data into a list of array(s) for histogramming
        repr_flav_int = kinds[0]
        sample = [self[repr_flav_int][colname] for colname in binning_cols]
        if weights_col is not None:
            weights = self[repr_flav_int][weights_col]
        else:
            weights = None

        hist, _ = np.histogramdd(sample=sample, weights=weights, bins=bin_edges)
        if errors:
            sumw2, _ = np.histogramdd(sample=sample,
                                      weights=np.square(weights),
                                      bins=bin_edges)
            hist = unp.uarray(hist, np.sqrt(sumw2))

        return hist

    def applyCut(self, keep_crit):
        """Apply a cut by specifying criteria for keeping events. The cut must
        be successfully applied to all flav/ints in the events object before
        the changes are kept, otherwise the cuts are reverted.


        Parameters
        ----------
        keep_crit : string
            Any string interpretable as numpy boolean expression.


        Examples
        --------
        Keep events with true energies in [1, 80] GeV (note that units are not
        recognized, so have to be handled outside this method)
        >>> applyCut("(true_energy >= 1) & (true_energy <= 80)")

        Do the opposite with "~" inverting the criteria
        >>> applyCut("~((true_energy >= 1) & (true_energy <= 80))")

        Numpy namespace is available for use via `np` prefix
        >>> applyCut("np.log10(true_energy) >= 0")

        """
        if keep_crit in self.metadata['cuts']:
            return

        assert isinstance(keep_crit, basestring)

        flavints_to_process = self.flavints()
        flavints_processed = []
        new_data = {}
        try:
            for flav_int in flavints_to_process:
                data_dict = self[flav_int]
                field_names = data_dict.keys()

                # TODO: handle unicode:
                #  * translate crit to unicode (easiest to hack but could be
                #    problematic elsewhere)
                #  * translate field names to ascii (probably should be done at
                #    the from_hdf stage?)

                # Replace simple field names with full paths into the data that
                # lives in this object
                crit_str = (keep_crit)
                for field_name in field_names:
                    crit_str = crit_str.replace(
                        field_name, 'self["%s"]["%s"]' %(flav_int, field_name)
                    )
                mask = eval(crit_str)
                new_data[flav_int] = {k:v[mask]
                                      for k,v in self[flav_int].iteritems()}
                flavints_processed.append(flav_int)
        except:
            if (len(flavints_processed) > 0
                and flavints_processed != flavints_to_process):
                logging.error('Events object is in an inconsistent state.'
                              ' Reverting cut for all flavInts.')
            raise
        else:
            for flav_int in flavints_to_process:
                self[flav_int] = new_data[flav_int]
                new_data[flav_int] = None
            self.metadata['cuts'].append(keep_crit)

    def keepInbounds(self, binning):
        """Cut out any events that fall outside `binning`. Note that events
        that fall exactly on the outer edge are kept.

        Parameters
        ----------
        binning : OneDimBinning or MultiDimBinning

        """
        if isinstance(binning, OneDimBinning):
            binning = [binning]
        else:
            assert isinstance(binning, MultiDimBinning)
        current_cuts = self.metadata['cuts']
        new_cuts = [dim.inbounds_criteria for dim in binning]
        unapplied_cuts = [c for c in new_cuts if c not in current_cuts]
        for cut in unapplied_cuts:
            self.applyCut(keep_crit=cut)

def test_Events():
    from pisa.utils.flavInt import NuFlavInt
    # Instantiate empty object
    events = Events()

    # Instantiate from PISA events HDF5 file
    events = Events('events/pingu_v39/events__pingu__v39__runs_620-622__proc_v5.1__joined_G_nue_cc+nuebar_cc_G_numu_cc+numubar_cc_G_nutau_cc+nutaubar_cc_G_nuall_nc+nuallbar_nc.hdf5')

    # Apply a simple cut
    events.applyCut('(true_coszen <= 0.5) & (true_energy <= 70)')
    for fi in events.flavints():
        assert np.max(events[fi]['true_coszen']) <= 0.5
        assert np.max(events[fi]['true_energy']) <= 70

    # Apply an "inbounds" cut via a OneDimBinning
    true_e_binning = OneDimBinning(
        name='true_energy', num_bins=80, is_log=True, domain=[10, 60]*ureg.GeV
    )
    events.keepInbounds(true_e_binning)
    for fi in events.flavints():
        assert np.min(events[fi]['true_energy']) >= 10
        assert np.max(events[fi]['true_energy']) <= 60

    # Apply an "inbounds" cut via a MultiDimBinning
    true_e_binning = OneDimBinning(
        name='true_energy', num_bins=80, is_log=True, domain=[20, 50]*ureg.GeV
    )
    true_cz_binning = OneDimBinning(
        name='true_coszen', num_bins=40, is_lin=True, domain=[-0.8, 0]
    )
    mdb = MultiDimBinning([true_e_binning, true_cz_binning])
    events.keepInbounds(mdb)
    for fi in events.flavints():
        assert np.min(events[fi]['true_energy']) >= 20
        assert np.max(events[fi]['true_energy']) <= 50
        assert np.min(events[fi]['true_coszen']) >= -0.8
        assert np.max(events[fi]['true_coszen']) <= 0

    # Now try to apply a cut that fails on one flav/int (since the field will
    # be missing) and make sure that the cut did not get applied anywhere in
    # the end (i.e., it is rolled back)
    sub_evts = events['nutaunc']
    sub_evts.pop('true_energy')
    events['nutaunc'] = sub_evts
    try:
        events.applyCut('(true_energy >= 30) & (true_energy <= 40)')
    except:
        pass
    else:
        raise Exception('Should not have been able to apply the cut!')
    for fi in events.flavints():
        if fi == NuFlavInt('nutaunc'):
            continue
        assert np.min(events[fi]['true_energy']) < 30

    logging.info('<< PASSED : test_Events >>')


if __name__ == "__main__":
    set_verbosity(3)
    test_Events()