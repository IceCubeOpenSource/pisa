# Author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# Date:   2017-03-26
"""
Produce a set of transforms mapping true events values (energy and coszen) onto
reconstructed values.

For each bin in true energy and true coszen, a corresponding distribution of
pid, reconstructed energy, and reconstructed coszen is estimated using
variable-bandwidth KDE to characterize this transformation using Monte Carlo
events.
"""


# TODO/BUG: setting any tgt_max_binwidth_factors to 0 can result in no events!
# TODO/BUG: Not regenerating KDEs with new cut applied to events...
# TODO: nutau needn't be treated below 3.5 GeV! ...
# TODO: write "closest bin" as a function
# TODO: muons
# TODO: Grab events from all dimensions but with weighting of which dimension
#       is preferred
# TODO: Figure out a dynamic similarity metric such that the parameters can be
#       figured out by the software, rather than set by the user. E.g., use
#       some statistical clustering technique?
# TODO: Move (most/all) functions defined here into module(s) in utils dir
# TODO: Add n_dct as a class instantiation argument?
# TODO: Separate VBWKDE parameters for each dimension, specified either in
#       char_deps_downsampling or as a separate arg?


from __future__ import division

from ast import literal_eval
from collections import Mapping, namedtuple, OrderedDict, Sequence
from copy import deepcopy
import threading

import numpy as np

from pisa import EPSILON, FTYPE, NUMBA_AVAIL, OMP_NUM_THREADS, numba_jit, ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.events import Events
from pisa.core.stage import Stage
from pisa.core.transform import BinnedTensorTransform, TransformSet

from pisa.utils.comparisons import EQUALITY_SIGFIGS, isbarenumeric, isscalar
from pisa.utils.flavInt import flavintGroupsFromString, NuFlavIntGroup
from pisa.utils.gaussians import gaussians
from pisa.utils.hash import hash_obj
from pisa.utils.parallel import parallel_run
from pisa.utils.vbwkde import vbwkde as vbwkde_func
from pisa.utils.log import logging


__all__ = ['KDEProfile', 'collect_enough_events', 'fold_coszen_diff',
           'weight_coszen_tails', 'coszen_error_edges', 'vbwkde']


KDEProfile = namedtuple('KDEProfile', ['x', 'counts'])
"""namedtuple type for storing the normalized KDE profile: (x, counts)"""


# TODO: modify this once we have fixed the Events object to be more agnostic to
#       flavint
# TODO: When running w/ GPU, this function (specifically `applyCut` method of
#       Events object) is the bottleneck. Make that faster!

def collect_enough_events(events, flavint, bin, min_num_events, tgt_num_events,
                          tgt_max_binwidth_factor):
    """Heuristic to collect enough events close to the provided bin such
    that KDEs can be applied and achieve robust results.

    Events are selected via the logic:
    * If there are `tgt_num_events` or more in the energy bin, simply return
      all events in the bin.
    * Otherwise, go up to `tgt_max_binwidth_factor` times the width of the
      `bin` to find up to `tgt_num_events`. (Note that the distance is
      measured from the bin's weighted center.)
    * However, keep going as far as necessary outside the energy bin (i.e.,
      ignore `tgt_max_binwidth_factor`) to find `min_num_events` events.

    Note that in the above, _all_ events in the bin will _always_ be returned.
    The rest of the algorithm decides how far outside it has to reach to get a
    "desired" number of events (`tgt_num_events`) without going too far, but
    will reach as far as necessary to get the "minimum" number of events
    (`min_num_events').

    Distances are measured on a logarithmic scale, so events that are one-half
    and twice the energy bin's weighted center (respectively) are considered to
    be the same distance to the weighted center.

    In the case of "exact ties" as described above, _both_ events will be kept.
    Therefore, one may get _more_ than `tgt_num_events` when that threshold is
    used or more than `min_num_events` when that threshold is used.

    Finally, note that both edges of the `bin` are _inclusive_, i.e., it
    defines a closed interval.


    Parameters
    ----------
    events : pisa.core.events.Events
        Note that a deepcopy is made on this object, so it is not modified in
        this function.

    bin : pisa.core.binning.OneDimBinning

    flavint : None or convertible to pisa.utils.flavInts.NuFlavIntGroup

    min_num_events : int
        At least this many events will collected, regardless how far outside
        the boundaries of `bin` is necessary to find them.

    tgt_num_events : int >= min_num_events
        Ideally each bin will have this many events. Allow the bin to expand a
        fraction (`tgt_max_binwidth_factor`) to try to hit this number.

    tgt_max_binwidth_factor : float >= 0
        Expand by up to this factor of the width of the bin in order to
        collect tgt_num_events.

    Returns
    -------
    events_subset : pisa.core.events.Events
        The subset of the passed `events` that fulfill the criteria outlined
        above.

    """
    # For now, be strict about what is allowed, so as to keep logic simple
    assert isinstance(events, Events)
    flavintgroup = NuFlavIntGroup(flavint)
    repr_flavint = flavintgroup[0]
    logging.trace('flavintgroup=%s, repr_flavint=%s, bin=%s',
                  flavintgroup, repr_flavint, bin)

    if isinstance(bin, MultiDimBinning):
        dims = bin.dimensions
        assert len(dims) == 1
        bin = dims[0]
    assert isinstance(bin, OneDimBinning)
    edges = bin.bin_edges.m
    assert len(edges) == 2
    assert bin.is_log
    assert min_num_events <= tgt_num_events

    # ASSUMPTION: units in Events object are same as in `bin`
    bin_wtd_center = bin.weighted_centers[0].m

    if bin.is_log:
        bin_width = edges[1] / edges[0]
        bin_half_width = np.sqrt(bin_width)
    else:
        bin_width = edges[1] - edges[0]
        bin_half_width = bin_width / 2

    # Define for convenience of use elsewhere
    field_values = events[repr_flavint][bin.name]
    n_events = len(field_values)

    if n_events == 0:
        raise ValueError(
            'No events were found for %s! Do you apply a cut that removes all'
            ' relevant events? E.g. this can occur if you cut away'
            ' downgoing events but then define true-coszen bins in the'
            ' downgoing region.' % flavintgroup
        )

    # If either min_num_events or tgt_num_events is greater than the total
    # number of events we have to work with, force to be equal to the number
    # of events we have.
    min_num_events = min_num_events if min_num_events < n_events else n_events
    tgt_num_events = tgt_num_events if tgt_num_events < n_events else n_events

    # Absolute distance from these events to the center of the bin, sorted in
    # ascending order (so events closest to bin center come first)
    if bin.is_log:
        sorted_abs_dist = np.sort(np.abs(np.log(field_values/bin_wtd_center)))
    else:
        sorted_abs_dist = np.sort(np.abs(field_values - bin_wtd_center))

    # Distance from the bin center you have to go to obtain `tgt_num_events`
    tgt_num_events_dist = np.exp(sorted_abs_dist[tgt_num_events-1])

    # Maximum distance the  tgt_max_binwidth_factor` allows us to go in order
    # to obtain `tgt_num_events` events
    if bin.is_log:
        tgt_max_dist = bin_half_width * (1 + tgt_max_binwidth_factor)**2
    else:
        tgt_max_dist = bin_half_width + bin_width*tgt_max_binwidth_factor

    # Define a single "target" distance taking into consideration that we
    # should neither exceed `tgt_max_dist` nor `tgt_num_events`
    if tgt_num_events_dist < tgt_max_dist:
        tgt_dist = tgt_num_events_dist
    else:
        tgt_dist = tgt_max_dist

    # If the target distance is within the bin, simply take all events from
    # the bin and we're done, as we will get >= tgt_num_events within the bin.
    if tgt_dist <= bin_half_width:
        thresh_dist = bin_half_width

    else:
        # Figure out how far out we have to go to get `min_num_events`
        min_num_events_dist = np.exp(sorted_abs_dist[min_num_events-1])

        # If this is _further_ than `tgt_dist`, then we have to suck it up
        # and go `min_num_events_dist` away to ensure we collect enough events
        if min_num_events_dist > tgt_dist:
            thresh_dist = min_num_events_dist

        # But if we can stop at tgt_dist away and get more events than
        # min_num_events, we should do that to maximize our statistics
        else:
            thresh_dist = tgt_dist

    if bin.is_log:
        lower_edge = bin_wtd_center / thresh_dist
        upper_edge = bin_wtd_center * thresh_dist
    else:
        lower_edge = bin_wtd_center - thresh_dist
        upper_edge = bin_wtd_center + thresh_dist

    keep_criteria = (
        '({field:s} >= {lower:.15e}) & ({field:s} <= {upper:0.15e})'.format(
            field=bin.name, lower=lower_edge, upper=upper_edge)
    )
    events_subset = events.applyCut(keep_criteria=keep_criteria)
    logging.trace('keep criteria:                 %s', keep_criteria)
    logging.trace('total events in that group:    %s',
                  len(events[repr_flavint][bin.name]))
    logging.trace('events in that group selected: %s',
                  len(events_subset[repr_flavint][bin.name]))

    return events_subset


def inf2finite(x):
    """Convert +/- infinities to largest/smallest representable numbers
    according to the current pisa.FTYPE"""
    return np.clip(x, a_min=np.finfo(FTYPE).min, a_max=np.finfo(FTYPE).max)


def fold_coszen_diff(coszen_diff, randomize=False):
    """Fold coszen difference above 1 down, and below -1 up.

    Parameters
    ----------
    coszen_diff
        Cosine-zenith difference, e.g. `reco_coszen - true_coszen`

    randomize : bool
        Randomizes the differences about 0, such that the full distribution of
        coszen diffs looks good to the eye when plotted against true coszen.
        This is not necessary, though, as a computational step (the underlying
        distribution of coszen diffs is the same with or without
        randomization).

    Returns
    -------
    folded_coszen_diffs

    """
    if randomize:
        rnd = np.random.RandomState()
        random_sign = rnd.choice((-1, +1), size=coszen_diff.shape)
        coszen_diff = coszen_diff * random_sign
        folded_coszen_diff = coszen_diff
    else:
        folded_coszen_diff = np.copy(coszen_diff)

    mask = coszen_diff > 1
    folded_coszen_diff[mask] = 2 - coszen_diff[mask]
    mask = coszen_diff < -1
    folded_coszen_diff[mask] = -2 - coszen_diff[mask]

    return folded_coszen_diff


@numba_jit(nogil=True)
def weight_coszen_tails(cz_diff, cz_bin_edges, input_weights=None):
    """Calculate weights that compensate for fewer points in the inherent tails
    of the coszen-difference (usually coszen-error) distribution.

    Parameters
    ----------
    cz_diff : array
        Cosine-zenith differences. E.g., `coszen_reco - coszen_true`

    cz_bin_edges : sequnce of two scalars
        Edges from the true-coszen bin in which the coszen differences were
        computed.

    input_weights : None or array of same size as `cz_diff`
        Existing weights that are to be multiplied by the tail weights to
        arrive at an overall weight for each event. If provided, must have same
        shape as `cz_diff`.

    Returns
    -------
    weights : array
    diff_limits : tuple of two scalars
        (diff_lower_lim, diff_upper_lim)

    """
    # Create all-ones weights vector if a weights field
    # hasn't been specified
    if input_weights is None:
        weights = np.ones_like(cz_diff)
    else:
        weights = np.copy(input_weights)

    # Shortcuts for accessing bin edges
    bin_lower_edge = np.min(cz_bin_edges)
    bin_upper_edge = np.max(cz_bin_edges)

    # Identify limits of possible diff distribution
    diff_lower_lim = -1 - bin_upper_edge
    diff_upper_lim = +1 - bin_lower_edge
    diff_limits = (diff_lower_lim, diff_upper_lim)

    # Identify inner limits of the tails
    lower_tail_upper_lim = -1 - bin_lower_edge
    upper_tail_lower_lim = +1 - bin_upper_edge

    # Identify tail widths
    lower_tail_width = lower_tail_upper_lim - diff_lower_lim
    upper_tail_width = diff_upper_lim - upper_tail_lower_lim

    # Create masks for events in the tails
    upper_tail_mask = cz_diff > upper_tail_lower_lim
    lower_tail_mask = cz_diff < lower_tail_upper_lim

    # Update the weights for events in the tails
    weights[lower_tail_mask] *= (
        lower_tail_width/(cz_diff[lower_tail_mask] - diff_lower_lim)
    )
    weights[upper_tail_mask] *= (
        upper_tail_width/(diff_upper_lim - cz_diff[upper_tail_mask])
    )

    return weights, diff_limits


def coszen_error_edges(true_edges, reco_edges):
    """Return a list of edges in coszen-error space given 2 true-coszen
    edges and reco-coszen edges. Systematics are not implemented at thistime.

    Parameters
    ----------
    true_edges : sequence of 2 scalars
    reco_edges : sequence of scalars
    bias : scalar // NOT IMPLEMENTED YET!
    scale : scalar > 0 // NOT IMPLEMENTED YET!

    Returns
    -------
    all_dcz_binedges : list of scalars
        The interleaved coszen-error (delta-cz) bin edges found both from the
        full reco range possible (given by `true_edges` and assuming reco can
        be from -1 to +1), and from the spans due to `reco_edges`.

    reco_indices : tuple with 2 lists of scalars
        Each list contains `len(reco_edges - 1)` scalars. Thesea are the
        indices for locating the edge in `all_binedges`, corresponding to
        `(reco_edges[:-1], reco_dges[1:])`.

    """
    n_reco_edges = len(reco_edges)
    reco_lower_binedges = reco_edges[:-1]
    reco_upper_binedges = reco_edges[1:]
    true_lower_binedge, true_upper_binedge = true_edges

    full_reco_range_lower_binedge = np.round(
        -1 - true_upper_binedge, EQUALITY_SIGFIGS
    )
    full_reco_range_upper_binedge = np.round(
        +1 - true_lower_binedge, EQUALITY_SIGFIGS
    )

    dcz_lower_binedges = np.round(
        reco_lower_binedges - true_upper_binedge, EQUALITY_SIGFIGS
    )
    dcz_upper_binedges = np.round(
        reco_upper_binedges - true_lower_binedge, EQUALITY_SIGFIGS
    )

    all_dcz_binedges, indices = np.unique(
        np.concatenate([
            [full_reco_range_lower_binedge],
            dcz_lower_binedges,
            dcz_upper_binedges,
            [full_reco_range_upper_binedge]
        ]),
        return_inverse=True
    )

    # Note: first index will be for `full_reco_range_lower_binedge`;
    # next `n_reco_edges - 1` edges correspond to `dcz_lower_binedges`
    # next `n_reco_edges - 1` edges correspond to `dcz_upper_binedges`, and
    # last index will be for `full_reco_range_upper_binedge`.
    dcz_lower_binedge_indices = indices[1:n_reco_edges]
    dcz_upper_binedge_indices = indices[n_reco_edges:-1]

    reco_indices = (dcz_lower_binedge_indices, dcz_upper_binedge_indices)

    return all_dcz_binedges, reco_indices


@numba_jit(nogil=True, nopython=True, fastmath=True)
def sorted_fast_histogram(a, bins, weights):
    """Fast but less precise histogramming of a sorted array with weights.

    Parameters
    ----------
    a : sorted array
    bins : sequence
    weights : array

    Returns
    -------
    hist, bin_edges

    """
    nbins = len(bins) - 1
    ndata = len(a)
    bin_min = bins[0]
    bin_max = bins[nbins]
    hist = np.zeros(nbins, np.float64)

    # Note that initialization of first bin is here, since set is monotonic
    lo = 0
    #for view_a, view_weights in zip(np.nditer(a), np.nditer(weights)):
    for idx in range(ndata):
        v_a = a[idx]
        if not bin_min <= v_a <= bin_max:
            # Value is out of bounds, ignore (this also catches NaNs)
            continue
        # Bisect in bins[:-1]
        hi = nbins - 1
        while lo < hi:
            # Note the `+ 1` is necessary to avoid an infinite
            # loop where mid = lo => lo = mid
            mid = (lo + hi + 1) >> 1
            if v_a < bins[mid]:
                hi = mid - 1
            else:
                lo = mid
        hist[lo] += weights[idx]
    return hist, bins


@numba_jit(nogil=True, nopython=True, fastmath=True)
def fast_histogram(a, bins, weights):
    """Fast but less precise histogramming of an array with weights.

    It is recommended that `a` and `weights` be sorted according to ascending
    `weights` to achieve the best numerical precision. This is due to the
    finite-precision effect whereby a small number added to a large number can
    have no effect. Therefore adding a sequence of small numbers, one at a
    time, to a large number also has no effect. However, adding the sequence of
    small numbers together before being added to the large number is more
    likely to account for all of the small numbers.

    Parameters
    ----------
    a : array
    bins : sequence
    weights : array

    Returns
    -------
    hist, bin_edges

    See Also
    --------
    sorted_fast_histogram
        Small speedup if `a` is sorted

    """
    nbins = len(bins) - 1
    ndata = len(a)
    bin_min = bins[0]
    bin_max = bins[nbins]
    hist = np.zeros(nbins, np.float64)

    # Note that initialization of first bin is here, since set is monotonic
    #for view_a, view_weights in zip(np.nditer(a), np.nditer(weights)):
    for idx in range(ndata):
        v_a = a[idx]
        if not bin_min <= v_a <= bin_max:
            # Value is out of bounds, ignore (this also catches NaNs)
            continue
        # Bisect in bins[:-1]
        lo = 0
        hi = nbins - 1
        while lo < hi:
            # Note the `+ 1` is necessary to avoid an infinite
            # loop where mid = lo => lo = mid
            mid = (lo + hi + 1) >> 1
            if v_a < bins[mid]:
                hi = mid - 1
            else:
                lo = mid
        hist[lo] += weights[idx]
    return hist, bins


if NUMBA_AVAIL:
    HIST_FUNC = fast_histogram
else:
    HIST_FUNC = np.histogram


class vbwkde(Stage):
    r"""
    From simulated events, a set of transforms are created which map
    bins of true events onto distributions of reconstructed events using
    variable-bandwidth kernel density estimation. These transforms can be
    accessed by [true_energy][true_coszen][reco_energy][reco_coszen].
    These distributions represent the probability that a true event
    (true_energy, true_coszen) with be reconstructed as (pid, reco_energy,
    reco_coszen).

    From these transforms and "true" event rate maps as inputs, calculates
    the reconstructed even rate maps.

    Parameters
    ----------
    params : ParamSet
        Must exclusively have parameters:

        reco_events : string or Events
            PISA Events object or filename/path to use to derive transforms, or
            a string specifying the resource location of the same.

        reco_weights_name : None or string
            Field to use in MC events to apply MC weighting for the computation

        transform_events_keep_criteria : None or string
            Additional cuts that are applied to events prior to computing
            transforms with them. E.g., "true_coszen <= 0" removes all MC-true
            downgoing events. See `pisa.core.events.Events` class for details
            on cut specifications.

        res_scale_ref : string
            One of "mean", "mode", or "zero". This is the reference point about
            which resolutions are scaled. "zero" scales about the zero-error
            point (i.e., the bin midpoint), "mean" scales about the mean of the
            KDE, and "mode" scales about the peak of the KDE.

        e_res_scale : float
            A scaling factor for energy resolutions.

        cz_res_scale : float
            A scaling factor for coszen resolutions.

        e_reco_bias : float

        cz_reco_bias : float

    particles : string
        Must be one of 'neutrinos' or 'muons' (though only neutrinos are
        supported at this time).

    input_names : string or list of strings
        Names of inputs expected. These should follow the standard PISA
        naming conventions for flavor/interaction types OR groupings
        thereof. Note that this service's outputs are named the same as its
        inputs. See Conventions section in the documentation for more info.

    transform_groups : string
        Specifies which particles/interaction types to combine together in
        computing the transforms. See Notes section for more details on how
        to specify this string

    sum_grouped_flavints : bool

    input_binning : MultiDimBinning or convertible thereto
        Input binning is in true variables, with names prefixed by "true_".
        Each must match a corresponding dimension in `output_binning`.

    output_binning : MultiDimBinning or convertible thereto
        Output binning is in reconstructed variables, which can include pid.

    min_num_events : int or mapping
        For KDEs, each bin is expanded up and down in true-energy as much as
        necessary to collect this many events. See `collect_enough_events` for
        more details.

    tgt_num_events : int or mapping
        Ideally each bin will have `tgt_num_events`. Allow the bin to expand up
        to a fraction times the bin's width (`tgt_max_binwidth_factors`) to try
        to hit this number. See `collect_enough_events` for more details.

    tgt_max_binwidth_factors : float
        Expand bin by up to this fraction in order to collect
        `tgt_num_events`. See `collect_enough_events` for more details.

    char_deps_downsampling : mapping
        For each dimension to be characterized, define the dimensions it
        depends upon. I.e., events will be collected in bins with dimensions of
        the defined dependencies. Note that the final dimension upon which
        there is a dependency is _not_ a strict binning. It's a starting event
        set, but in order to hit min_num_events or tgt_num_events, events from
        outside the boundaries of this dimension may be exceeded. Therefore,
        this should be the _least_ correlated with the dimension being
        characterized.

        Dimension names in either `char_binning` or `output_binning` may be
        included as dependencies, but if a binning is defined in both places,
        the binning in `char_binning` will be used.

        E.g.:
        ```
            {
                'pid': 'true_energy',
                'energy': ['pid', 'true_energy'],
                'coszen': ['pid', 'true_coszen', 'true_energy']
            }
        ```

    char_binning : mapping
        Binning used for characterizing each dimension. E.g.

        ```
            {
                'pid': MultiDimBinning([
                        dict(name='true_energy', num_bins=80, is_log=True,
                             domain=[1, 80]*ureg.GeV')
                    ]),
                'energy': MultiDimBinning([
                        dict(name='true_energy', num_bins=40, is_log=True,
                             domain=[1, 80]*ureg.GeV')
                    ]),
                'coszen': MultiDimBinning([
                        dict(name='true_coszen', num_bins=80, is_lin=True,
                             domain=[-1, 1], tex=r'\cos\,\theta_{\rm true}'),
                        dict(name='true_energy', num_bins=20, is_log=True,
                             domain=[1, 80]*ureg.GeV')
                    ])
            }
        ```

        It is not necessary to
        define a binning here that is the same as that defined in
        `output_binning`, but a different definition here takes precedence over
        `output_binning`.

        As for `char_deps_downsampling`, specify characterization dimensions by
        their basenames.

        See `CHAR_BINNING` for the default, and use this an example.

    disk_cache : bool, string, etc.
        Simplest to set to True or False for enabling/disabling disk caching,
        respectively, but other inputs are possible (see docs for
        `pisa.core.stage.Stage` class for more info). The KDE profiles are
        cached to disk by this service, _not_ the full transform (since the
        latter can be multiple GB, depending on input/output binning).

    transforms_cache_depth : int >= 0
        Default is 1 since transforms for this service can be huge (gigabytes)

    outputs_cache_depth : int >= 0
        Default is 20 since the outputs from this stage are generally not too
        large.

    memcache_deepcopy : bool

    debug_mode : None or bool
        Whether to run extra checks and store extra debug info

    Notes
    -----
    The `transform_groups` string is interpreted (and therefore defined) by
    pisa.utils.flavInt.flavint_groups_string. E.g. commonly one might use:

    'nue cc + nuebar cc, numu_cc+numubar_cc, nutau_cc+nutaubar_cc, nuall_nc+nuallbar_nc'

    Any particle type not explicitly mentioned in `transform_groups` is taken
    as a singleton group.

    Plus signs add types to a group, while groups are separated by commas.
    Whitespace is ignored.

    Input "true" event rate maps should be binned finely enough that the
    smearing that is removed when we lump all events in the char_binning
    together and KDE the reco error is not an issue.

    """
    def __init__(self, params, particles, input_names, transform_groups,
                 sum_grouped_flavints, input_binning, output_binning,
                 char_deps_downsampling, min_num_events, tgt_num_events,
                 tgt_max_binwidth_factors,
                 error_method=None,
                 disk_cache=True,
                 transforms_cache_depth=1,
                 outputs_cache_depth=20,
                 memcache_deepcopy=False,
                 debug_mode=None):
        # All of the following params (and no more) must be passed via the
        # `params` argument.
        expected_params = (
            'reco_events', 'reco_weights_name',
            'transform_events_keep_criteria',
            'res_scale_ref', 'e_res_scale', 'cz_res_scale',
            'e_reco_bias', 'cz_reco_bias'
        )

        #============================================================
        # Parse, translate, normalize, and/or validate init args...
        #============================================================

        # `particles` ...

        assert isinstance(particles, basestring)
        particles = particles.strip().lower()
        assert particles in ['neutrinos']
        self.particles = particles

        # `transform_groups` ...

        if isinstance(transform_groups, basestring):
            transform_groups = flavintGroupsFromString(transform_groups)
        elif transform_groups is None:
            transform_groups = flavintGroupsFromString('')
        self.transform_groups = transform_groups

        # `sum_grouped_flavints` ...

        assert isinstance(sum_grouped_flavints, bool)
        if not sum_grouped_flavints:
            raise NotImplementedError(
                'Grouped flavints must be summed at this time, as logic for'
                ' not doing so is not yet implemented.'
            )
        self.sum_grouped_flavints = sum_grouped_flavints

        # `char_deps_downsampling` ...

        if isinstance(char_deps_downsampling, basestring):
            char_deps_downsampling = literal_eval(char_deps_downsampling)

        new_dict = dict()
        for char_dim_name, deps in char_deps_downsampling.items():
            assert isinstance(char_dim_name, basestring)
            new_dict[char_dim_name] = OrderedDict()

            if isinstance(deps, basestring):
                char_deps_downsampling[char_dim_name] = [deps]
                new_dict[char_dim_name][deps] = 1
                continue

            if isinstance(deps, Sequence):
                if (len(deps) == 2 and isinstance(deps[0], basestring)
                        and isscalar(deps[1])):
                    new_dict[char_dim_name][deps[0]] = deps[1]
                    continue

                for subseq in deps:
                    if isinstance(subseq, basestring):
                        new_dict[char_dim_name][subseq] = 1
                    elif isinstance(subseq, Sequence):
                        assert isinstance(subseq[0], basestring)
                        if len(subseq) == 2:
                            assert isscalar(subseq[1])
                            new_dict[char_dim_name][subseq[0]] = subseq[1]
                        else:
                            new_dict[char_dim_name][subseq[0]] = 1
        char_deps_downsampling = new_dict

        # Until `collect_enough_events` is fixed to work with dimensions other
        # than `true_energy` as the final dimension, have to enforce this
        for dim_name, dependencies in char_deps_downsampling.iteritems():
            if dependencies.keys()[-1] != 'true_energy':
                raise NotImplementedError(
                    "Only 'true_energy' can come last in the list of"
                    " dependencies at this time."
                    " (Dimension: %s, dependencies supplied = %s)"
                    % (dim_name, dependencies.keys())
                )

        # `min_num_events` ...

        if isinstance(min_num_events, basestring):
            min_num_events = literal_eval(min_num_events)

        if isscalar(min_num_events):
            min_num_events = {d: int(min_num_events)
                              for d in char_deps_downsampling.keys()}

        assert isinstance(min_num_events, Mapping), str(min_num_events)

        # `tgt_num_events` ...

        if isinstance(tgt_num_events, basestring):
            tgt_num_events = literal_eval(tgt_num_events)

        if isscalar(tgt_num_events):
            tgt_num_events = {d: int(tgt_num_events)
                              for d in char_deps_downsampling.keys()}

        assert isinstance(tgt_num_events, Mapping), str(tgt_num_events)

        # `tgt_max_binwidth_factors` ...

        if isinstance(tgt_max_binwidth_factors, basestring):
            tgt_max_binwidth_factors = literal_eval(tgt_max_binwidth_factors)

        if isscalar(tgt_max_binwidth_factors):
            tgt_max_binwidth_factors = {
                d: float(tgt_max_binwidth_factors)
                for d in char_deps_downsampling.keys()
            }

        assert isinstance(tgt_max_binwidth_factors, Mapping), \
                str(max_binwidth_factor)

        # `input_names` ...

        if isinstance(input_names, basestring):
            input_names = (''.join(input_names.split(' '))).split(',')

        # Define the names of objects expected in inputs and produced as
        # outputs
        if self.particles == 'neutrinos':
            if self.sum_grouped_flavints:
                output_names = [str(g) for g in self.transform_groups]
            else:
                output_names = input_names
        elif self.particles == 'muons':
            raise NotImplementedError('`particles` = %s not implemented.'
                                      % self.particles)
        else:
            raise ValueError('Invalid value for `particles`: "%s"'
                             % self.particles)

        # Invoke the init method from the parent class, which does a lot of
        # work for you.
        super(self.__class__, self).__init__(
            use_transforms=True,
            params=params,
            expected_params=expected_params,
            input_names=input_names,
            output_names=output_names,
            error_method=error_method,
            disk_cache=disk_cache,
            outputs_cache_depth=outputs_cache_depth,
            transforms_cache_depth=transforms_cache_depth,
            memcache_deepcopy=memcache_deepcopy,
            input_binning=input_binning,
            output_binning=output_binning,
            debug_mode=debug_mode
        )

        # We have some number of dimensions to characterize (KDE). Each one of
        # these has "dimension dependencies," whereby we need to isolate events
        # in _those_ dimensions before we can characterize the dimension. By
        # isolating events as such, we reduce or remove correlations among the
        # involved dimensions. Ego we reduce the dimensionality that needs to
        # be characterized and so can get away with fewer events for the
        # characterization. (This also means that we can simply use
        # one-dimensional KDE for the characterization.)

        self.char_binning = dict()
        """dict formatted as `{out_dim.basename: MultiDimBinning}`, used
        for binning events for characterizing resolutions (namely via VBWKDE)
        in the output dimensions"""

        self.min_num_events = dict()
        """min_num_events for each dimension dependency for each dimension
        being characterized"""

        self.tgt_num_events = dict()
        """tgt_num_events for each dimension dependency for each dimension
        being characterized"""

        self.tgt_max_binwidth_factors = dict()
        """tgt_max_binwidth_factors for each dimension dependency for each
        dimension being characterized"""

        # Only characterize dimensions that are to be output
        for out_dim in output_binning.dims:
            char_dim_name = out_dim.basename

            # Populate class attrs with info for this char dimension
            self.min_num_events[char_dim_name] = min_num_events[char_dim_name]
            self.tgt_num_events[char_dim_name] = tgt_num_events[char_dim_name]
            self.tgt_max_binwidth_factors[char_dim_name] = (
                tgt_max_binwidth_factors[char_dim_name]
            )

            # Construct the binning for this char dimension
            dep_binnings = []
            for dep_dim_name, downsamp_factor \
                    in char_deps_downsampling[char_dim_name].items():
                if dep_dim_name in input_binning:
                    dep_binnings.append(
                        input_binning[dep_dim_name].downsample(downsamp_factor)
                    )
                elif dep_dim_name in output_binning:
                    dep_binnings.append(
                        output_binning[dep_dim_name].downsample(downsamp_factor)
                    )
                else:
                    raise ValueError('Dimension "%s" was neither found in'
                                     ' `input_binning` nor in `output_binning`'
                                     % dep_dim_name)

            self.char_binning[char_dim_name] = MultiDimBinning(dep_binnings)

        self.include_attrs_for_hashes('particles')
        self.include_attrs_for_hashes('transform_groups')
        self.include_attrs_for_hashes('sum_grouped_flavints')
        self.include_attrs_for_hashes('char_binning')
        self.include_attrs_for_hashes('min_num_events')
        self.include_attrs_for_hashes('tgt_num_events')
        self.include_attrs_for_hashes('tgt_max_binwidth_factors')

        self.kde_profiles = dict()
        """dict containing `KDEProfile`s. Structure is:
            {dim_basename: {flavintgroup: {(Coord): (KDEProfile)}}}

        For example:
            {'pid': {
                numu_cc: {
                    (true_energy=0): (x=[...], counts=[...]),
                    (true_energy=1): (x=[...], counts=[...])
                },
                numubar_cc: {
                    (true_energy=0): (x=[...], counts=[...]),
                    (true_energy=1): (x=[...], counts=[...])
                }},
             'energy': {
                numu_cc: {
                    (pid=0, true_energy=0): (x=[...], counts=[...]),
                    (pid=0, true_energy=1): (x=[...], counts=[...]),
                    (pid=1, true_energy=0): (x=[...], counts=[...]),
                    (pid=1, true_energy=1): (x=[...], counts=[...])
                },
                numubar_cc: {
                    (pid=0, true_energy=0): (x=[...], counts=[...]),
                    (pid=0, true_energy=1): (x=[...], counts=[...]),
                    (pid=1, true_energy=0): (x=[...], counts=[...]),
                    (pid=1, true_energy=1): (x=[...], counts=[...])
                }}
            }
        """

        self._kde_profiles_lock = threading.Lock()

        self._kde_hashes = dict()

        self.xform_kernels = dict()
        """Storage of the N-dim smearing kernels, one per flavintgroup"""

        self._xform_kernels_lock = threading.Lock()

    def validate_binning(self):
        """Require input dimensions of "true_energy" and "true_coszen" (in any
        order).

        Require output dimensions of "reco_energy" and "reco_coszen", and
        optionally allow output dimension of "pid"; can be in any order.

        """
        input_names = set(self.input_binning.names)
        assert input_names == set(['true_energy', 'true_coszen']), \
                str(input_names)

        output_names = set(self.output_binning.names)
        outs1 = set(['reco_energy', 'reco_coszen'])
        outs2 = set(['reco_energy', 'reco_coszen', 'pid'])
        assert output_names == outs1 or output_names == outs2

        input_basenames = set(self.input_binning.basenames)
        output_basenames = set(self.output_binning.basenames)
        for base_d in input_basenames:
            assert base_d in output_basenames

    def _compute_transforms(self):
        """Generate reconstruction smearing kernels by estimating the
        distribution of reconstructed events corresponding to each bin of true
        events using VBW-KDE.

        The resulting transform is an MxN-dimensional histogram, where M is the
        dimensionality of the input binning and N is the dimensionality of the
        output binning. The transform maps the truth bin counts to the
        reconstructed bin counts.

        I.e., for the case of 1D input binning, the ith element of the
        reconstruction kernel will be a map showing the distribution of events
        over all the reco space from truth bin i. This will be normalised to
        the total number of events in truth bin i.

        """
        self.load_events(self.params.reco_events)
        self.cut_events(self.params.transform_events_keep_criteria)
        self.characterize_resolutions()
        self.generate_all_kernels()

        xforms = []
        for xform_flavints in self.transform_groups:
            xform_input_names = []
            for input_name in self.input_names:
                input_flavs = NuFlavIntGroup(input_name)
                if len(set(xform_flavints).intersection(input_flavs)) > 0:
                    xform_input_names.append(input_name)

            for output_name in self.output_names:
                if output_name not in xform_flavints:
                    continue
                xform = BinnedTensorTransform(
                    input_names=xform_input_names,
                    output_name=output_name,
                    input_binning=self.input_binning,
                    output_binning=self.output_binning,
                    xform_array=self.xform_kernels[xform_flavints].hist,
                    sum_inputs=True
                )
                xforms.append(xform)

        return TransformSet(transforms=xforms)

    def characterize_resolutions(self):
        """Compute the KDEs for each (pid, E) bin. If PID is not present, this
        is just (E). The results are propagated to each (pid, E, cz) bin, as
        the transforms are assumed to not be cz-dependent.

        The results are cached to disk and simply loaded from that cache to
        avoid re-computation.

        """
        weights_name = self.params.reco_weights_name.value

        # TODO: add sourcecode hash for pisa.utils.vbwkde module (entire module
        #       is probably safest, due to all the functions there)

        hash_items = [FTYPE, self.source_code_hash, self.events.hash,
                      self.transform_groups, self.particles,
                      self.sum_grouped_flavints]

        for char_dim, dep_dims_binning in self.char_binning.items():
            logging.debug('Working on KDE dimension "%s"', char_dim)
            new_hash = hash_obj(deepcopy(hash_items) + [dep_dims_binning.hash])

            # See if we already have correct kde_profiles for this dim
            if (char_dim in self._kde_hashes
                    and new_hash == self._kde_hashes[char_dim]):
                logging.debug('  > Already have KDEs for "%s"', char_dim)
                continue

            # Try to load from disk cache
            if self.disk_cache is not None:
                try:
                    if new_hash in self.disk_cache:
                        logging.debug(
                            '  > Loading KDEs for "%s" from disk cache',
                            char_dim
                        )
                        self.kde_profiles[char_dim] = self.disk_cache[new_hash]
                        self._kde_hashes[char_dim] = new_hash
                        continue
                except:
                    logging.error('Loading from disk cache failed.')

            # Reset the hash for this dim so if anything fails below, the wrong
            # info won't be loaded
            self._kde_hashes[char_dim] = None

            # Clear out all previous kde info
            self.kde_profiles[char_dim] = OrderedDict()
            for flavintgroup in self.transform_groups:
                self.kde_profiles[char_dim][flavintgroup] = OrderedDict()

            if FTYPE == np.float64:
                ftype_bytes = 8
            elif FTYPE == np.float32:
                ftype_bytes = 4

            sizeof_kde_profiles = 0

            # First element in the list is the full set of events. Then the
            # next element is the previous element's events but with inbounds
            # cut from the first dim applied, and the next element has second
            # dim's inbounds cut applied to the first dim's inbounds cut, and
            # so forth.
            #
            # The final dimensional dependency is _not_ cut on, allowing for
            # events to be collected exceeding its boundaries so that enough
            # statistics can be acquired.
            cut_events = [self.events] * len(dep_dims_binning)

            for bin_num, bin_binning in enumerate(dep_dims_binning.iterbins()):
                bin_dims = bin_binning.dims
                bin_coord = dep_dims_binning.index2coord(bin_num)
                logging.debug('  > characterizing bin %s (%d of %d)',
                              bin_coord, bin_num+1, dep_dims_binning.size)

                # Apply cuts for all but the last dimension dependency
                for dim_num, dim in enumerate(bin_dims[:-1]):
                    if dim.inbounds_criteria not in cut_events[dim_num + 1]:
                        # Apply this dim cut to "parent" dim's events. Note
                        # that indices are off by 1, so this dim's events are
                        # stored in element dim_num + 1 and its parent is at
                        # [dim_num]
                        cut_events[dim_num + 1] = (
                            cut_events[dim_num].applyCut(dim.inbounds_criteria)
                        )

                for flavintgroup in self.transform_groups:
                    logging.trace('    > flavintgroup = %s', flavintgroup)
                    repr_flavint = flavintgroup[0]

                    flav_events = collect_enough_events(
                        events=cut_events[-1],
                        flavint=repr_flavint,
                        bin=bin_dims[-1],
                        min_num_events=self.min_num_events[char_dim],
                        tgt_num_events=self.tgt_num_events[char_dim],
                        tgt_max_binwidth_factor=self.tgt_max_binwidth_factors[char_dim]
                    )[repr_flavint]

                    if weights_name in flav_events:
                        weights = flav_events[weights_name]
                        weights = weights * (len(weights)/np.sum(weights))
                    else:
                        weights = None

                    if char_dim == 'pid':
                        feature = flav_events['pid']
                        fmin, fmax = min(feature), max(feature)
                        half_width = (fmax - fmin)/2
                        lowerlim = fmin - half_width
                        upperlim = fmax + half_width
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(5e3))
                        )

                    elif char_dim == 'energy':
                        feature = np.log(flav_events['reco_energy']
                                         / flav_events['true_energy'])
                        fmin, fmax = min(feature), max(feature)
                        lowerlim = fmin
                        upperlim = fmax
                        # Note that this only evaluates the KDE profile within
                        # the range of datapoints, so as to not extrapolate
                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=lowerlim, max=upperlim,
                            evaluate_at=np.linspace(lowerlim, upperlim,
                                                    int(5e3))
                        )

                    elif char_dim == 'coszen':
                        feature = (flav_events['reco_coszen']
                                   - flav_events['true_coszen'])
                        weights, error_limits = weight_coszen_tails(
                            cz_diff=feature,
                            cz_bin_edges=bin_binning.true_coszen.bin_edges.m,
                            input_weights=weights
                        )
                        weights = weights * (len(weights)/np.sum(weights))

                        # TODO: try the following to fix the tails falling off
                        # too abruptly:
                        # 1. Simply mirror half the points about the error
                        #    limits, KDE, then take the central portion
                        # 2. Mirror about mode (but only place "new" datapoints
                        #    *outside* the current error limits)
                        # 3. Evaluate KDE as now, but evaluate a range outside
                        #    the allowed limits; fold the shapes in by
                        #    reflecting at the limits and adding this in.

                        # Trying combination of methods 1+3 now: compute
                        # bandwidths with half of dataset mirrored about upper
                        # limit, and half mirrored about lower limit. Then only
                        # evaluate gaussians attached datapoints within the
                        # limits, but fold their tails in at the limits & sum

                        error_width = error_limits[1] - error_limits[0]
                        lower_mask = feature <= error_limits[0] + error_width/2
                        upper_mask = feature > error_limits[0] + error_width/2

                        orig_feature = feature
                        orig_weights = weights

                        feature_to_cat = [orig_feature]
                        weights_to_cat = [orig_weights]
                        if np.sum(lower_mask) > 0:
                            feature_to_cat.append(2*error_limits[0] - feature[lower_mask])
                            weights_to_cat.append(weights[lower_mask])
                        if np.sum(upper_mask) > 0:
                            feature_to_cat.append(2*error_limits[1] - feature[upper_mask])
                            weights_to_cat.append(weights[upper_mask])
                        feature = np.concatenate(feature_to_cat)
                        weights = np.concatenate(weights_to_cat)

                        extended_lower_lim = error_limits[0] - 0.5*error_width
                        extended_upper_lim = error_limits[1] + 0.5*error_width

                        vbwkde_kwargs = dict(
                            n_dct=int(2**6),
                            min=extended_lower_lim,
                            max=extended_upper_lim,
                            evaluate_at=None
                        )

                    else:
                        raise NotImplementedError(
                            'Applying KDEs to dimension "%s" is not'
                            ' implemented.' % char_dim
                        )

                    bw, x, counts = vbwkde_func(
                        feature, weights=weights, **vbwkde_kwargs
                    )

                    if char_dim == 'coszen':
                        x = np.linspace(extended_lower_lim, extended_upper_lim,
                                        5e4)
                        counts = gaussians(x=x, mu=orig_feature,
                                           sigma=bw[:len(orig_feature)],
                                           weights=orig_weights)

                        mirrored_length = len(x) // 4
                        below_range_mask = x < error_limits[0]
                        above_range_mask = x > error_limits[1]
                        in_range_mask = ~(below_range_mask | above_range_mask)

                        x_in_range = x[in_range_mask]
                        counts_in_range = counts[in_range_mask]

                        counts_in_range[:mirrored_length] += (
                            counts[below_range_mask][-mirrored_length:][::-1]
                        )
                        counts_in_range[-mirrored_length:] += (
                            counts[above_range_mask][:mirrored_length][::-1]
                        )

                        x = x_in_range
                        counts = counts_in_range

                    # Sort according to ascending weight to improve numerical
                    # precision of "poor-man's" histogram
                    sortind = counts.argsort()
                    x = x[sortind]
                    counts = counts[sortind]

                    self.kde_profiles[char_dim][flavintgroup][bin_coord] = (
                        KDEProfile(x=x, counts=counts)
                    )

                    sizeof_kde_profiles += 2*len(x) * ftype_bytes

            self._kde_hashes[char_dim] = new_hash

            if self.disk_cache is not None:
                try:
                    self.disk_cache[new_hash] = self.kde_profiles[char_dim]
                except:
                    logging.error(
                        'Failed to write KDE profiles for dimension %s'
                        ' (%d bytes) to disk cache; skipping caching.',
                        char_dim, sizeof_kde_profiles
                    )

    def generate_all_kernels(self):
        """Dispatches `generate_single_kernel` for all specified transform
        flavintgroups, in parallel if that is possible."""
        if not NUMBA_AVAIL or OMP_NUM_THREADS == 1:
            for flavintgroup in self.transform_groups:
                self.generate_single_kernel(flavintgroup)
            return

        parallel_run(
            func=self.generate_single_kernel,
            kind='threads',
            num_parallel=OMP_NUM_THREADS,
            scalar_func=True,
            divided_args_mask=None,
            divided_kwargs_names=['flavintgroup'],
            flavintgroup=self.transform_groups
        )

    def generate_single_kernel(self, flavintgroup):
        """Construct a smearing kernel for the flavintgroup specified.

        The resulting array can be indexed for clarity using two indexes,
        one for input and one for output dimensions. E.g.:
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l]
        or if PID is included,
           kernel[true_energy_i, true_coszen_j][reco_energy_k, reco_coszen_l, pid_m]
        where ordering within the two sets of brackets is arbitrary.

        In other words, the indices point from a single MC-true histogram bin
        indexed by (i,j) to a single reco histogram bin indexed by (k,l{,m}).

        Parameters
        ----------
        flavintgroup

        Returns
        -------
        kernel : MxN-dimensional array of float
            Mapping from the number of events in each bin of the 2D
            MC-true-events histogram to the number of events reconstructed in
            each bin of the 2D reconstructed-events histogram. Dimensions are
                input_binning.shape x output_binning.shape
            Note that PID binning can optionally be included, and behaves as
            another output (reco) dimension.

        """
        logging.debug('Generating smearing kernel for %s', flavintgroup)

        pid_kde_profiles = self.kde_profiles['pid'][flavintgroup]
        e_kde_profiles = self.kde_profiles['energy'][flavintgroup]
        dcz_kde_profiles = self.kde_profiles['coszen'][flavintgroup]

        char_binning = self.char_binning

        # Events in each input bin will be smeared out into all output bins.
        # To characterize the smearing, get the KDE profile from each input
        # dimension that was created from events closest to this input bin.

        kernel_binning = self.input_binning * self.output_binning
        kernel = kernel_binning.empty(name='kernel')

        # Shortcut names
        true_energy = self.input_binning.true_energy
        true_coszen = self.input_binning.true_coszen
        reco_energy = self.output_binning.reco_energy
        reco_coszen = self.output_binning.reco_coszen
        pid_binning = self.output_binning.pid

        # Get the following once so we don't have to repeat within the loops

        pid_kde_e_centers = inf2finite(
            char_binning['pid'].true_energy.weighted_centers.m
        )
        e_kde_e_centers = inf2finite(
            char_binning['energy'].true_energy.weighted_centers.m
        )
        cz_kde_e_centers = char_binning['coszen'].true_energy.weighted_centers.m
        cz_kde_cz_centers = (
            char_binning['coszen'].true_coszen.weighted_centers.m
        )

        num_pid_bins = len(pid_binning)

        e_res_scale = self.params.e_res_scale.value.m

        # TODO: implement these systematics
        cz_res_scale = self.params.cz_res_scale.value.m
        cz_reco_bias = self.params.cz_reco_bias.value.m

        pid_edges = inf2finite(pid_binning.bin_edges.m)

        # NOTE: when we get scaling-about-the-mode working, will have to change
        # this.
        reco_e_edges = (
            np.log(inf2finite(reco_energy.bin_edges.m)
                   - self.params.e_reco_bias.value.m)
            / e_res_scale
        )

        reco_cz_edges = reco_coszen.bin_edges.m
        if self.debug_mode:
            assert np.all(np.isfinite(reco_cz_edges)), str(reco_cz_edges)

        # Compute info for the delta-coszen bin edges, for each true-coszen
        # input bin and each reco-coszen output bin; also get the bin edges
        # across the entire _possible_ reco-coszen range, for purposes of
        # normalization.

        # NOTE: reco-coszen systematics will be applied to the reco-coszen
        # bin edges, but (for proper treatment) must know the mode location of
        # the coszen KDE profile (which is only known in the innermost loop).
        # The bias isn't costly to do, but scaling will require shifting the
        # mode to 0, applying the scale factor, and then shifting back (or
        # shifting by the bias). So we need to compute as much here as possible
        # (since it's relatively cheap) but we may still need to do some of the
        # work within the innermost loop (yuck).

        true_e_centers = true_energy.weighted_centers.m
        true_cz_centers = true_coszen.weighted_centers.m
        true_cz_edges = true_coszen.bin_edges.m
        true_cz_edge_pairs = [(e0, e1) for e0, e1 in zip(true_cz_edges[:-1],
                                                         true_cz_edges[1:])]

        allbins_dcz_edge_info = []
        cz_closest_cz_indices = []
        for center, true_edgetuple in zip(true_cz_centers,
                                          true_coszen.iteredgetuples()):
            all_dcz_binedges, cz_reco_indices = coszen_error_edges(
                true_edges=true_edgetuple, reco_edges=reco_cz_edges
            )

            # Note that the final (uppermost) edge doesn't matter as it does
            # not define the *start* of a range.
            edge_counts = []
            for idx in range(len(all_dcz_binedges) - 1):
                count = 0
                for rng in zip(*cz_reco_indices):
                    if idx >= rng[0] and idx < rng[1]:
                        count += 1
                edge_counts.append(count)

            allbins_dcz_edge_info.append(
                dict(all_dcz_binedges=all_dcz_binedges,
                     cz_reco_indices=cz_reco_indices,
                     edge_counts=np.array(edge_counts))
            )

            cz_closest_cz_indices.append(
                np.argmin(np.abs(center - cz_kde_cz_centers))
            )

        for true_e_bin_num, true_e_center in enumerate(true_e_centers):
            logging.debug('  > Working on true_e_bin_num %d of %d',
                          true_e_bin_num+1, true_energy.size)

            idx = np.argmin(np.abs(np.log(true_e_center/pid_kde_e_centers)))
            pid_closest_kde_coord = char_binning['pid'].Coord(true_energy=idx)

            # Figure out PID fractions
            pid_kde_profile = pid_kde_profiles[pid_closest_kde_coord]

            pid_norm = 1/np.sum(pid_kde_profile.counts)
            pid_counts, _ = HIST_FUNC(
                pid_kde_profile.x, weights=pid_kde_profile.counts,
                bins=pid_edges
            )
            pid_fractions = pid_norm * pid_counts

            if self.debug_mode:
                assert np.all(pid_fractions >= 0), str(pid_fractions)
                assert np.all(pid_fractions <= 1), str(pid_fractions)
                assert np.sum(pid_fractions) < 1 + 10*EPSILON, \
                        str(pid_fractions)

            for pid_bin_num in range(num_pid_bins):
                pid_fraction = pid_fractions[pid_bin_num]

                energy_indexer = kernel_binning.defaults_indexer(
                    true_energy=true_e_bin_num,
                    pid=pid_bin_num
                )

                # If PID is zero, no need to figure out anything further for
                # (..., true_energy=this, ..., pid=this, ...) bins
                if not np.any(np.abs(pid_fraction) > EPSILON):
                    kernel[energy_indexer] = 0
                    continue

                # Get the energy smearing for this (PID, true-energy) bin
                closest_e_idx = np.argmin(
                    np.abs(np.log(true_e_center / e_kde_e_centers))
                )
                e_closest_kde_coord = char_binning['energy'].Coord(
                    pid=pid_bin_num,
                    true_energy=closest_e_idx
                )
                e_kde_profile = e_kde_profiles[e_closest_kde_coord]

                # TODO: scale about the mode of the KDE! i.e., implement
                #       `res_scale_ref`

                # Tranform reco-energy bin edges into the log-ratio space
                # where we characterized the energy resolutions
                e_edges = reco_e_edges - np.log(true_e_center)/e_res_scale

                energy_norm = 1 / np.sum(e_kde_profile.counts)

                reco_energy_counts, _ = HIST_FUNC(
                    e_kde_profile.x, weights=e_kde_profile.counts,
                    bins=e_edges
                )
                reco_energy_fractions = energy_norm * reco_energy_counts

                if self.debug_mode:
                    assert np.all(reco_energy_fractions >= 0), \
                            str(reco_energy_fractions)
                    assert np.all(reco_energy_fractions <= 1), \
                            str(reco_energy_fractions)
                    assert np.sum(reco_energy_fractions < 1 + 10*EPSILON), \
                            str(reco_energy_fractions)

                # pid and true_energy are covered by the `energy_indexer`;
                # then we broadcast reco_energy to
                # (true_coszen, reco_coszen, reco_energy)
                kernel[energy_indexer] = kernel_binning.broadcast(
                    pid_fraction * reco_energy_fractions,
                    from_dim='reco_energy',
                    to_dims=['true_coszen', 'reco_coszen']
                )

                # Do this just once for the energy bin, prior to looping over
                # coszen
                cz_closest_e_idx = np.argmin(np.abs(
                    np.log(true_e_center / cz_kde_e_centers)
                ))

                # Get the closest coszen smearing for this
                # (PID, true-coszen, true-energy) bin

                # TODO: implement `res_scale_ref` and `cz_reco_bias`! Note that
                # this is why `true_cz_lower_edge` and `true_cz_upper_edge` are
                # enumerated over in the below loop, since these will be
                # necessary to implement the systamtic(s).
                for true_cz_bin_num, (true_cz_lower_edge, true_cz_upper_edge) \
                        in enumerate(true_cz_edge_pairs):
                    cz_closest_cz_idx = cz_closest_cz_indices[true_cz_bin_num]
                    cz_closest_kde_coord = char_binning['coszen'].Coord(
                        pid=pid_bin_num,
                        true_coszen=cz_closest_cz_idx,
                        true_energy=cz_closest_e_idx
                    )

                    # Get KDE profile (in "delta-cz" space)
                    dcz_kde_profile = dcz_kde_profiles[cz_closest_kde_coord]

                    dcz_edge_info = allbins_dcz_edge_info[true_cz_bin_num]

                    reco_coszen_counts, _ = HIST_FUNC(
                        dcz_kde_profile.x, weights=dcz_kde_profile.counts,
                        bins=dcz_edge_info['all_dcz_binedges']
                    )

                    # Collect the relevant hist sections to describe each
                    # quantity of interest, starting with normalization
                    reco_indices = dcz_edge_info['cz_reco_indices']

                    coszen_norm = 1 / np.sum(
                        reco_coszen_counts * dcz_edge_info['edge_counts']
                    )

                    reco_coszen_fractions = []
                    for reco_lower, reco_upper in zip(*reco_indices):
                        reco_coszen_fractions.append(
                            coszen_norm * np.sum(
                                reco_coszen_counts[reco_lower:reco_upper]
                            )
                        )
                    reco_coszen_fractions = np.array(reco_coszen_fractions)

                    if self.debug_mode:
                        assert np.all(reco_coszen_fractions <= 1), \
                                str(reco_coszen_fractions)
                        assert np.all(reco_coszen_fractions >= 0), \
                                str(reco_coszen_fractions)
                        assert np.sum(reco_coszen_fractions) < 1 + 10*EPSILON, \
                                str(reco_coszen_fractions)

                    # Here we index directly into (i.e. the smearing profile
                    # applies direclty to) a single
                    # `(true_energy, true_coszen, pid)` coordinate.
                    coszen_indexer = kernel_binning.defaults_indexer(
                        true_energy=true_e_bin_num,
                        true_coszen=true_cz_bin_num,
                        pid=pid_bin_num
                    )

                    # At that coordinate, we broadcast the information from
                    # the `reco_coszen` dimension into the entire `reco_energy`
                    # dimension.
                    kernel.hist[coszen_indexer] *= kernel_binning.broadcast(
                        reco_coszen_fractions,
                        from_dim='reco_coszen',
                        to_dims=['reco_energy']
                    )

        with self._xform_kernels_lock:
            self.xform_kernels[flavintgroup] = kernel
