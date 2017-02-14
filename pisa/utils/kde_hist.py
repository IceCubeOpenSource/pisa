
from kde.cudakde import gaussian_kde
import numpy as np
from uncertainties import unumpy as unp

from pisa.core.binning import OneDimBinning, MultiDimBinning


__all__ = ['get_hist', 'kde_histogramdd', 'test_kde_histogramdd']


# TODO:
# * Module-level docstring
# * Docstring for `get_hist` function
# * Handle zenith like coszen? Or better: Define set of variables to perform
#   reflection on and reflection parameters (e.g. `reflect_fract` or somesuch
#   to stand in for for `coszen_reflection` and `reflect_dims` as standin for
#   `coszen_name`; also need some way to specify whether to reflect about lower
#   and/or upper edge); each such parameter can either be a single value, or a
#   sequence with one value per variable.
# * Any good reason for 0.25 and 'scott' defaults? If not, don't define a
#   default and force the user to explicitly set this when function is called.

def get_hist(sample, binning, weights=None, bw_method='scott', adaptive=True,
             alpha=0.3, use_cuda=False, coszen_reflection=0.25,
             coszen_name='coszen', oversample=1):
    if weights is None:
        weights = []

    if len(weights) == 0:
        norm = sample.shape[0]
    else:
        # TODO: make explicit axis reference in sum
        norm = np.sum(weights, axis=0)

    binning = binning.oversample(oversample)

    # Flip around to satisfy the kde implementation
    x = sample.T

    # Must have same amount of dimensions as binning dimensions
    assert x.shape[0] == len(binning)

    # TODO: What if coszen isn't in binning? Does this fail?
    cz_bin = binning.names.index(coszen_name)

    # Normal hist

    # Swap out cz bin to first place (index 0)
    if cz_bin != 0:
        # Also swap binning:
        new_binning = [binning[coszen_name]]
        for b in binning:
            if b.name != coszen_name:
                new_binning.append(b)
        binning = MultiDimBinning(new_binning)
        x[[0, cz_bin]] = x[[cz_bin, 0]]

    # Check if edge needs to be reflected
    reflect_lower = binning[coszen_name].bin_edges[0] == -1
    reflect_upper = binning[coszen_name].bin_edges[-1] == 1

    # Get the kernel weights
    kernel_weights_adaptive = gaussian_kde(
        x, weights=weights, bw_method=bw_method, adaptive=adaptive,
        alpha=alpha, use_cuda=use_cuda
    )

    # Get the bin centers, where we're going to evaluate the KDEs, and extend
    # the bin range for reflection
    bin_points = []
    for b in binning:
        c = unp.nominal_values(b.weighted_centers)
        if b.name == coszen_name:
            # how many bins to add for reflection
            l = int(len(c)*coszen_reflection)
            if reflect_lower:
                c0 = 2*c[0] - c[1:l+1][::-1]
            else:
                c0 = []
            if reflect_upper:
                c1 = 2*c[-1] - c[-l-1:-1][::-1]
            else:
                c1 = []
            c = np.concatenate([c0, c, c1])
        bin_points.append(c)

    # Shape including reflection edges
    megashape = (
        binning.shape[0] + (int(reflect_upper)+int(reflect_lower))*l,
        binning.shape[1]
    )

    # Shape of the reflection edges alone
    minishape = (binning.shape[0] - l, binning.shape[1])

    # Create a set of points
    grid = np.meshgrid(*bin_points, indexing='ij')
    points = np.array([g.ravel() for g in grid])

    # Evaluate KDEs at given points
    hist = kernel_weights_adaptive(points)

    # Reshape 1d array into nd
    hist = hist.reshape(megashape)

    # Cut off the reflection edges, mirror them, fill up remaining space with
    # zeros and add to histo
    if reflect_lower:
        hist0 = hist[0:l, :]
        hist0_0 = np.zeros(minishape)
        hist0 = np.flipud(np.concatenate([hist0_0, hist0]))
        hist = hist[l:, :]
    else:
        hist0 = 0

    if reflect_upper:
        hist1 = hist[-l:, :]
        hist1_0 = np.zeros(minishape)
        hist1 = np.flipud(np.concatenate([hist1, hist1_0]))
        hist = hist[:-l, :]
    else:
        hist1 = 0

    hist = hist + hist1 + hist0

    # Bin volumes
    volume = binning.bin_volumes(attach_units=False)
    hist = hist*volume

    # Downsample
    if oversample != 1:
        for i, b in enumerate(binning):
            hist = np.add.reduceat(
                hist,
                np.arange(0, len(b.bin_edges)-1, oversample),
                axis=i
            )

    # Swap back the axes
    if cz_bin != 0:
        hist = np.swapaxes(hist, 0, cz_bin)

    # TODO: either this is needed or not, don't leave commented-out code
    #hist = hist/np.sum(hist)*norm
    return hist*norm


# TODO:
# * Stack_pid parameter seems oddly named, not clearly documented
# * The body of this function seems like it could be cleaned up with methods on
#   the binning object which should make working with things like this more
#   generic, though I haven't thought through the whole problem
# * Explicitly call functions with named parameters, especially when there are
#   more than just a couple

def kde_histogramdd(sample, binning, weights=None, bw_method='scott',
                    adaptive=True, alpha=0.3, use_cuda=False,
                    coszen_reflection=0.25, coszen_name='coszen', oversample=1,
                    stack_pid=True):
    """Run kernel density estimation (KDE) for an array of data points, and
    then evaluate them on a histogram-like grid to effectively produce a
    histogram-like output.

    Based on Sebastian Schoenen's KDE implementation:
    http://code.icecube.wisc.edu/svn/sandbox/schoenen/kde

    Parameters
    ----------
    sample : array
        Shape (N_evts, vars), with vars in the right order corresponding to the
        binning order.

    binning : MultiDimBinning

    weights : None or array
        Same shape as `sample`

    bw_method: string
        'scott' or 'silverman' (see kde module)

    adaptive : bool
        (see kde module)

    alpha : float
        A parameter for the KDEs (see kde module)

    use_cuda : bool
        Run on GPU (only works with <= 2d)

    coszen_reflection : float
        Part (number between 0 and 1) of binning that is reflected at the
        coszen -1 and 1 edges

    coszen_name : string
        Binning name to identify the coszen bin that needs to undergo special
        treatment for reflection

    oversample : int
        Evaluate KDE at more points per bin, takes longer, but is more accurate

    stack_pid : bool
        Treat each pid bin separately, not as another dimension of the KDEs

    Returns
    -------
    histogram : numpy.ndarray

    """
    if weights is not None and len(weights) != sample.shape[0]:
        raise ValueError('Length of sample (%s) and weights (%s) incompatible'
                         %(sample.shape[0], len(weights)))

    if not stack_pid:
        return get_hist(
            sample=sample,
            binning=binning,
            weights=weights,
            bw_method=bw_method,
            adaptive=adaptive,
            alpha=alpha,
            use_cuda=use_cuda,
            coszen_reflection=coszen_reflection,
            coszen_name=coszen_name,
            oversample=oversample
        )

    bin_names = binning.names
    bin_edges = []
    # TODO: why the following? and why must it be GeV?
    for name in bin_names:
        if 'energy' in name:
            bin_edge = binning[name].bin_edges.to('GeV').magnitude
        else:
            bin_edge = binning[name].bin_edges.magnitude
        bin_edges.append(bin_edge)
    pid_bin = bin_names.index('pid')
    other_bins = [0, 1, 2]
    other_bins.pop(pid_bin)
    bin_names.pop(pid_bin)
    assert len(bin_names) == 2
    pid_bin_edges = bin_edges.pop(pid_bin)
    d2d_binning = []
    for b in binning:
        if b.name != 'pid':
            d2d_binning.append(b)
    d2d_binning = MultiDimBinning(d2d_binning)
    pid_stack = []
    for pid in range(len(pid_bin_edges)-1):
        mask_pid = (
            (sample.T[pid_bin] >= pid_bin_edges[pid])
            & (sample.T[pid_bin] < pid_bin_edges[pid+1])
        )
        data = np.array([
            sample.T[other_bins[0]][mask_pid],
            sample.T[other_bins[1]][mask_pid]
        ])

        if weights is None:
            weights_pid = None
        else:
            weights_pid = weights[mask_pid]

        pid_stack.append(
            get_hist(
                sample=data.T,
                weights=weights_pid,
                binning=d2d_binning,
                coszen_name=coszen_name,
                use_cuda=use_cuda,
                bw_method=bw_method,
                alpha=alpha,
                oversample=oversample,
                coszen_reflection=coszen_reflection,
                adaptive=adaptive
            )
        )
    hist = np.dstack(pid_stack)
    if pid_bin != 2:
        hist = np.swapaxes(hist, pid_bin, 2)
    return hist


# TODO: make the plotting optional but add comparisons against some known
# results. This can be accomplished by seeding before calling random to obtain
# a reference result, and check that the same values are returned when run
# below.

def test_kde_histogramdd():
    """Unit tests for kde_histogramdd"""
    from argparse import ArgumentParser
    from shutil import rmtree
    from tempfile import mkdtemp
    from pisa import ureg
    from pisa.core.map import Map, MapSet
    from pisa.utils.log import logging, set_verbosity
    from pisa.utils.plotter import Plotter

    parser = ArgumentParser()
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    temp_dir = mkdtemp()
    try:
        my_plotter = Plotter(stamp='', outdir=temp_dir, fmt='pdf', log=False,
                             annotate=False, symmetric=False, ratio=True)

        b1 = OneDimBinning(name='coszen', num_bins=20, is_lin=True,
                           domain=[-1, 1], tex=r'\cos(\theta)')
        b2 = OneDimBinning(name='energy', num_bins=10, is_log=True,
                           domain=[0.1, 10]*ureg.GeV, tex=r'E')
        b3 = OneDimBinning(name='pid', num_bins=2,
                           bin_edges=[0, 1, 3], tex=r'pid')
        binning = b2 * b1 * b3
        x = np.random.normal(1, 1, (2, 1000))
        p = np.random.uniform(0, 3, 10000)
        x = np.array([np.abs(x[0])-1, x[1], p])
        # Cut away outside coszen
        x = x.T[(x[0] <= 1) & (x[0] >= -1), :].T
        # Swap
        x[[0, 1]] = x[[1, 0]]
        bins = [unp.nominal_values(b.bin_edges) for b in binning]
        raw_hist, _ = np.histogramdd(x.T, bins=bins)

        hist = kde_histogramdd(x.T, binning, bw_method='silverman',
                               coszen_name='coszen', oversample=10,
                               stack_pid=True)

        # TODO: use or remove the following line
        #hist = hist/np.sum(hist)*np.sum(raw_hist)
        m1 = Map(name='KDE', hist=hist, binning=binning)
        m2 = Map(name='raw', hist=raw_hist, binning=binning)
        with np.errstate(divide='ignore', invalid='ignore'):
            m3 = m2/m1
        m3.name = 'hist/KDE'
        m3.tex = m3.name
        m4 = m1 - m2
        m4.name = 'KDE - hist'
        m4.tex = m4.name
        ms = MapSet([m1, m2, m3, m4])
        my_plotter.plot_2d_array(ms, fname='test_kde', cmap='summer')
    except:
        rmtree(temp_dir)
        raise
    else:
        logging.warn('Inspect and manually clean up output(s) saved to %s'
                     % temp_dir)


if __name__ == '__main__':
    test_kde_histogramdd()
