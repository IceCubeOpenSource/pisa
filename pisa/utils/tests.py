# author : S.Wren, J.L.Lanfranchi
#
# date   : September 06, 2016


import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pisa.core.map import Map
from pisa.utils.fileio import get_valid_filename, mkdir
from pisa.utils.log import logging


def has_cuda():
    # pycuda is present if it can be imported
    try:
        import pycuda.driver as cuda
    except:
        CUDA = False
    else:
        CUDA = True
    return CUDA


def order(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        o = np.ceil(np.log10(np.abs(x)))
    return o


def order_str(x):
    order_float = order(x)
    try:
        return str(int(order_float))
    except OverflowError:
        pass
    return str(order_float)


def check_agreement(testname, thresh_ratio, ratio, thresh_diff, diff):
    ratio_pass = ratio <= thresh_ratio
    diff_pass = diff <= thresh_diff

    thresh_ratio_str = order_str(thresh_ratio)
    ratio_ord_str = order_str(ratio)
    ratio_pass_str = 'PASS' if diff_pass else 'FAIL'

    thresh_diff_str = order_str(thresh_diff)
    diff_ord_str = order_str(diff)
    diff_pass_str = 'PASS' if diff_pass else 'FAIL'

    s = '<< {testname:s}, {kind:s}: {pass_str:s} >>' \
        ' agreement to 10^{level:s} (threshold={thresh:e})'

    s_ratio = s.format(
        testname=testname, kind='fract diff', pass_str=ratio_pass_str,
        level=ratio_ord_str, thresh=thresh_ratio
    )
    s_diff = s.format(
        testname=testname, kind='diff', pass_str=diff_pass_str,
        level=diff_ord_str, thresh=thresh_diff
    )

    if ratio_pass:
        logging.info(s_ratio)
    else:
        logging.error(s_ratio)
        raise ValueError(s_ratio)

    if diff_pass:
        logging.info(s_diff)
    else:
        logging.error(s_diff)
        raise ValueError(s_diff)


def print_agreement(testname, ratio):
    ratio_ord_str = order_str(ratio)
    s = '<< {testname:s}, {kind:s} >>' \
        ' agreement to 10^{level:s}'

    s_ratio = s.format(
        testname=testname, kind='fract diff', level=ratio_ord_str
    )

    logging.info(s_ratio)


def validate_maps(amap, bmap):
    """Validate that two PISA 2 style maps are compatible binning."""
    if not (np.allclose(amap['ebins'], bmap['ebins']) and
            np.allclose(amap['czbins'], bmap['czbins'])):
        raise ValueError("Maps' binnings do not match!")


def make_delta_map(amap, bmap):
    """Get the difference between two PISA 2 style maps (amap-bmap) and return
    as another PISA 2 style map."""
    validate_maps(amap, bmap)
    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map': amap['map'] - bmap['map']}


def make_ratio_map(amap, bmap):
    """Get the ratio of two PISA 2 style maps (amap/bmap) and return as another
    PISA 2 style map."""
    validate_maps(amap, bmap)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = {'ebins': amap['ebins'],
                  'czbins': amap['czbins'],
                  'map': amap['map']/bmap['map']}
    return result


def baseplot(m, title, ax, symm=False, evtrate=False):
    """Simple plotting of a 2D histogram (map)"""
    hist = np.ma.masked_invalid(m['map'])
    energy = m['ebins']
    coszen = m['czbins']
    islog = False
    if symm:
        cmap = plt.cm.seismic
        extr = np.nanmax(np.abs(hist))
        vmax = extr
        vmin = -extr
    else:
        cmap = plt.cm.hot
        if evtrate:
            vmin = 0
        else:
            vmin = np.nanmin(hist)
        vmax = np.nanmax(hist)
    cmap.set_bad(color=(0,1,0), alpha=1)
    x = coszen
    y = np.log10(energy)
    X, Y = np.meshgrid(x, y)
    pcmesh = ax.pcolormesh(X, Y, hist, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    cbar.ax.tick_params(labelsize='large')
    ax.set_xlabel(r'$\cos\theta_Z$')
    ax.set_ylabel(r'Energy (GeV)')
    ax.set_title(title, y=1.03)
    min_e = np.min(energy)
    max_e = np.max(energy)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    lin_yticks = 2**(np.arange(np.ceil(np.log2(min_e)),
                               np.floor(np.log2(max_e))+1))
    ax.set_yticks(np.log10(lin_yticks))
    ax.set_yticklabels([str(int(yt)) for yt in lin_yticks])


def baseplot2(map, title, ax, symm=False, evtrate=False):
    """Simple plotting of a 2D map.

    Parameters
    ----------
    map : Map
    title : str
    ax : axis
    symm : bool
    evtrate : bool

    Returns
    -------
    ax, pcmesh, cbar

    """
    assert len(map.binning) == 2
    hist = np.ma.masked_invalid(map.hist)
    islog = False
    if symm:
        cmap = plt.cm.seismic
        extr = np.nanmax(np.abs(hist))
        vmax = extr
        vmin = -extr
    else:
        cmap = plt.cm.afmhot
        if evtrate:
            vmin = 0
        else:
            vmin = np.nanmin(hist)
        vmax = np.nanmax(hist)
    cmap.set_bad(color=(0,1,0), alpha=1)

    x = map.binning.dims[0].bin_edges.magnitude
    y = map.binning.dims[1].bin_edges.magnitude

    if map.binning.dims[0].is_log:
        xticks = 2**(np.arange(np.ceil(np.log2(min(x))),
                               np.floor(np.log2(max(x)))+1))
        x = np.log10(x)
    if map.binning.dims[1].is_log:
        yticks = 2**(np.arange(np.ceil(np.log2(min(y))),
                               np.floor(np.log2(max(y)))+1))
        y = np.log10(y)

    X, Y = np.meshgrid(x, y)
    pcmesh = ax.pcolormesh(X, Y, hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    cbar.ax.tick_params(labelsize='large')
    ax.set_xlabel(map.binning.dims[0].tex)
    ax.set_ylabel(map.binning.dims[1].tex)
    ax.set_title(title, y=1.03)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))

    if map.binning.dims[0].is_log:
        ax.set_xticks(np.log10(xticks))
        ax.set_xticklabels([str(int(xt)) for xt in xticks])
    if map.binning.dims[1].is_log:
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels([str(int(yt)) for yt in yticks])

    return ax, pcmesh, cbar


def plot_comparisons(ref_map, new_map, ref_abv, new_abv, outdir, subdir, name,
                     texname, stagename, servicename, ftype='png'):
    """Plot comparisons between two identically-binned histograms (maps)"""
    path = [outdir]

    if subdir is None:
        subdir = stagename.lower()
    path.append(subdir)

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    fname = ['%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower()),
             'stage_'+stagename]
    if servicename is not None:
        fname.append('service_'+servicename)
    if name is not None:
        fname.append(name.lower())
    fname = '__'.join(fname) + '.' + ftype

    path.append(fname)

    basetitle = []
    if stagename is not None:
        basetitle.append('%s' % stagename)
    if texname is not None:
        basetitle.append(r'$%s$' % texname)
    basetitle = ' '.join(basetitle)

    ratio_map = make_ratio_map(new_map, ref_map)
    diff_map = make_delta_map(new_map, ref_map)
    diff_ratio_map = make_ratio_map(diff_map, ref_map)

    max_diff_ratio = np.nanmax(diff_ratio_map['map'])

    # Handle cases where ratio returns infinite
    # This isn't necessarily a fail, since all it means is the referene was
    # zero If the new value is sufficiently close to zero then it's still fine
    if max_diff_ratio == float('inf'):
        logging.warn('Infinite value found in ratio tests. Difference tests '
                     'now also being calculated')
        # First find all the finite elements
        FiniteMap = np.isfinite(diff_ratio_map['map'])
        # Then find the nanmax of this, will be our new test value
        max_diff_ratio = np.nanmax(diff_ratio_map['map'][FiniteMap])
        # Also find all the infinite elements
        InfiniteMap = np.logical_not(FiniteMap)
        # This will be a second test value
        max_diff = np.nanmax(diff_map['map'][InfiniteMap])
    else:
        # Without any infinite elements we can ignore this second test
        max_diff = 0.0

    if outdir is not None:
        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(20,5))
        baseplot(m=ref_map,
                 title=basetitle+' '+ref_abv,
                 evtrate=True,
                 ax=axes[0])
        baseplot(m=new_map,
                 title=basetitle+' '+new_abv,
                 evtrate=True,
                 ax=axes[1])
        baseplot(m=ratio_map,
                 title=basetitle+' %s/%s' %(new_abv, ref_abv),
                 ax=axes[2])
        baseplot(m=diff_map,
                 title=basetitle+' %s-%s' %(new_abv, ref_abv),
                 symm=True, ax=axes[3])
        baseplot(m=diff_ratio_map,
                 title=basetitle+' (%s-%s)/%s' %(new_abv, ref_abv, ref_abv),
                 symm=True,
                 ax=axes[4])
        logging.debug('>>>> Plot for inspection saved at %s'
                      %os.path.join(*path))
        fig.savefig(os.path.join(*path))
        plt.close(fig.number)

    return max_diff_ratio, max_diff


def plot_cmp(new, ref, new_label, ref_label, plot_label, file_label, outdir,
             ftype='png'):
    """Plot comparisons between two (identically-binned) maps or map sets.

    Parameters
    ----------
    new : Map or MapSet
    ref : Map or MapSet
    new_label : str
    ref_label : str
    plot_label : str
    file_label : str
    outdir : str
    ftype : str

    """
    path = [outdir]

    if isinstance(ref, Map):
        assert isinstance(new, Map)
        ref_maps = [ref]
        new_maps = [new]

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    for ref, new in zip(ref_maps, new_maps):
        assert ref.binning == new.binning
        fname = get_valid_filename(
            '__'.join([
                get_valid_filename(file_label),
                '%s_vs_%s' %(get_valid_filename(new_label.lower()),
                             get_valid_filename(ref_label.lower()))
            ]) + '.' + ftype
        )
        path.append(fname)

        ratio = new / ref
        diff = new - ref
        fract_diff = diff / ref

	finite_ratio = ratio.hist[np.isfinite(ratio.hist)]
	ratio_mean = np.mean(finite_ratio)
	ratio_median = np.median(finite_ratio)

        finite_diff = diff.hist[np.isfinite(diff.hist)]
        diff_mean = np.mean(finite_diff)
        diff_median = np.median(finite_diff)

        finite_fract_diff = fract_diff.hist[np.isfinite(fract_diff.hist)]
        fract_diff_mean = np.mean(finite_fract_diff)
        fract_diff_median = np.median(finite_fract_diff)

        max_diff_ratio = np.nanmax(fract_diff.hist)

        # Handle cases where ratio returns infinite
        # This isn't necessarily a fail, since all it means is the referene was
        # zero If the new value is sufficiently close to zero then it's still fine
        if max_diff_ratio == np.inf:
            logging.warn('Infinite value found in ratio tests. Difference tests'
                         ' now also being calculated')
            # First find all the finite elements
            finite_mask = np.isfinite(fract_diff.hist)
            # Then find the nanmax of this, will be our new test value
            max_diff_ratio = np.nanmax(fract_diff.hist[finite_mask])
            # Also find all the infinite elements; compute a second test value
            max_diff = np.nanmax(diff.hist[~finite_mask])
        else:
            # Without any infinite elements we can ignore this second test
            max_diff = 0.0

        if outdir is not None:
            gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False, figsize=(20,5))
            baseplot2(map=ref,
                      title=ref_label,
                      evtrate=True,
                      ax=axes[0])
            baseplot2(map=new,
                      title=new_label,
                      evtrate=True,
                      ax=axes[1])
            ax, _, _ = baseplot2(map=ratio,
                                 title='%s/%s' %(new_label, ref_label),
                                 ax=axes[2])
            ax.text(0.95, 0.95, "Mean: %.6f"%ratio_mean, horizontalalignment='right',
                    transform=ax.transAxes, color='g')
            ax.text(0.95, 0.91, "Median: %.6f"%ratio_median, horizontalalignment='right',
                    transform=ax.transAxes, color='g')

            ax, _, _ = baseplot2(map=diff,
                                 title='%s-%s' %(new_label, ref_label),
                                 symm=True, ax=axes[3])
            ax.text(0.95, 0.95, "Mean: %.6f"%diff_mean, horizontalalignment='right',
                    transform=ax.transAxes)
            ax.text(0.95, 0.91, "Median: %.6f"%diff_median, horizontalalignment='right',
                    transform=ax.transAxes)

            ax, _, _ = baseplot2(map=fract_diff,
                                 title='(%s-%s)/%s' %(new_label, ref_label, ref_label),
                                 symm=True,
                                 ax=axes[4])
            ax.text(0.95, 0.95, "Mean: %.6f"%fract_diff_mean, horizontalalignment='right',
                    transform=ax.transAxes)
            ax.text(0.95, 0.91, "Median: %.6f"%fract_diff_median, horizontalalignment='right',
                    transform=ax.transAxes)

            logging.debug('>>>> Plot for inspection saved at %s'
                          %os.path.join(*path))
            fig.savefig(os.path.join(*path))
            plt.close(fig.number)

        return max_diff_ratio, max_diff
