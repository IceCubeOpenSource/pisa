#! /usr/bin/env python
# author: S.Wren
# date:   March 20, 2016
"""
Runs the pipeline multiple times to test everything still agrees with PISA 2.
Test data for comparing against should be in the tests/data directory.
A set of plots will be output in your output directory for you to check.
Agreement is expected to order 10^{-14} in the far right plots.
"""

from argparse import ArgumentParser
from collections import Sequence
from copy import deepcopy
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import numpy as np

from pisa.core.map import Map, MapSet
from pisa.core.pipeline import Pipeline
from pisa.utils.fileio import mkdir
from pisa.utils.jsons import from_json
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource
from pisa.utils.config_parser import parse_pipeline_config


def has_cuda():
    """pycuda is present if it can be imported"""
    try:
        import pycuda.driver as cuda
    except ImportError:
        return False
    return True


def order(x):
    with np.errstate(divide='ignore'):
        o = np.ceil(np.log10(x))
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


def validate_pisa2_maps(amap, bmap):
    """Validate that two PISA 2 maps are compatible binning."""
    if not (np.allclose(amap['ebins'], bmap['ebins']) and
            np.allclose(amap['czbins'], bmap['czbins'])):
        raise ValueError("Maps' binnings do not match!")


def delta_map(amap, bmap):
    """Get the difference between two PISA 2 maps (amap-bmap) and return as
    another PISA 2 map."""
    validate_pisa2_maps(amap, bmap)
    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map': amap['map'] - bmap['map']}


def ratio_map(amap, bmap):
    """Get the ratio of two PISA 2 maps (amap/bmap) and return as another PISA
    2 map."""
    validate_pisa2_maps(amap, bmap)
    return {'ebins': amap['ebins'],
            'czbins': amap['czbins'],
            'map': amap['map']/bmap['map']}


def clean_dir(path):
    """Remove whatever is located at `path` recursively, and create a new
    directory at `path`. WARNING: this will delete files without interaction by
    the user, so make sure `path` and everything below it are ok to be removed
    prior to calling this function!!!

    """
    if isinstance(path, Sequence):
        path = os.path.join(*path)
    assert isinstance(path, basestring)

    if os.path.exists(path):
        # Remove if (possibly non-empty) directory
        if os.path.isdir(path):
            shutil.rmtree(path)
        # Remove if file
        else:
            os.remove(path)
    # Create the new directory at the path
    mkdir(path)


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


def plot_comparisons(ref_map, new_map, ref_abv, new_abv, outdir, subdir, name,
                     texname, stagename, servicename, ftype='png'):
    """Plot comparisons between two identically-binned histograms (maps)"""
    path = [outdir]

    if subdir is None:
        subdir = stagename.lower()
    path.append(subdir)

    if outdir is not None:
        mkdir(os.path.join(*path), warn=False)

    fname = ['pisa_%s_%s_comparisons' %(ref_abv.lower(), new_abv.lower()),
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
    basetitle.append('PISA')
    basetitle = ' '.join(basetitle)

    RatioMapObj = ratio_map(new_map, ref_map)
    DiffMapObj = delta_map(new_map, ref_map)
    DiffRatioMapObj = ratio_map(DiffMapObj, ref_map)

    max_diff_ratio = np.nanmax(DiffRatioMapObj['map'])

    # Handle cases where ratio returns infinite
    # This isn't necessarily a fail, since all it means is the referene was zero
    # If the new value is sufficiently close to zero then it's still fine
    if max_diff_ratio == float('inf'):
        logging.warn('Infinite value found in ratio tests. Difference tests '
                     'now also being calculated')
        # First find all the finite elements
        FiniteMap = np.isfinite(DiffRatioMapObj['map'])
        # Then find the nanmax of this, will be our new test value
        max_diff_ratio = np.nanmax(DiffRatioMapObj['map'][FiniteMap])
        # Also find all the infinite elements
        InfiniteMap = not FiniteMap
        # This will be a second test value
        max_diff = np.nanmax(DiffMapObj['map'][InfiniteMap])
    else:
        # Without any infinite elements we can ignore this second test
        max_diff = 0.0

    if outdir is not None:
        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=5, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(20,5))
        baseplot(m=ref_map, title=basetitle+' '+ref_abv, evtrate=True,
                 ax=axes[0])
        baseplot(m=new_map, title=basetitle+' '+new_abv, evtrate=True,
                 ax=axes[1])
        baseplot(m=RatioMapObj, title=basetitle+' %s/%s' %(new_abv, ref_abv),
                 ax=axes[2])
        baseplot(m=DiffMapObj, title=basetitle+' %s-%s' %(new_abv, ref_abv),
                 symm=True, ax=axes[3])
        baseplot(m=DiffRatioMapObj, title=basetitle+' (%s-%s)/%s'
             %(new_abv, ref_abv, ref_abv), symm=True, ax=axes[4])
        logging.debug('>>>> Plot for inspection saved at %s'
                      %os.path.join(*path))
        fig.savefig(os.path.join(*path))
        plt.close(fig.number)

    return max_diff_ratio, max_diff


def compare_flux(config, servicename, pisa2file, systname,
                 outdir, ratio_test_threshold, diff_test_threshold):
    """Compare flux stages run in isolation with dummy inputs"""

    logging.debug('>> Working on flux stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            config['flux']['params'][systname] = \
                    config['flux']['params'][systname].value + \
                    config['flux']['params'][systname].prior.stddev
        except:
            config['flux']['params'][systname] = \
                    1.25*config['flux']['params'][systname].value

        pisa2file = pisa2file.split('.json')[0] + \
                        '-%s%.2f.json' \
                        %(systname, config['flux']['params'][systname].value)
        servicename += '-%s%.2f' \
                           %(systname, config['flux']['params'][systname].value)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    outputs = stage.get_outputs()
    pisa2_comparisons = from_json(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_comparisons[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map = outputs[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['true_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['true_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='flux',
            stagename='flux',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='V3-V2 flux:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_osc(config, servicename, pisa2file, systname,
                outdir, ratio_test_threshold, diff_test_threshold):
    """Compare osc stages run in isolation with dummy inputs"""

    logging.debug('>> Working on osc stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            config['osc']['params'][systname] = \
                    config['osc']['params'][systname].value + \
                    config['osc']['params'][systname].prior.stddev
        except:
            config['osc']['params'][systname] = \
                    1.25*config['osc']['params'][systname].value

        if config['osc']['params'][systname].value.magnitude < 0.01:
            systval = '%e'%config['osc']['params'][systname].value.magnitude
            systval = systval[0:4]
        else:
            systval = '%.2f'%config['osc']['params'][systname].value.magnitude

        pisa2file = pisa2file.split('.json')[0] + \
                '-%s%s.json' %(systname, systval)
        servicename += '-%s%s' %(systname, systval)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(
        inputs=MapSet(maps=input_maps, name='ones', hash=1)
    )
    pisa2_comparisons = from_json(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_comparisons[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map = outputs[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['true_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['true_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='osc',
            stagename='osc',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='V3-V2 osc:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_aeff(config, servicename, pisa2file, systname,
                 outdir, ratio_test_threshold, diff_test_threshold):
    """Compare aeff stages run in isolation with dummy inputs"""

    logging.debug('>> Working on aeff stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    if systname is not None:
        logging.debug('>>> Checking %s systematic'%systname)
        test_syst = systname
        try:
            config['aeff']['params'][systname] = \
                    config['aeff']['params'][systname].value + \
                    config['aeff']['params'][systname].prior.stddev
        except:
            config['aeff']['params'][systname] = \
                    1.25*config['aeff']['params'][systname].value

        pisa2file = pisa2file.split('.json')[0] + \
                '-%s%.2f.json' \
                %(systname, config['aeff']['params'][systname].value)
        servicename += '-%s%.2f' \
                %(systname, config['aeff']['params'][systname].value)
    else:
        logging.debug('>>> Checking baseline')
        test_syst = 'baseline'

    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))
    pisa2_comparisons = from_json(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        for intkey in pisa2_comparisons[nukey].keys():
            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
            else:
                new_nukey = nukey
            cakekey = new_nukey + '_' + intkey
            pisa_map_to_plot = pisa2_comparisons[nukey][intkey]

            cake_map = outputs[cakekey]
            cake_map_to_plot = {}
            cake_map_to_plot['ebins'] = \
                    cake_map.binning['true_energy'].bin_edges.magnitude
            cake_map_to_plot['czbins'] = \
                    cake_map.binning['true_coszen'].bin_edges.magnitude
            cake_map_to_plot['map'] = cake_map.hist

            max_diff_ratio, max_diff = plot_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='V2', new_abv='V3',
                outdir=outdir,
                subdir='aeff',
                stagename='aeff',
                servicename=servicename,
                name=cakekey,
                texname=outputs[cakekey].tex,
            )

            check_agreement(
                testname='V3-V2 aeff:%s %s %s'
                    %(test_service, test_syst, cakekey),
                thresh_ratio=ratio_test_threshold,
                ratio=max_diff_ratio,
                thresh_diff=diff_test_threshold,
                diff=max_diff
            )

    return pipeline


def compare_reco(config, servicename, pisa2file, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare reco stages run in isolation with dummy inputs"""
    logging.debug('>> Working on reco stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    logging.debug('>>> Checking baseline')
    test_syst = 'baseline'
    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        if 'nc' in name:
            # NC is combination of three flavours
            hist *= 3.0
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))
    nue_nuebar_cc = outputs.combine_re(r'nue(bar){0,1}_cc')
    numu_numubar_cc = outputs.combine_re(r'numu(bar){0,1}_cc')
    nutau_nutaubar_cc = outputs.combine_re(r'nutau(bar){0,1}_cc')
    nuall_nuallbar_nc = outputs.combine_re(r'nu.*_nc')

    modified_cake_outputs = {
        'nue_cc': {
            'map': nue_nuebar_cc.hist,
            'ebins': nue_nuebar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nue_nuebar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'numu_cc': {
            'map': numu_numubar_cc.hist,
            'ebins': numu_numubar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': numu_numubar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'nutau_cc': {
            'map': nutau_nutaubar_cc.hist,
            'ebins': nutau_nutaubar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nutau_nutaubar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'nuall_nc': {
            'map': nuall_nuallbar_nc.hist,
            'ebins': nuall_nuallbar_nc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nuall_nuallbar_nc.binning.reco_coszen.bin_edges.magnitude
        }
    }

    pisa2_comparisons = from_json(pisa2file)

    for nukey in pisa2_comparisons.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa2_comparisons[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = modified_cake_outputs[nukey]

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='reco',
            stagename='reco',
            servicename=servicename,
            name=nukey,
            texname=outputs[nukey].tex
        )

        check_agreement(
            testname='V3-V2 reco:%s %s %s'
                %(test_service, test_syst, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )

    return pipeline


def compare_pid(config, servicename, pisa2file, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare pid stages run in isolation with dummy inputs"""
    logging.debug('>> Working on pid stage comparisons')
    logging.debug('>>> Checking %s service'%servicename)
    test_service = servicename

    logging.debug('>>> Checking baseline')
    test_syst = 'baseline'
    pipeline = Pipeline(config)
    stage = pipeline.stages[0]
    input_maps = []
    for name in stage.input_names:
        hist = np.ones(stage.input_binning.shape)
        # Input names still has nu and nubar separated.
        # PISA 2 is not expecting this
        hist *= 0.5
        input_maps.append(
            Map(name=name, hist=hist, binning=stage.input_binning)
        )
    outputs = stage.get_outputs(inputs=MapSet(maps=input_maps, name='ones',
                                              hash=1))

    cake_trck = outputs.combine_wildcard('*_trck')
    cake_cscd = outputs.combine_wildcard('*_cscd')
    total_cake_trck_dict = {
        'map': cake_trck.hist,
        'ebins': cake_trck.binning.reco_energy.bin_edges.magnitude,
        'czbins': cake_trck.binning.reco_coszen.bin_edges.magnitude
    }
    total_cake_cscd_dict = {
        'map': cake_cscd.hist,
        'ebins': cake_cscd.binning.reco_energy.bin_edges.magnitude,
        'czbins': cake_cscd.binning.reco_coszen.bin_edges.magnitude
    }

    pisa2_comparisons = from_json(pisa2file)
    total_pisa_trck_dict = pisa2_comparisons['trck']
    total_pisa_cscd_dict = pisa2_comparisons['cscd']

    max_diff_ratio, max_diff= plot_comparisons(
        ref_map=total_pisa_cscd_dict,
        new_map=total_cake_cscd_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='cscd',
        texname=r'{\rm cscd}'
    )

    check_agreement(
        testname='V3-V2 pid:%s %s cscd'
            %(test_service, test_syst),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=total_pisa_trck_dict,
        new_map=total_cake_trck_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='pid',
        stagename='pid',
        servicename=servicename,
        name='trck',
        texname=r'{\rm trck}'
    )

    check_agreement(
        testname='V3-V2 pid:%s %s trck'
            %(test_service, test_syst),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    return pipeline


def compare_flux_full(cake_maps, pisa_maps, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the flux stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of flux stage')
    test_service = 'honda'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa_maps[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map = cake_maps[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['true_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['true_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='flux',
            servicename=test_service,
            name=nukey,
            texname=cake_maps[nukey].tex
        )

        check_agreement(
            testname='V3-V2 flux:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_osc_full(cake_maps, pisa_maps, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the osc stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of osc stage')
    test_service = 'prob3cpu'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa_maps[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map = cake_maps[nukey]
        cake_map_to_plot = {}
        cake_map_to_plot['ebins'] = \
                cake_map.binning['true_energy'].bin_edges.magnitude
        cake_map_to_plot['czbins'] = \
                cake_map.binning['true_coszen'].bin_edges.magnitude
        cake_map_to_plot['map'] = cake_map.hist

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='osc',
            servicename=test_service,
            name=nukey,
            texname=cake_maps[nukey].tex
        )

        check_agreement(
            testname='V3-V2 full pipeline through osc:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_aeff_full(cake_maps, pisa_maps, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the aeff stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of aeff stage')
    test_service = 'hist_1X585'

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        for intkey in pisa_maps[nukey].keys():
            if '_' in nukey:
                if nukey.split('_')[1] == 'bar':
                    new_nukey = ""
                    for substr in nukey.split('_'):
                        new_nukey += substr
            else:
                new_nukey = nukey
            cakekey = new_nukey + '_' + intkey
            pisa_map_to_plot = pisa_maps[nukey][intkey]

            cake_map = cake_maps[cakekey]
            cake_map_to_plot = {}
            cake_map_to_plot['ebins'] = \
                    cake_map.binning['true_energy'].bin_edges.magnitude
            cake_map_to_plot['czbins'] = \
                    cake_map.binning['true_coszen'].bin_edges.magnitude
            cake_map_to_plot['map'] = cake_map.hist

            max_diff_ratio, max_diff = plot_comparisons(
                ref_map=pisa_map_to_plot,
                new_map=cake_map_to_plot,
                ref_abv='V2', new_abv='V3',
                outdir=outdir,
                subdir='fullpipeline',
                stagename='aeff',
                servicename=test_service,
                name=cakekey,
                texname=cake_maps[cakekey].tex,
            )

            check_agreement(
                testname='V3-V2 full pipeline through aeff:hist %s' %nukey,
                thresh_ratio=ratio_test_threshold,
                ratio=max_diff_ratio,
                thresh_diff=diff_test_threshold,
                diff=max_diff
            )


def compare_reco_full(cake_maps, pisa_maps, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the reco stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of reco stage')
    test_service = 'hist_1X585'

    nue_nuebar_cc = cake_maps.combine_re(r'nue(bar){0,1}_cc')
    numu_numubar_cc = cake_maps.combine_re(r'numu(bar){0,1}_cc')
    nutau_nutaubar_cc = cake_maps.combine_re(r'nutau(bar){0,1}_cc')
    nuall_nuallbar_nc = cake_maps.combine_re(r'nu.*_nc')

    modified_cake_outputs = {
        'nue_cc': {
            'map': nue_nuebar_cc.hist,
            'ebins': nue_nuebar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nue_nuebar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'numu_cc': {
            'map': numu_numubar_cc.hist,
            'ebins': numu_numubar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': numu_numubar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'nutau_cc': {
            'map': nutau_nutaubar_cc.hist,
            'ebins': nutau_nutaubar_cc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nutau_nutaubar_cc.binning.reco_coszen.bin_edges.magnitude
        },
        'nuall_nc': {
            'map': nuall_nuallbar_nc.hist,
            'ebins': nuall_nuallbar_nc.binning.reco_energy.bin_edges.magnitude,
            'czbins': nuall_nuallbar_nc.binning.reco_coszen.bin_edges.magnitude
        }
    }

    for nukey in pisa_maps.keys():
        if 'nu' not in nukey:
            continue

        pisa_map_to_plot = pisa_maps[nukey]

        if '_' in nukey:
            if nukey.split('_')[1] == 'bar':
                new_nukey = ""
                for substr in nukey.split('_'):
                    new_nukey += substr
                nukey = new_nukey

        cake_map_to_plot = modified_cake_outputs[nukey]

        if 'nc' in nukey:
            if 'bar' in nukey:
                texname = r'\bar{\nu} NC'
            else:
                texname = r'\nu NC'
        else:
            texname = cake_maps[nukey].tex

        max_diff_ratio, max_diff = plot_comparisons(
            ref_map=pisa_map_to_plot,
            new_map=cake_map_to_plot,
            ref_abv='V2', new_abv='V3',
            outdir=outdir,
            subdir='fullpipeline',
            stagename='reco',
            servicename=test_service,
            name=nukey,
            texname=texname
        )

        check_agreement(
            testname='V3-V2 full pipeline through reco:%s %s'
                %(test_service, nukey),
            thresh_ratio=ratio_test_threshold,
            ratio=max_diff_ratio,
            thresh_diff=diff_test_threshold,
            diff=max_diff
        )


def compare_pid_full(cake_maps, pisa_maps, outdir, ratio_test_threshold, diff_test_threshold):
    """Compare a fully configured pipeline (with stages flux, osc, aeff, reco,
    and pid) through the pid stage.

    """
    logging.debug('>> Working on full pipeline comparisons')
    logging.debug('>>> Checking to end of pid stage')
    test_service = 'hist_1X585'

    cake_trck = cake_maps.combine_wildcard('*_trck')
    cake_cscd = cake_maps.combine_wildcard('*_cscd')
    total_cake_trck_dict = {
        'map': cake_trck.hist,
        'ebins': cake_trck.binning.reco_energy.bin_edges.magnitude,
        'czbins': cake_trck.binning.reco_coszen.bin_edges.magnitude
    }
    total_cake_cscd_dict = {
        'map': cake_cscd.hist,
        'ebins': cake_cscd.binning.reco_energy.bin_edges.magnitude,
        'czbins': cake_cscd.binning.reco_coszen.bin_edges.magnitude
    }

    total_pisa_trck_dict = pisa_maps['trck']
    total_pisa_cscd_dict = pisa_maps['cscd']

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=total_pisa_cscd_dict,
        new_map=total_cake_cscd_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='pid',
        servicename=test_service,
        name='cscd',
        texname=r'{\rm cscd}'
    )
    check_agreement(
        testname='V3-V2 full pipeline through pid:%s cscd'
            %(test_service),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )

    max_diff_ratio, max_diff = plot_comparisons(
        ref_map=total_pisa_trck_dict,
        new_map=total_cake_trck_dict,
        ref_abv='V2', new_abv='V3',
        outdir=outdir,
        subdir='fullpipeline',
        stagename='pid',
        servicename=test_service,
        name='trck',
        texname=r'{\rm trck}'
    )
    check_agreement(
        testname='V3-V2 full pipeline through pid:%s trck'
            %(test_service),
        thresh_ratio=ratio_test_threshold,
        ratio=max_diff_ratio,
        thresh_diff=diff_test_threshold,
        diff=max_diff
    )


if __name__ == '__main__':
    parser = ArgumentParser(
        description='''Run a set of tests on the PISA 3 pipeline against
        benchmark PISA 2 data. If no test flags are specified, *all* tests will
        be run.

        This script should always be run when you make any major modifications
        to be sure nothing has broken.

        If you find this script does not work, please either fix it or report
        it! In general, this will signify you have "changed" something, somehow
        in the basic functionality which you should understand!'''
    )
    parser.add_argument('--flux', action='store_true', default=False,
                        help='''Run flux tests i.e. the interpolation methods
                        and the flux systematics.''')
    parser.add_argument('--osc', action='store_true', default=False,
                        help='''Run osc tests i.e. the oscillograms with one
                        sigma deviations in the parameters.''')
    parser.add_argument('--osc-gpu', action='store_true', default=False,
                        help='''Run GPU-based osc tests i.e. the oscillograms
                        with one sigma deviations in the parameters.''')
    parser.add_argument('--aeff', action='store_true', default=False,
                        help='''Run effective area tests i.e. the different
                        transforms with the aeff systematics.''')
    parser.add_argument('--reco', action='store_true', default=False,
                        help='''Run reco tests i.e. the different reco kernels
                        and their systematics.''')
    parser.add_argument('--pid', action='store_true', default=False,
                        help='''Run PID tests i.e. the different pid kernels
                        methods and their systematics.''')
    parser.add_argument('--full', action='store_true', default=False,
                        help='''Run full pipeline tests for the baseline i.e.
                        all stages simultaneously rather than each in
                        isolation.''')
    parser.add_argument('--outdir', metavar='DIR', type=str,
                        help='''Store all output plots to this directory. If
                        they don't exist, the script will make them, including
                        all subdirectories. If none is supplied no plots will
                        be saved.''')
    parser.add_argument('--ratio_threshold', type=float, default=1E-8,
                        help='''Sets the agreement threshold on the ratio test
                        plots. If this is not reached the tests will fail.''')
    parser.add_argument('--diff_threshold', type=float, default=1E-3,
                        help='''Sets the agreement threshold on the diff test
                        plots. If this is not reached the tests will fail. This
                        test is only important if any ratios return inf.''')
    parser.add_argument('-v', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()
    set_verbosity(args.v)

    # Figure out which tests to do
    test_all = True
    if args.flux or args.osc or args.osc_gpu or args.aeff or args.reco or \
            args.pid or args.full:
        test_all = False

    # Perform flux tests
    if args.flux or test_all:
        flux_settings = os.path.join(
            'tests', 'settings', 'flux_test.ini'
        )
        flux_config = parse_pipeline_config(flux_settings)
        flux_config['flux']['params']['flux_file'] = \
                'flux/honda-2015-spl-solmax-aa.d'
        flux_config['flux']['params']['flux_mode'] = \
                'integral-preserving'

        for syst in [None, 'atm_delta_index', 'nue_numu_ratio',
                     'nu_nubar_ratio', 'energy_scale']:
            pisa2file = os.path.join(
                'tests', 'data', 'flux', 'PISAV2IPHonda2015SPLSolMaxFlux.json'
            )
            pisa2file = find_resource(pisa2file)
            flux_pipeline = compare_flux(
                config=deepcopy(flux_config),
                servicename='IP_Honda',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

        flux_config['flux']['params']['flux_mode'] = 'bisplrep'
        pisa2file = os.path.join(
            'tests', 'data', 'flux', 'PISAV2bisplrepHonda2015SPLSolMaxFlux.json'
        )
        pisa2file = find_resource(pisa2file)
        flux_pipeline = compare_flux(
            config=deepcopy(flux_config),
            servicename='bisplrep_Honda',
            pisa2file=pisa2file,
            systname=None,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

    # Perform (CPU-based) oscillations tests
    if args.osc or test_all:
        osc_settings = os.path.join(
            'tests', 'settings', 'osc_test.ini'
        )
        osc_config = parse_pipeline_config(osc_settings)
        for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21',
                     'deltam31']:
            pisa2file = os.path.join(
                'tests', 'data', 'osc', 'PISAV2OscStageProb3Service.json'
            )
            pisa2file = find_resource(pisa2file)
            osc_pipeline = compare_osc(
                config=deepcopy(osc_config),
                servicename='prob3',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform GPU-based oscillations tests
    if args.osc_gpu or test_all:
        osc_settings = os.path.join(
            'tests', 'settings', 'osc_gpu_test.ini'
        )
        osc_config = parse_pipeline_config(osc_settings)
        for syst in [None, 'theta12', 'theta13', 'theta23', 'deltam21',
                     'deltam31']:
            pisa2file = os.path.join(
                'tests', 'data', 'osc', 'PISAV2OscStageProb3Service.json'
            )
            pisa2file = find_resource(pisa2file)
            osc_pipeline = compare_osc(
                config=deepcopy(osc_config),
                servicename='prop_grid',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform effective-area tests
    if args.aeff or test_all:
        aeff_settings = os.path.join(
            'tests', 'settings', 'aeff_test.ini'
        )
        aeff_config = parse_pipeline_config(aeff_settings)
        aeff_config['aeff']['params']['aeff_weight_file'] = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'UnJoined',
            'DC_MSU_1X585_unjoined_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'aeff', 'PISAV2AeffStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        for syst in [None, 'aeff_scale']:
            aeff_pipeline = compare_aeff(
                config=deepcopy(aeff_config),
                servicename='hist_1X585',
                pisa2file=pisa2file,
                systname=syst,
                outdir=args.outdir,
                ratio_test_threshold=args.ratio_threshold,
                diff_test_threshold=args.diff_threshold
            )

    # Perform reconstruction tests
    if args.reco or test_all:
        reco_settings = os.path.join(
            'tests', 'settings', 'reco_test.ini'
        )
        reco_config = parse_pipeline_config(reco_settings)
        reco_config['reco']['params']['reco_weights_name'] = None
        reco_config['reco']['params']['reco_weight_file'] = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'Joined',
            'DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'reco', 'PISAV2RecoStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist_1X585',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

        reco_config['reco']['params']['reco_weight_file'] = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXX', 'Joined',
            'DC_MSU_1X60_joined_nu_nubar_events_mc.hdf5'
        )
        pisa2file = os.path.join(
            'tests', 'data', 'reco', 'PISAV2RecoStageHist1X60Service.json'
        )
        pisa2file = find_resource(pisa2file)
        reco_pipeline = compare_reco(
            config=deepcopy(reco_config),
            servicename='hist_1X60',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

    # Perform PID tests
    if args.pid or test_all:
        pid_settings = os.path.join(
            'tests', 'settings', 'pid_test.ini'
        )
        pid_config = parse_pipeline_config(pid_settings)
        pisa2file = os.path.join(
            'tests', 'data', 'pid', 'PISAV2PIDStageHistV39Service.json'
        )
        pisa2file = find_resource(pisa2file)
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='hist_V39',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        pid_config['pid']['params']['pid_events'] = os.path.join(
            'events', 'deepcore_ic86', 'MSU', '1XXXX', 'Joined',
            'DC_MSU_1X585_joined_nu_nubar_events_mc.hdf5'
        )
        pid_config['pid']['params']['pid_weights_name'] = 'weighted_aeff'
        pid_config['pid']['params']['pid_ver'] = 'msu_mn8d-mn7d'
        pisa2file = os.path.join(
            'tests', 'data', 'pid', 'PISAV2PIDStageHist1X585Service.json'
        )
        pisa2file = find_resource(pisa2file)
        pid_pipeline = compare_pid(
            config=deepcopy(pid_config),
            servicename='hist_1X585',
            pisa2file=pisa2file,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )

    # Perform full-pipeline tests
    if args.full or test_all:
        full_settings = os.path.join(
            'tests', 'settings', 'full_pipeline_test.ini'
        )
        pipeline = Pipeline(full_settings)
        pipeline.get_outputs()

        pisa2file = os.path.join(
            'tests', 'data', 'full',
            'PISAV2FullDeepCorePipeline-IPSPL2015SolMax-Prob3CPUNuFit2014-AeffHist1X585-RecoHist1X585-PIDHist1X585.json'
        )
        pisa2file = find_resource(pisa2file)
        pisa2_comparisons = from_json(pisa2file)
        # Up to flux stage comparisons
        compare_flux_full(
            pisa_maps=pisa2_comparisons[0],
            cake_maps=pipeline['flux'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Up to osc stage comparisons
        compare_osc_full(
            pisa_maps=pisa2_comparisons[1],
            cake_maps=pipeline['osc'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Up to aeff stage comparisons
        compare_aeff_full(
            pisa_maps=pisa2_comparisons[2],
            cake_maps=pipeline['aeff'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Up to reco stage comparisons
        compare_reco_full(
            pisa_maps=pisa2_comparisons[3],
            cake_maps=pipeline['reco'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
        # Up to PID stage comparisons
        compare_pid_full(
            pisa_maps=pisa2_comparisons[4],
            cake_maps=pipeline['pid'].outputs,
            outdir=args.outdir,
            ratio_test_threshold=args.ratio_threshold,
            diff_test_threshold=args.diff_threshold
        )
