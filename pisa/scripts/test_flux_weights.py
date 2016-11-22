#! /usr/bin/env python
# author: S.Wren
# date:   November 15, 2016
"""
A set of tests on the flux weights calculated by PISA.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib.colors as colors
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging
from pisa.utils.flux_weights import load_2D_table, calculate_2D_flux_weights, primaries, texprimaries, load_3D_table, calculate_3D_flux_weights

def Plot1DSlices(xintvals, yintvals, xtabvals, ytabvals, xtabbins,
                 xlabel, ylabel, xtext, ytext, text, tablename,
                 savename, log):
    '''
    Test function to show interpolation and tables overlaid in 1D slices.
    See main function for how to use this function.
    
    Parameters
    ----------

    xintvals : list
        A list of the x points where the spline was evaluated.
    yintvals : list
        A list of the y points which the spline evaluated to.
    xtabvals : list
        A list of the x points where the table is defined.
    ytabvals : list
        A list of the y points where the table is defined.
    xtabbins : list
        A list of the bin edges. Should have xtabvals as the bin centres.
    xlabel : string 
        A label for the x-axis of the plot.
    ylabel : string 
        A label for the y-axis of the plot.
    xtext : float 
        The position for the text label showing the slice along x.
    ytext : float
        The position for the text label showing the slice along y.
    text : string
        The text label showing the slice.
    tablename : string
        The text label naming the tables used
    savename : string 
        The place and name to save the plot.
    log : bool
        A boolean to whether the axes should be made logarithmic. 
        Will do both.
    '''

    plt.plot(xintvals,
             yintvals,
             color='r',
             linewidth=2,
             label='IP Interpolation')
    plt.hist(xtabvals,
             weights = ytabvals,
             bins = xtabbins,
             color = 'k',
             linewidth = 2,
             histtype='step',
             label=tablename)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    if log:
        plt.xlim(xtabbins[0],xtabbins[-1])
        plt.xscale("log")
        plt.yscale("log")
        ymin = min(min(np.log10(yintvals)),min(np.log10(ytabvals)))
        ymax = max(max(np.log10(yintvals)),max(np.log10(ytabvals)))
        ydiff = ymax - ymin
        plt.xlim(min(xtabbins),max(xtabbins))
        plt.ylim(np.power(10,ymin-0.1*ydiff),np.power(10,ymax+0.1*ydiff))
        if 'numu' in savename:
            plt.legend(loc='lower right')
        elif 'nue' in savename:
            plt.legend(loc='lower left')
    else:
        ymin = min(min(yintvals),min(ytabvals))
        ymax = max(max(yintvals),max(ytabvals))
        ydiff = ymax-ymin
        plt.xlim(min(xtabbins),max(xtabbins))
        if min(xtabbins) == 0.0 and max(xtabbins) == 360.0:
            plt.ylim(ymin-0.1*ydiff,ymax+0.8*ydiff)
        else:
            plt.ylim(ymin-0.1*ydiff,ymax+0.1*ydiff)
        plt.legend(loc='upper right')
    plt.figtext(xtext,
                ytext,
                text,
                verticalalignment='center',
                horizontalalignment='center',
                color='k',
                fontsize=24)
    plt.savefig(savename)
    plt.close()
    

def logplot(m, title, ax, clabel, cmap=plt.cm.afmhot, largelabels=False):
    """Simple plotting of a 2D histogram (map)"""
    hist = np.ma.masked_invalid(m['map'])
    y = m['ebins']
    x = m['czbins']
    X, Y = np.meshgrid(x, y)
    ax.set_yscale('log')
    vmin = hist.min()
    vmax = hist.max()
    if clabel is not None:
        pcmesh = ax.pcolormesh(X, Y, hist,
                               norm=colors.LogNorm(vmin=vmin,vmax=vmax),
                               cmap=cmap)
    else:
        pcmesh = ax.pcolormesh(X, Y, hist,
                               cmap=cmap)
    cbar = plt.colorbar(mappable=pcmesh, ax=ax)
    if clabel is not None:
        if largelabels:
            cbar.set_label(clabel,labelpad=-1,fontsize=36)
            cbar.ax.tick_params(labelsize=36)
        else:
            cbar.set_label(clabel,labelpad=-1)
            cbar.ax.tick_params(labelsize='large')
    if largelabels:
        ax.set_xlabel(r'$\cos\theta_Z$',fontsize=36)
        ax.set_ylabel(r'Energy (GeV)',labelpad=-3,fontsize=36)
        ax.set_title(title, y=1.03, fontsize=36)
        plt.tick_params(axis='both', which='major', labelsize=36)
    else:
        ax.set_xlabel(r'$\cos\theta_Z$')
        ax.set_ylabel(r'Energy (GeV)',labelpad=-3)
        ax.set_title(title, y=1.03)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))


def take_average(interp_map, oversampling):
    average_map = interp_map.reshape(
        [len(interp_map)/oversampling,
         oversampling,
         len(interp_map[0])/oversampling,
         oversampling]
    ).mean(3).mean(1)
    return average_map


def do_1D_2D_honda_test(spline_dict, flux_dict, LegendFileName,
                        SaveName, outdir, enpow=1):

    czs = np.linspace(-1,1,81)
    low_ens = 5.0119*np.ones_like(czs)
    high_ens = 50.119*np.ones_like(czs)
    
    ens = np.logspace(-1.025,4.025,1020)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)

    for flav, flavtex in zip(primaries, texprimaries):
    
        low_en_flux_weights = calculate_2D_flux_weights(low_ens,
                                                        czs,
                                                        spline_dict[flav],
                                                        enpow=enpow)
            
        high_en_flux_weights = calculate_2D_flux_weights(high_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        flux5 = flux_dict[flav].T[np.where(flux_dict['energy']==5.0119)][0]
        flux50 = flux_dict[flav].T[np.where(flux_dict['energy']==50.119)][0]

        Plot1DSlices(
            xintvals = czs,
            yintvals = low_en_flux_weights,
            xtabvals = flux_dict['coszen'],
            ytabvals = flux5,
            xtabbins = np.linspace(-1,1,21),
            xlabel = r'$\cos\theta_Z$',
            ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext = 0.75,
            ytext = 0.7,
            text = 'Slice at \n 5.0119 GeV',
            tablename = LegendFileName,
            savename = os.path.join(
                outdir,'%s_%sfluxweightstest5GeV.png'%(SaveName,flav)
            ),
            log = False
        )
        
        Plot1DSlices(
            xintvals = czs,
            yintvals = high_en_flux_weights,
            xtabvals = flux_dict['coszen'],
            ytabvals = flux50,
            xtabbins = np.linspace(-1,1,21),
            xlabel = r'$\cos\theta_Z$',
            ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext = 0.75,
            ytext = 0.7,
            text = 'Slice at \n 50.119 GeV',
            tablename = LegendFileName,
            savename = os.path.join(
                outdir,'%s_%sfluxweightstest50GeV.png'%(SaveName,flav)
            ),
            log = False
        )

        upgoing_flux_weights = calculate_2D_flux_weights(ens,
                                                         upgoing,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        downgoing_flux_weights = calculate_2D_flux_weights(ens,
                                                           downgoing,
                                                           spline_dict[flav],
                                                           enpow=enpow)

        upgoing_flux_weights *= np.power(ens,3)
        downgoing_flux_weights *= np.power(ens,3)

        coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
        coszen_strs = np.array(coszen_strs)

        fluxupgoing = flux_dict[flav][np.where(coszen_strs=='-0.95')][0]
        fluxdowngoing = flux_dict[flav][np.where(coszen_strs=='0.35')][0]

        fluxupgoing *= np.power(flux_dict['energy'],3)
        fluxdowngoing *= np.power(flux_dict['energy'],3)

        if 'numu' in flav:
            xtext = 0.68
            ytext = 0.25
        elif 'nue' in flav:
            xtext = 0.35
            ytext = 0.25

        Plot1DSlices(
            xintvals = ens,
            yintvals = upgoing_flux_weights,
            xtabvals = flux_dict['energy'],
            ytabvals = fluxupgoing,
            xtabbins = np.logspace(-1.025,4.025,102),
            xlabel = 'Neutrino Energy (GeV)',
            ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext = xtext,
            ytext = ytext,
            text = r'Slice at $\cos\theta_Z=-0.95$',
            tablename = LegendFileName,
            savename = os.path.join(
                outdir,'%s_%sfluxweightstest-0.95cz.png'%(SaveName,flav)
            ),
            log = True
        )
            
        Plot1DSlices(
            xintvals = ens,
            yintvals = downgoing_flux_weights,
            xtabvals = flux_dict['energy'],
            ytabvals = fluxdowngoing,
            xtabbins = np.logspace(-1.025,4.025,102),
            xlabel = 'Neutrino Energy (GeV)',
            ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext = xtext,
            ytext = ytext,
            text = r'Slice at $\cos\theta_Z=0.35$',
            tablename = LegendFileName,
            savename = os.path.join(
                outdir,'%s_%sfluxweightstest0.35cz.png'%(SaveName,flav)
            ),
            log = True
        )


def do_2D_2D_honda_test(spline_dict, flux_dict, outdir, ip_checks,
                     oversample, SaveName, TitleFileName, enpow=1):

    all_ens_bins = np.logspace(-1.025,4.025,101*oversample+1)
    all_log_ens_bins = np.linspace(-1.025,4.025,101*oversample+1)
    log_en_bin_width = all_log_ens_bins[1] - all_log_ens_bins[0]
    all_ens = np.logspace(all_log_ens_bins[0] + log_en_bin_width/2.0,
                          all_log_ens_bins[-1] - log_en_bin_width/2.0,
                          101*oversample)
    all_czs_bins = np.linspace(-1.0,1.0,20*oversample+1)
    cz_bin_width = all_czs_bins[1] - all_czs_bins[0]
    all_czs = np.linspace(all_czs_bins[0] + cz_bin_width/2.0,
                          all_czs_bins[-1] - cz_bin_width/2.0,
                          20*oversample)

    all_ens_mg, all_czs_mg = np.meshgrid(all_ens, all_czs)
    
    for flav, flavtex in zip(primaries, texprimaries):

        all_flux_weights = calculate_2D_flux_weights(all_ens_mg.flatten(),
                                                     all_czs_mg.flatten(),
                                                     spline_dict[flav],
                                                     enpow=enpow)

        all_flux_weights = np.array(np.split(all_flux_weights,
                                             len(all_czs)))
        all_flux_weights_map = {}
        all_flux_weights_map['map'] = all_flux_weights.T
        all_flux_weights_map['ebins'] = all_ens_bins
        all_flux_weights_map['czbins'] = all_czs_bins
        
        gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(12,10))

        logplot(m=all_flux_weights_map,
                title='Finely Interpolated %s Flux'%flavtex,
                ax=axes,
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                largelabels=True)

        fig.savefig(
            os.path.join(
                outdir,
                '%s_%s2dinterpolation.png'%(SaveName,flav)
            )
        )

        if ip_checks:

            downsampled_flux_map = {}
            downsampled_flux_map['map'] = take_average(
                all_flux_weights.T, oversample
            )
            downsampled_flux_map['ebins'] = np.logspace(-1.025,4.025,102)
            downsampled_flux_map['czbins'] = np.linspace(-1.0,1.0,21)

            honda_tables = {}
            honda_tables['map'] = flux_dict[flav].T
            honda_tables['ebins'] = np.logspace(-1.025,4.025,102)
            honda_tables['czbins'] = np.linspace(-1.0,1.0,21)

            diff_map = {}
            diff_map['map'] = honda_tables['map']-downsampled_flux_map['map']
            diff_map['ebins'] = np.logspace(-1.025,4.025,102)
            diff_map['czbins'] = np.linspace(-1.0,1.0,21)

            diff_ratio_map = {}
            diff_ratio_map['map'] = diff_map['map'] / honda_tables['map']
            diff_ratio_map['ebins'] = np.logspace(-1.025,4.025,102)
            diff_ratio_map['czbins'] = np.linspace(-1.0,1.0,21)
                
            gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=5,
                                     gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False,
                                     figsize=(20,5))

            logplot(m=all_flux_weights_map,
                    title='Oversampled by %i'%oversample,
                    ax=axes[0],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=downsampled_flux_map,
                    title='Downsampled to Honda Binning',
                    ax=axes[1],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=honda_tables,
                    title='Honda Tables',
                    ax=axes[2],
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
            logplot(m=diff_map,
                    title='Difference',
                    ax=axes[3],
                    clabel=None)
            logplot(m=diff_ratio_map,
                    title='Percentage Difference',
                    ax=axes[4],
                    clabel=None)

            plt.suptitle(
                'Integral Preserving Tests for %s %s Flux Tables'
                %(flavtex,TitleFileName), fontsize=36
            )
            plt.subplots_adjust(top=0.8)
            fig.savefig(
                os.path.join(
                    outdir,
                    '%s_%siptest_fullrange.png'%(SaveName,flav)
                )
            )
            plt.close(fig.number)
            

def do_1D_2D_bartol_test(spline_dict, flux_dict, outdir, enpow=1):

    czs = np.linspace(-1,1,81)
    low_ens = 4.732*np.ones_like(czs)
    high_ens = 44.70*np.ones_like(czs)

    ens = np.logspace(-1,4,701)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)

    for flav, flavtex in zip(primaries, texprimaries):
        
        low_en_flux_weights = calculate_2D_flux_weights(low_ens,
                                                        czs,
                                                        spline_dict[flav],
                                                        enpow=enpow)
            
        high_en_flux_weights = calculate_2D_flux_weights(high_ens,
                                                         czs,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        flux5 = flux_dict[flav].T[np.where(flux_dict['energy']==4.732)][0]
        flux50 = flux_dict[flav].T[np.where(flux_dict['energy']==44.70)][0]

        Plot1DSlices(
            xintvals = czs,
            yintvals = low_en_flux_weights,
            xtabvals = flux_dict['coszen'],
            ytabvals = flux5,
            xtabbins = np.linspace(-1,1,21),
            xlabel = r'$\cos\theta_Z$',
            ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext = 0.75,
            ytext = 0.7,
            text = 'Slice at \n 4.732 GeV',
            tablename = 'Bartol SNO 2004',
            savename = os.path.join(
                outdir,'bartol_%sfluxweightstest5GeV.png'%flav
            ),
            log = False
        )
        
        Plot1DSlices(
            xintvals = czs,
            yintvals = high_en_flux_weights,
            xtabvals = flux_dict['coszen'],
            ytabvals = flux50,
            xtabbins = np.linspace(-1,1,21),
            xlabel = r'$\cos\theta_Z$',
            ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
            xtext = 0.75,
            ytext = 0.7,
            text = 'Slice at \n 44.70 GeV',
            tablename = 'Bartol SNO 2004',
            savename = os.path.join(
                outdir,'bartol_%sfluxweightstest50GeV.png'%flav
            ),
            log = False
        )

        upgoing_flux_weights = calculate_2D_flux_weights(ens,
                                                         upgoing,
                                                         spline_dict[flav],
                                                         enpow=enpow)

        downgoing_flux_weights = calculate_2D_flux_weights(ens,
                                                           downgoing,
                                                           spline_dict[flav],
                                                           enpow=enpow)

        upgoing_flux_weights *= np.power(ens,3)
        downgoing_flux_weights *= np.power(ens,3)
        
        coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
        coszen_strs = np.array(coszen_strs)

        fluxupgoing = flux_dict[flav][np.where(coszen_strs=='-0.95')][0]
        fluxdowngoing = flux_dict[flav][np.where(coszen_strs=='0.35')][0]

        fluxupgoing *= np.power(flux_dict['energy'],3)
        fluxdowngoing *= np.power(flux_dict['energy'],3)

        low_log_energy = np.logspace(-1,1,41)
        high_log_energy = np.logspace(1.1,4,30)
        xtabbins = np.concatenate(
            [low_log_energy,high_log_energy]
        )

        if 'numu' in flav:
            xtext = 0.68
            ytext = 0.25
        elif 'nue' in flav:
            xtext = 0.35
            ytext = 0.25

        Plot1DSlices(
            xintvals = ens,
            yintvals = upgoing_flux_weights,
            xtabvals = flux_dict['energy'],
            ytabvals = fluxupgoing,
            xtabbins = xtabbins,
            xlabel = 'Neutrino Energy (GeV)',
            ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext = xtext,
            ytext = ytext,
            text = r'Slice at $\cos\theta_Z=-0.95$',
            tablename = 'Bartol SNO 2004',
            savename = os.path.join(
                outdir,'bartol_%sfluxweightstest-0.95cz.png'%flav
            ),
            log = True
        )
            
        Plot1DSlices(
            xintvals = ens,
            yintvals = downgoing_flux_weights,
            xtabvals = flux_dict['energy'],
            ytabvals = fluxdowngoing,
            xtabbins = xtabbins,
            xlabel = 'Neutrino Energy (GeV)',
            ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
            xtext = xtext,
            ytext = ytext,
            text = r'Slice at $\cos\theta_Z=0.35$',
            tablename = 'Bartol SNO 2004',
            savename = os.path.join(
                outdir,'bartol_%sfluxweightstest0.35cz.png'%flav
            ),
            log = True
        )
            

def do_2D_2D_bartol_test(spline_dict, flux_dict, outdir, ip_checks,
                      oversample, enpow=1):

    all_en_bins_low = np.logspace(-1.0,1.0,40*oversample+1)
    all_log_en_bins_low = np.linspace(-1.0,1.0,40*oversample+1)
    log_en_bin_width_low = all_log_en_bins_low[1] - all_log_en_bins_low[0]
    all_ens_low = np.logspace(
        all_log_en_bins_low[0]+log_en_bin_width_low/2.0,
        all_log_en_bins_low[-1]-log_en_bin_width_low/2.0,
        40*oversample
    )

    all_en_bins_high = np.logspace(1.0,4.0,30*oversample+1)
    all_log_en_bins_high = np.linspace(1.0,4.0,30*oversample+1)
    log_en_bin_width_high = all_log_en_bins_high[1] - all_log_en_bins_high[0]
    all_ens_high = np.logspace(
        all_log_en_bins_high[0]+log_en_bin_width_high/2.0,
        all_log_en_bins_high[-1]-log_en_bin_width_high/2.0,
        30*oversample
    )

    all_en_bins = [all_en_bins_low, all_en_bins_high]
    all_ens = [all_ens_low, all_ens_high]

    all_all_ens_bins = np.concatenate(
        (
            np.logspace(-1.0,1.0,40*oversample+1),
            np.logspace(1.0+log_en_bin_width_high,4.0,30*oversample)
        )
    )

    all_czs_bins = np.linspace(-1.0,1.0,20*oversample+1)
    cz_bin_width = all_czs_bins[1] - all_czs_bins[0]
    all_czs = np.linspace(
        all_czs_bins[0] + cz_bin_width/2.0,
        all_czs_bins[-1] - cz_bin_width/2.0,
        20*oversample
    )

    en_labels = ['3DCalc', '1DCalc']

    all_fluxes = {}
    for flav in primaries:
        all_fluxes[flav] = []

    for all_en, all_ens_bins, en_label in zip(all_ens,
                                              all_en_bins,
                                              en_labels):
        all_ens_mg, all_czs_mg = np.meshgrid(all_en, all_czs)
        
        for flav, flavtex in zip(primaries, texprimaries):

            all_flux_weights = calculate_2D_flux_weights(all_ens_mg.flatten(),
                                                         all_czs_mg.flatten(),
                                                         spline_dict[flav],
                                                         enpow=enpow)

            all_flux_weights = np.array(np.split(all_flux_weights,
                                                 len(all_czs)))

            if len(all_fluxes[flav]) == 0:
                all_fluxes[flav] = all_flux_weights.T
            else:
                all_fluxes[flav] = np.concatenate((all_fluxes[flav],
                                                   all_flux_weights.T))

                
            all_flux_weights_map = {}
            all_flux_weights_map['map'] = all_flux_weights.T
            all_flux_weights_map['ebins'] = all_ens_bins
            all_flux_weights_map['czbins'] = all_czs_bins
        
            gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
            fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                     sharex=False, sharey=False, figsize=(12,10))

            logplot(m=all_flux_weights_map,
                    title='Finely Interpolated %s Flux'%flavtex,
                    ax=axes,
                    clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                    largelabels=True)

            fig.savefig(os.path.join(outdir,
                                     'bartol_%s_%s2dinterpolation.png'%(en_label,flav)))

            if ip_checks:

                bartol_tables = {}
                if en_label == '3DCalc':
                    bartol_tables['map'] = flux_dict[flav].T[:40]
                    bartol_tables['ebins'] = np.logspace(-1.0,1.0,41)
                elif en_label == '1DCalc':
                    bartol_tables['map'] = flux_dict[flav].T[40:]
                    bartol_tables['ebins'] = np.logspace(1.0,4.0,31)
                bartol_tables['czbins'] = np.linspace(-1.0,1.0,21)

                downsampled_flux_map = {}
                downsampled_flux_map['map'] = take_average(
                    all_flux_weights.T, oversample
                )
                downsampled_flux_map['ebins'] = bartol_tables['ebins']
                downsampled_flux_map['czbins'] = np.linspace(-1.0,1.0,21)
                
                diff_map = {}
                diff_map['map'] = bartol_tables['map']-downsampled_flux_map['map']
                diff_map['ebins'] = bartol_tables['ebins']
                diff_map['czbins'] = np.linspace(-1.0,1.0,21)
                
                diff_ratio_map = {}
                diff_ratio_map['map'] = diff_map['map'] / bartol_tables['map']
                diff_ratio_map['ebins'] = bartol_tables['ebins']
                diff_ratio_map['czbins'] = np.linspace(-1.0,1.0,21)
                
                gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
                fig, axes = plt.subplots(nrows=1, ncols=5,
                                         gridspec_kw=gridspec_kw,
                                         sharex=False, sharey=False,
                                         figsize=(20,5))
                
                logplot(m=all_flux_weights_map,
                        title='Oversampled by %i'%oversample,
                        ax=axes[0],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=downsampled_flux_map,
                        title='Downsampled to Bartol Binning',
                        ax=axes[1],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=bartol_tables,
                        title='Bartol Tables',
                        ax=axes[2],
                        clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,)
                logplot(m=diff_map,
                        title='Difference',
                        ax=axes[3],
                        clabel=None)
                logplot(m=diff_ratio_map,
                        title='Percentage Difference',
                        ax=axes[4],
                        clabel=None)

                plt.suptitle('Integral Preserving Tests for %s Bartol Sudbury 2015 Flux Tables'%flavtex, fontsize=36)
                plt.subplots_adjust(top=0.8)
                fig.savefig(os.path.join(outdir,'bartol_%s_%siptest_fullrange.png'%(en_label,flav)))
                plt.close(fig.number)

    for flav, flavtex in zip(primaries, texprimaries):
                
        all_flux_weights_map = {}
        all_flux_weights_map['map'] = all_fluxes[flav]
        all_flux_weights_map['ebins'] = all_all_ens_bins
        all_flux_weights_map['czbins'] = all_czs_bins
            
        gridspec_kw = dict(left=0.15, right=0.90, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=1, gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False, figsize=(12,10))

        logplot(m=all_flux_weights_map,
                title='Finely Interpolated %s Flux'%flavtex,
                ax=axes,
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                largelabels=True)

        fig.savefig(os.path.join(outdir,
                                 'bartol_%s2dinterpolation.png'%flav))

            
def do_2D_2D_comparisons(honda_spline_dict, bartol_spline_dict,
                      outdir, oversample, enpow=1):
        
    all_ens_bins = np.logspace(-1.0,4.0,100*oversample+1)
    all_czs_bins = np.linspace(-1.0,1.0,20*oversample+1)
    # need log energy bin width for defining evaluation points
    log_en_bin_width = np.linspace(-1.0,4.0,100*oversample+1)[1]-np.linspace(-1.0,4.0,100*oversample+1)[0]
    cz_bin_width = all_czs_bins[1]-all_czs_bins[0]
    all_ens = np.logspace(-1.0+log_en_bin_width/2.0,
                          4.0-log_en_bin_width/2.0,
                          100*oversample)
    all_czs = np.linspace(-1.0+cz_bin_width/2.0,
                          1.0-cz_bin_width/2.0,
                          20*oversample)

    all_ens_mg, all_czs_mg = np.meshgrid(all_ens, all_czs)
    
    for flav, flavtex in zip(primaries, texprimaries):

        honda_flux_weights = calculate_2D_flux_weights(
            all_ens_mg.flatten(),
            all_czs_mg.flatten(),
            honda_spline_dict[flav],
            enpow=enpow
        )
        bartol_flux_weights = calculate_2D_flux_weights(
            all_ens_mg.flatten(),
            all_czs_mg.flatten(),
            bartol_spline_dict[flav],
            enpow=enpow
        )

        honda_flux_weights = np.array(np.split(honda_flux_weights,
                                               len(all_czs)))
        bartol_flux_weights = np.array(np.split(bartol_flux_weights,
                                                len(all_czs)))
            
        honda_flux_weights_map = {}
        honda_flux_weights_map['map'] = honda_flux_weights.T
        honda_flux_weights_map['ebins'] = all_ens_bins
        honda_flux_weights_map['czbins'] = all_czs_bins

        bartol_flux_weights_map = {}
        bartol_flux_weights_map['map'] = bartol_flux_weights.T
        bartol_flux_weights_map['ebins'] = all_ens_bins
        bartol_flux_weights_map['czbins'] = all_czs_bins

        diff_map = {}
        diff_map['map'] = honda_flux_weights_map['map']-bartol_flux_weights_map['map']
        diff_map['ebins'] = all_ens_bins
        diff_map['czbins'] = all_czs_bins

        diff_ratio_map = {}
        diff_ratio_map['map'] = diff_map['map'] / honda_flux_weights_map['map']
        diff_ratio_map['ebins'] = all_ens_bins
        diff_ratio_map['czbins'] = all_czs_bins
        
        gridspec_kw = dict(left=0.03, right=0.968, wspace=0.32)
        fig, axes = plt.subplots(nrows=1, ncols=4,
                                 gridspec_kw=gridspec_kw,
                                 sharex=False, sharey=False,
                                 figsize=(16,5))

        logplot(m=honda_flux_weights_map,
                title='Honda SNO 2015 %s Flux'%flavtex,
                ax=axes[0],
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex)
        logplot(m=bartol_flux_weights_map,
                title='Bartol SNO 2004 %s Flux'%flavtex,
                ax=axes[1],
                clabel=r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex)
        logplot(m=diff_map,
                title='Difference',
                ax=axes[2],
                clabel=None)
        logplot(m=diff_ratio_map,
                title='Percentage Difference to Honda',
                ax=axes[3],
                clabel=None)

        plt.suptitle('Comparisons for %s Honda 2015 and Bartol 2004 Sudbury Flux Tables'%flavtex, fontsize=36)
        plt.subplots_adjust(top=0.8)
        
        fig.savefig(os.path.join(outdir,
                                 'honda_bartol_%s2dcomparisons.png'%flav))


def do_1D_3D_honda_test(spline_dict, flux_dict, LegendFileName,
                        SaveName, outdir, enpow=1, az_linear=True):

    czs = np.linspace(-1,1,81)
    low_ens = 5.0119*np.ones_like(czs)
    high_ens = 50.119*np.ones_like(czs)
    low_azs = 75.0*np.ones_like(czs)*np.pi/180.0
    high_azs = 285.0*np.ones_like(czs)*np.pi/180.0
    
    ens = np.logspace(-1.025,4.025,1020)
    upgoing = -0.95*np.ones_like(ens)
    downgoing = 0.35*np.ones_like(ens)
    low_azs_two = 75.0*np.ones_like(ens)*np.pi/180.0
    high_azs_two = 285.0*np.ones_like(ens)*np.pi/180.0

    azs = np.linspace(0.0,360.0,121)*np.pi/180.0
    low_ens_two = 5.0119*np.ones_like(azs)
    high_ens_two = 50.119*np.ones_like(azs)
    upgoing_two = -0.95*np.ones_like(azs)
    downgoing_two  = 0.35*np.ones_like(azs)

    lin_azs = [True, False]
    name_additions = ['lin_az', 'ip_az']

    for lin_az, name_addition in zip(lin_azs, name_additions):

        NewSaveName = SaveName + '_%s'%name_addition

        for flav, flavtex in zip(primaries, texprimaries):
    
            low_en_low_az_flux_weights = calculate_3D_flux_weights(
                low_ens,
                czs,
                low_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            high_en_low_az_flux_weights = calculate_3D_flux_weights(
                high_ens,
                czs,
                low_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            low_en_high_az_flux_weights = calculate_3D_flux_weights(
                low_ens,
                czs,
                high_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            high_en_high_az_flux_weights = calculate_3D_flux_weights(
                high_ens,
                czs,
                high_azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )

            flux5lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0][
                    np.where(flux_dict['energy']==50.119)][0]
            
            flux5highaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50highaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0][
                    np.where(flux_dict['energy']==50.119)][0]

            Plot1DSlices(
                xintvals = czs,
                yintvals = low_en_low_az_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux5lowaz,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = 'Slice at \n 5.0119 GeV \n $\phi_{Az}=75^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest5GeV75Az.png'%(NewSaveName,flav)
                ),
                log = False
            )
        
            Plot1DSlices(
                xintvals = czs,
                yintvals = high_en_low_az_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux50lowaz,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = 'Slice at \n 50.119 GeV \n $\phi_{Az}=75^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest50GeV75Az.png'%(NewSaveName,flav)
                ),
                log = False
            )

            Plot1DSlices(
                xintvals = czs,
                yintvals = low_en_high_az_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux5highaz,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = 'Slice at \n 5.0119 GeV \n $\phi_{Az}=285^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest5GeV285Az.png'%(NewSaveName,flav)
                ),
                log = False
            )
        
            Plot1DSlices(
                xintvals = czs,
                yintvals = high_en_high_az_flux_weights,
                xtabvals = flux_dict['coszen'],
                ytabvals = flux50highaz,
                xtabbins = np.linspace(-1,1,21),
                xlabel = r'$\cos\theta_Z$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = 'Slice at \n 50.119 GeV \n $\phi_{Az}=285^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest50GeV285Az.png'%(NewSaveName,flav)
                ),
                log = False
            )

            upgoing_flux_weights_low_azs = calculate_3D_flux_weights(
                ens,
                upgoing,
                low_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            downgoing_flux_weights_low_azs = calculate_3D_flux_weights(
                ens,
                downgoing,
                low_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            upgoing_flux_weights_high_azs = calculate_3D_flux_weights(
                ens,
                upgoing,
                high_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            downgoing_flux_weights_high_azs = calculate_3D_flux_weights(
                ens,
                downgoing,
                high_azs_two,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            upgoing_flux_weights_low_azs *= np.power(ens,3)
            downgoing_flux_weights_low_azs *= np.power(ens,3)
            upgoing_flux_weights_high_azs *= np.power(ens,3)
            downgoing_flux_weights_high_azs *= np.power(ens,3)
            
            coszen_strs = ['%.2f'%coszen for coszen in flux_dict['coszen']]
            coszen_strs = np.array(coszen_strs)
            
            flux5lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50lowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0][
                    np.where(flux_dict['energy']==50.119)][0]
            
            flux5highaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50highaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0][
                    np.where(flux_dict['energy']==50.119)][0]
            
            fluxupgoinglowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0].T[
                    np.where(coszen_strs=='-0.95')][0]
            fluxdowngoinglowaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==75.0)][0].T[
                    np.where(coszen_strs=='0.35')][0]
            fluxupgoinghighaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0].T[
                    np.where(coszen_strs=='-0.95')][0]
            fluxdowngoinghighaz = flux_dict[flav][
                np.where(flux_dict['azimuth']==285.0)][0].T[
                    np.where(coszen_strs=='0.35')][0]
            
            fluxupgoinglowaz *= np.power(flux_dict['energy'],3)
            fluxdowngoinglowaz *= np.power(flux_dict['energy'],3)
            fluxupgoinghighaz *= np.power(flux_dict['energy'],3)
            fluxdowngoinghighaz *= np.power(flux_dict['energy'],3)
            
            if 'numu' in flav:
                xtext = 0.68
                ytext = 0.28
            elif 'nue' in flav:
                xtext = 0.35
                ytext = 0.28
                
            Plot1DSlices(
                xintvals = ens,
                yintvals = upgoing_flux_weights_low_azs,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxupgoinglowaz,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=-0.95$'+'\n'+r'$\phi_{Az}=75^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest-0.95cz75Az.png'%(NewSaveName,flav)
                ),
                log = True
            )
            
            Plot1DSlices(
                xintvals = ens,
                yintvals = downgoing_flux_weights_low_azs,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxdowngoinglowaz,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=0.35$'+'\n'+r'$\phi_{Az}=75^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest0.35cz75Az.png'%(NewSaveName,flav)
                ),
                log = True
            )

            Plot1DSlices(
                xintvals = ens,
                yintvals = upgoing_flux_weights_high_azs,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxupgoinghighaz,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=-0.95$'+'\n'+r'$\phi_{Az}=285^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest-0.95cz285Az.png'%(NewSaveName,flav)
                ),
                log = True
            )
            
            Plot1DSlices(
                xintvals = ens,
                yintvals = downgoing_flux_weights_high_azs,
                xtabvals = flux_dict['energy'],
                ytabvals = fluxdowngoinghighaz,
                xtabbins = np.logspace(-1.025,4.025,102),
                xlabel = 'Neutrino Energy (GeV)',
                ylabel = r'%s Flux $\times E_{\nu}^3$ $\left([m^2\,s\,sr\,GeV]^{-1}[GeV]^3\right)$'%flavtex,
                xtext = xtext,
                ytext = ytext,
                text = r'Slice at $\cos\theta_Z=0.35$'+'\n'+r'$\phi_{Az}=285^{\circ}$',
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest0.35cz285Az.png'%(NewSaveName,flav)
                ),
                log = True
            )
            
            low_en_upgoing_flux_weights = calculate_3D_flux_weights(
                low_ens_two,
                upgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            high_en_upgoing_flux_weights = calculate_3D_flux_weights(
                high_ens_two,
                upgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            low_en_downgoing_flux_weights = calculate_3D_flux_weights(
                low_ens_two,
                downgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            high_en_downgoing_flux_weights = calculate_3D_flux_weights(
                high_ens_two,
                downgoing_two,
                azs,
                spline_dict[flav],
                enpow=enpow,
                az_linear=lin_az
            )
            
            flux5downgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen']==0.35)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50downgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen']==0.35)][0][
                    np.where(flux_dict['energy']==50.119)][0]
            
            flux5upgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen']==-0.95)][0][
                    np.where(flux_dict['energy']==5.0119)][0]
            flux50upgoing = flux_dict[flav].T[
                np.where(flux_dict['coszen']==-0.95)][0][
                    np.where(flux_dict['energy']==50.119)][0]

            Plot1DSlices(
                xintvals = np.linspace(0.0,360.0,121),
                yintvals = low_en_upgoing_flux_weights,
                xtabvals = flux_dict['azimuth'],
                ytabvals = flux5upgoing,
                xtabbins = np.linspace(0.0,360.0,13),
                xlabel = r'$\phi_{Az}$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = "Slice at \n 5.0119 GeV \n"+r" $\cos\theta_Z=-0.95$",
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest5GeV-0.95cz.png'%(NewSaveName,flav)
                ),
                log = False
            )

            Plot1DSlices(
                xintvals = np.linspace(0.0,360.0,121),
                yintvals = high_en_upgoing_flux_weights,
                xtabvals = flux_dict['azimuth'],
                ytabvals = flux50upgoing,
                xtabbins = np.linspace(0.0,360.0,13),
                xlabel = r'$\phi_{Az}$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = "Slice at \n 50.119 GeV \n"+r"$\cos\theta_Z=-0.95$",
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest50GeV-0.95cz.png'%(NewSaveName,flav)
                ),
                log = False
            )

            Plot1DSlices(
                xintvals = np.linspace(0.0,360.0,121),
                yintvals = low_en_downgoing_flux_weights,
                xtabvals = flux_dict['azimuth'],
                ytabvals = flux5downgoing,
                xtabbins = np.linspace(0.0,360.0,13),
                xlabel = r'$\phi_{Az}$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = "Slice at \n 5.0119 GeV \n"+r"$\cos\theta_Z=0.35$",
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest5GeV0.35cz.png'%(NewSaveName,flav)
                ),
                log = False
            )

            Plot1DSlices(
                xintvals = np.linspace(0.0,360.0,121),
                yintvals = high_en_downgoing_flux_weights,
                xtabvals = flux_dict['azimuth'],
                ytabvals = flux50downgoing,
                xtabbins = np.linspace(0.0,360.0,13),
                xlabel = r'$\phi_{Az}$',
                ylabel = r'%s Flux $\left([m^2\,s\,sr\,GeV]^{-1}\right)$'%flavtex,
                xtext = 0.75,
                ytext = 0.68,
                text = "Slice at \n 50.119 GeV \n"+r"$\cos\theta_Z=0.35$",
                tablename = LegendFileName,
                savename = os.path.join(
                    outdir,'%s_%sfluxweightstest50GeV0.35cz.png'%(NewSaveName,flav)
                ),
                log = False
            )
        

if __name__ == '__main__':
        
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flux_file_2D',type=str,
                        default=None,
                        help='''2D flux file you want to run tests on. If one 
                        is not specified then no 2D tests will be done.''')
    parser.add_argument('--flux_file_3D',type=str,
                        default=None,
                        help='''3D flux file you want to run tests on. If one 
                        is not specified then no 3D tests will be done.''')
    parser.add_argument('--onedim_checks', action='store_true',
                        help='''Run verifications on 1D slices.''')
    parser.add_argument('--twodim_checks', action='store_true',
                        help='''Make finely interpolated 2D plots.
                        WARNING - THESE ARE SLOW.''')
    parser.add_argument('--ip_checks', action='store_true',
                        help='''Run checks on integral-preserving nature.
                        WARNING - THESE ARE VERY SLOW.''')
    parser.add_argument('--comparisons', action='store_true',
                        help='''Run comparisons between a Bartol and Honda 
                        flux file. WARNING - ALSO VERY SLOW.''')
    parser.add_argument('--oversample', type=int, default=10,
                        help='''Integer to oversample for integral-preserving
                        checks and comparisons between flux files.''')
    parser.add_argument('--enpow', type=int, default=1,
                        help='''Power of energy to use in making the energy
                        splines i.e. flux * (energy**enpow).''')
    parser.add_argument('--outdir', metavar='DIR', type=str, required=True,
                        help='''Store all output plots to this directory.''')

    args = parser.parse_args()

    if args.flux_file_2D is not None:

        if ('honda' not in args.flux_file_2D) and ('bartol' not in args.flux_file_2D):
            raise ValueError('Type of flux file not recognised.')

        spline_dict_2D, flux_dict_2D = load_2D_table(
            args.flux_file_2D,
            enpow=args.enpow,
            returnTable=True
        )
        
        if 'honda' in args.flux_file_2D:

            flux_file_2D_name = args.flux_file_2D.split('/')[-1]
            flux_file_2D_bits = flux_file_2D_name.split('-')
            year = flux_file_2D_bits[1]
            site = flux_file_2D_bits[2]

            TitleFileName = 'Honda'
            LegendFileName = 'Honda'

            if site == 'spl':
                TitleFileName += ' South Pole'
                LegendFileName += ' SPL'
            elif site == 'sno':
                TitleFileName += ' Sudbury'
                LegendFileName += ' SNO'
            else:
                logging.warn('Don\'t know what to do with site %s.'
                             'Omitting from titles'%site)

            TitleFileName += ' %s'%year
            LegendFileName += ' %s'%year
            SaveName = 'honda_2D_%s_%s'%(site,year)
        
            if args.onedim_checks:
                do_1D_2D_honda_test(
                    spline_dict = spline_dict_2D,
                    flux_dict = flux_dict_2D,
                    LegendFileName = LegendFileName,
                    SaveName = SaveName,
                    outdir = args.outdir,
                    enpow = args.enpow
                )

            if args.twodim_checks:
                do_2D_2D_honda_test(
                    spline_dict = spline_dict_2D,
                    flux_dict = flux_dict_2D,
                    outdir = args.outdir,
                    ip_checks = args.ip_checks,
                    oversample = args.oversample,
                    SaveName = SaveName,
                    TitleFileName = TitleFileName,
                    enpow = args.enpow
                )

        else:

            if args.onedim_checks:
                do_1D_2D_bartol_test(
                    spline_dict = spline_dict_2D,
                    flux_dict = flux_dict_2D,
                    outdir = args.outdir,
                    enpow = args.enpow
                )

            if args.twodim_checks:
                do_2D_2D_bartol_test(
                    spline_dict = spline_dict_2D,
                    flux_dict = flux_dict_2D,
                    outdir = args.outdir,
                    ip_checks = args.ip_checks,
                    oversample = args.oversample,
                    enpow = args.enpow
                )

    if args.flux_file_3D is not None:

        spline_dict_3D, flux_dict_3D = load_3D_table(
            args.flux_file_3D,
            enpow=args.enpow,
            returnTable=True
        )

        flux_file_3D_name = args.flux_file_3D.split('/')[-1]
        flux_file_3D_bits = flux_file_3D_name.split('-')
        year = flux_file_3D_bits[1]
        site = flux_file_3D_bits[2]

        TitleFileName = 'Honda'
        LegendFileName = 'Honda'

        if site == 'spl':
            TitleFileName += ' South Pole'
            LegendFileName += ' SPL'
        elif site == 'sno':
            TitleFileName += ' Sudbury'
            LegendFileName += ' SNO'
        else:
            logging.warn('Don\'t know what to do with site %s.'
                         'Omitting from titles'%site)

        TitleFileName += ' %s'%year
        LegendFileName += ' %s'%year
        SaveName = 'honda_3D_%s_%s'%(site,year)

        if args.onedim_checks:
            do_1D_3D_honda_test(
                spline_dict = spline_dict_3D,
                flux_dict = flux_dict_3D,
                LegendFileName = LegendFileName,
                SaveName = SaveName,
                outdir = args.outdir,
                enpow = args.enpow
            )

    if args.comparisons:

        logging.warning('Comparisons will be of Honda 2015 SNO and '
                        'Bartol 2004 SNO 2D tables regardless of what you set '
                        'in the flux_file argument(s).')

        honda_spline_dict_2D = load_2D_table(
            'flux/honda-2015-sno-solmax-aa.d',
            enpow=args.enpow
        )

        bartol_spline_dict_2D = load_2D_table(
            'flux/bartol-2004-sno-solmax-aa.d',
            enpow=args.enpow
        )
        
        do_2D_2D_comparisons(
            honda_spline_dict = honda_spline_dict_2D,
            bartol_spline_dict = bartol_spline_dict_2D,
            outdir = args.outdir,
            oversample = args.oversample,
            enpow = args.enpow
        )
