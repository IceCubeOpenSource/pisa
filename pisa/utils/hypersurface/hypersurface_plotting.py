'''
Hypersurface Plotting functions
'''

__all__ = ['plot_bin_fits', 'plot_bin_fits_2d']

__author__ = 'T. Stuttard, A. Trettin'

__license__ = '''Copyright (c) 2014-2017, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''
 
import numpy as np

def plot_bin_fits(ax, hypersurface, bin_idx, param_name, color=None, label=None, show_nominal=False, show_offaxis=True, show_zero=False, show_uncertainty=True):
    '''
    Plot the hypersurface for a given bin, in 1D w.r.t. to a single specified parameter.
    Plots the following:
      - on-axis data points used in the fit
      - hypersurface w.r.t to the specified parameter (1D)
      - nominal value of the specified parameter

    Parameters
    ----------
    ax : matplotlib.Axes
        matplotlib ax to draw the plot on

    hypersurface : Hypersurface
        Hypersurface to make the plots from

    bin_idx : tuple
        Index (numpy array indexing format) of the bin to plot

    param_name : str
        Name of the parameter of interest

    color : str
        color to use for hypersurface curve

    label : str
        label to use for hypersurface curve

    show_nominal : bool
        Indicate the nominal value of the param on the plot

    show_uncertainty : bool
        Indicate the hypersurface uncertainty on the plot
    '''

    import matplotlib.pyplot as plt

    # Get the param
    param = hypersurface.params[param_name]

    # Check bin index
    assert len(bin_idx) == len(hypersurface.binning.shape)

    # Get bin values for this bin only
    try:
        chosen_bin_values = np.squeeze(
            [m.nominal_values[bin_idx] for m in hypersurface.fit_maps])
        chosen_bin_sigma = np.squeeze([m.std_devs[bin_idx]
                                       for m in hypersurface.fit_maps])
    except:
        # sometimes maps aren't stored, like when we are recovering interpolated
        # hypersurfaces
        chosen_bin_values = np.full(hypersurface.num_fit_sets, np.nan)
        chosen_bin_sigma = np.full(hypersurface.num_fit_sets, np.nan)

    # Define a mask for selecting on-axis points only
    on_axis_mask = hypersurface.get_on_axis_mask(param.name)
    with np.errstate(invalid='ignore'):  # empty bins are a regular occurrence
        include_mask = np.ones_like(on_axis_mask) if show_zero else (
            np.asarray(chosen_bin_values) > 0.)

    # Plot the points from the datasets used for fitting
    x = np.asarray(param.fit_param_values)[on_axis_mask & include_mask]
    y = np.asarray(chosen_bin_values)[on_axis_mask & include_mask]
    yerr = np.asarray(chosen_bin_sigma)[on_axis_mask & include_mask]

    ax.errorbar(x=x, y=y, yerr=yerr, marker="o", color=(
        "black" if color is None else color), linestyle="None", label=label)

    # Plot off-axis points by projecting them along the fitted surface on the axis.
    if show_offaxis:
        x = np.asarray(param.fit_param_values)
        y = np.asarray(chosen_bin_values)
        yerr = np.asarray(chosen_bin_sigma)
        prediction = hypersurface.evaluate(
            hypersurface.fit_param_values, bin_idx=bin_idx)
        params_for_projection = {param.name: x}
        for p in list(hypersurface.params.values()):
            if p.name != param.name:
                params_for_projection[p.name] = np.full_like(
                    x, hypersurface.nominal_values[p.name])
        prediction_on_axis = hypersurface.evaluate(
            params_for_projection, bin_idx=bin_idx)
        y_projected = y - prediction + prediction_on_axis
        ax.errorbar(x=x[~on_axis_mask & include_mask],
                    y=y_projected[~on_axis_mask & include_mask],
                    yerr=yerr[~on_axis_mask & include_mask],
                    marker="o", color=("black" if color is None else color), linestyle="None",
                    alpha=0.5,
                    )
    # Plot the hypersurface
    # Generate as bunch of values along the sys param axis to make the plot
    # Then calculate the hypersurface value at each point, using the nominal values for all other sys params
    x_plot = np.linspace(np.nanmin(param.fit_param_values),
                         np.nanmax(param.fit_param_values), num=100)
    params_for_plot = {param.name: x_plot, }
    for p in list(hypersurface.params.values()):
        if p.name != param.name:
            params_for_plot[p.name] = np.full_like(
                x_plot, hypersurface.nominal_values[p.name])
    y_plot, y_sigma = hypersurface.evaluate(
        params_for_plot, bin_idx=bin_idx, return_uncertainty=True)
    ax.plot(x_plot, y_plot, color=("red" if color is None else color))
    # y_sigma = hypersurface.uncertainty(params_for_plot, bin_idx=bin_idx)
    if show_uncertainty:
        ax.fill_between(x_plot, y_plot - y_sigma, y_plot + y_sigma,
                        color=("red" if color is None else color), alpha=0.2)

    # Show the nominal value
    if show_nominal:
        ax.axvline(x=param.nominal_value, color="blue",
                   alpha=0.7, linestyle="-", zorder=-1)

    # Format ax
    ax.set_xlabel(param.name)
    ax.grid(True)
    ax.legend()


def plot_bin_fits_2d(ax, hypersurface, bin_idx, param_names):
    '''
    Plot the hypersurface for a given bin, in 2D w.r.t. to a pair of params
    Plots the following:
      - All data points used in the fit
      - hypersurface w.r.t to the specified parameters (2D)
      - nominal value of the specified parameters

    Parameters
    ----------
    ax : matplotlib.Axes
        matplotlib ax to draw the plot on

    hypersurface : Hypersurface
        Hypersurface to make the plots from

    bin_idx : tuple
        Index (numpy array indexing format) of the bin to plot

    param_names : list of str
        List containing the names of the two parameters of interest
    '''

    import matplotlib.pyplot as plt

    assert len(param_names) == 2
    assert len(bin_idx) == len(hypersurface.binning.shape)

    # Get bin values for this bin only
    chosen_bin_values = [m.nominal_values[bin_idx]
                         for m in hypersurface.fit_maps]
    chosen_bin_sigma = [m.std_devs[bin_idx] for m in hypersurface.fit_maps]

    # Shortcuts to the param values and bin values
    p0 = hypersurface.params[param_names[0]]
    p1 = hypersurface.params[param_names[1]]
    z = np.asarray(chosen_bin_values)
    # zerr = #TODO error bars

    # Choose categories of points to plot
    nominal_mask = hypersurface.get_nominal_mask()
    p0_on_axis_mask = hypersurface.get_on_axis_mask(p0.name) & (~nominal_mask)
    p1_on_axis_mask = hypersurface.get_on_axis_mask(p1.name) & (~nominal_mask)

    off_axis_mask = np.ones_like(p1_on_axis_mask, dtype=bool)
    # Ignore points that are off-axis for other params
    for p in list(hypersurface.params.values()):
        if p.name not in param_names:
            off_axis_mask = off_axis_mask & (
                p.fit_param_values == p.nominal_value)
    off_axis_mask = off_axis_mask & ~(
        p0_on_axis_mask | p1_on_axis_mask | nominal_mask)

    # Plot data points
    ax.scatter(p0.fit_param_values[p0_on_axis_mask], p1.fit_param_values[p0_on_axis_mask],
               z[p0_on_axis_mask], marker="o", color="blue", label="%s on-axis" % p0.name)
    ax.scatter(p0.fit_param_values[p1_on_axis_mask], p1.fit_param_values[p1_on_axis_mask],
               z[p1_on_axis_mask], marker="^", color="red", label="%s on-axis" % p1.name)
    ax.scatter(p0.fit_param_values[off_axis_mask], p1.fit_param_values[off_axis_mask],
               z[off_axis_mask], marker="s", color="black", label="Off-axis")
    ax.scatter(p0.fit_param_values[nominal_mask], p1.fit_param_values[nominal_mask],
               z[nominal_mask], marker="*", color="magenta", label="Nominal")

    # Plot hypersurface (as a 2D surface)
    x_plot = np.linspace(p0.fit_param_values.min(),
                         p0.fit_param_values.max(), num=100)
    y_plot = np.linspace(p1.fit_param_values.min(),
                         p1.fit_param_values.max(), num=100)
    x_grid, y_grid = np.meshgrid(x_plot, y_plot)
    x_grid_flat = x_grid.flatten()
    y_grid_flat = y_grid.flatten()
    params_for_plot = {p0.name: x_grid_flat, p1.name: y_grid_flat, }
    for p in list(hypersurface.params.values()):
        if p.name not in list(params_for_plot.keys()):
            params_for_plot[p.name] = np.full_like(
                x_grid_flat, hypersurface.nominal_values[p.name])
    z_grid_flat = hypersurface.evaluate(params_for_plot, bin_idx=bin_idx)
    z_grid = z_grid_flat.reshape(x_grid.shape)
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", linewidth=0,
                           antialiased=False, alpha=0.2)  # , label="Hypersurface" )

    # Format
    ax.set_xlabel(p0.name)
    ax.set_ylabel(p1.name)
    ax.legend()