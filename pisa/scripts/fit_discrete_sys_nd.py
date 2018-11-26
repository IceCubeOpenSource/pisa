#!/usr/bin/env python

"""
Fit a hyperplane to discrete systematics datasets

Produce fit results for sets of discrete systematics (e.g. several simulations
for different DOM efficiencies); there is one dimension in the hyperplane per
systematic.

The parameters and settings going into the fit are given by an external cfg
file (fit config).

n-dimensional MapSets are supported to be fit by an m-dimensional
hyperplane (one dimension per discrete parameter).

A script for making plots from the fit results produced by this file can be
found in fridge/analysis/common/scripts/plotting/plot_hyperplane_fits.py
"""

from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import Mapping, Sequence, OrderedDict
import copy
from os.path import join
import re
from StringIO import StringIO

import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from uncertainties import ufloat

from pisa import ureg
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.map import Map, MapSet
from pisa.utils.fileio import mkdir, from_file, to_file
from pisa.utils.log import logging, set_verbosity


__all__ = [
    "GENERAL_SECTION_NAME",
    "APPLY_ALL_SECTION_NAME",
    "COMBINE_REGEX_OPTION",
    "SYS_LIST_OPTION",
    "UNITS_OPTION",
    "SYS_SET_OPTION",
    "SET_OPTION_RE",
    "REMOVE_OPTION_RE",
    "WS_RE",
    "parse_args",
    "hyperplane_fun",
    "parse_fit_config",
    "load_and_modify_pipeline_cfg",
    "make_discrete_sys_distributions",
    "norm_sys_distributions",
    "fit_discrete_sys_distributions",
    "hyperplane",
    "save_hyperplane_fits",
    "main",
]

__author__ = "P. Eller, T. Stuttard, T. Ehrhardt, J.L. Lanfranchi"

__license__ = """Copyright (c) 2014-2018, The IceCube Collaboration

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License."""


GENERAL_SECTION_NAME = "general"

APPLY_ALL_SECTION_NAME = "apply_to_all_sets"

COMBINE_REGEX_OPTION = "combine_regex"

SYS_LIST_OPTION = "sys_list"

UNITS_OPTION = "units"

NOMINAL_SET_PFX = "nominal_set:"

SYS_SET_PFX = "sys_set:"

SYS_SET_OPTION = "pipeline_cfg"
"""systematics set config file is specified by this option"""

SET_OPTION_RE = re.compile(r"\s*set\s*\[(.*)\]\s*(\S*.*)")
"""defining a section and/or an option within a section in a pipeline configs is
specified by following this pattern. E.g. ::

  set [data.simple_data_loader] events_file = settings/pipeline/example.cfg

where the section is created if it doesn't exist and the option "events_file" is added
if it doesn't exist and is set to have value "settings/pipeline/example.cfg"; use ::

  set [data.simple_data_loader] =

to create the section "data.simple_data_loader" if it doesn't already exist. Note that
you have to have an equals sign (or colon) for the file to parse correctly, but
everything from the equals (or colon) on is ignored.
"""

REMOVE_OPTION_RE = re.compile(r"\s*remove\s*\[(.*)\]\s*(\S*.*)")
"""modifications to pipeline configs are specified by options following this pattern,
e.g. to remove an entire section ::

  remove [discr_sys.pi_hyperplanes] =

or to remove a single option within a section ::

  remove [data.simple_data_loader] events_file =

Note that you have to have an equals sign (or colon) for the file to parse correctly,
but everything from the equals (or colon) on is ignored.
"""

WS_RE = re.compile(r"\s+", flags=re.UNICODE)
"""regex to match all whitespace characters"""


def parse_args():
    """Parse arguments from command line.

    Returns
    -------
    args : namespace

    """
    parser = ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "-f",
        "--fit-cfg",
        type=str,
        metavar="configfile",
        required=True,
        help="Settings for the hyperplane fit",
    )
    parser.add_argument(
        "-sp",
        "--set-param",
        type=str,
        default=None,
        action="append",
        help="""Currently *NOT* implemented. Set one or multiple parameters
        to a certain value (e.g. to test stability of parameterisation).""",
    )
    parser.add_argument(
        "--tag", type=str, default="deepcore", help="Tag for the filename."
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True, help="Set output directory"
    )
    parser.add_argument("-v", action="count", default=None, help="set verbosity level")
    args = parser.parse_args()
    return args


def hyperplane_fun(x, *p):
    """Hyperplane fit function (just defines plane in n dimensions).

    Parameters
    ----------
    x : list
        nested list holding the different assumed values of each parameter
        in the second dimension (i.e., m values for m discrete sets)
        e.g. [[vals for param 0], [vals for param 1], ...]
    p : list
        list of fit function parameters values
        (one offset, n slopes, where n is the number of systematic parameters)

    Returns
    -------
    fun : list
        function value vector (one value in each systematics dimension)

    """
    # TODO Avoid duplication with pi_hyperplanes.eval_hyperplane
    fun = p[0]
    for xval, pval in zip(x, p[1:]):
        fun += xval * pval
    return fun


def parse_fit_config(fit_cfg):
    """Perform sanity checks on and parse fit configuration file.

    Parameters
    ----------
    fit_cfg : str
        path to a fit configuration file

    Returns
    -------
    fit_cfg : PISAConfigParser
        parsed fit configuration
    sys_list : list of str
        parsed names of systematic parameters
    units : list
    combine_regex : list of str
        parsed regular expressions for combining pipeline outputs

    """
    fit_cfg = from_file(fit_cfg)
    no_ws_section_map = {WS_RE.sub("", s): s for s in fit_cfg.sections()}

    if GENERAL_SECTION_NAME not in no_ws_section_map.values():
        raise KeyError('Fit config is missing the "%s" section!' % GENERAL_SECTION_NAME)

    general_section = fit_cfg[GENERAL_SECTION_NAME]
    if SYS_LIST_OPTION not in general_section:
        raise KeyError(
            "Fit config has to specify systematic parameters as"
            ' "%s" option in "%s" section (comma-separated list of names).'
            % (SYS_LIST_OPTION, GENERAL_SECTION_NAME)
        )

    sys_list = WS_RE.sub("", general_section[SYS_LIST_OPTION]).split(",")

    if UNITS_OPTION in general_section:
        units_list = []
        for unit_str in general_section[UNITS_OPTION].split(","):
            unit_str = unit_str.strip().split(".")[-1]
            if unit_str.lower() in ("", "none"):
                unit_str = "dimensionless"
            # Make sure unit is interpret-able by Pint
            ureg.Unit(unit_str)
            units_list.append(unit_str)
    else:
        units_list = ["dimensionless" for s in sys_list]
        logging.warn(
            "No %s option found in %s section; assuming systematic parameters are"
            " dimensionless",
            UNITS_OPTION,
            GENERAL_SECTION_NAME,
        )

    logging.info(
        "Found systematic parameters %s",
        ["{} ({})".format(s, u) for s, u in zip(sys_list, units_list)],
    )

    combine_regex = general_section.get(COMBINE_REGEX_OPTION, None)
    if combine_regex:
        combine_regex = WS_RE.sub("", combine_regex).split(",")

    if APPLY_ALL_SECTION_NAME in no_ws_section_map:
        apply_all_section = fit_cfg[no_ws_section_map[APPLY_ALL_SECTION_NAME]]
        for no_ws_sname, sname in no_ws_section_map.items():
            if not (
                no_ws_sname.startswith(NOMINAL_SET_PFX)
                or no_ws_sname.startswith(SYS_SET_PFX)
            ):
                continue
            sys_set_section = fit_cfg[sname]
            for option, val in apply_all_section.items():
                sys_set_section[option] = val

    return fit_cfg, sys_list, units_list, combine_regex


def load_and_modify_pipeline_cfg(fit_cfg, section):
    """Load and modify the pipeline config file as specified in that section of the fit
    config.

    Parameters
    ----------
    fit_cfg : pisa.utils.config_parser.PISAConfigParser
        any subclass of :class:`configparser.RawConfigParser` should work as well

    section : str
        name of the section to extract from the `fit_cfg`

    Returns
    -------
    pipeline_cfg : pisa.utils.config_parser.PISAConfigParser
        pipeline config

    pipeline_cfg_path : str
        path to the pipeline config as it is specified in the fit config

    """
    pipeline_cfg_path = fit_cfg.get(section, SYS_SET_OPTION)
    other_options = fit_cfg.options(section)
    other_options.remove(SYS_SET_OPTION)

    pipeline_cfg = from_file(pipeline_cfg_path)

    # Get a no-whitespace version of the section names
    section_map = {WS_RE.sub("", s): s for s in pipeline_cfg.sections()}

    for option in other_options:
        set_match = SET_OPTION_RE.match(option)
        remove_match = REMOVE_OPTION_RE.match(option) if not set_match else None
        if set_match:
            section_spec, set_option = set_match.groups()
            no_ws_section_spec = WS_RE.sub("", section_spec)
            set_option = set_option.strip()
            if no_ws_section_spec not in section_map:
                logging.debug(
                    'Adding section [%s] to in-memory copy of pipeline config "%s"',
                    section_spec,
                    pipeline_cfg_path,
                )
                pipeline_cfg.add_section(section_spec)
                section_map[no_ws_section_spec] = section_spec
            if set_option:
                set_value = fit_cfg.get(section, option).strip()
                logging.debug(
                    'Setting section [%s] option "%s = %s" in in-memory'
                    ' copy of pipeline config "%s"',
                    section_spec,
                    set_option,
                    set_value,
                    pipeline_cfg_path,
                )
                pipeline_cfg.set(section_map[no_ws_section_spec], set_option, set_value)
        elif remove_match:
            section_spec, remove_option = remove_match.groups()
            no_ws_section_spec = WS_RE.sub("", section_spec)
            remove_option = remove_option.strip()
            if no_ws_section_spec in section_map:
                if remove_option:
                    logging.debug(
                        'Removing section [%s] option "%s" from in-memory copy of'
                        ' pipeline config "%s"',
                        section_spec,
                        remove_option,
                        pipeline_cfg_path,
                    )
                    pipeline_cfg.remove_option(
                        section_map[no_ws_section_spec], remove_option
                    )
                else:
                    logging.debug(
                        "Removing section [%s] from in-memory copy of pipeline config"
                        ' "%s"',
                        section_spec,
                        pipeline_cfg_path,
                    )
                    pipeline_cfg.remove_section(section_map[no_ws_section_spec])
            else:
                logging.warn(
                    "Told to remove section [%s] but section does not exist in"
                    ' pipline config "%s"',
                    section_spec,
                    pipeline_cfg_path,
                )
        else:
            raise ValueError("Unhandled option in fit config: {}".format(option))

    return pipeline_cfg, pipeline_cfg_path


def make_discrete_sys_distributions(fit_cfg, set_params=None):
    """Generate and store mapsets for different discrete systematics sets
    (with a single set characterised by a dedicated pipeline configuration)

    Parameters
    ----------
    fit_cfg : string
        Path to a fit config file

    Returns
    -------
    input_data : OrderedDict
        Container with the processed input data including MapSets
        resulting from each input pipelines

    """
    # check optional `set_params`
    if set_params is not None:
        if not isinstance(set_params, Mapping):
            raise TypeError("`set_params` must be dict-like")
        for param_name, param_value in set_params.items():
            if not isinstance(param_name, basestring):
                raise TypeError("`set_params` keys must be strings (parameter name)")
            if not isinstance(param_value, ureg.Quantity):
                raise TypeError("`set_params` values must be Quantities")

    parsed_fit_cfg, sys_list, units_list, combine_regex = parse_fit_config(fit_cfg)
    fit_cfg_txt_buf = StringIO()
    parsed_fit_cfg.write(fit_cfg_txt_buf)
    fit_cfg_txt = fit_cfg_txt_buf.getvalue()

    # prepare the data container
    input_data = OrderedDict()
    input_data["fit_cfg_path"] = fit_cfg
    input_data["fit_cfg_txt"] = fit_cfg_txt
    input_data["param_names"] = sys_list
    input_data["param_units"] = units_list
    input_data["datasets"] = []

    # -- Load systematics sets -- #

    found_nominal = False
    sys_sets_info = OrderedDict()

    for section in parsed_fit_cfg.sections():
        no_ws_section = WS_RE.sub("", section)

        is_nominal = no_ws_section.startswith(NOMINAL_SET_PFX)
        is_sys_set = is_nominal or no_ws_section.startswith(SYS_SET_PFX)

        if is_nominal:
            if found_nominal:
                raise ValueError(
                    "Found multiple nominal sets in fit cfg! There must be"
                    " exactly one."
                )
            found_nominal = True

        if is_sys_set:
            # Parse the list of systematics parameter values from the section name
            sys_param_point = tuple(float(x) for x in section.split(":")[1].split(","))

            if len(sys_param_point) != len(sys_list):
                raise ValueError(
                    "Section heading [{}] specifies {:d} systematic"
                    " parameter values, but there are {:d} systematics".format(
                        section, len(sys_param_point), len(sys_list)
                    )
                )

            parsed_pipeline_cfg, pipeline_cfg_path = load_and_modify_pipeline_cfg(
                fit_cfg=parsed_fit_cfg, section=section
            )

            pipeline_cfg_txt_buf = StringIO()
            parsed_pipeline_cfg.write(pipeline_cfg_txt_buf)
            pipeline_cfg_txt = pipeline_cfg_txt_buf.getvalue()

            sys_sets_info[sys_param_point] = dict(
                is_nominal=is_nominal,
                parsed_pipeline_cfgs=[parsed_pipeline_cfg],
                pipeline_cfg_paths=[pipeline_cfg_path],
                pipeline_cfg_txts=[pipeline_cfg_txt],
            )

        # In this loop, nothing to do for general / apply all sections
        elif no_ws_section not in (GENERAL_SECTION_NAME, APPLY_ALL_SECTION_NAME):
            raise ValueError("Invalid section in fit cfg file: %s" % section)

    if not found_nominal:
        raise ValueError(
            "Could not find a nominal discrete systematics set in fit cfg."
            " There must be exactly one."
        )

    nsets = len(sys_sets_info)
    nsys = len(sys_list)
    if nsets <= nsys:
        logging.warn(
            "Fit will either fail or be unreliable since the number of"
            " systematics sets to be fit is small (%d <= %d).",
            nsets,
            nsys + 1,
        )

    for sys_param_point, info in sys_sets_info.items():
        point_str = " | ".join(
            ["%s=%.2f" % (p, v) for p, v in zip(sys_list, sys_param_point)]
        )

        logging.info(
            "Generating maps for discrete systematics point: %s. Using"
            ' pipeline config(s) at "%s"',
            point_str,
            info["pipeline_cfg_paths"],
        )

        # make a dedicated distribution maker for each systematics set
        distribution_maker = DistributionMaker(info["parsed_pipeline_cfgs"])

        # update params if requested
        if set_params is not None:
            for pname, pval in set_params.items():
                if pname not in distribution_maker.params.names:
                    raise ValueError("Unknown param '%s' in `set_params`" % pname)
                if (
                    pval.dimensionality
                    != distribution_maker.params[pname].dimensionality
                ):
                    raise ValueError(
                        'Incorrect units for param "%s" in `set_params`' % pname
                    )
                distribution_maker.params[pname].value = pval
                logging.info("Changed param '%s' to %s", pname, pval)

        distribution_maker_param_values = OrderedDict()
        for dmpname in sorted(distribution_maker.params.names):
            dmpval = distribution_maker.params[dmpname].value
            distribution_maker_param_values[dmpname] = dmpval

        # run the distribution maker to get the mapset
        # TODO This assumes only one pipeline, either make more general or enforce
        mapset = distribution_maker.get_outputs(return_sum=False)[0]

        if combine_regex:
            logging.info(
                "Combining maps according to regular expression(s) %s", combine_regex
            )
            mapset = mapset.combine_re(combine_regex)

        # Store the info
        dataset = OrderedDict()
        dataset["pipeline_cfg_paths"] = info["pipeline_cfg_paths"]
        dataset["pipeline_cfg_txts"] = info["pipeline_cfg_txts"]
        dataset["distribution_maker_param_values"] = distribution_maker_param_values
        dataset["param_values"] = sys_param_point
        dataset["mapset"] = mapset
        dataset["nominal"] = info["is_nominal"]
        input_data["datasets"].append(dataset)

    return input_data


def norm_sys_distributions(input_data):
    """Normalises systematics mapsets to the nominal mapset,
    performing error propagation.

    Parameters
    ----------
    input_data : dict
        The data container returned by `make_discrete_sys_distributions`.
        Note that this is modified by this function to add the normalised
        distrbutions.

    Notes
    -----
    Nothing is returned; `input_data` is modified in-place

    """
    #
    # Get the input mapsets
    #

    nominal_mapset = [
        dataset["mapset"] for dataset in input_data["datasets"] if dataset["nominal"]
    ]
    if len(nominal_mapset) != 1:
        raise ValueError(
            "need 1 but got {} nominal mapsets".format(len(nominal_mapset))
        )
    nominal_mapset = nominal_mapset[0]

    for dataset_dict in input_data["datasets"]:
        dataset_dict["norm_mapset"] = []

    #
    # loop over types of event
    #

    for map_name in nominal_mapset.names:
        logging.info('Normalizing "%s" maps.', map_name)
        nominal_map = nominal_mapset[map_name]
        nominal_map_nominal_vals = nominal_map.nominal_values

        # Note that all
        #   `sys_mapset[map_name].nominal_values`
        # and
        #   `nominal_map.nominal_values`
        # are finite (neither infinite nor NaN), so only issue for diviiding
        # the former by the latter is if there are "empty bins" (zeros) in
        #   `nominal_map.nominal_values`
        finite_mask = nominal_map_nominal_vals != 0

        #
        # loop over datasets
        #

        for dataset_dict in input_data["datasets"]:
            #
            # Normalise maps
            #

            sys_mapset_map = dataset_dict["mapset"][map_name]

            # TODO: think about the best way to perform unc. propagation

            # Crete a new array with uncertainties the same shape as map;
            # values are irrelevant as all will be overwritten
            norm_sys_hist = copy.deepcopy(sys_mapset_map.hist)

            # Note that we divide by nominal_vals to avoid double counting the
            # uncertainty on the nominal template when applying the hyperplane
            # fits
            norm_sys_hist[finite_mask] = (
                sys_mapset_map.hist[finite_mask] / nominal_map_nominal_vals[finite_mask]
            )
            norm_sys_hist[~finite_mask] = ufloat(np.NaN, np.NaN)

            # TODO Check for bins that are empty in the nominal hist but not in
            # at least one of the sys sets; currently we do not support this...

            norm_sys_map = Map(
                name=sys_mapset_map.name,
                binning=sys_mapset_map.binning,
                hist=norm_sys_hist,
            )

            # TODO Save the map
            dataset_dict["norm_mapset"].append(norm_sys_map)

    # Re-format
    for dataset_dict in input_data["datasets"]:
        dataset_dict["norm_mapset"] = MapSet(
            maps=dataset_dict["norm_mapset"], name=dataset_dict["mapset"].name
        )


def fit_discrete_sys_distributions(input_data, p0=None, fit_method=None):
    """Fits a hyperplane to MapSets generated at given systematics parameters
    values.

    Parameters
    ----------
    input_data : OrderedDict
        The data container returned by `make_discrete_sys_distributions`
        and modified by `norm_sys_distributions`.
    p0 : list or dict
        Initial guess list (same initial guess for all maps) or dictionary
        (keys have to correspond to event groups/channels in maps)
        with one offset and len(sys_list) slopes. Default is list of ones.
    fit_method : None or string
        `method` arg to pass to `curve_fit` (see curve_fit docs).
        If None, will default to `trf` (this method supports covariance matrix
        calculation in the dimensionality we're dealing with).

    Returns
    -------
    fit_results : OrderedDict
        Container of the hyerplane fit results + supporting data

    """
    #
    # Prepare a few things before fitting
    #

    # Set a default fit method for curve_fit
    if fit_method is None:
        fit_method = "trf"  # lm, trf, dogbox
    # TODO Store in output data

    # prepare an output data container
    fit_results = OrderedDict()
    fit_results["hyperplanes"] = OrderedDict()

    # store info from the input data in the fit results
    fit_results["datasets"] = input_data["datasets"]
    fit_results["param_names"] = input_data["param_names"]
    fit_results["fit_cfg_path"] = input_data["fit_cfg_path"]
    fit_results["fit_cfg_txt"] = input_data["fit_cfg_txt"]

    # get number of systematic parameters and datasets
    n_sys_params = len(fit_results["param_names"])
    n_datasets = len(fit_results["datasets"])

    # get number of params in hyperplane fit
    # this is one slope per systematic, plus a single intercept
    n_fit_params = 1 + len(fit_results["param_names"])

    # get binning info
    binning = fit_results["datasets"][0]["mapset"][0].binning
    binning_shape = list(binning.shape)

    # normalise the systematics variations to the nominal distribution
    # with error propagation
    norm_sys_distributions(input_data)

    # re-organise normalised maps to be stored per event type (a list for each dataset)
    norm_sys_maps = OrderedDict()
    for map_name in input_data["datasets"][0]["norm_mapset"].names:
        norm_sys_maps[map_name] = [
            dataset_dict["norm_mapset"][map_name]
            for dataset_dict in input_data["datasets"]
        ]

    # get an array of the systematic parameter points sampled across all datasets
    # transpose to get format compatible with scipy.optimize.curve_fit
    sys_param_points = np.asarray(
        [dataset_dict["param_values"] for dataset_dict in fit_results["datasets"]]
    )  # [datasets, params]
    sys_param_points_T = sys_param_points.T
    assert sys_param_points_T.shape[0] == n_sys_params
    assert sys_param_points_T.shape[1] == n_datasets

    # store some of this stuff
    fit_results["sys_param_points"] = sys_param_points
    fit_results["binning"] = binning
    fit_results["binning_hash"] = binning.hash

    #
    # Prepare initial parameter guesses
    #

    if p0:
        if isinstance(p0, Mapping):
            p0_keys = sorted(p0.keys())
            map_keys = sorted(norm_sys_maps.keys())
            if not p0_keys == map_keys:
                raise KeyError(
                    "Initial guess mapping contains keys %s which are not the"
                    " same as %s in maps." % (p0_keys, map_keys)
                )
            for ini_guess in p0.values():
                assert len(ini_guess) == n_fit_params
        elif isinstance(p0, Sequence):
            assert len(p0) == n_fit_params
            p0 = {map_name: p0 for map_name in norm_sys_maps.keys()}
        else:
            raise TypeError(
                "Initial guess must be a mapping or a sequence. Found %s." % type(p0)
            )
    else:
        p0 = {map_name: np.ones(n_fit_params) for map_name in norm_sys_maps.keys()}

    fit_results["p0"] = p0

    #
    # Loop over event types
    #

    for map_name, chan_norm_sys_maps in norm_sys_maps.items():
        logging.info('Fitting "%s" maps with initial guess %s.', map_name, p0[map_name])

        # create a container for fit results for this event type
        fit_results["hyperplanes"][map_name] = OrderedDict()

        # initialise data arrays with NaNs
        fit_results["hyperplanes"][map_name]["fit_params"] = np.full(
            shape=binning_shape + [n_fit_params],  # [bins..., hyperplane params]
            fill_value=np.nan,
        )
        fit_results["hyperplanes"][map_name]["chi2s"] = np.full(
            shape=binning_shape + [n_datasets], fill_value=np.nan  # [bins..., datasets]
        )
        fit_results["hyperplanes"][map_name]["cov_matrices"] = np.full(
            shape=binning_shape
            + [
                n_fit_params,
                n_fit_params,
            ],  # [bins..., hyperplane params, hyperplane params]
            fill_value=np.nan,
        )
        fit_results["hyperplanes"][map_name]["finite_mask"] = np.full(
            shape=binning_shape + [n_datasets], fill_value=np.nan  # [bins..., datasets]
        )

        #
        # loop over bins
        #

        for idx in np.ndindex(*binning_shape):
            # get the bin content, including uncertainty and mask indicating if
            # the bin is finite treat the bin content as y values in the fit,
            # e.g. y(x0,...,xN) where N is the number of parameters each of
            # these 1D arrays has one element per input dataset
            y = np.asarray([m.hist[idx] for m in chan_norm_sys_maps])
            y_values = unp.nominal_values(y)
            y_sigma = unp.std_devs(y)
            finite_mask = np.isfinite(y_values) & np.isfinite(y_sigma)

            # empty bins have sigma=0 which causes the hyperplane fit to fail (silently)
            # replace with sigma=inf (e.g. we know nothing in this bin)
            empty_bin_mask = np.isclose(y_values, 0.)
            if np.any(empty_bin_mask):
                empty_bin_zero_sigma_mask = empty_bin_mask & np.isclose(y_sigma, 0.)
                if np.any(empty_bin_zero_sigma_mask):
                    y_sigma[empty_bin_zero_sigma_mask] = np.inf

            # check no zero sigma values remaining
            if np.any(np.isclose(y_sigma, 0.)):
                raise ValueError(
                    "Found histogram sigma values that are 0., which is unphysical"
                )

            #
            # Perform hyperplane fit in this bin
            #

            # case 1: uncertainties are available in the bins (ideal case)
            if np.any(y_sigma[finite_mask]):

                # fit
                popt, pcov = curve_fit(
                    hyperplane_fun,
                    sys_param_points_T[:, finite_mask],
                    y_values[finite_mask],
                    sigma=y_sigma[finite_mask],
                    p0=p0[map_name],
                    absolute_sigma=True,  # TODO Should we use this?
                    method=fit_method,
                )

                # Calculate chi-square values comparing the input data and the
                # fit results at each data point (e.g. per dataset, and of
                # course in each bin)
                for point_idx in range(n_datasets):  # Loop over datasets
                    # Get param values for this dataset
                    point = sys_param_points[point_idx, :]
                    # Predict counts in this bin accoridng to hyperplane for
                    # this dataset
                    predicted = hyperplane_fun(point, *popt)
                    observed = y_values[point_idx]
                    sigma = y_sigma[point_idx]
                    # TODO Is chi2 computation correct?
                    chi2 = ((predicted - observed) / sigma) ** 2
                    chi2_idx = tuple(list(idx) + [point_idx])
                    fit_results["hyperplanes"][map_name]["chi2s"][chi2_idx] = chi2

            else:
                # if here, no uncertainties are available for this bin
                # note that cannot calculate chi2 without uncertainties

                # case 2: there are at least central values in the bins
                if np.any(y_values[finite_mask]):

                    # without error estimates each point has the same weight
                    # and we cannot get chi-square values (but can still fit)
                    logging.warn(
                        "No uncertainties for any of the normalised counts in bin"
                        ' %s ("%s") found. Fit is performed unweighted and no'
                        " chisquare values will be available.",
                        idx,
                        map_name,
                    )

                    # fit
                    popt, pcov = curve_fit(
                        hyperplane_fun,
                        sys_param_points_T[:, finite_mask],
                        y_values,
                        p0=p0[map_name],
                        methods=fit_method,
                    )

                # case 3: no data in this bin
                # this is the worst case, where there are no central values or
                # errors. Most likely this came about because this bin is
                # empty, which is not necessarily an error.
                else:

                    # Store NaN for fit params and chi2
                    popt = np.full_like(p0[map_name], np.NaN)
                    pcov = np.NaN  # TODO Shape?

            # store the results for this bin
            # note that chi2 is already stored above
            fit_results["hyperplanes"][map_name]["fit_params"][idx] = popt
            fit_results["hyperplanes"][map_name]["cov_matrices"][idx] = pcov
            fit_results["hyperplanes"][map_name]["finite_mask"][idx] = finite_mask

    return fit_results


def hyperplane(fit_cfg, set_params=None, fit_method=None):
    """Wrapper around distribution generation and fitting functions.

    Parameters
    ----------
    fit_cfg : string
        Path to a fit configuration file
    set_params : mapping, optional
        Params to be manually set; keys are param names and corresponding
        values are those to be manually set for the respective params

    Returns
    -------
    input_data : OrderedDict
        Container holding the input data provided by the user to be fitted with
        hyperplanes, as produced by `make_discrete_sys_distributions`
    fit_results : OrderedDict
        Container holding the results of the hyperplane fits, as produced by
        `fit_discrete_sys_distributions`
    fit_method : None or string
        See description in `fit_discrete_sys_distributions` method documentation

    """
    input_data = make_discrete_sys_distributions(fit_cfg=fit_cfg, set_params=set_params)
    fit_results = fit_discrete_sys_distributions(
        input_data=input_data, fit_method=fit_method
    )

    return input_data, fit_results


def save_hyperplane_fits(input_data, fit_results, outdir, tag):
    """Store discrete systematics fits and chi-square values to a specified
    output location, with results identified by a tag.

    Parameters
    ----------
    input_data : mapping
        input data container returned by `hyperplane` function
    fit_results : dict
        fit results data container returned by `hyperplane` function
    outdir : string
        output directory
    tag : string
        identifier for filenames holding fit results

    """
    # Get some strings to use when naming
    dim = len(input_data["param_names"])
    param_str = "_".join(input_data["param_names"])

    # Store as JSON
    mkdir(outdir)
    res_path = join(outdir, "%s__%dd__%s__hyperplane_fits.json" % (tag, dim, param_str))
    to_file(fit_results, res_path)


def main():
    """Perform a hyperplane fit to discrete systematics sets."""
    args = parse_args()
    set_verbosity(args.v)

    # Read in data and fit hyperplanes to it
    input_data, fit_results = hyperplane(fit_cfg=args.fit_cfg)

    # Save to disk
    save_hyperplane_fits(
        input_data=input_data, fit_results=fit_results, outdir=args.outdir, tag=args.tag
    )


if __name__ == "__main__":
    main()
