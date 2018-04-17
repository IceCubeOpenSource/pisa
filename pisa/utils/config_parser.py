"""
Parse a ConfigFile object into a dict containing an item for every analysis
stage, that itself contains all necessary instantiation arguments/objects for
that stage. for en example config file, please consider
:file:`$PISA/pisa_examples/resources/settings/pipeline/example.cfg`

Config File Structure
=====================

A pipeline config file is expected to contain something like the following,
with the sections ``[pipeline]`` and corresponding ``[stage:service]``
required, in addition to a ``[binning]`` section:

.. code-block:: cfg

    #include file_x.cfg as x
    #include file_y.cfg as y

    [pipeline]
    order = stageA:serviceA, stageB:serviceB

    [binning]
    #include generic_binning.cfg

    binning1.order = axis1, axis2
    binning1.axis1 = {'num_bins':40, 'is_log':True,
                      'domain':[1,80] units.GeV, 'tex': r'A_1'}
    binning1.axis2 = {'num_bins':10, 'is_lin':True,
                      'domain':[1,5], 'tex': r'A_2'}

    [stageA:serviceA]
    input_binning = bining1
    output_binning = binning1
    error_method = None
    debug_mode = False

    param.p1 = 0.0 +/- 0.5 units.deg
    param.p1.fixed = False
    param.p1.range = nominal + [-2.0, +2.0] * sigma

    [stageB:serviceB]
    ...

* ``#include`` statements can be used to include other config files. The
  #include statement must be the first non-whitespace on a line, and these
  statements can be used anywhere within a config file.
* ``#include resource as xyz`` statements behave similarly, but prepend the
  included file's text with a setion header containing ``xyz`` in this case.
* ``pipeline`` is the top-most section that defines the hierarchy of stages and
  what services to be instantiated.
* ``binning`` can contain different binning definitions, that are then later
  referred to from within the ``stage.service`` sections.
* ``stage.service`` one such section per stage.service is necessary. It
  contains some options that are common for all stages (`binning`,
  `error_method` and `debug_mode`) as well as all the necessary arguments and
  parameters for a given stage.
* Duplicate section headers and duplicate keys within a section are illegal.


Param definitions
-----------------

Every key in a stage section that starts with `param.<name>` is interpreted and
parsed into a PISA :class:`pisa.core.param.Param` object. These can be strings
(e.g. a filename--but don't use any quotation marks) or quantities (numbers
with units).

Quantities expect an expression that can be converted by the
:func:`parse_quantity` function. The expression for a quantity can optionally
include a simple Gaussian prior and units. The simplest definition of a
quantity with neither Gaussian prior nor units would look something like this:

.. code-block:: cfg

    param.p1 = 12.5

Gaussian priors can be included for a quantity using ``+/-`` notation, where
the number that follows ``+/-`` is the standard deviation. E.g.:

.. code-block:: cfg

    param.p1 = 12.5 +/- 2.3

If no units are explicitly set for a quantity, it is taken to be a quantity
with special units ``dimensionless``. Units can be set by multiplying (using
``*``) by ``units.<unit>`` where ``<unit>`` is the short or long name
(optionally including metric prefix) of a unit. E.g. the following set
equivalent values for params `p1` and `p2`:

.. code-block:: cfg

    param.p1 = 12.5 * unit.GeV
    param.p2 = 12.5 * unit.gigaelectronvolt

and this can be combined with the Gaussian-prior ``+/-`` notation:

.. code-block:: cfg

    param.p1 = 12.5 +/- 2.3 * unit.GeV

Additional arguments to a parameter are passed in with the ``.`` notation, for
example ``param.p1.fixed = False``, which makes p1 a free parameter in the
fit (by default a parameter is fixed unless specified like this).

Uniform and spline priors can also be set using the ``.prior`` attribute:

.. code-block:: cfg

    param.p1 = 12.5
    param.p1.prior = uniform

    param.p2 = 12.5
    param.p2.prior = spline
    param.p2.prior.data = resource_loc

If no prior is specified, it is taken to have no prior (or, equivalently, a
uniform prior with no penalty). A uniform prior can be explicitly set or
arbitrary (Priors (including a Gaussian prior, as an alternative to the above
notation) can be explicitly set using the ``.prior`` attribute of a ``param``:

A range must be given for a free parameter. Either as absolute range `[x,y]` or
in conjunction with the keywords `nominal` (= nominal parameter value) and
`sigma` if the param was specified with the `+/-` notation.

`.prior` is another argument, that can take the values `uniform` or `spline`,
for the latter case a `.prior.data` will be expected, pointing to the spline
data file.

N.B.
++++
Params that have the same name in multiple stages of the pipeline are
instantiated as references to a single param in memory, so updating one updates
all of them.

Note that this mechanism of synchronizing parameters holds only within the
scope of a single pipeline; synchronization of parameters across pipelines is
done by adding the pipelines to a single DistributionMaker object and updating
params through the DistributionMaker's update_params method.

If you DO NOT want parameters to be synchronized, provide a unique_id for them.
This is imply done by setting `.unique_id`


Param selector
--------------

A special mechanism allows the user to specify multiple, different values for
the same param via the param selector method. This can be used for example for
hypothesis testing, there for hypothesis A a param takes a certain value, while
for hypothesis B a different value.

A given param, say `foo`, then needs two definitions like the following,
assuming we name our selections `A` and `B`:

.. code-block:: cfg

    param.A.foo = 1
    param.B.foo = 2

The default param selector needs to be spcified under section `pipeline` as e.g.

.. code-block:: cfg

    param_selections = A

Which will default the value of 1 for param `foo`. An instatiated pipeline can
dynamically switch to another selection after instantiation.

Multiple different param selectors are allowed in a single config. In the
default selection they must be separated by commas.

"""

# TODO: consistency, etc.
# * Order-independent hashing of the PISAConfigParser object (recursively sort
#   contents?). This is still a worse idea than hashing on instantiated PISA
#   objects since things like meaningless whitespace will modify the hash of
#   the config.
# * Add explicit gaussian prior (should NOT just rely on +/- notation to make
#   consistent with other priors)
# * Furthermore, all priors should be able to be defined in one line, e.g.:
#     p1.prior = guassian: std_dev = 1.2
#     p2.prior = uniform
#     p3.prior = spline: data = resource/location/config.cfg
#     p4.prior = None
# * Make interoperable with pisa.utils.resources. I.e., able to work with
#   Python package resources, not just filesystem files.
# * Docstrings
# * TODO: add try: except: blocks around class instantiation calls to give
#   maximally useful error info to the user (spit out a good message, but then
#   re-raise the exception)


from __future__ import absolute_import, division

from collections import Counter, OrderedDict
from io import StringIO
from os.path import abspath, expanduser, expandvars, isfile, join
import re
import sys
import warnings

from backports.configparser import (
    RawConfigParser, ExtendedInterpolation, DuplicateOptionError,
    SectionProxy, MissingSectionHeaderError, DuplicateSectionError,
    NoSectionError
)
from backports.configparser.helpers import open as c_open
from backports.configparser.helpers import PY2
import numpy as np
from uncertainties import ufloat, ufloat_fromstr

from pisa import ureg
from pisa.utils.fileio import from_file
from pisa.utils.format import split
from pisa.utils.hash import hash_obj
from pisa.utils.log import logging, set_verbosity
from pisa.utils.resources import find_resource


__all__ = ['PARAM_RE', 'PARAM_ATTRS', 'STAGE_SEP',
           'parse_quantity', 'parse_string_literal',
           'interpret_param_subfields', 'parse_param', 'parse_pipeline_config',
           'parse_optimizer_config', 'parse_minimizer_config',
           'MutableMultiFileIterator', 'PISAConfigParser']

__author__ = 'P. Eller, J. Lanfranchi'

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


PARAM_RE = re.compile(
    r'^param\.(?P<subfields>(([^.\s]+)(\.|$))+)',
    re.IGNORECASE
)

PARAM_ATTRS = ['range', 'prior', 'fixed']

STAGE_SEP = '.'

# Define names that users can specify in configs such that the eval of those
# strings works.
numpy = np # pylint: disable=invalid-name
inf = np.inf # pylint: disable=invalid-name
units = ureg # pylint: disable=invalid-name


def parse_quantity(string):
    """Parse a string into a pint/uncertainty quantity.

    Parameters
    ----------
    string : string

    Returns
    -------
    value : pint.quantity of uncertainties.core.AffineScalarFunc

    Examples
    --------
    >>> quant = parse_quantity('1.2 +/- 0.7 * units.meter')
    >>> print str(quant)
    1.2+/-0.7 meter
    >>> print '{:~}'.format(quant)
    1.2+/-0.7 m
    >>> print quant.magnitude
    1.2+/-0.7
    >>> print quant.units
    meter
    >>> print quant.nominal_value
    1.2
    >>> print quant.std_dev
    0.7

    Also note that spaces and the "*" are optional:

    >>> print parse_quantity('1+/-1units.GeV')
    1.0+/-1.0 gigaelectron_volt

    """
    value = string.replace(' ', '')
    if 'units.' in value:
        value, unit = value.split('units.')
    else:
        unit = None
    value = value.rstrip('*')
    if '+/-' in value:
        value = ufloat_fromstr(value)
    else:
        value = ufloat(float(value), 0)
    value *= ureg(unit)
    return value


def parse_string_literal(string):
    """Evaluate a string with certain special values, or return the string. Any
    further parsing must be done outside this module, as this is as specialized
    as we're willing to be in assuming/interpreting what a string is supposed
    to mean.

    Parameters
    ----------
    string : string

    Returns
    -------
    val : bool, None, or str

    Examples
    --------
    >>> print parse_string_literal('true')
    True

    >>> print parse_string_literal('False')
    False

    >>> print parse_string_literal('none')
    None

    >>> print parse_string_literal('something else')
    'something else'

    """
    if string.strip().lower() == 'true':
        return True
    if string.strip().lower() == 'false':
        return False
    if string.strip().lower() == 'none':
        return None
    return string


def interpret_param_subfields(subfields, selector=None, pname=None, attr=None):
    infodict = dict(subfields=subfields, selector=selector, pname=pname,
                    attr=attr)

    # Everything has been parsed
    if not infodict['subfields']:
        return infodict

    # If only one field, this must be the param's name, and we're done
    if len(infodict['subfields']) == 1:
        infodict['pname'] = infodict['subfields'].pop()
        return interpret_param_subfields(**infodict)

    # Look for and remove attr field and any subsequent fields
    attr_indices = []
    for n, field in enumerate(infodict['subfields']):
        if field in PARAM_ATTRS:
            attr_indices.append(n)

    # TODO: not clear what's being done here; also, would slicing be more clear
    # than iterating & calling pop()?
    if len(attr_indices) == 1:
        attr_idx = attr_indices[0]
        infodict['attr'] = [
            infodict['subfields'].pop(attr_idx)
            for _ in range(attr_idx, len(infodict['subfields']))
        ]
        return interpret_param_subfields(**infodict)

    elif len(attr_indices) > 1:
        raise ValueError('Found multiple attrs in config name "%s"' %pname)

    if len(infodict['subfields']) == 2:
        infodict['pname'] = infodict['subfields'].pop()
        infodict['selector'] = infodict['subfields'].pop()
        return interpret_param_subfields(**infodict)

    raise ValueError('Unable to parse param subfields %s'
                     %infodict['subfields'])


def parse_minimizer_config(config):
    """Parse a minimizer configuration. Note that some of
    the fields which are passed on to the corresponding
    `scipy.optimize` interface might have to be integers.
    This needs to be ensured outside of here.

    Parameters
    ----------
    config : string or ConfigParser

    Returns
    -------
    settings_dict : OrderedDict

    """
    if isinstance(config, basestring):
        config = from_file(config)
    elif isinstance(config, PISAConfigParser):
        pass
    else:
        raise TypeError(
            '`config` must either be a string or PISAConfigParser. Got %s '
            'instead.' % type(config)
        )

    if not config.has_section('method'):
        raise NoSectionError(
            "Could not find 'method'. Only found sections: %s"
            % config.sections()
        )

    if not config.has_section('options'):
        raise NoSectionError(
            "Could not find 'options'. Only found sections: %s"
            % config.sections()
        )

    settings_dict = OrderedDict()
    solver = config['method']['name']
    settings_dict['method'] = solver
    settings_dict['options'] = OrderedDict()
    for opt, val in config['options'].items():
        try:
            val = parse_quantity(val)
            settings_dict['options'][opt] = val.nominal_value
        except ValueError:
            val = parse_string_literal(val)
            settings_dict['options'][opt] = val

    return settings_dict


def parse_fit_config(config):
    """Parse a fit configuration from a path to a parsable
    configuration file or from a `PISAConfigParser`.
    Requires at least sections 'fit' and 'fit.params'
    and additional ones depending on the chosen fitting methods.

    Parameters
    ----------
    config : string or ConfigParser

    Returns
    -------
    settings_dict : OrderedDict

    """
    from pisa.analysis.analysis import ANALYSIS_METHODS
    if isinstance(config, basestring):
        config = from_file(config)
    elif isinstance(config, PISAConfigParser):
        pass
    else:
        raise TypeError(
            '`config` must either be a string or PISAConfigParser. Got %s '
            'instead.' % type(config)
        )

    if not config.has_section('fit'):
        raise NoSectionError(
            'Could not find "fit". Please specify which fit methods to use'
            ' here. Only found sections: %s.' % config.sections()
        )

    if not config.has_section('fit.params'):
        raise NoSectionError(
            'Could not find "fit.params". Please specify by which method to'
            ' treat which parameter here. Only found sections: %s.'
            % config.sections()
        )

    # parse fit methods first
    settings_dict = OrderedDict()
    fit_methods = config['fit']['method']
    fit_methods = ''.join(fit_methods.split()).split('+')

    # check whether all fit methods are recognised
    excess = set(fit_methods).difference(set(ANALYSIS_METHODS))
    if excess:
        raise ValueError('Unrecognised fit method(s): %s' % excess)

    # check for duplicates (could also ignore those, but let's be more
    # cautious and raise)
    method_count = Counter(fit_methods)
    duplicates = [m for (m,c) in method_count.items() if c > 1]
    if duplicates:
        raise ValueError('Found duplicated fit method(s): %s' % duplicates)

    # these are the allowed + must have options for the different fit methods
    # these can be specified globally ("<opt> = ...") or per param
    # ("<param>.<opt> = ..."), but if the latter doesn't exist a global
    # default must be there
    method_defaults = {'scan': {'range': None, 'nvalues': None},
                       'pull': {'lin_range': None},
                        # probably won't want to allow different minimizers for
                        # different parameters, but at least we already have
                        # the structure here for more complex fit settings for
                        # minimization
                       'minimize': {'global': None, 'local': None},
                      }
    # if the wildcard is employed, require global defaults to be set
    wildcard = '*'

    # now check whether for each method there's a specification of its
    # parameters
    fit_params = config['fit.params']
    wildcard_used = False
    # to prevent single parameters from being assigned to multiple methods
    fit_pnames_collected = set()
    for fit_method in sorted(fit_methods):
        wildcard_here = False
        if not fit_method in fit_params:
            # no implicit assignment of fit parameters will be tolerated -
            # at least wildcard required
            raise ValueError('Please specify which parameters should be fit'
                             ' via "%s".' % fit_method)
        method_pnames = ''.join(fit_params[fit_method].split()).split(',')
        for pname in method_pnames:
            if pname in fit_pnames_collected:
                raise ValueError(
                    'Parameter "%s" already assigned to a fit method other than'
                    ' "%s"!' % (pname, fit_method)
                )
                fit_pnames_collected.add(pname)
        # TODO: make sure only one occurrence of any parameter within a method
        if wildcard in method_pnames:
            if wildcard_used:
                raise ValueError(
                    'Cannot use wildcard "%s" more than once in a fit config.'
                    % wildcard
                )
            wildcard_used = True
            wildcard_here = True
        settings_dict[fit_method] = {
            'params': {p: {} for p in sorted(method_pnames)}
        }
        # require a section for the fit method
        if not config.has_section(fit_method):
            raise NoSectionError(
                'Could not find "%s". Only found sections: %s'
                % (fit_method, config.sections())
            )

        # Look for specification of the allowed global defaults from
        # `method_defaults` and for param-specific ones (take precedence)
        if fit_method in method_defaults:
            allowed_opts = sorted(method_defaults[fit_method].keys())
            for opt in allowed_opts:
                found_default = False
                if opt in config[fit_method]:
                    val = config[fit_method][opt]
                    found_default = True
                    if val == "None":
                        method_defaults[fit_method][opt] = None
                    # parse minimizer config
                    elif fit_method == 'minimize' and opt in ['global', 'local']:
                        method_defaults[fit_method][opt] = parse_minimizer_config(val)
                    else:
                        # remove *any* whitespace
                        val = ''.join(parse_string_literal(val).split())
                        method_defaults[fit_method][opt] = val
                    # processed, so remove
                    config[fit_method].pop(opt)
                else:
                    # this allowed default hasn't been set
                    # -> only problematic if wildcard is used
                    if wildcard_here:
                        raise ValueError(
                            'You have to globally set option "%s" for fit'
                            ' method "%s" since you used the wildcard!'
                            % (opt, fit_method)
                        )
                # start searching for param specific specs
                for pname in method_pnames:
                    # options set as <param>.<opt> take precedence over global
                    # setting of <opt>
                    param_opt = '%s.%s' % (pname, opt)
                    if param_opt in config[fit_method]:
                        if 'fit_method' == 'minimize':
                            raise ValueError(
                                'Currently only global default options allowed'
                                ' for minimization! Found: "%s". Please just'
                                ' specify "%s" exactly once.'
                                % (param_opt, opt)
                            )
                        val = config[fit_method][param_opt]
                        # remove *any* whitespace
                        val = ''.join(parse_string_literal(val).split())
                        config[fit_method].pop(param_opt)
                    else:
                        # but if no <param>.<opt> entry is found, there
                        # *has* to be a global default for this fit method
                        val = method_defaults[fit_method][opt]
                        if not found_default:
                            raise ValueError(
                                'No option "%s" found for "%s". Either'
                                ' set "%s" explicitly or set a "%s" default.'
                                % (opt, pname, param_opt, opt)
                            )
                    settings_dict[fit_method]['params'][pname][opt] = val
        # have to record the global defaults so we can later on apply them
        # to the remaining parameters if the wildcard is used
        if wildcard_here:
            settings_dict[fit_method]['defaults'] = method_defaults[fit_method]
        # make sure no excess specs remain
        unhandled = config[fit_method].keys()
        if unhandled:
            raise ValueError(
                'Unhandled "%s" specs: %s.' % (fit_method, unhandled)
            )

    return settings_dict


def parse_param(config, section, selector, fullname, pname, value):
    """Parse a param specification from a PISA config file.

    Note that if the param sepcification does not include ``fixed``,
    ``prior``, and/or ``range``, the defaults for these are:
    ``fixed = True``, ``prior = None``, and ``range = None``.

    If a prior is specified explicitly via ``.prior``, this takes precendence,
    but if no ``.prior`` is specified and the param's value is parsed to be a
    :class:`uncertainties.AffineScalarFunc` (i.e. have `std_dev` attribute), a
    Gaussian prior is constructed from that and then the AffineScalarFunc is
    stripped out of the param's value (such that it is just a
    :class:`~pint.quantity.Quantity`).

    Parameters
    ----------
    config : pisa.utils.config_parser.PISAConfigParser
    section : string
    selector : string or None
    fullname : string
    pname : string
    value : string

    Returns
    -------
    param : pisa.core.param.Param

    """
    # Note: imports placed here to avoid circular imports
    from pisa.core.param import Param
    from pisa.core.prior import Prior
    kwargs = dict(name=pname, is_fixed=True, prior=None, range=None)
    try:
        value = parse_quantity(value)
        kwargs['value'] = value.nominal_value * value.units
    except ValueError:
        value = parse_string_literal(value)
        kwargs['value'] = value

    # Search for explicit attr specifications
    if config.has_option(section, fullname + '.fixed'):
        kwargs['is_fixed'] = config.getboolean(section, fullname + '.fixed')

    if config.has_option(section, fullname + '.unique_id'):
        kwargs['unique_id'] = config.get(section, fullname + '.unique_id')

    if config.has_option(section, fullname + '.range'):
        range_ = config.get(section, fullname + '.range')
        # Note: `nominal` and `sigma` are called out in the `range_` string
        if 'nominal' in range_:
            nominal = value.n * value.units # pylint: disable=unused-variable
        if 'sigma' in range_:
            sigma = value.s * value.units # pylint: disable=unused-variable
        range_ = range_.replace('[', 'np.array([')
        range_ = range_.replace(']', '])')
        # Strip out uncertainties from value itself (as we will rely on the
        # prior from here on out)
        kwargs['range'] = eval(range_).to(value.units) # pylint: disable=eval-used

    if config.has_option(section, fullname + '.prior'):
        prior = str(config.get(section, fullname + '.prior')).strip().lower()
        if prior == 'uniform':
            kwargs['prior'] = Prior(kind='uniform')
        elif prior == 'jeffreys':
            kwargs['prior'] = Prior(kind='jeffreys', A=kwargs['range'][0], B=kwargs['range'][1])
        elif prior == 'spline':
            priorname = pname
            if selector is not None:
                priorname += '_' + selector
            data = config.get(section, fullname + '.prior.data')
            data = from_file(data)
            data = data[priorname]
            knots = ureg.Quantity(np.asarray(data['knots']), data['units'])
            knots = knots.to(value.units)
            coeffs = np.asarray(data['coeffs'])
            deg = data['deg']
            kwargs['prior'] = Prior(kind='spline', knots=knots, coeffs=coeffs,
                                    deg=deg)
        elif prior == 'none':
            kwargs['prior'] = None
        elif 'gauss' in prior:
            raise Exception('Please use new style +/- notation for gaussian'
                            ' priors in config')
        else:
            raise Exception('Prior type unknown')

    elif hasattr(value, 'std_dev') and value.std_dev != 0:
        kwargs['prior'] = Prior(kind='gaussian',
                                mean=value.nominal_value * value.units,
                                stddev=value.std_dev * value.units)

    # Strip out any uncertainties from value itself (an explicit ``.prior``
    # specification takes precedence over this)
    if hasattr(value, 'std_dev'):
        value = value.nominal_value * value.units
    try:
        param = Param(**kwargs)
    except:
        logging.error('Failed to instantiate new Param object with kwargs %s',
                      kwargs)
        raise

    return param


def parse_pipeline_config(config):
    """Parse pipeline config.

    Parameters
    ----------
    config : string or ConfigParser

    Returns
    -------
    stage_dicts : OrderedDict
        Keys are (stage_name, service_name) tuples and values are OrderedDicts
        with keys the argnames and values the arguments' values. Some known arg
        values are parsed out fully into Python objects, while the rest remain
        as strings that must be used or parsed elsewhere.

    """
    # Note: imports placed here to avoid circular imports
    from pisa.core.binning import MultiDimBinning, OneDimBinning
    from pisa.core.param import ParamSelector

    if isinstance(config, basestring):
        config = from_file(config)
    elif isinstance(config, PISAConfigParser):
        pass
    else:
        raise TypeError(
            '`config` must either be a string or PISAConfigParser. Got %s '
            'instead.' % type(config)
        )

    if not config.has_section('binning'):
        raise NoSectionError(
            "Could not find 'binning'. Only found sections: %s"
            % config.sections()
        )

    # Create binning objects
    binning_dict = {}
    for name, value in config['binning'].items():
        if name.endswith('.order'):
            order = split(config.get('binning', name))
            binning, _ = split(name, sep='.')
            bins = []
            for bin_name in order:
                try:
                    def_raw = config.get('binning', binning + '.' + bin_name)
                except:
                    dims_defined = [
                        split(dim, sep='.')[1] for dim in
                        config['binning'].keys() if
                        dim.startswith(binning + '.') and not
                        dim.endswith('.order')
                    ]
                    logging.error(
                        "Failed to find definition of '%s' dimension of '%s'"
                        " binning entry. Only found definition(s) of: %s",
                        bin_name, binning, dims_defined
                    )
                    del dims_defined
                    raise
                try:
                    kwargs = eval(def_raw) # pylint: disable=eval-used
                except:
                    logging.error(
                        "Failed to evaluate definition of '%s' dimension of"
                        " '%s' binning entry:\n'%s'",
                        bin_name, binning, def_raw
                    )
                    raise
                try:
                    bins.append(OneDimBinning(bin_name, **kwargs))
                except:
                    logging.error(
                        "Failed to instantiate new `OneDimBinning` from '%s'"
                        " dimension of '%s' binning entry with definition:\n"
                        "'%s'\n", bin_name, binning, kwargs
                    )
                    raise
            binning_dict[binning] = MultiDimBinning(bins)

    # Pipeline section
    section = 'pipeline'

    # Get and parse the order of the stages (and which services implement them)
    order = [split(x, STAGE_SEP) for x in split(config.get(section, 'order'))]

    param_selections = []
    if config.has_option(section, 'param_selections'):
        param_selections = split(config.get(section, 'param_selections'))

    # Parse [stage.<stage_name>] sections and store to stage_dicts
    stage_dicts = OrderedDict()
    for stage, service in order:
        old_section_header = 'stage%s%s' % (STAGE_SEP, stage)
        new_section_header = '%s%s%s' % (stage, STAGE_SEP, service)
        if config.has_section(old_section_header):
            logging.warning('%s is an old-style section header, in the future use %s'%(old_section_header, new_section_header))
            section = old_section_header
        elif config.has_section(new_section_header):
            section = new_section_header
        else:
            raise IOError('missing section in cfg for stage %s service %s'%(stage, service))

        # Instantiate dict to store args to pass to this stage
        service_kwargs = OrderedDict()

        param_selector = ParamSelector(selections=param_selections)
        service_kwargs['params'] = param_selector

        n_params = 0
        for fullname, value in config.items(section):
            # See if this matches a param specification
            param_match = PARAM_RE.match(fullname)
            if param_match is not None:
                n_params += 1

                param_match_dict = param_match.groupdict()
                param_subfields = param_match_dict['subfields'].split('.')

                # Figure out what the dotted fields represent...
                infodict = interpret_param_subfields(subfields=param_subfields)

                # If field is an attr, skip since these are located manually
                if infodict['attr'] is not None:
                    continue

                # Check if this param already exists in a previous stage; if
                # so, make sure there are no specs for this param, but just a
                # link to previous the param object that is already
                # instantiated.
                for kw in stage_dicts.values():
                    # Stage did not get a `params` argument from config
                    if not kw.has_key('params'):
                        continue

                    # Retrieve the param from the ParamSelector
                    try:
                        param = kw['params'].get(
                            name=infodict['pname'],
                            selector=infodict['selector']
                        )
                    except KeyError:
                        continue

                    # Make sure there are no other specs (in this section) for
                    # the param defined defined in previous section
                    for a in PARAM_ATTRS:
                        if config.has_option(section, '%s.%s' %(fullname, a)):
                            raise ValueError("Parameter spec. '%s' of '%s' "
                                             "found in section '%s', but "
                                             "parameter exists in previous "
                                             "stage!"%(a, fullname, section))

                    break

                # Param *not* found in a previous stage (i.e., no explicit
                # `break` encountered in `for` loop above); therefore must
                # instantiate it.
                else:
                    param = parse_param(
                        config=config,
                        section=section,
                        selector=infodict['selector'],
                        fullname=fullname,
                        pname=infodict['pname'],
                        value=value
                    )

                param_selector.update(param, selector=infodict['selector'])

            # If it's not a param spec but contains 'binning', assume it's a
            # binning spec for CAKE stages
            elif 'binning' in fullname:
                service_kwargs[fullname] = binning_dict[value]

            # it's gonna be a PI stage
            elif '_specs' in fullname:
                value = parse_string_literal(value)
                # is it None?
                if value is None:
                    service_kwargs[fullname] = value
                # is it evts?
                elif value in ['evnts', 'events']:
                    service_kwargs[fullname] = 'events'
                # so it gotta be a binning
                else:
                    service_kwargs[fullname] = binning_dict[value]

            # it's a list on in/output names list
            elif fullname.endswith('_names'):
                value = split(value)
                service_kwargs[fullname] = value
            # Otherwise it's some other stage instantiation argument; identify
            # this by its full name and try to interpret and instantiate a
            # Python object using the string
            else:
                try:
                    value = parse_quantity(value)
                    value = value.nominal_value * value.units
                except ValueError:
                    value = parse_string_literal(value)
                service_kwargs[fullname] = value

        # If no params actually specified in config, remove 'params' from the
        # service's keyword args
        if n_params == 0:
            service_kwargs.pop('params')

        # Store the service's kwargs to the stage_dicts
        stage_dicts[(stage, service)] = service_kwargs

    return stage_dicts


class MutableMultiFileIterator(object):
    """
    Iterate through the lines of an already-open file (`fp`) but then can pause
    at any point and open and iterate through another file via the
    `switch_to_file` method (and this file can be paused to iterate through
    another, etc.).

    This has the effect of in-lining files within files for e.g. parsing
    multiple files as if they're a singe file. Which line comes from which file
    is also tracked for generating maximally-useful error messages, via the
    `location` method.

    Note that circular references are not allowed.

    Parameters
    ----------
    fp : file-like object
        The (opened) main config to be read. E.g. can be an opened file,
        io.StringIO object, etc.

    fpname : string
        Identifier for the initially `fp` object

    """
    def __init__(self, fp, fpname, fpath=None):
        self._iter_stack = []
        """Stack for storing dicts with 'fp', 'fpname', 'fpath', 'lineno', and
        'line' for keeping track of the hierarchy of master config & included
        configs"""

        # It's ok to not find the fpname / fpname to not be a file for the
        # *master* config, since this could e.g. be a io.StringIO file-like
        # object (`read_string`) which comes from no actual file/resource on
        # disk.
        if not fpname and hasattr(fp, 'name'):
            fpname = fp.name

        if fpath is None:
            try:
                resource = find_resource(fpname)
            except IOError:
                pass
            else:
                if isfile(resource):
                    fpath = abspath(expanduser(expandvars(fpname)))

        if fpath is None:
            try:
                resource = find_resource(fpname)
            except IOError:
                pass
            else:
                if isfile(resource):
                    fpath = resource

        if fpath is None:
            self.fpaths_processed = []
        else:
            self.fpaths_processed = [fpath]

        self.fps_processed = [fp]

        record = dict(fp=fp, fpname=fpname, fpath=fpath, lineno=0, line='')
        self._iter_stack.append(record)
        self.file_hierarchy = OrderedDict([(fpname, OrderedDict())])

    def next(self):
        """Iterate through lines in the file(s).

        Returns
        -------
        line : string
            The next line from the current file.

        fpname : string
            The `fpname` of the file from which the line was gotten.

        lineno : int
            The line number in the file.

        """
        if not self._iter_stack:
            self._cleanup()
            raise StopIteration
        try:
            record = self._iter_stack[-1]
            record['line'] = next(record['fp'])
            record['lineno'] += 1
            return record
        except StopIteration:
            record = self._iter_stack.pop()
            logging.trace(('Finished processing "{fpname:s}" with {lineno:d}'
                           ' line(s)').format(**record))
            return next(self)
        except:
            self._cleanup()
            raise

    def switch_to_file(self, fp=None, fpname=None):
        """Switch iterator to a new resource location to continue processing.

        Parameters
        ----------
        fp : None or file-like object
            If `fp` is specified, this takes precedence over `fpname`.

        fpname : None or string
            Path of the file or resource to read from. This resource will be
            located and opened if `fp` is None.

        encoding
            Argument is passed to the builtin ``open`` function for opening
            the file.

        """
        fpath = None
        if fp is None:
            assert fpname
            resource = find_resource(fpname)
            if isfile(resource):
                fpath = abspath(expanduser(expandvars(resource)))
                if fpath in self.fpaths_processed:
                    self._cleanup()
                    raise ValueError(
                        'Circular reference; already processed "%s" at path'
                        ' "%s"' % (fpname, fpath)
                    )
            else:
                self._cleanup()
                raise ValueError('`fpname` "%s" is not a file')
            fp_ = c_open(fpath, encoding=None)
        else:
            fp_ = fp
            if fpname is None:
                if hasattr(fp_, 'name'):
                    fpname = fp_.name
                else:
                    fpname = ''
            try:
                resource = find_resource(fpname)
            except IOError:
                pass
            else:
                if isfile(resource):
                    fpath = resource
            if fp in self.fps_processed:
                self._cleanup()
                raise ValueError(
                    'Circular reference; already processed file pointer "%s"'
                    ' at path "%s"' % (fp_, fpname)
                )

        if fpath is not None:
            if fpath in self.fpaths_processed:
                self._cleanup()
                raise ValueError(
                    'Circular reference; already processed "%s" at path'
                    ' "%s"' % (fpname, fpath)
                )
            self.fpaths_processed.append(fpath)

        self.fps_processed.append(fp)
        if fpath is not None:
            self.fpaths_processed.append(fpath)

        logging.trace('Switching to "%s" at path "%s"' % (fpname, fpath))

        record = dict(fp=fp_, fpname=fpname, fpath=fpath, lineno=0, line='')
        self._iter_stack.append(record)

    @property
    def location(self):
        """string : Full hierarchical location, formatted for display"""
        info = ['File hierarchy (most recent last):\n']
        for record_num, record in enumerate(self._iter_stack):
            s = '  Line {lineno:d}, fpname "{fpname:s}"'
            if record_num > 0:
                s += ' at path "{fpath:s}"'
            s += '\n    {line:s}'
            info.append(s.format(**record))
        return ''.join(info)

    def __iter__(self):
        return self

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        """Close all file handles opened by this object (i.e. all except the
        first file pointer, which is provided as an argument to `__init__`)"""
        for record in self._iter_stack[1:]:
            record['fp'].close()


class PISAConfigParser(RawConfigParser):
    """
    Parses a PISA config file, extending :class:`configparser.RawConfigParser`
    (the backport of RawConfigParser from Python 3.x) by adding the ability to
    include external files inline via, for example:

    .. code-block:: cfg

        #include /path/to/file.cfg
        #include path/to/resource.cfg
        #include path/to/resource2.cfg as section2

        [section1]
        key11 = value1
        key12 = ${section2:key21}
        key13 = value3

    where the files or resources located at "/path/to/file.cfg",
    "path/to/resource.cfg", and "path/to/resource2.cfg" are effectively inlined
    wherever the #include statements occur.

    The ``#include path/to/resource2.cfg as section_name`` syntax
    prefixes the contents of ``resource2.cfg`` by a section header named
    "section2", expanding ``resource2.cfg`` as:

    .. code-block:: cfg

        [section2]
        line1 of resource2.cfg
        line2 of resource2.cfg
        ... etc.

    Special parsing rules we have added to make ``#include`` behavior sensible:

    1. Using an ``#include file`` that contains a sction header
       (``[section_name]``) *or* using ``#include file as section_name``
       requires that the next non-blank / non-comment / non-#include line be a
       new section header (``[section_name2]``).
    2. Empty sections after fully parsing a config will raise a ``ValueError``.
       This is likely never a desired behavior, and should alert the user to
       inadvertent use of ``#include``.

    Also note that, unlike the default :class:`~configparser.ConfigParser`
    behavior, :class:`~configparser.ExtendedInterpolation` is used, whitespace
    surrounding text in a section header is ignored, empty lines are *not*
    allowed between multi-line values, and section names, keys, and values are
    all case-sensitive.

    All other options are taken as the defaults / default behaviors of
    :class:`~configparser.ConfigParser`.

    See help for :class:`configparser.ConfigParser` for further help on valid
    config file syntax and parsing behavior.

    """

    _DEFAULT_INTERPOLATION = ExtendedInterpolation()
    INCLUDE_RE = re.compile(r'\s*#include\s+(?P<include>\S.*)')
    INCLUDE_AS_RE = re.compile(r'\s*(?P<file>.+)((?:\s+as\s+)(?P<as>\S+))')
    SECTCRE = re.compile(r'\[\s*(?P<header>[^]]+?)\s*\]')

    def __init__(self):
        #self.default_section = None #DEFAULTSECT
        # Instantiate parent class with PISA-specific options
        #super(PISAConfigParser, self).__init__(
        RawConfigParser.__init__(
            self,
            interpolation=ExtendedInterpolation(),
            empty_lines_in_values=False,
        )
        self.file_iterators = []

    def set(self, section, option, value=None):
        """Set an option.  Extends RawConfigParser.set by validating type and
        interpolation syntax on the value."""
        _, option, value = self._validate_value_types(option=option,
                                                      value=value)
        super(PISAConfigParser, self).set(section, option, value)

    def add_section(self, section):
        """Create a new section in the configuration.  Extends
        RawConfigParser.add_section by validating if the section name is
        a string."""
        section, _, _ = self._validate_value_types(section=section)
        super(PISAConfigParser, self).add_section(section)

    def optionxform(self, optionstr):
        """Enable case-sensitive options in .cfg files, and force all values to
        be ASCII strings."""
        return optionstr #.encode('ascii')

    @property
    def hash(self):
        """int : Hash value of the contents (does not depend on order of
        sections, but does depend on order of keys within each section)"""
        return self.__hash__()

    def __hash__(self):
        return hash_obj([(sec, (self.items(sec)))
                         for sec in sorted(self.sections())])

    @staticmethod
    def _get_include_info(line):
        match = PISAConfigParser.INCLUDE_RE.match(line)
        if not match:
            return None
        include = match.groupdict()['include']
        match = PISAConfigParser.INCLUDE_AS_RE.match(include)
        if match is None:
            return {'file': include, 'as': None}
        return match.groupdict()

    def read(self, filenames, encoding=None):
        """Override `read` method to interpret `filenames` as PISA resource
        locations, then call overridden `read` method. Also, IOError fails
        here, whereas it is ignored in RawConfigParser.

        For further help on this method and its arguments, see
        :method:`~backports.configparser.configparser.read`

        """
        if isinstance(filenames, basestring):
            filenames = [filenames]
        resource_locations = []
        for filename in filenames:
            resource_location = find_resource(filename)
            if not isfile(resource_location):
                raise ValueError(
                    '"%s" is not a file or could not be located' % filename
                )
            resource_locations.append(resource_location)

        filenames = resource_locations

        # NOTE: From here on, most of the `read` method is copied, but
        # ignoring IOError exceptions is removed here. Python copyrights apply.

        if PY2 and isinstance(filenames, bytes):
            # we allow for a little unholy magic for Python 2 so that
            # people not using unicode_literals can still use the library
            # conveniently
            warnings.warn(
                "You passed a bytestring as `filenames`. This will not work"
                " on Python 3. Use `cp.read_file()` or switch to using Unicode"
                " strings across the board.",
                DeprecationWarning,
                stacklevel=2,
            )
            filenames = [filenames]
        elif isinstance(filenames, str):
            filenames = [filenames]
        read_ok = []
        for filename in filenames:
            with c_open(filename, encoding=encoding) as fp:
                self._read(fp, filename)
            read_ok.append(filename)
        return read_ok

    # NOTE: the `_read` method is copy-pasted (then modified slightly) from
    # Python's backports.configparser (version 3.5.0), and so any copyright
    # notices at the top of this file might need modification to be compatible
    # with copyrights on that module.
    #
    # Also, diff this function with future releases in case something needs
    # modification.
    def _read(self, fp, fpname):
        """Parse a sectioned configuration file.

        Each section in a configuration file contains a header, indicated by
        a name in square brackets (`[]'), plus key/value options, indicated by
        `name' and `value' delimited with a specific substring (`=' or `:' by
        default).

        Values can span multiple lines, as long as they are indented deeper
        than the first line of the value. Depending on the parser's mode, blank
        lines may be treated as parts of multiline values or ignored.

        Configuration files may include comments, prefixed by specific
        characters (`#' and `;' by default). Comments may appear on their own
        in an otherwise empty line or may be entered in lines holding values or
        section names.

        This implementation is extended from the original to also accept

        .. code:: ini

          #include <file or pisa_resource>

        or

        .. code:: ini

          #include <file or pisa_resource> as <section_name>

        syntax anywhere in the file, which switches (via
        :class:`MutableMultiFileIterator`) to the new file as if it were
        in-lined within the original file. The latter syntax also prepends a
        section header

        .. code:: ini

            [section_name]

        before the text of the specified file or pisa_resource.

        """
        elements_added = set()
        cursect = None                        # None, or a dictionary
        sectname = None
        optname = None
        lineno = 0
        indent_level = 0
        e = None                              # None, or an exception

        file_iter = MutableMultiFileIterator(fp=fp, fpname=fpname)
        self.file_iterators.append(file_iter)
        for record in file_iter:
            fpname = record['fpname']
            lineno = record['lineno']
            line = record['line']

            comment_start = sys.maxsize
            # strip inline comments
            inline_prefixes = dict(
                (p, -1) for p in self._inline_comment_prefixes)
            while comment_start == sys.maxsize and inline_prefixes:
                next_prefixes = {}
                for prefix, index in inline_prefixes.items():
                    index = line.find(prefix, index+1)
                    if index == -1:
                        continue
                    next_prefixes[prefix] = index
                    if index == 0 or (index > 0 and line[index-1].isspace()):
                        comment_start = min(comment_start, index)
                inline_prefixes = next_prefixes
            # parse #include statement
            include_info = self._get_include_info(line)
            if include_info:
                file_iter.switch_to_file(fpname=include_info['file'])
                if include_info['as']:
                    as_header = '[%s]\n' % include_info['as']
                    file_iter.switch_to_file(
                        fp=StringIO(as_header.decode('utf-8'))
                    )
                continue
            # strip full line comments
            for prefix in self._comment_prefixes:
                if line.strip().startswith(prefix):
                    comment_start = 0
                    break
            if comment_start == sys.maxsize:
                comment_start = None
            value = line[:comment_start].strip()
            if not value:
                if self._empty_lines_in_values:
                    # add empty line to the value, but only if there was no
                    # comment on the line
                    if (comment_start is None and
                        cursect is not None and
                        optname and
                        cursect[optname] is not None):
                        cursect[optname].append('') # newlines added at join
                else:
                    # empty line marks end of value
                    indent_level = sys.maxsize
                continue
            # continuation line?
            first_nonspace = self.NONSPACECRE.search(line)
            cur_indent_level = first_nonspace.start() if first_nonspace else 0
            if (cursect is not None and optname and
                cur_indent_level > indent_level):
                cursect[optname].append(value)
            # a section header or option header?
            else:
                indent_level = cur_indent_level
                # is it a section header?
                mo = self.SECTCRE.match(value)
                if mo:
                    sectname = mo.group('header')
                    if sectname in self._sections:
                        if self._strict and sectname in elements_added:
                            raise DuplicateSectionError(sectname, fpname,
                                                        lineno)
                        cursect = self._sections[sectname]
                        elements_added.add(sectname)
                    elif sectname == self.default_section:
                        cursect = self._defaults
                    else:
                        cursect = self._dict()
                        self._sections[sectname] = cursect
                        self._proxies[sectname] = SectionProxy(self, sectname)
                        elements_added.add(sectname)
                    # So sections can't start with a continuation line
                    optname = None
                # no section header in the file?
                elif cursect is None:
                    raise MissingSectionHeaderError(fpname, lineno, line)
                # an option line?
                else:
                    mo = self._optcre.match(value)
                    if mo:
                        optname, vi, optval = mo.group('option', 'vi', 'value')
                        if not optname:
                            e = self._handle_error(e, fpname, lineno, line)
                        optname = self.optionxform(optname.rstrip())
                        if (self._strict and
                            (sectname, optname) in elements_added):
                            raise DuplicateOptionError(sectname, optname,
                                                       fpname, lineno)
                        elements_added.add((sectname, optname))
                        # This check is fine because the OPTCRE cannot
                        # match if it would set optval to None
                        if optval is not None:
                            optval = optval.strip()
                            cursect[optname] = [optval]
                        else:
                            # valueless option handling
                            cursect[optname] = None
                    else:
                        # a non-fatal parsing error occurred. set up the
                        # exception but keep going. the exception will be
                        # raised at the end of the file and will contain a
                        # list of all bogus lines
                        e = self._handle_error(e, fpname, lineno, line)
        # if any parsing errors occurred, raise an exception
        if e:
            raise e
        self._join_multiline_values()


def test_parse_pipeline_config():
    """Unit test for function `parse_pipeline_config`"""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--pipeline', metavar='CONFIGFILE',
        default='settings/pipeline/example.cfg',
        help='File containing settings for the pipeline.'
    )
    parser.add_argument(
        '-v', action='count', default=0,
        help='Set verbosity level. Minimum is forced to level 1 (info)'
    )
    args = parser.parse_args()
    args.v = max(1, args.v)
    set_verbosity(args.v)

    # Load via PISAConfigParser
    config0 = PISAConfigParser()
    config0.read(args.pipeline)
    _ = parse_pipeline_config(config0)

    # Load directly
    config = parse_pipeline_config(args.pipeline)

    logging.debug('Keys and values found in config:')
    for key, vals in config.items():
        logging.debug('%s: %s', key, vals)

    logging.info('<< PASS : test_parse_pipeline_config >>')


def test_MutableMultiFileIterator():
    """Unit test for class `MutableMultiFileIterator`"""
    import shutil
    import tempfile

    prefixes = ['a', 'b', 'c']
    file_len = 4

    reference_lines = [
        # start in file a
        'a0', 'a1',
        # switch to file b after second line of a
        'b0', 'b1',
        # switch to file c after second line of b
        'c0', 'c1', 'c2', 'c3',
        # switch back to b after exhausting c
        'b2', 'b3',
        # switch back to a after exhausting b
        'a2', 'a3'
    ]

    tempdir = tempfile.mkdtemp()
    try:
        # Create test files
        paths = [join(tempdir, prefix) for prefix in prefixes]
        for prefix, path in zip(prefixes, paths):
            with open(path, 'w') as f:
                for i in range(file_len):
                    f.write('%s%d\n' % (prefix, i))
            logging.trace(path)

        actual_lines = []
        with open(paths[0]) as fp:
            file_iter = MutableMultiFileIterator(fp=fp, fpname=paths[0])

            remaining_paths = paths[1:]

            for record in file_iter:
                actual_lines.append(record['line'].strip())
                logging.trace(str(record))
                if record['line'][1:].strip() == '1':
                    if remaining_paths:
                        path = remaining_paths.pop(0)
                        file_iter.switch_to_file(fpname=path)
                    else:
                        for l in str(file_iter.location).split('\n'):
                            logging.trace(l)
    except:
        shutil.rmtree(tempdir)
        raise

    if actual_lines != reference_lines:
        raise ValueError('<< FAIL : test_MutableMultiFileIterator >>')

    logging.info('<< PASS : test_MutableMultiFileIterator >>')


if __name__ == '__main__':
    # Note: put test_parse_pipeline_config first since it reads command line
    # args and -v option will be used also by subsequent unit tests
    test_parse_pipeline_config()
    test_MutableMultiFileIterator()
