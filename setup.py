#!/usr/bin/env python
# pylint: disable=line-too-long

"""
Allows for PISA installation. Tested with `pip`. Use the environment variable
`CC` to pass a custom compiler to the instller. (GCC and Clang should both
work; OpenMP support--an optional dependency--is only available for recent
versions of the latter).

Checkout the source code tree in the current directory via

    $ git clone https://github.com/icecubeopensource/pisa.git

and install basic PISA package (in editable mode via -e flag) via

    $ pip install -e ./pisa -r ./pisa/requirements.txt

or include optional dependencies by specifying them in brackets

    $ pip install -e ./pisa[cuda,numba,develop] -r ./pisa/requirements.txt

If you wish to upgrade PISA and/or its dependencies:

    $ pip install ./pisa[cuda,numba,develop] -r ./pisa/requirements.txt --upgrade
"""


from __future__ import absolute_import

from distutils.command.build import build
import os
import shutil
import subprocess
import sys
import tempfile

from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension, find_packages
import versioneer


__all__ = ['setup_cc', 'check_cuda', 'OMP_TEST_PROGRAM', 'check_openmp',
           'CustomBuild', 'CustomBuildExt', 'do_setup']

__author__ = 'S. Boeser, J.L. Lanfranchi, P. Eller, M. Hieronymus'

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


# TODO: Compile CUDA kernel(s) here (since no need for dynamic install yet...
# unless e.g. datatype becomes optional and therefore compilation of the kernel
# needs to be done at run-time).

# TODO: address some/all of the following in the `setup()` method?
# * package_data
# * exclude_package_data : dict
# * include_package_data : bool; include everything in source control
# * eager_resources : list of str paths (using '/' notation relative to source root) unzip these together if any one is requested (for C
#   extensions, etc.)


def setup_cc():
    """Set env var CC=cc if it is undefined"""
    if 'CC' not in os.environ or os.environ['CC'].strip() == '':
        os.environ['CC'] = 'cc'


def check_cuda():
    """pycuda is considered to be present if it can be imported"""
    try:
        import pycuda.driver # pylint: disable=unused-variable
    except Exception:
        cuda = False
    else:
        cuda = True
    return cuda


# See http://openmp.org/wp/openmp-compilers/
OMP_TEST_PROGRAM = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}"""


def check_openmp():
    """OpenMP is present if a test program can compile with -fopenmp flag (e.g.
    some versions of Clang / gcc don't support OpenMP).

    Source: http://stackoverflow.com/questions/16549893

    """
    openmp = False
    setup_cc()
    tmpfname = r'test.c'
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    cc = os.environ['CC']
    try:
        with open(tmpfname, 'w', 0) as f:
            f.write(OMP_TEST_PROGRAM)
        with open(os.devnull, 'w') as fnull:
            returncode = subprocess.call([cc, '-fopenmp', tmpfname],
                                         stdout=fnull, stderr=fnull)
        # Successful build (possibly with warnings) means we can use OpenMP
        openmp = returncode == 0
    finally:
        # Restore directory location and clean up
        os.chdir(curdir)
        shutil.rmtree(tmpdir)
    return openmp


class CustomBuild(build):
    """Define custom build order, so that the python interface module created
    by SWIG is staged in build_py.

    """
    # different order: build_ext *before* build_py
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts)
    ]


class CustomBuildExt(build_ext):
    """Replace default build_ext to allow for numpy install before setup.py
    needs it to include its dir.

    Code copied from coldfix / http://stackoverflow.com/a/21621689

    """
    def finalize_options(self):
        build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def do_setup():
    """Perform the setup process"""
    setup_cc()
    sys.stdout.write('Using compiler %s\n' %os.environ['CC'])

    has_openmp = check_openmp()
    if not has_openmp:
        sys.stderr.write(
            'WARNING: Could not compile test program with -fopenmp;'
            ' installing PISA without OpenMP support.\n'
        )

    # Collect (build-able) external modules and package_data
    ext_modules = []

    # Prob3 oscillation code (pure C++, no CUDA)
    prob3cpu_module = Extension(
        name='pisa.stages.osc.prob3cc._BargerPropagator',
        sources=[
            'pisa/stages/osc/prob3cc/BargerPropagator.i',
            'pisa/stages/osc/prob3cc/BargerPropagator.cc',
            'pisa/stages/osc/prob3cc/EarthDensity.cc',
            'pisa/stages/osc/prob3cc/mosc.c',
            'pisa/stages/osc/prob3cc/mosc3.c'
        ],
        include_dirs=[
            'pisa/stages/osc/prob3cc/'
        ],
        extra_compile_args=['-Wall', '-O3', '-fPIC'],
        swig_opts=['-c++'],
    )
    ext_modules.append(prob3cpu_module)

    # Include these things in source (and binary?) distributions
    package_data = {}

    # Include documentation and license files wherever they may be
    package_data[''] = ['*.md', '*.rst', 'LICENSE*']

    package_data['pisa_examples'] = [
        'resources/aeff/*.json*',
        'resources/cross_sections/*json*',
        'resources/discr_sys/*.json*',

        'resources/events/*.hdf5',
        'resources/events/*.json*',

        'resources/flux/*.d',
        'resources/osc/*.hdf5',
        'resources/osc/*.dat',
        'resources/pid/*.json*',
        'resources/priors/*.json*',
        'resources/priors/*.md',
        'resources/reco/*.json*',

        'resources/settings/binning/*.cfg',
        'resources/settings/discrete_sys/*.cfg',
        'resources/settings/logging/logging.json',
        'resources/settings/mc/*.cfg',
        'resources/settings/minimizer/*.json*',
        'resources/settings/osc/*.cfg',
        'resources/settings/osc/*.md',
        'resources/settings/pipeline/*.cfg',
        'resources/settings/pipeline/*.md',

        'notebooks/*ipynb',
    ]

    package_data['pisa_tests'] = [
        '*.py',
        '*.sh'
    ]

    package_data['pisa.utils'] = [
        '*.h',
        '*.pyx'
    ]

    package_data['pisa.stages.osc.prob3cuda'] = [
        '*.h',
        '*.cu'
    ]

    extra_compile_args = ['-O3', '-ffast-math', '-msse3']
    extra_link_args = ['-ffast-math', '-msse2']
    if has_openmp:
        gaussians_cython_module = Extension(
            'pisa.utils.gaussians_cython',
            ['pisa/utils/gaussians_cython.pyx'],
            libraries=['m'],
            extra_compile_args=extra_compile_args + ['-fopenmp'],
            extra_link_args=extra_link_args + ['-fopenmp'],
        )
    else:
        gaussians_cython_module = Extension(
            'pisa.utils.gaussians_cython',
            ['pisa/utils/gaussians_cython.pyx'],
            libraries=['m'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ext_modules.append(gaussians_cython_module)

    cmdclasses = {'build': CustomBuild, 'build_ext': CustomBuildExt}
    cmdclasses.update(versioneer.get_cmdclass())

    # Now do the actual work
    setup(
        name='pisa',
        version=versioneer.get_version(),
        description='Tools for analyzing and drawing statistical conclusions from experimental data',
        license='Apache 2.0',
        author='The IceCube Collaboration',
        author_email='jll1062+pisa@phys.psu.edu',
        url='http://github.com/icecubeopensource/pisa',
        cmdclass=cmdclasses,
        python_requires='>=2.7, <3.0',
        setup_requires=[
            'pip>=1.8',
            'setuptools>18.5', # versioneer requires >18.5
            'cython',
            'numpy>=1.11',
        ],
        install_requires=[
            'configparser',
            'scipy>=0.17',
            'dill',
            'h5py',
            'line_profiler',
            'matplotlib>=2.0', # 1.5: inferno colormap; 2.0: 'C0' colorspec
            'pint>=0.8', # earlier versions buggy
            'kde',
            'simplejson>=3.2',
            'tables',
            'uncertainties',
            'decorator',
        ],
        extras_require={
            'cuda': [
                'pycuda'
            ],
            'numba': [
                'llvmlite>=0.16', # fastmath jit flag
                'numba==0.35' # fastmath jit flag, version fixed because of issue #439
            ],
            'develop': [
                'pylint>=1.7',
                'recommonmark',
                'sphinx>=1.3',
                'sphinx_rtd_theme',
                'versioneer',
                'yapf',
            ],
            # TODO: get mceq install to work... this is non-trivial since that
            # project isn't exactly cleanly instllable via pip already, plus it
            # has "sub-projects" that won't get picked up by a simple single
            # URL (e.g. the data). Plus it's huge (~1GB).
            #'mceq': [
            #    'llvmlite>=0.16',
            #    'numba==0.35',
            #    'progressbar',
            #    'MCEq'
            #]
        },
        #dependency_links=[
        #    'git+https://github.com/afedynitch/MCEq.git#egg=MCEq'
        #],
        packages=find_packages(),
        ext_modules=ext_modules,
        package_data=package_data,
        # Cannot be compressed due to c, pyx, and cuda source files/headers
        # that need to be compiled at run-time but are inaccessible in a zip
        # (I think...)
        zip_safe=False,
        entry_points={
            'console_scripts': [
                # Scripts in core dir
                'pisa-distribution_maker = pisa.core.distribution_maker:main',
                'pisa-pipeline = pisa.core.pipeline:main',

                # Scripts in scripts dir
                'pisa-add_flux_to_events_file = pisa.scripts.add_flux_to_events_file:main',
                'pisa-analysis = pisa.scripts.analysis:main',
                'pisa-postproc = pisa.scripts.analysis_postprocess:main',
                'pisa-compare = pisa.scripts.compare:main',
                'pisa-convert_config_format = pisa.scripts.convert_config_format:main',
                'pisa-fit_discrete_sys = pisa.scripts.fit_discrete_sys:main',
                'pisa-make_asymmetry_plots = pisa.scripts.make_asymmetry_plots:main',
                'pisa-make_events_file = pisa.scripts.make_events_file:main',
                'pisa-make_nufit_theta23_spline_priors = pisa.scripts.make_nufit_theta23_spline_priors:main',
                'pisa-make_systematic_variation_plots = pisa.scripts.make_systematic_variation_plots:main',
                'pisa-make_toy_events = pisa.scripts.make_toy_events:main',
                'pisa-profile_scan = pisa.scripts.profile_scan:main',
                'pisa-scan_allsyst = pisa.scripts.scan_allsyst:main',

                # Scripts in pisa_tests dir
                'pisa-test_changes_with_combined_pidreco = pisa_tests.test_changes_with_combined_pidreco:main',
                'pisa-test_example_pipelines = pisa_tests.test_example_pipelines:main'
            ]
        }
    )
    if not check_cuda():
        sys.stderr.write('WARNING: Could not import pycuda; attempt will be '
                         ' made to install, but if this fails, PISA may not be'
                         ' able to support CUDA (GPU) accelerations.\n')


if __name__ == '__main__':
    do_setup()
