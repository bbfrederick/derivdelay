#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from derivdelay.filter import NoncausalFilter
from derivdelay.miscmath import stdnormalize
from derivdelay.refinedelay import (
    filterderivratios,
    getderivratios,
    ratiotodelay,
    trainratiotooffset,
)
from derivdelay.resample import FastResampler
from derivdelay.tests.utils import mse


def _get_parser():
    """
    Argument parser for demo
    """
    parser = argparse.ArgumentParser(
        prog="demo",
        description=("Generate a histogram of the values in a timecourse"),
        allow_abbrev=False,
    )

    parser.add_argument(
        "--displayplots",
        dest="displayplots",
        action="store_true",
        help=("Make pretty plots."),
        default=False,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Print additional debugging information."),
        default=False,
    )

    return parser


def eval_refinedelay(
    sampletime=0.72,
    tclengthinsecs=300.0,
    mindelay=-5.0,
    maxdelay=5.0,
    numpoints=501,
    smoothpts=3,
    nativespaceshape=(10, 10, 10),
    displayplots=False,
    padtime=30.0,
    noiselevel=0.0,
    outputdir=".",
    outputsuffix="",
    debug=False,
):
    np.random.seed(12345)
    tclen = int(tclengthinsecs // sampletime)

    Fs = 1.0 / sampletime
    print("Testing transfer function:")
    lowestfreq = 1.0 / (sampletime * tclen)
    nyquist = 0.5 / sampletime
    print(
        "    sampletime=",
        sampletime,
        ", timecourse length=",
        tclengthinsecs,
        "s,  possible frequency range:",
        lowestfreq,
        nyquist,
    )

    # make an sLFO timecourse
    timeaxis = np.linspace(0.0, sampletime * tclen, num=tclen, endpoint=False)
    rawgms = stdnormalize(np.random.normal(size=tclen))
    testfilter = NoncausalFilter(filtertype="lfo")
    sLFO = stdnormalize(testfilter.apply(Fs, rawgms))
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Initial regressor")
        plt.plot(timeaxis, rawgms)
        plt.plot(timeaxis, sLFO)
        plt.show()

    # now turn it into a lagtc generator
    numpadtrs = int(padtime // sampletime)
    padtime = sampletime * numpadtrs
    lagtcgenerator = FastResampler(timeaxis, sLFO, padtime=padtime)

    # find the mapping of glm ratios to delays
    trainratiotooffset(
        lagtcgenerator,
        timeaxis,
        os.path.join(outputdir, "refinedelaytest" + outputsuffix),
        "norm",
        mindelay=mindelay,
        maxdelay=maxdelay,
        numpoints=numpoints,
        smoothpts=smoothpts,
        debug=debug,
    )

    # make a delay map
    numlags = nativespaceshape[0] * nativespaceshape[1] * nativespaceshape[2]
    lagtimes = np.linspace(mindelay, maxdelay, numlags, endpoint=True)
    if debug:
        print("    lagtimes=", lagtimes)

    # now make synthetic fMRI data
    internalvalidfmrishape = (numlags, tclen)
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numlags, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    for i in range(numlags):
        noisevec = stdnormalize(testfilter.apply(Fs, stdnormalize(np.random.normal(size=tclen))))
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i]) + noiselevel * noisevec

    """if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Timecourses")
        for i in range(0, numlags, 200):
            plt.plot(timeaxis, fmridata[i, :])
        plt.show()"""

    thedims = np.ones(nativespaceshape, dtype=float)

    rt_floattype = "float64"
    rvalue = np.zeros(numlags, dtype=rt_floattype)
    r2value = np.zeros(numlags, dtype=rt_floattype)
    fitcoeff = np.zeros((numlags, 2), dtype=rt_floattype)

    glmderivratios = getderivratios(
        fmridata,
        rvalue,
        r2value,
        fitcoeff[:, :2],
        timeaxis,
        lagtcgenerator,
        numderivs=1,
        debug=debug,
    )

    medfilt, filteredglmderivratios, themad = filterderivratios(
        glmderivratios,
        nativespaceshape,
        validvoxels,
        thedims,
        patchthresh=3.0,
        fileiscifti=False,
        textio=False,
        rt_floattype="float64",
        debug=debug,
    )

    delayoffset = filteredglmderivratios * 0.0
    for i in range(filteredglmderivratios.shape[0]):
        delayoffset[i] = ratiotodelay(filteredglmderivratios[i])

    # do the tests
    msethresh = 0.1
    aethresh = 2
    print(f"{mse(lagtimes, delayoffset)=}")
    assert mse(lagtimes, delayoffset) < msethresh
    # np.testing.assert_almost_equal(lagtimes, delayoffset, aethresh)

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Lagtimes")
        plt.plot(lagtimes)
        plt.plot(delayoffset)
        plt.legend(["Target", "Fit"])
        plt.show()


def demo(displayplots=False, debug=False):
    for noiselevel in np.linspace(0.0, 0.5, num=5, endpoint=True):
        eval_refinedelay(
            sampletime=0.72,
            tclengthinsecs=300.0,
            mindelay=-3.0,
            maxdelay=3.0,
            numpoints=501,
            smoothpts=9,
            nativespaceshape=(10, 10, 10),
            displayplots=displayplots,
            outputsuffix="_1",
            noiselevel=noiselevel,
            debug=debug,
        )
    eval_refinedelay(
        sampletime=0.72,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=9,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_2",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=0.72,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=5,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_3",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=1.5,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=3,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_1p5_501_3",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=3.0,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=3,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_3p0_501_3",
        debug=debug,
    )
