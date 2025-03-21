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
import sys

# this is here until numpy deals with their fft issue
import warnings

import numpy as np
import pylab as pl
import scipy as sp
from scipy import signal

import derivdelay.filter as dd_filt
import derivdelay.io as dd_io

warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ---------------------------------------- Global constants -------------------------------------------
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from numba import jit
except ImportError:
    donotusenumba = True
else:
    donotusenumba = False


def conditionaljit():
    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=True)

    return resdec


def conditionaljit2():
    def resdec(f):
        if donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=True)

    return resdec


def disablenumba():
    global donotusenumba
    donotusenumba = True


# --------------------------- Resampling and time shifting functions -------------------------------------------
class FastResampler:
    def __init__(
        self,
        timeaxis,
        timecourse,
        padtime=30.0,
        upsampleratio=100,
        doplot=False,
        debug=False,
        method="univariate",
    ):
        self.timeaxis = timeaxis
        self.timecourse = timecourse
        self.upsampleratio = upsampleratio
        self.padtime = padtime
        self.initstep = timeaxis[1] - timeaxis[0]
        self.initstart = timeaxis[0]
        self.initend = timeaxis[-1]
        self.hiresstep = self.initstep / np.float64(self.upsampleratio)
        self.hires_x = np.arange(
            timeaxis[0] - self.padtime,
            self.initstep * len(timeaxis) + self.padtime,
            self.hiresstep,
        )
        self.hiresstart = self.hires_x[0]
        self.hiresend = self.hires_x[-1]
        self.method = method
        if self.method == "poly":
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[
                int(self.padtime // self.hiresstep)
                + 1 : -(int(self.padtime // self.hiresstep) + 1)
            ] = signal.resample_poly(timecourse, int(self.upsampleratio * 10), 10)
        elif self.method == "fourier":
            self.hires_y = 0.0 * self.hires_x
            self.hires_y[
                int(self.padtime // self.hiresstep)
                + 1 : -(int(self.padtime // self.hiresstep) + 1)
            ] = signal.resample(timecourse, self.upsampleratio * len(timeaxis))
        else:
            self.hires_y = doresample(timeaxis, timecourse, self.hires_x, method=method)
        self.hires_y[: int(self.padtime // self.hiresstep)] = self.hires_y[
            int(self.padtime // self.hiresstep)
        ]
        self.hires_y[-int(self.padtime // self.hiresstep) :] = self.hires_y[
            -int(self.padtime // self.hiresstep)
        ]
        if debug:
            print("FastResampler __init__:")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)

        # self.hires_y[:int(self.padtime // self.hiresstep)] = 0.0
        # self.hires_y[-int(self.padtime // self.hiresstep):] = 0.0
        if doplot:
            import matplolib.pyplot as pl

            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title("FastResampler initial timecourses")
            pl.plot(timeaxis, timecourse, self.hires_x, self.hires_y)
            pl.legend(("input", "hires"))
            pl.show()

    def info(self, prefix=""):
        print(f"{prefix}{self.timeaxis=}")
        print(f"{prefix}{self.timecourse=}")
        print(f"{prefix}{self.upsampleratio=}")
        print(f"{prefix}{self.padtime=}")
        print(f"{prefix}{self.initstep=}")
        print(f"{prefix}{self.initstart=}")
        print(f"{prefix}{self.initend=}")
        print(f"{prefix}{self.hiresstep=}")
        print(f"{prefix}{self.hires_x[0]=}")
        print(f"{prefix}{self.hires_x[-1]=}")
        print(f"{prefix}{self.hiresstart=}")
        print(f"{prefix}{self.hiresend=}")
        print(f"{prefix}{self.method=}")
        print(f"{prefix}{self.hires_y[0]=}")
        print(f"{prefix}{self.hires_y[-1]=}")

    def save(self, outputname):
        dd_io.writebidstsv(
            outputname,
            self.timecourse,
            1.0 / self.initstep,
            starttime=self.initstart,
            columns=["timecourse"],
            extraheaderinfo={"Description": "The lagged timecourse generator"},
            append=False,
        )

    def yfromx(self, newtimeaxis, doplot=False, debug=False):
        if debug:
            print("FastResampler: yfromx called with following parameters")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)
            print("    requested axis limits:", newtimeaxis[0], newtimeaxis[-1])
        outindices = ((newtimeaxis - self.hiresstart) // self.hiresstep).astype(int)
        if debug:
            print("len(self.hires_y):", len(self.hires_y))
        try:
            out_y = self.hires_y[outindices]
        except IndexError:
            print("")
            print("indexing out of bounds in FastResampler")
            print("    padtime:, ", self.padtime)
            print("    initstep, hiresstep:", self.initstep, self.hiresstep)
            print("    initial axis limits:", self.initstart, self.initend)
            print("    hires axis limits:", self.hiresstart, self.hiresend)
            print("    requested axis limits:", newtimeaxis[0], newtimeaxis[-1])
            sys.exit()
        if doplot:
            fig = pl.figure()
            ax = fig.add_subplot(111)
            ax.set_title("FastResampler timecourses")
            pl.plot(self.hires_x, self.hires_y, newtimeaxis, out_y)
            pl.legend(("hires", "output"))
            pl.show()
        return out_y


def FastResamplerFromFile(inputname, colspec=None, debug=False, **kwargs):
    (
        insamplerate,
        instarttime,
        incolumns,
        indata,
        incompressed,
        incolsource,
    ) = dd_io.readbidstsv(inputname, colspec=colspec, debug=debug)
    if len(incolumns) > 1:
        raise ValueError("Multiple columns in input file")
    intimecourse = indata[0, :]
    intimeaxis = np.linspace(
        instarttime,
        instarttime + len(intimecourse) / insamplerate,
        len(intimecourse),
        endpoint=False,
    )
    if debug:
        print(f"FastResamplerFromFile: {len(intimeaxis)=}, {intimecourse.shape=}")
    return FastResampler(intimeaxis, intimecourse, **kwargs)


def doresample(
    orig_x,
    orig_y,
    new_x,
    method="cubic",
    padlen=0,
    padtype="reflect",
    antialias=False,
    debug=False,
):
    """
    Resample data from one spacing to another.  By default, does not apply any antialiasing filter.

    Parameters
    ----------
    orig_x
    orig_y
    new_x
    method
    padlen

    Returns
    -------

    """
    tstep = orig_x[1] - orig_x[0]
    if padlen > 0:
        rawxpad = np.linspace(0.0, padlen * tstep, num=padlen, endpoint=False)
        frontpad = rawxpad + orig_x[0] - padlen * tstep
        backpad = rawxpad + orig_x[-1] + tstep
        pad_x = np.concatenate((frontpad, orig_x, backpad))
        pad_y = dd_filt.padvec(orig_y, padlen=padlen, padtype=padtype)
    else:
        pad_x = orig_x
        pad_y = orig_y

    if debug:
        print("padlen=", padlen)
        print("tstep=", tstep)
        print("lens:", len(pad_x), len(pad_y))
        print(pad_x)
        print(pad_y)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Original and padded vector")
        pl.plot(orig_x, orig_y + 1.0, pad_x, pad_y)
        pl.show()

    # antialias and ringstop filter
    init_freq = len(pad_x) / (pad_x[-1] - pad_x[0])
    final_freq = len(new_x) / (new_x[-1] - new_x[0])
    if antialias and (init_freq > final_freq):
        aafilterfreq = final_freq / 2.0
        aafilter = dd_filt.NoncausalFilter(filtertype="arb", transferfunc="trapezoidal")
        aafilter.setfreqs(0.0, 0.0, 0.95 * aafilterfreq, aafilterfreq)
        pad_y = aafilter.apply(init_freq, pad_y)

    if method == "cubic":
        cj = signal.cspline1d(pad_y)
        # return dd_filt.unpadvec(
        #   np.float64(signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
        return signal.cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == "quadratic":
        qj = signal.qspline1d(pad_y)
        # return dd_filt.unpadvec(
        #    np.float64(signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])), padlen=padlen)
        return signal.qspline1d_eval(qj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])
    elif method == "univariate":
        interpolator = sp.interpolate.UnivariateSpline(pad_x, pad_y, k=3, s=0)  # s=0 interpolates
        # return dd_filt.unpadvec(np.float64(interpolator(new_x)), padlen=padlen)
        return np.float64(interpolator(new_x))
    else:
        print("invalid interpolation method")
        return None
