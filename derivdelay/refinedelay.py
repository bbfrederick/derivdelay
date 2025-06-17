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
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import median_filter
from scipy.special import factorial
from statsmodels.robust import mad

import derivdelay.filter as dd_filt
import derivdelay.fit as dd_fit
import derivdelay.io as dd_io

global ratiotooffsetfunc, maplimits


def makevoxelspecificderivs(theevs, nderivs=1, debug=False):
    r"""Perform multicomponent expansion on theevs (each ev replaced by itself,
    its square, its cube, etc.).

    Parameters
    ----------
    theevs : 2D numpy array
        NxP array of voxel specific explanatory variables (one timecourse per voxel)
        :param theevs:

    nderivs : integer
        Number of components to use for each ev.  Each successive component is a
        higher power of the initial ev (initial, square, cube, etc.)
        :param nderivs:

    debug: bool
        Flag to toggle debugging output
        :param debug:
    """
    if debug:
        print(f"{theevs.shape=}")
    if nderivs == 0:
        thenewevs = theevs
    else:
        taylorcoffs = np.zeros((nderivs + 1), dtype=np.float64)
        taylorcoffs[0] = 1.0
        thenewevs = np.zeros((theevs.shape[0], theevs.shape[1], nderivs + 1), dtype=float)
        for i in range(1, nderivs + 1):
            taylorcoffs[i] = 1.0 / factorial(i)
        for thevoxel in range(0, theevs.shape[0]):
            thenewevs[thevoxel, :, 0] = theevs[thevoxel, :] * 1.0
            for i in range(1, nderivs + 1):
                thenewevs[thevoxel, :, i] = taylorcoffs[i] * np.gradient(
                    thenewevs[thevoxel, :, i - 1]
                )
    if debug:
        print(f"{nderivs=}")
        print(f"{thenewevs.shape=}")

    return thenewevs


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def trainratiotooffset(
    lagtcgenerator,
    timeaxis,
    outputname,
    outputlevel,
    mindelay=-3.0,
    maxdelay=3.0,
    numpoints=501,
    smoothpts=3,
    edgepad=5,
    numderivs=1,
    debug=False,
):
    global ratiotooffsetfunc, maplimits

    if debug:
        print("ratiotooffsetfunc:")
        lagtcgenerator.info(prefix="\t")
        print("\ttimeaxis:", timeaxis)
        print("\toutputname:", outputname)
        print("\tmindelay:", mindelay)
        print("\tmaxdelay:", maxdelay)
        print("\tsmoothpts:", smoothpts)
        print("\tedgepad:", edgepad)
        print("\tnumderivs:", numderivs)
        print("\tlagtcgenerator:", lagtcgenerator)
    # make a delay map
    delaystep = (maxdelay - mindelay) / (numpoints - 1)
    if debug:
        print(f"{delaystep=}")
        print(f"{mindelay=}")
        print(f"{maxdelay=}")
    lagtimes = np.linspace(
        mindelay - edgepad * delaystep,
        maxdelay + edgepad * delaystep,
        numpoints + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{mindelay=}")
        print(f"{maxdelay=}")
        print("lagtimes=", lagtimes)

    # now make synthetic fMRI data
    internalvalidfmrishape = (numpoints + 2 * edgepad, timeaxis.shape[0])
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numpoints + 2 * edgepad, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    for i in range(numpoints + 2 * edgepad):
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i])

    rt_floattype = "float64"
    linfitmean = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    rvalue = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    r2value = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    fitNorm = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    fitcoeff = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    sampletime = timeaxis[1] - timeaxis[0]
    optiondict = {
        "linfitthreshval": 0.0,
        "saveminimumlinfitfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_linfit": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "memprofile": False,
        "focaldebug": debug,
        "fmrifreq": 1.0 / sampletime,
        "textio": False,
    }

    derivcoffratios = getderivratios(
        fmridata,
        rvalue,
        r2value,
        fitcoeff[:, :2],
        timeaxis,
        lagtcgenerator,
        numderivs=numderivs,
        debug=debug,
    )
    if debug:
        print("before trimming")
        print(f"{derivcoffratios.shape=}")
        print(f"{lagtimes.shape=}")
    if numderivs == 1:
        smoothderivcoffratios = dd_filt.unpadvec(
            smooth(dd_filt.padvec(derivcoffratios, padlen=20, padtype="constant"), smoothpts),
            padlen=20,
        )
        derivcoffratios = derivcoffratios[edgepad:-edgepad]
        smoothderivcoffratios = smoothderivcoffratios[edgepad:-edgepad]
    else:
        smoothderivcoffratios = np.zeros_like(derivcoffratios)
        for i in range(numderivs):
            smoothderivcoffratios[i, :] = dd_filt.unpadvec(
                smooth(
                    dd_filt.padvec(derivcoffratios[i, :], padlen=20, padtype="constant"),
                    smoothpts,
                ),
                padlen=20,
            )
        derivcoffratios = derivcoffratios[:, edgepad:-edgepad]
        smoothderivcoffratios = smoothderivcoffratios[:, edgepad:-edgepad]
    lagtimes = lagtimes[edgepad:-edgepad]
    if debug:
        print("after trimming")
        print(f"{derivcoffratios.shape=}")
        print(f"{smoothderivcoffratios.shape=}")
        print(f"{lagtimes.shape=}")

    # make sure the mapping function is legal
    xaxis = smoothderivcoffratios[::-1]
    yaxis = lagtimes[::-1]
    midpoint = int(len(xaxis) // 2)
    lowerlim = midpoint + 0
    while (lowerlim > 1) and xaxis[lowerlim] > xaxis[lowerlim - 1]:
        lowerlim -= 1
    upperlim = midpoint + 0
    while (upperlim < len(xaxis) - 2) and xaxis[upperlim] < xaxis[upperlim + 1]:
        upperlim += 1
    xaxis = xaxis[lowerlim : upperlim + 1]
    yaxis = yaxis[lowerlim : upperlim + 1]
    ratiotooffsetfunc = CubicSpline(xaxis, yaxis)
    maplimits = (xaxis[0], xaxis[-1])

    if outputlevel != "min":
        resampaxis = np.linspace(xaxis[0], xaxis[-1], num=len(xaxis), endpoint=True)
        dd_io.writebidstsv(
            f"{outputname}_desc-ratiotodelayfunc_timeseries",
            ratiotooffsetfunc(resampaxis),
            1.0 / (resampaxis[1] - resampaxis[0]),
            starttime=resampaxis[0],
            columns=["delay"],
            extraheaderinfo={"Description": "The function mapping derivative ratio to delay"},
            append=False,
        )


def ratiotodelay(theratio):
    global ratiotooffsetfunc, maplimits
    if theratio < maplimits[0]:
        return ratiotooffsetfunc(maplimits[0])
    elif theratio > maplimits[1]:
        return ratiotooffsetfunc(maplimits[1])
    else:
        return ratiotooffsetfunc(theratio)


def coffstodelay(thecoffs, mindelay=-3.0, maxdelay=3.0, debug=False):
    justaone = np.array([1.0], dtype=thecoffs.dtype)
    allcoffs = np.concatenate((justaone, thecoffs))
    theroots = (poly.Polynomial(allcoffs, domain=(mindelay, maxdelay))).roots()
    if theroots is None:
        return 0.0
    elif len(theroots) == 1:
        return theroots[0].real
    else:
        candidates = []
        for i in range(len(theroots)):
            if np.isreal(theroots[i]) and (mindelay <= theroots[i] <= maxdelay):
                if debug:
                    print(f"keeping root {i} ({theroots[i]})")
                candidates.append(theroots[i].real)
            else:
                if debug:
                    print(f"discarding root {i} ({theroots[i]})")
                else:
                    pass
        if len(candidates) > 0:
            chosen = candidates[np.argmin(np.fabs(np.array(candidates)))].real
            if debug:
                print(f"{theroots=}, {candidates=}, {chosen=}")
            return chosen
        return 0.0


def fitOneTimecourse(theevs, thedata, rt_floatset=np.float64, rt_floattype="float64"):
    # NOTE: if theevs is 2D, dimension 0 is number of points, dimension 1 is number of evs
    thefit, R2 = dd_fit.mlregress(theevs, thedata)
    if theevs.ndim > 1:
        if thefit is None:
            thefit = np.matrix(np.zeros((1, theevs.shape[1] + 1), dtype=rt_floattype))
        fitcoeffs = rt_floatset(thefit[0, 1:])
        if fitcoeffs[0, 0] < 0.0:
            coeffsign = -1.0
        else:
            coeffsign = 1.0
        if np.any(fitcoeffs) != 0.0:
            pass
        else:
            R2 = 0.0
        return (
            rt_floatset(coeffsign * np.sqrt(R2)),
            rt_floatset(R2),
            fitcoeffs,
        )
    else:
        fitcoeff = rt_floatset(thefit[0, 1])
        if fitcoeff < 0.0:
            coeffsign = -1.0
        else:
            coeffsign = 1.0
        if fitcoeff == 0.0:
            R2 = 0.0
        return (
            rt_floatset(coeffsign * np.sqrt(R2)),
            rt_floatset(R2),
            fitcoeff,
        )


def getderivratios(
    inputdata,
    rvalue,
    r2value,
    fitcoeff,
    initial_fmri_x,
    genlagtc,
    numderivs=1,
    debug=False,
):
    if numderivs > 0:
        if debug:
            print(f"adding derivatives up to order {numderivs} prior to regression")
        baseev = genlagtc.yfromx(initial_fmri_x)
        evset = makevoxelspecificderivs(baseev.reshape((1, -1)), numderivs).reshape((-1, 2))
    else:
        if debug:
            print(f"using raw lagged regressors for regression")
        evset = genlagtc.yfromx(initial_fmri_x)

    for vox in range(fitcoeff.shape[0]):
        (
            rvalue[vox],
            r2value[vox],
            fitcoeff[vox, :],
        ) = fitOneTimecourse(evset, inputdata[vox, :])

    # calculate the ratio of the first derivative to the main regressor
    if numderivs == 1:
        derivcoffratios = np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0])
    else:
        numvoxels = fitcoeff.shape[0]
        derivcoffratios = np.zeros((numderivs, numvoxels), dtype=np.float64)
        for i in range(numderivs):
            derivcoffratios[i, :] = np.nan_to_num(fitcoeff[:, i + 1] / fitcoeff[:, 0])

    return derivcoffratios


def filterderivratios(
    derivcoffratios,
    nativespaceshape,
    validvoxels,
    thedims,
    patchthresh=3.0,
    gausssigma=0,
    fileiscifti=False,
    textio=False,
    rt_floattype="float64",
    debug=False,
):

    if debug:
        print("filterderivratios:")
        print(f"\t{patchthresh=}")
        print(f"\t{validvoxels.shape=}")
        print(f"\t{nativespaceshape=}")

    # filter the ratio to find weird values
    themad = mad(derivcoffratios).astype(np.float64)
    print(f"MAD of GLM derivative ratios = {themad}")
    outmaparray, internalspaceshape = dd_io.makedestarray(
        nativespaceshape,
        filetype="nifti",
        rt_floattype=rt_floattype,
    )
    mappedderivcoffratios = dd_io.populatemap(
        derivcoffratios,
        internalspaceshape,
        validvoxels,
        outmaparray,
        debug=debug,
    )
    if textio or fileiscifti:
        medfilt = derivcoffratios
        filteredarray = derivcoffratios
    else:
        if debug:
            print(f"{derivcoffratios.shape=}, {mappedderivcoffratios.shape=}")
        medfilt = median_filter(
            mappedderivcoffratios.reshape(nativespaceshape), size=(3, 3, 3)
        ).reshape(internalspaceshape)[validvoxels]
        filteredarray = np.where(
            np.fabs(derivcoffratios - medfilt) > patchthresh * themad, medfilt, derivcoffratios
        )
        if gausssigma > 0:
            mappedfilteredarray = dd_io.populatemap(
                filteredarray,
                internalspaceshape,
                validvoxels,
                outmaparray,
                debug=debug,
            )
            filteredarray = dd_filt.ssmooth(
                thedims[0],
                thedims[1],
                thedims[2],
                gausssigma,
                mappedfilteredarray.reshape(nativespaceshape),
            ).reshape(internalspaceshape)[validvoxels]

    return medfilt, filteredarray, themad
