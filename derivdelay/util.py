#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
import bisect
import logging
import os
import platform
import resource
import site
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import derivdelay._version as dd_versioneer
import derivdelay.io as dd_io

LGR = logging.getLogger(__name__)
TimingLGR = logging.getLogger("TIMING")
MemoryLGR = logging.getLogger("MEMORY")


# ---------------------------------------- Global constants -------------------------------------------
defaultbutterorder = 6
MAXLINES = 10000000
donotbeaggressive = True

# ----------------------------------------- Conditional imports ---------------------------------------
try:
    from numba import jit
except ImportError:
    donotusenumba = True
else:
    donotusenumba = False

try:
    import pyfftw
except ImportError:
    pyfftwpresent = False
else:
    pyfftwpresent = True


def checkimports(optiondict):
    if pyfftwpresent:
        print("pfftw exists")
    else:
        print("pfftw does not exist")
    optiondict["pfftwexists"] = pyfftwpresent

    if donotbeaggressive:
        print("no aggressive optimization")
    else:
        print("aggressive optimization")
    optiondict["donotbeaggressive"] = donotbeaggressive

    global donotusenumba
    if donotusenumba:
        print("will not use numba even if present")
    else:
        print("using numba if present")
    optiondict["donotusenumba"] = donotusenumba


# ----------------------------------------- Conditional jit handling ----------------------------------
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


# --------------------------- Utility functions -------------------------------------------------
def findavailablemem():
    if os.path.isfile("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as limit:
            mem = int(limit.read())
            return mem, mem
    else:
        retdata = subprocess.run(["free", "-m"], capture_output=True).stdout.decode().split("\n")
        free = int((retdata[1].split())[3]) * 1024 * 1024
        swap = int((retdata[2].split())[3]) * 1024 * 1024
        return free, swap


def checkifincontainer():
    # Determine if the program is running in a container.  If so, we may need to adjust the python memory
    # limits because they are not set properly.  But check if we're running on CircleCI - it does not seem
    # to like you twiddling with the container parameters.
    #
    # possible return values are: None, "Docker", "Singularity", and "CircleCI"
    #
    if os.environ.get("SINGULARITY_CONTAINER") is not None:
        containertype = "Singularity"
    elif os.environ.get("RUNNING_IN_CONTAINER") is not None:
        containertype = "Docker"
    else:
        containertype = None
    if os.environ.get("CIRCLECI") is not None:
        containertype = "CircleCI"
    return containertype


def setmemlimit(memlimit):
    resource.setrlimit(resource.RLIMIT_AS, (memlimit, memlimit))


def formatmemamt(meminbytes):
    units = ["B", "kB", "MB", "GB", "TB"]
    index = 0
    unitnumber = np.uint64(1)
    while True:
        if meminbytes < np.uint64(1024) * unitnumber:
            return f"{round(meminbytes/unitnumber, 3):.3f}{units[index]}"
        unitnumber *= np.uint64(1024)
        index += 1
        if index >= len(units):
            break
    return f"{round(meminbytes/unitnumber, 3):.3f}{units[-1]}"


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "kilo", 2: "mega", 3: "giga", 4: "tera"}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n] + "bytes"


def logmem(msg=None):
    """Log memory usage with a logging object.

    Parameters
    ----------
    msg : str or None, optional
        A message to include in the first column.
        If None, the column headers are logged.
        Default is None.
    """
    global lastmaxrss_parent, lastmaxrss_child
    if platform.system() != "Windows":
        import resource

        if msg is None:
            outvals = [
                "",
                "Self Max RSS",
                "Self Diff RSS",
                "Self Shared Mem",
                "Self Unshared Mem",
                "Self Unshared Stack",
                "Self Non IO Page Fault",
                "Self IO Page Fault",
                "Self Swap Out",
                "Children Max RSS",
                "Children Diff RSS",
                "Children Shared Mem",
                "Children Unshared Mem",
                "Children Unshared Stack",
                "Children Non IO Page Fault",
                "Children IO Page Fault",
                "Children Swap Out",
            ]
            lastmaxrss_parent = 0
            lastmaxrss_child = 0
        else:
            rcusage = resource.getrusage(resource.RUSAGE_SELF)
            outvals = [msg]
            outvals.append(formatmemamt(rcusage.ru_maxrss))
            outvals.append(formatmemamt(rcusage.ru_maxrss - lastmaxrss_parent))
            lastmaxrss_parent = rcusage.ru_maxrss
            outvals.append(formatmemamt(rcusage.ru_ixrss))
            outvals.append(formatmemamt(rcusage.ru_idrss))
            outvals.append(formatmemamt(rcusage.ru_isrss))
            outvals.append(formatmemamt(rcusage.ru_minflt))
            outvals.append(formatmemamt(rcusage.ru_majflt))
            outvals.append(formatmemamt(rcusage.ru_nswap))
            rcusage = resource.getrusage(resource.RUSAGE_CHILDREN)
            outvals.append(formatmemamt(rcusage.ru_maxrss))
            outvals.append(formatmemamt(rcusage.ru_maxrss - lastmaxrss_child))
            lastmaxrss_child = rcusage.ru_maxrss
            outvals.append(formatmemamt(rcusage.ru_ixrss))
            outvals.append(formatmemamt(rcusage.ru_idrss))
            outvals.append(formatmemamt(rcusage.ru_isrss))
            outvals.append(formatmemamt(rcusage.ru_minflt))
            outvals.append(formatmemamt(rcusage.ru_majflt))
            outvals.append(formatmemamt(rcusage.ru_nswap))
    else:
        outvals = ["Not available on Windows"]

    MemoryLGR.info("\t".join(outvals))


def findexecutable(command):
    """

    Parameters
    ----------
    command

    Returns
    -------

    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        return shutil.which(command)
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            if os.access(os.path.join(path, command), os.X_OK):
                return os.path.join(path, command)
        return None


def isexecutable(command):
    """

    Parameters
    ----------
    command

    Returns
    -------

    """
    import shutil

    theversion = sys.version_info
    if (theversion[0] >= 3) and (theversion[1] >= 3):
        if shutil.which(command) is not None:
            return True
        else:
            return False
    else:
        return any(
            os.access(os.path.join(path, command), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )


def makeadir(pathname):
    try:
        os.makedirs(pathname)
    except OSError:
        if os.path.exists(pathname):
            # We are nearly safe
            return True
        else:
            # There was an error on creation, so make sure we know about it
            print("ERROR: ", pathname, " does not exist, and could not create it")
            return False
    return True


def findreferencedir():
    # Get the list of directories
    site_packages_dirs = site.getsitepackages()

    # Find the "site-packages" directory in the list
    for dir in site_packages_dirs:
        if dir.endswith("site-packages"):
            sitepackages_dir = dir
            break
        else:
            sitepackages_dir = None
    referencedir = os.path.join(
        sitepackages_dir,
        "derivdelay",
        "data",
        "reference",
    )
    return referencedir


def savecommandline(theargs, thename):
    """

    Parameters
    ----------
    theargs
    thename

    Returns
    -------

    """
    dd_io.writevec([" ".join(theargs)], thename + "_commandline.txt")


def startendcheck(timepoints, startpoint, endpoint):
    """

    Parameters
    ----------
    timepoints
    startpoint
    endpoint

    Returns
    -------

    """
    if startpoint > timepoints - 1:
        print("startpoint is too large (maximum is ", timepoints - 1, ")")
        sys.exit()
    if startpoint < 0:
        realstart = 0
        print("startpoint set to minimum, (0)")
    else:
        realstart = startpoint
        print("startpoint set to ", startpoint)
    if endpoint == -1:
        endpoint = 100000000
    if endpoint > timepoints - 1:
        realend = timepoints - 1
        print("endpoint set to maximum, (", timepoints - 1, ")")
    else:
        realend = endpoint
        print("endpoint set to ", endpoint)
    if realstart >= realend:
        print("endpoint (", realend, ") must be greater than startpoint (", realstart, ")")
        sys.exit()
    return realstart, realend


def valtoindex(
    thearray,
    thevalue,
    evenspacing=True,
    discrete=True,
    discretization="round",
    debug=False,
):
    """

    Parameters
    ----------
    thearray: array-like
        An ordered list of values (does not need to be equally spaced)
    thevalue: float
        The value to search for in the array
    evenspacing: boolean, optional
        If True (default), assume data is evenly spaced for faster calculation.
    discrete: boolean, optional
        If True make the index an integer (round by default).
    discretization: string, optional
        Select rounding method - floor, ceiling, or round(default)

    Returns
    -------
    closestidx: int
        The index of the sample in thearray that is closest to val

    """
    if evenspacing:
        limval = np.max([thearray[0], np.min([thearray[-1], thevalue])])
        position = (limval - thearray[0]) / (thearray[1] - thearray[0])
        if debug:
            print("valtoindex:")
            print("\tthevalue:", thevalue)
            print("\tarraymin:", thearray[0])
            print("\tarraymax:", thearray[-1])
            print("\tlimval:", limval)
            print(
                "\tindex:",
                int(np.round((limval - thearray[0]) / (thearray[1] - thearray[0]), 0)),
            )
        if discrete:
            if discretization == "round":
                position = int(np.round(position, 0))
            elif discretization == "floor":
                position = int(np.floor(position))
            elif discretization == "ceiling":
                position = int(np.ceil(position))
            else:
                print("valtoindex - illegal discretization mode")
                position = None
            position = int(np.min([len(thearray) - 1, np.max([0, position])]))
        return position
    else:
        return int((np.abs(thearray - thevalue)).argmin())


def progressbar(thisval, end_val, label="Percent", barsize=60):
    """

    Parameters
    ----------
    thisval
    end_val
    label
    barsize

    Returns
    -------

    """
    percent = float(thisval) / end_val
    hashes = "#" * int(round(percent * barsize))
    spaces = " " * (barsize - len(hashes))
    sys.stdout.write("\r{0}: [{1}] {2:.2f}%".format(label, hashes + spaces, 100.0 * percent))
    sys.stdout.flush()


def makelaglist(lagstart, lagend, lagstep):
    """

    Parameters
    ----------
    lagstart
    lagend
    lagstep

    Returns
    -------

    """
    numsteps = int((lagend - lagstart) // lagstep + 1)
    lagend = lagstart + lagstep * (numsteps - 1)
    print(
        "creating list of ",
        numsteps,
        " lag steps (",
        lagstart,
        " to ",
        lagend,
        " in steps of ",
        lagstep,
        ")",
    )
    # thelags = np.r_[0.0:1.0 * numsteps] * lagstep + lagstart
    thelags = np.arange(0.0, 1.0 * numsteps) * lagstep + lagstart
    return thelags


# ------------------------------------------ Version function ----------------------------------
def version():
    """

    Returns
    -------

    """
    try:
        dummy = os.environ["RUNNING_IN_CONTAINER"]
    except KeyError:
        iscontainer = False
    else:
        iscontainer = True

    if iscontainer:
        try:
            theversion = os.environ["GITVERSION"]
            if theversion.find("+") < 0:
                theverion = theversion.split(".")[0]
        except KeyError:
            theversion = "UNKNOWN"
        try:
            thedirectversion = os.environ["GITDIRECTVERSION"]
            directversionparts = thedirectversion.split("-")
            if len(directversionparts) == 3:
                thedirectversion = (
                    directversionparts[0]
                    + "."
                    + directversionparts[1]
                    + "+"
                    + directversionparts[2]
                )
                isdirty = True
            elif len(directversionparts) == 2:
                thedirectversion = directversionparts[0] + "." + directversionparts[1]
                isdirty = True
            elif len(directversionparts) == 1:
                thedirectversion = directversionparts[0]
                isdirty = False
            else:
                pass
        except KeyError:
            thedirectversion = "UNKNOWN"
        try:
            thesha = os.environ["GITSHA"]
        except KeyError:
            thesha = "UNKNOWN"
        try:
            thedate = os.environ["GITDATE"]
        except KeyError:
            thedate = "UNKNOWN"
        if thedirectversion != "UNKNOWN":
            theversion = thedirectversion
    else:
        try:
            versioninfo = dd_versioneer.get_versions()
        except:
            return "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
        isdirty = versioninfo["dirty"]
        if isdirty is None:
            isdirty = "UNKNOWN"
        theversion = versioninfo["version"]
        if theversion is None:
            theversion = "UNKNOWN"
        else:
            splitversion = theversion.split("+")
            if len(splitversion) > 1:
                resplit = splitversion[1].split(".")
                if len(resplit) == 3:
                    if resplit[0] == "0":
                        theversion = splitversion[0]
        thesha = versioninfo["full-revisionid"]
        if thesha is None:
            thesha = "UNKNOWN"
        thedate = versioninfo["date"]
        if thedate is None:
            thedate = "UNKNOWN"

    return theversion, thesha, thedate, isdirty


# --------------------------- timing functions -------------------------------------------------
def timefmt(thenumber):
    """

    Parameters
    ----------
    thenumber

    Returns
    -------
    outputlines:
        The formatted lines to save to the formatted runtimings file
    totaldiff:
        The total time from start to finish, in seconds

    """
    return "{:10.2f}".format(thenumber)


def proctiminglogfile(logfilename, timewidth=10):
    timingdata = pd.read_csv(
        logfilename,
        sep=None,
        header=None,
        names=["time", "description", "number", "units"],
        engine="python",
    )
    starttime = datetime.strptime(timingdata["time"].iloc[0], "%Y%m%dT%H%M%S.%f")
    outputlines = [f"{'Total (s)'.rjust(timewidth)}\t{'Diff. (s)'.rjust(timewidth)}\tDescription"]
    outputlines += [
        f"{'0.0'.rjust(timewidth)}\t{'0.0'.rjust(timewidth)}\t{timingdata['description'].iloc[0]}"
    ]
    for therow in range(1, timingdata.shape[0]):
        thistime = datetime.strptime(timingdata["time"].iloc[therow], "%Y%m%dT%H%M%S.%f")
        prevtime = datetime.strptime(timingdata["time"].iloc[therow - 1], "%Y%m%dT%H%M%S.%f")
        totaldiff = (thistime - starttime).total_seconds()
        incdiff = (thistime - prevtime).total_seconds()
        totaldiffstr = f"{totaldiff:.2f}".rjust(timewidth)
        incdiffstr = f"{incdiff:.2f}".rjust(timewidth)
        theoutputline = f"{totaldiffstr}\t{incdiffstr}\t{timingdata['description'].iloc[therow]}"
        try:
            dummy = np.isnan(timingdata["number"].iloc[therow])
        except:
            pass
        else:
            if not np.isnan(timingdata["number"].iloc[therow]):
                speedunit = f"{timingdata['units'].iloc[therow]}/s"
                if incdiff == 0.0:
                    speed = "undefined"
                else:
                    speed = f"{float(timingdata['number'].iloc[therow]) / incdiff:.2f}"
                theoutputline += f" ({timingdata['number'].iloc[therow]} {timingdata['units'].iloc[therow]} @ {speed} {speedunit})"
        outputlines += [theoutputline]

    return outputlines, totaldiff


def proctiminginfo(thetimings, outputfile="", extraheader=None):
    """

    Parameters
    ----------
    thetimings
    outputfile
    extraheader

    Returns
    -------

    """
    theinfolist = []
    start = thetimings[0]
    starttime = float(start[1])
    lasteventtime = starttime
    if extraheader is not None:
        print(extraheader)
        theinfolist.append(extraheader)
    headerstring = "Clock time\tProgram time\tDuration\tDescription"
    print(headerstring)
    theinfolist.append(headerstring)
    for theevent in thetimings:
        theduration = float(theevent[1] - lasteventtime)
        outstring = (
            time.strftime("%Y%m%dT%H%M%S", time.localtime(theevent[1]))
            + timefmt(float(theevent[1]) - starttime)
            + "\t"
            + timefmt(theduration)
            + "\t"
            + theevent[0]
        )
        if theevent[2] is not None:
            outstring += " ({0:.2f} {1}/second)".format(
                float(theevent[2]) / theduration, theevent[3]
            )
        print(outstring)
        theinfolist.append(outstring)
        lasteventtime = float(theevent[1])
    if outputfile != "":
        dd_io.writevec(theinfolist, outputfile)


# timecourse functions
def maketcfrom3col(inputdata, timeaxis, outputvector, debug=False):
    theshape = np.shape(inputdata)
    for idx in range(0, theshape[1]):
        starttime = inputdata[0, idx]
        endtime = starttime + inputdata[1, idx]
        if (starttime <= timeaxis[-1]) and (endtime >= 0.0) and (endtime > starttime):
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(outputvector)))
            outputvector[startindex:endindex] = inputdata[2, idx]
            print(starttime, startindex, endtime, endindex)
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("temporal output vector")
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


def maketcfrom2col(inputdata, timeaxis, outputvector, debug=False):
    theshape = np.shape(inputdata)
    rangestart = int(inputdata[0, 0])
    for i in range(1, theshape[1]):
        if rangestart < len(outputvector) - 1:
            rangeend = int(np.min([inputdata[0, i], len(outputvector)]))
            if rangeend > rangestart:
                theval = inputdata[1, i - 1]
                if debug:
                    print(f"{i}: setting outputvector[{rangestart}:{rangeend}] to {theval}")
                outputvector[rangestart:rangeend] = theval
                rangestart = rangeend
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("temporal output vector")
        plt.plot(timeaxis, outputvector)
        plt.show()
    return outputvector


# --------------------------- simulation functions ----------------------------------------------
def makeslicetimes(numslices, sliceordertype, tr=1.0, multibandfac=1, debug=False):
    outlist = np.zeros((numslices), dtype=np.float)
    if (numslices % multibandfac) != 0:
        print("ERROR: numslices is not evenly divisible by multband factor")
        return None
    mbcycle = int(numslices / multibandfac)
    normal = True
    if sliceordertype == "ascending":
        controllist = [[0, 1]]
    elif sliceordertype == "descending":
        controllist = [[mbcycle - 1, -1]]
    elif sliceordertype == "ascending_interleaved":
        controllist = [[0, 2], [1, 2]]
    elif sliceordertype == "descending_interleaved":
        controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
    elif sliceordertype == "ascending_sparkplug":
        normal = False
        controllist = [[0, int(mbcycle // 2) - 1]]
    elif sliceordertype == "descending_sparkplug":
        normal = False
        controllist = [[mbcycle - 1, -int(mbcycle // 2) - 1]]
    elif sliceordertype == "ascending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[0, 2], [1, 2]]
        else:
            controllist = [[1, 2], [0, 2]]
    elif sliceordertype == "descending_interleaved_siemens":
        if numslices % 2 == 0:
            controllist = [[mbcycle - 1, -2], [mbcycle - 2, -2]]
        else:
            controllist = [[mbcycle - 2, -2], [mbcycle - 1, -2]]
    elif sliceordertype == "ascending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([i, numgroups])
    elif sliceordertype == "descending_interleaved_philips":
        controllist = []
        numgroups = int(np.floor(np.sqrt(numslices)))
        for i in range(numgroups):
            controllist.append([mbcycle - i - 1, -numgroups])
    else:
        print("ERROR: illegal sliceordertype")
        return None

    # now make the slicetimes
    timelist = np.linspace(0, tr, num=mbcycle, endpoint=False)
    slicelist = []
    if debug:
        print("sliceordertype:", sliceordertype)
        print("number of mbcycles:", numslices // mbcycle)
        print("size of mbcycles:", mbcycle)
    for thecontrollist in controllist:
        start = thecontrollist[0]
        step = thecontrollist[1]
        theindex = start
        if normal:
            while 0 <= theindex < mbcycle:
                slicelist.append(theindex)
                theindex += step
        else:
            while len(slicelist) < mbcycle:
                slicelist.append(theindex)
                theindex = (theindex + step) % mbcycle

    if debug:
        print(slicelist)
    for index in range(numslices):
        posinmbcycle = index % mbcycle
        outlist[index] = timelist[slicelist[posinmbcycle]] + 0.0
    return outlist


# --------------------------- testing functions -------------------------------------------------
def comparemap(map1, map2, mask=None, debug=False):
    ndims = len(map1.shape)
    if debug:
        print("map has", ndims, "axes")
    if map1.shape != map2.shape:
        print("comparemap: maps do not have the same shape - aborting")
        sys.exit()
    if ndims == 1:
        if debug:
            print("dealing with ndims == 1 case")
        map1valid = map1
        map2valid = map2
    else:
        if mask is None:
            map1valid = map1
            map2valid = map2
        else:
            if debug:
                print("mask is not None")
            ndims_mask = len(mask.shape)
            if debug:
                print("mask has", ndims_mask, "axes")
            if ndims_mask == ndims:
                if debug:
                    print("dealing with ndims == ndims_mask case")
                if map1.shape != mask.shape:
                    print("comparemap: mask does not have the same shape as the maps - aborting")
                    sys.exit()
                validvoxels = np.where(mask > 0)[0]
                map1valid = map1[validvoxels, :]
                map2valid = map2[validvoxels, :]
            elif ndims_mask == ndims - 1:
                # need to make expanded mask
                if debug:
                    print("dealing with ndims == ndims_mask + 1 case")
                    print("shape of map:", map1.shape)
                    print("shape of mask:", mask.shape)
                numvox = 1
                for i in range(ndims - 1):
                    numvox *= mask.shape[i]
                reshapemask = mask.reshape(numvox)
                reshapemap1 = map1.reshape(numvox, -1)
                reshapemap2 = map2.reshape(numvox, -1)
                validvoxels = np.where(reshapemask > 0)[0]
                map1valid = reshapemap1[validvoxels, :]
                map2valid = reshapemap2[validvoxels, :]
            else:
                print("mask is not compatible with map")
                sys.exit()

    # at this point, map2valid and map1valid are the same dimensions
    diff = map2valid - map1valid
    reldiff = np.where(map1valid != 0.0, diff / map1valid, 0.0)
    maxdiff = np.max(diff)
    mindiff = np.min(diff)
    meandiff = np.mean(diff)
    mse = np.mean(np.square(diff))

    maxreldiff = np.max(reldiff)
    minreldiff = np.min(reldiff)
    meanreldiff = np.mean(reldiff)
    relmse = np.mean(np.square(reldiff))

    return mindiff, maxdiff, meandiff, mse, minreldiff, maxreldiff, meanreldiff, relmse


# shared memory routines
def numpy2shared(inarray, theouttype, name=None):
    # Create a shared memory block to store the array data
    outnbytes = np.dtype(theouttype).itemsize * inarray.size
    shm = shared_memory.SharedMemory(name=None, create=True, size=outnbytes)
    shm.unlink()
    inarray_shared = np.ndarray(inarray.shape, dtype=theouttype, buffer=shm.buf)
    np.copyto(inarray_shared, inarray)  # Copy data to shared memory array
    return inarray_shared, shm  # Return both the array and the shared memory object


def allocshared(theshape, thetype, name=None):
    # Calculate size based on shape
    thesize = np.prod(theshape)
    # Determine the data type size
    dtype_size = np.dtype(thetype).itemsize
    # Create a shared memory block of the required size
    shm = shared_memory.SharedMemory(name=None, create=True, size=thesize * dtype_size)
    shm.unlink()
    outarray = np.ndarray(theshape, dtype=thetype, buffer=shm.buf)
    return outarray, shm  # Return both the array and the shared memory object


def cleanup_shm(shm):
    # Cleanup
    pass
    # if shm is not None:
    #    shm.close()
    #    shm.unlink()
