#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2018-2024 Blaise Frederick
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
"""
Functions for parsers.
"""
import argparse
import os.path as op
import sys
from argparse import Namespace

import derivdelay.filter as dd_filt
import derivdelay.io as dd_io
import derivdelay.util as dd_util


class IndicateSpecifiedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, self.dest + "_nondefault", True)


def detailedversion():
    (
        release_version,
        git_sha,
        git_date,
        git_isdirty,
    ) = dd_util.version()
    python_version = str(sys.version_info)
    print(f"release version: {release_version}")
    print(f"git_sha: {git_sha}")
    print(f"git_date: {git_date}")
    print(f"git_isdirty: {git_isdirty}")
    print(f"python_version: {python_version}")
    sys.exit()


def setifnotset(thedict, thekey, theval):
    if (thekey + "_nondefault") not in thedict.keys():
        print("overriding " + thekey)
        thedict[thekey] = theval


def is_valid_file(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is not None:
        thefilename, colspec = dd_io.parsefilespec(arg)
    else:
        thefilename = None

    if not op.isfile(thefilename) and thefilename is not None:
        parser.error("The file {0} does not exist!".format(thefilename))

    return arg


def invert_float(parser, arg):
    """
    Check if argument is float or auto.
    """
    arg = is_float(parser, arg)

    if arg != "auto":
        arg = 1.0 / arg
    return arg


def is_float(parser, arg, minval=None, maxval=None):
    """
    Check if argument is float or auto.
    """
    if arg != "auto":
        try:
            arg = float(arg)
        except parser.error:
            parser.error('Value {0} is not a float or "auto"'.format(arg))
        if minval is not None and arg < minval:
            parser.error("Value {0} is smaller than {1}".format(arg, minval))
        if maxval is not None and arg > maxval:
            parser.error("Value {0} is larger than {1}".format(arg, maxval))

    return arg


def is_valid_file_or_float(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is not None:
        thefilename, colspec = dd_io.parsefilespec(arg)
    else:
        thefilename = None

    if not op.isfile(thefilename) and thefilename is not None:
        # this is not a file - is it a float?
        try:
            arg = float(arg)
        except ValueError:
            parser.error("Value {0} is not a float or a valid filename".format(arg))

    return arg


def is_int(parser, arg, minval=None, maxval=None):
    """
    Check if argument is int or auto.
    """
    if arg != "auto":
        try:
            arg = int(arg)
        except parser.error:
            parser.error('Value {0} is not an int or "auto"'.format(arg))
        if minval is not None and arg < minval:
            parser.error("Value {0} is smaller than {1}".format(arg, minval))
        if maxval is not None and arg > maxval:
            parser.error("Value {0} is larger than {1}".format(arg, maxval))

    return arg


def is_range(parser, arg):
    """
    Check if argument is min/max pair.
    """
    if arg is not None and len(arg) != 2:
        parser.error("Argument must be min/max pair.")
    elif arg is not None and float(arg[0]) > float(arg[1]):
        parser.error("Argument min must be lower than max.")

    return arg


def is_valid_tag(parser, arg):
    """
    Check if argument is existing file.
    """
    if arg is not None:
        argparts = arg.split(",")
        if len(argparts) < 2:
            parser.error("No tag value specified.")
        tagname = argparts[0]
        tagval = ",".join(argparts[1:])

    return (tagname, tagval)


DEFAULT_FILTER_ORDER = 6
DEFAULT_PAD_SECONDS = 30.0
DEFAULT_PREFILTERPADTYPE = "reflect"
DEFAULT_PERMUTATIONMETHOD = "shuffle"
DEFAULT_NORMTYPE = "stddev"
DEFAULT_FILTERBAND = "lfo"
DEFAULT_FILTERTYPE = "trapezoidal"
DEFAULT_PADVAL = 0
DEFAULT_WINDOWFUNC = "hamming"


def addreqinputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Input NIFTI file name.  " + addedtext,
    )


def addreqoutputniftifile(parser, varname, addedtext=""):
    parser.add_argument(
        varname,
        type=str,
        help="Output NIFTI file name.  " + addedtext,
    )


def addreqinputtextfile(parser, varname, onecol=False):
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns"
        )
    parser.add_argument(
        varname,
        type=lambda x: is_valid_file(parser, x),
        help="Text file containing one or more timeseries columns. " + colspecline,
    )


def addreqinputtextfiles(parser, varname, numreq="Two", nargs="*", onecol=False):
    if onecol:
        colspecline = (
            "Use [:COLUMN] to select which column to use, where COLUMN is an "
            "integer or a column name (if input file is BIDS)."
        )
    else:
        colspecline = (
            "Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an "
            "integer, a column separated list of ranges, or a comma "
            "separated set of column names (if input file is BIDS).  Default is to use all columns."
        )
    parser.add_argument(
        varname,
        nargs=nargs,
        type=lambda x: is_valid_file(parser, x),
        help=numreq + " text files containing one or more timeseries columns. " + colspecline,
    )


def addreqoutputtextfile(parser, varname, rootname=False):
    if rootname:
        helpline = "Root name for the output files"
    else:
        helpline = "Name of the output text file."
    parser.add_argument(
        varname,
        type=str,
        help=helpline,
    )


def addtagopts(
    opt_group,
    helptext="Additional key, value pairs to add to the options json file (useful for tracking analyses).",
):
    opt_group.add_argument(
        "--infotag",
        action="append",
        nargs=2,
        metavar=("tagkey", "tagvalue"),
        help=helptext,
        default=None,
    )


def postprocesstagopts(args):
    if args.infotag is not None:
        argvars = vars(args)
        for thetag in argvars["infotag"]:
            argvars[f"INFO_{thetag[0]}"] = thetag[1]
        del argvars["infotag"]
        return Namespace(**argvars)
    else:
        return args


def addnormalizationopts(parser, normtarget="timecourse", defaultmethod=DEFAULT_NORMTYPE):
    norm_opts = parser.add_argument_group("Normalization options")
    norm_opts.add_argument(
        "--normmethod",
        dest="normmethod",
        action="store",
        type=str,
        choices=["None", "percent", "variance", "stddev", "z", "p2p", "mad"],
        help=(
            f"Demean and normalize {normtarget} "
            "using one of the following methods: "
            '"None" - demean only; '
            '"percent" - divide by mean; '
            '"variance" - divide by variance; '
            '"stddev" or "z" - divide by standard deviation; '
            '"p2p" - divide by range; '
            '"mad" - divide by median absolute deviation. '
            f'Default is "{defaultmethod}".'
        ),
        default=defaultmethod,
    )


def addversionopts(parser):
    version_opts = parser.add_argument_group("Version options")
    version_opts.add_argument(
        "--version",
        action="version",
        help="Show simplified version information and exit",
        version=f"%(prog)s {dd_util.version()[0]}",
    )
    version_opts.add_argument(
        "--detailedversion",
        action="version",
        help="Show detailed version information and exit",
        version=f"%(prog)s {dd_util.version()}",
    )


def addsamplerateopts(parser, details=False):
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )


def addfilteropts(
    parser, filtertarget="timecourses", defaultmethod=DEFAULT_FILTERBAND, details=False
):
    filt_opts = parser.add_argument_group("Filtering options")
    filt_opts.add_argument(
        "--filterband",
        dest="filterband",
        action="store",
        type=str,
        choices=[
            "None",
            "vlf",
            "lfo",
            "resp",
            "cardiac",
            "hrv_ulf",
            "hrv_vlf",
            "hrv_lf",
            "hrv_hf",
            "hrv_vhf",
            "lfo_legacy",
            "lfo_tight",
        ],
        help=(
            f'Filter {filtertarget} to specific band. Use "None" to disable filtering.  '
            f'Default is "{defaultmethod}".  Ranges are: '
            f'vlf: {dd_filt.getfilterbandfreqs("vlf", asrange=True)}, '
            f'lfo: {dd_filt.getfilterbandfreqs("lfo", asrange=True)}, '
            f'cardiac: {dd_filt.getfilterbandfreqs("cardiac", asrange=True)}, '
            f'hrv_ulf: {dd_filt.getfilterbandfreqs("hrv_ulf", asrange=True)}, '
            f'hrv_vlf: {dd_filt.getfilterbandfreqs("hrv_vlf", asrange=True)}, '
            f'hrv_lf: {dd_filt.getfilterbandfreqs("hrv_lf", asrange=True)}, '
            f'hrv_hf: {dd_filt.getfilterbandfreqs("hrv_hf", asrange=True)}, '
            f'hrv_vhf: {dd_filt.getfilterbandfreqs("hrv_vhf", asrange=True)}, '
            f'lfo_legacy: {dd_filt.getfilterbandfreqs("lfo_legacy", asrange=True)}, '
            f'lfo_tight: {dd_filt.getfilterbandfreqs("lfo_tight", asrange=True)}'
        ),
        default=defaultmethod,
    )
    filt_opts.add_argument(
        "--filterfreqs",
        dest="passvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERPASS", "UPPERPASS"),
        help=(
            "Filter " + filtertarget + " to retain LOWERPASS to "
            "UPPERPASS. If --filterstopfreqs is not also specified, "
            "LOWERSTOP and UPPERSTOP will be calculated "
            "automatically. "
        ),
        default=None,
    )
    filt_opts.add_argument(
        "--filterstopfreqs",
        dest="stopvec",
        action="store",
        nargs=2,
        type=float,
        metavar=("LOWERSTOP", "UPPERSTOP"),
        help=(
            "Filter " + filtertarget + " to with stop frequencies LOWERSTOP and UPPERSTOP. "
            "LOWERSTOP must be <= LOWERPASS, UPPERSTOP must be >= UPPERPASS. "
            "Using this argument requires the use of --filterfreqs."
        ),
        default=None,
    )
    if details:
        filt_opts.add_argument(
            "--filtertype",
            dest="filtertype",
            action="store",
            type=str,
            choices=["trapezoidal", "brickwall", "butterworth"],
            help=(
                f"Filter {filtertarget} "
                "using a trapezoidal FFT, brickwall FFT, or "
                "butterworth bandpass filter. "
                f'Default is "{DEFAULT_FILTERTYPE}".'
            ),
            default=DEFAULT_FILTERTYPE,
        )
        filt_opts.add_argument(
            "--butterorder",
            dest="filtorder",
            action="store",
            type=int,
            metavar="ORDER",
            help=(
                "Set order of butterworth filter (if used). " f"Default is {DEFAULT_FILTER_ORDER}."
            ),
            default=DEFAULT_FILTER_ORDER,
        )
        filt_opts.add_argument(
            "--padseconds",
            dest="padseconds",
            action="store",
            type=float,
            metavar="SECONDS",
            help=(
                "The number of seconds of padding to add to each end of a "
                "timecourse to be filtered "
                f"to reduce end effects.  Default is {DEFAULT_PAD_SECONDS}."
            ),
            default=DEFAULT_PAD_SECONDS,
        )
        filt_opts.add_argument(
            "--padtype",
            dest="ncfiltpadtype",
            action="store",
            type=str,
            choices=["reflect", "zero", "constant", "constant+"],
            help=(
                f"The type of padding at each end of a "
                "timecourse to be filtered "
                f'to reduce end effects.  Default is "{DEFAULT_PREFILTERPADTYPE}".'
            ),
            default=DEFAULT_PREFILTERPADTYPE,
        )


def postprocesssamplerateopts(args, debug=False):
    # set the sample rate
    if args.samplerate == "auto":
        samplerate = 1.0
        args.samplerate = samplerate
    else:
        samplerate = args.samplerate

    return args


def postprocessfilteropts(args, debug=False):
    # configure the filter
    # set the trapezoidal flag, if using
    try:
        thetype = args.filtertype
    except AttributeError:
        args.filtertype = "trapezoidal"
    try:
        theorder = args.filtorder
    except AttributeError:
        args.filtorder = DEFAULT_FILTER_ORDER
    try:
        thepadseconds = args.padseconds
    except AttributeError:
        args.padseconds = DEFAULT_PAD_SECONDS
    try:
        prefilterpadtype = args.prefilterpadtype
    except AttributeError:
        args.prefilterpadtype = DEFAULT_PREFILTERPADTYPE

    # if passvec, or passvec and stopvec, are set, we are going set up an arbpass filter
    args.arbvec = None
    if debug:
        print("before preprocessing")
        print("\targs.arbvec:", args.arbvec)
        print("\targs.passvec:", args.passvec)
        print("\targs.stopvec:", args.stopvec)
        print("\targs.filterband:", args.filterband)
    if args.stopvec is not None:
        if args.passvec is not None:
            args.arbvec = [args.passvec[0], args.passvec[1], args.stopvec[0], args.stopvec[1]]
        else:
            raise ValueError("--filterfreqs must be used if --filterstopfreqs is specified")
    else:
        if args.passvec is not None:
            args.arbvec = [
                args.passvec[0],
                args.passvec[1],
                args.passvec[0] * 0.95,
                args.passvec[1] * 1.05,
            ]
    if args.arbvec is not None:
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = dd_filt.NoncausalFilter(
            "arb",
            transferfunc=args.filtertype,
            padtime=args.padseconds,
            padtype=args.prefilterpadtype,
        )
        theprefilter.setfreqs(args.arbvec[2], args.arbvec[0], args.arbvec[1], args.arbvec[3])
    else:
        theprefilter = dd_filt.NoncausalFilter(
            args.filterband,
            transferfunc=args.filtertype,
            padtime=args.padseconds,
            padtype=args.prefilterpadtype,
        )

    # set the butterworth order
    theprefilter.setbutterorder(args.filtorder)

    if debug:
        print("before preprocessing")
        print("\targs.arbvec:", args.arbvec)
        print("\targs.passvec:", args.passvec)
        print("\targs.stopvec:", args.stopvec)
        print("\targs.filterband:", args.filterband)

    (
        args.lowerstop,
        args.lowerpass,
        args.upperpass,
        args.upperstop,
    ) = theprefilter.getfreqs()

    if debug:
        print("after getfreqs")
        print("\targs.arbvec:", args.arbvec)

    return args, theprefilter


def addwindowopts(parser, windowtype=DEFAULT_WINDOWFUNC):
    wfunc = parser.add_argument_group("Windowing options")
    wfunc.add_argument(
        "--windowfunc",
        dest="windowfunc",
        action="store",
        type=str,
        choices=["hamming", "hann", "blackmanharris", "None"],
        help=(
            "Window function to use prior to correlation. "
            "Options are hamming, hann, "
            f"blackmanharris, and None. Default is {windowtype}"
        ),
        default=windowtype,
    )
    wfunc.add_argument(
        "--zeropadding",
        dest="zeropadding",
        action="store",
        type=int,
        metavar="PADVAL",
        help=(
            "Pad input functions to correlation with PADVAL zeros on each side. "
            "A PADVAL of 0 does circular correlations, positive values reduce edge artifacts. "
            f"Set PADVAL < 0 to set automatically. Default is {DEFAULT_PADVAL}."
        ),
        default=DEFAULT_PADVAL,
    )


def addplotopts(parser, multiline=True):
    plotopts = parser.add_argument_group("General plot appearance options")
    plotopts.add_argument(
        "--title",
        dest="thetitle",
        metavar="TITLE",
        type=str,
        action="store",
        help="Use TITLE as the overall title of the graph.",
        default=None,
    )
    plotopts.add_argument(
        "--xlabel",
        dest="xlabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot x axis.",
        default=None,
    )
    plotopts.add_argument(
        "--ylabel",
        dest="ylabel",
        metavar="LABEL",
        type=str,
        action="store",
        help="Label for the plot y axis.",
        default=None,
    )
    if multiline:
        plotopts.add_argument(
            "--legends",
            dest="legends",
            metavar="LEGEND[,LEGEND[,LEGEND...]]",
            type=str,
            action="store",
            help="Comma separated list of legends for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--legend",
            dest="legends",
            metavar="LEGEND",
            type=str,
            action="store",
            help="Legends for the timecourse.",
            default=None,
        )
    plotopts.add_argument(
        "--legendloc",
        dest="legendloc",
        metavar="LOC",
        type=int,
        action="store",
        help=(
            "Integer from 0 to 10 inclusive specifying legend location.  Legal values are: "
            "0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right, "
            "5: right, 6: center left, 7: center right, 8: lower center, 9: upper center, "
            "10: center.  Default is 2."
        ),
        default=2,
    )
    if multiline:
        plotopts.add_argument(
            "--colors",
            dest="colors",
            metavar="COLOR[,COLOR[,COLOR...]]",
            type=str,
            action="store",
            help="Comma separated list of colors for each timecourse.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--color",
            dest="colors",
            metavar="COLOR",
            type=str,
            action="store",
            help="Color of the timecourse plot.",
            default=None,
        )
    plotopts.add_argument(
        "--nolegend",
        dest="dolegend",
        action="store_false",
        help="Turn off legend label.",
        default=True,
    )
    plotopts.add_argument(
        "--noxax",
        dest="showxax",
        action="store_false",
        help="Do not show x axis.",
        default=True,
    )
    plotopts.add_argument(
        "--noyax",
        dest="showyax",
        action="store_false",
        help="Do not show y axis.",
        default=True,
    )
    if multiline:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH[,LINEWIDTH[,LINEWIDTH...]]",
            type=str,
            help="A comma separated list of linewidths (in points) for plots.  Default is 1.",
            default=None,
        )
    else:
        plotopts.add_argument(
            "--linewidth",
            dest="linewidths",
            metavar="LINEWIDTH",
            type=str,
            help="Linewidth (in points) for plot.  Default is 1.",
            default=None,
        )
    plotopts.add_argument(
        "--tofile",
        dest="outputfile",
        metavar="FILENAME",
        type=str,
        action="store",
        help="Write figure to file FILENAME instead of displaying on the screen.",
        default=None,
    )
    plotopts.add_argument(
        "--fontscalefac",
        dest="fontscalefac",
        metavar="FAC",
        type=float,
        action="store",
        help="Scaling factor for annotation fonts (default is 1.0).",
        default=1.0,
    )
    plotopts.add_argument(
        "--saveres",
        dest="saveres",
        metavar="DPI",
        type=int,
        action="store",
        help="Write figure to file at DPI dots per inch (default is 1000).",
        default=1000,
    )


def addpermutationopts(parser, numreps=10000):
    sigcalc_opts = parser.add_argument_group("Significance calculation options")
    permutationmethod = sigcalc_opts.add_mutually_exclusive_group()
    permutationmethod.add_argument(
        "--permutationmethod",
        dest="permutationmethod",
        action="store",
        type=str,
        choices=["shuffle", "phaserandom"],
        help=(
            "Permutation method for significance testing. "
            f'Default is "{DEFAULT_PERMUTATIONMETHOD}".'
        ),
        default=DEFAULT_PERMUTATIONMETHOD,
    )
    sigcalc_opts.add_argument(
        "--numnull",
        dest="numestreps",
        action="store",
        type=int,
        metavar="NREPS",
        help=(
            "Estimate significance threshold by running "
            f"NREPS null correlations (default is {numreps}, "
            "set to 0 to disable). "
        ),
        default=numreps,
    )


def addsearchrangeopts(parser, details=False, defaultmin=-30.0, defaultmax=30.0):
    parser.add_argument(
        "--searchrange",
        dest="lag_extrema",
        action=IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Limit fit to a range of lags from LAGMIN to "
            "LAGMAX.  Default is -30.0 to 30.0 seconds. "
        ),
        default=(defaultmin, defaultmax),
    )
    if details:
        parser.add_argument(
            "--fixdelay",
            dest="initialdelayvalue",
            action="store",
            type=float,
            metavar="DELAYTIME",
            help=("Don't fit the delay time - set it to " "DELAYTIME seconds for all voxels. "),
            default=None,
        )


def postprocesssearchrangeopts(args):
    # Additional argument parsing not handled by argparse
    # first handle fixed delay
    try:
        test = args.initialdelayvalue
    except:
        args.initialdelayvalue = None
    if args.initialdelayvalue is not None:
        args.fixdelay = True
        args.lag_extrema = (args.initialdelayvalue - 10.0, args.initialdelayvalue + 10.0)
    else:
        args.fixdelay = False

    # now set the extrema
    try:
        test = args.lag_extrema_nondefault
        args.lagmin_nondefault = True
        args.lagmax_nondefault = True
    except AttributeError:
        pass
    args.lagmin = args.lag_extrema[0]
    args.lagmax = args.lag_extrema[1]
    return args


def addtimerangeopts(parser):
    parser.add_argument(
        "--timerange",
        dest="timerange",
        action="store",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help=(
            "Limit analysis to data between timepoints "
            "START and END in the input file. If END is set to -1, "
            "analysis will go to the last timepoint.  Negative values "
            "of START will be set to 0. Default is to use all timepoints."
        ),
        default=(-1, -1),
    )


def postprocesstimerangeopts(args):
    args.startpoint = int(args.timerange[0])
    if args.timerange[1] == -1:
        args.endpoint = 10000000000
    else:
        args.endpoint = int(args.timerange[1])
    return args


def parserange(timerange, descriptor="timerange", debug=False):
    if timerange[0] < 0:
        startpoint = 0
    else:
        startpoint = timerange[0]
    if timerange[1] < 0:
        endpoint = 100000000
    else:
        endpoint = timerange[1]
    if debug:
        print("startpoint:", startpoint)
        print("endpoint:", endpoint)
        print("timerange:", timerange)
    if endpoint <= startpoint:
        raise ValueError(f"{descriptor} startpoint must be < endpoint")
    return startpoint, endpoint


def addsimilarityopts(parser):
    parser.add_argument(
        "--mutualinfosmoothingtime",
        dest="smoothingtime",
        action="store",
        type=float,
        metavar="TAU",
        help=(
            "Time constant of a temporal smoothing function to apply to the "
            "mutual information function. "
            "Default is 3.0 seconds.  TAU <=0.0 disables smoothing."
        ),
        default=3.0,
    )


def setargs(thegetparserfunc, inputargs=None):
    """
    Compile arguments for derivdelay workflow.
    """
    if inputargs is None:
        # get arguments from the command line
        # LGR.info("processing command line arguments")
        try:
            args = thegetparserfunc().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise
    else:
        # get arguments from the passed list
        # LGR.info("processing passed argument list:")
        # LGR.info(inputargs)
        try:
            args = thegetparserfunc().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise

    return args, argstowrite


def generic_init(theparser, themain, inputargs=None):
    """
    Compile arguments either from the command line, or from an argument list.
    """
    if inputargs is None:
        print("processing command line arguments")
        # write out the command used
        try:
            args = theparser().parse_args()
            argstowrite = sys.argv
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise
    else:
        print("processing passed argument list:")
        try:
            args = theparser().parse_args(inputargs)
            argstowrite = inputargs
        except SystemExit:
            print("Use --help option for detailed information on options.")
            raise

    # save the raw and formatted command lines
    args.commandline = " ".join(argstowrite)

    themain(args)
