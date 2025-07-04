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
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from derivdelay.resample import FastResampler, FastResamplerFromFile
from derivdelay.tests.utils import mse
from derivdelay.tests.utils import get_test_temp_path


def test_FastResampler(debug=False):
    tr = 1.0
    padtime = 50.0
    testlen = 1000
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    timecoursein = np.float64(timeaxis * 0.0)
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint - 1] = np.float64(1.0)
    timecoursein[midpoint] = np.float64(1.0)
    timecoursein[midpoint + 1] = np.float64(1.0)
    timecoursein -= 0.5

    shiftlist = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    # generate the fast resampled regressor
    genlaggedtc = FastResampler(timeaxis, timecoursein, padtime=padtime)
    if debug:
        print(f"{genlaggedtc.initstart=}, {genlaggedtc.initend=}, {genlaggedtc.initstep=}")
        print(f"{genlaggedtc.hiresstart=}, {genlaggedtc.hiresend=}, {genlaggedtc.hiresstep=}")

    # save and reload with another name
    resamplername = os.path.join(get_test_temp_path(), "savedresampler")
    genlaggedtc.save(resamplername)
    genlaggedtc2 = FastResamplerFromFile(resamplername, padtime=padtime, debug=debug)
    if debug:
        print(f"{genlaggedtc2.initstart=}, {genlaggedtc2.initend=}, {genlaggedtc2.initstep=}")
        print(f"{genlaggedtc2.hiresstart=}, {genlaggedtc2.hiresend=}, {genlaggedtc2.hiresstep=}")

    if debug:
        plt.figure()
        plt.ylim([-1.0, 2.0 * len(shiftlist) + 1.0])
        plt.plot(timecoursein)
        legend = ["Original"]
        offset = 0.0

    for shiftdist in shiftlist:
        # generate the ground truth rolled regressor
        tcrolled = np.float64(np.roll(timecoursein, shiftdist))

        # generate the fast resampled regressor
        tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist, debug=debug)
        tcshifted2 = genlaggedtc2.yfromx(timeaxis - shiftdist, debug=debug)

        # print out all elements
        if debug:
            for i in range(0, len(tcrolled)):
                print(
                    i,
                    tcrolled[i],
                    tcshifted[i],
                    tcshifted[i] - tcrolled[i],
                    tcshifted2[i],
                    tcshifted2[i] - tcrolled[i],
                )

        # plot if we are doing that
        if debug:
            offset += 1.0
            plt.plot(tcrolled + offset)
            legend.append("Roll " + str(shiftdist))
            offset += 1.0
            plt.plot(tcshifted + offset)
            legend.append("Fastresampler " + str(shiftdist))
            offset += 1.0
            plt.plot(tcshifted2 + offset)
            legend.append("FastresamplerFromFile " + str(shiftdist))

        # do the tests
        msethresh = 1e-6
        aethresh = 2
        assert mse(tcrolled, tcshifted) < msethresh
        np.testing.assert_almost_equal(tcrolled, tcshifted, aethresh)
        assert mse(tcrolled, tcshifted2) < msethresh
        np.testing.assert_almost_equal(tcrolled, tcshifted2, aethresh)

    if debug:
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_FastResampler(debug=True)
