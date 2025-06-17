The derivdelay package
=====================

derivdelay implements a method for calculating the time delay of a signal
relative to a reference using the coefficients of a linear fit of the reference
and its time derivatives.  This works over a limited time range, but if quite
fast and robust for finding time delays.

Full documentation is at: http://derivdelay.readthedocs.io/en/latest/

|PyPi Latest Version| |PyPi - Python Versions| |License| |Documentation Status| |CircleCI| |Coverage| |DOI| |Funded by NIH|

The derivdelay program
---------------------

Rapidtide is also the name of the first program in the package, which is
used to perform rapid time delay
analysis on functional imaging data to find time lagged correlations
between the voxelwise time series and other time series, primarily in the LFO
band.


Stability, etc.
===============
This is an evolving code base. I\'m constantly tinkering with it. That
said, now that I\'ve sent this off into to the world, I\'m being somewhat
more responsible about locking down stable release points. In between
releases, however, I\'ll be messing with things, although for the most
part this will be restricted to the dev branch.
**It\'s very possible that at any given time the dev branch will be very broken,
so stay away from it unless you have a good reason to be using it.**
I\'ve finally become a little more modern and started
adding automated testing, so as time goes by hopefully the \"in between\"
releases will be somewhat more reliable.  That said, my tests routinely fail, even
when things actually work.  Probably should deal with that. Check back often for exciting
new features and bug fixes!


Financial Support
=================

This code base is being developed and supported by grants from the US
NIH (RF1 MH130637-01)


.. |PyPi Latest Version| image:: https://img.shields.io/pypi/v/derivdelay.svg
   :target: https://pypi.python.org/pypi/derivdelay/
.. |PyPi - Python Versions| image:: https://img.shields.io/pypi/pyversions/derivdelay.svg
   :target: https://pypi.python.org/pypi/derivdelay/
.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |Documentation Status| image:: https://readthedocs.org/projects/derivdelay/badge/?version=stable
   :target: http://derivdelay.readthedocs.io/en/stable/?badge=stable
.. |CircleCI| image:: https://circleci.com/gh/bbfrederick/derivdelay.svg?branch=main&style=shield
   :target: https://circleci.com/gh/bbfrederick/derivdelay
.. |Coverage| image:: https://codecov.io/gh/bbfrederick/derivdelay/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/bbfrederick/derivdelay
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.814990.svg
   :target: https://doi.org/10.5281/zenodo.814990
.. |Funded by NIH| image:: https://img.shields.io/badge/NIH-RF1--MH130637--01-yellowgreen.svg
   :target: https://reporter.nih.gov/project-details/10509534
