============
Installation
============

The package :code:`bop_pipeline` is written in Python and requires Python 3.10 or later.
Currently the package hasn't been deployed on the PyPI. To install the package,

.. code::

   pip install git+https://github.com/ihmeuw-msca/bop_pipeline.git@main

For developers, you can clone the repository and install the package in the
development mode.

.. code::

    git clone https://github.com/ihmeuw-msca/bop_pipeline.git
    cd bop_pipeline
    pip install -e ".[test,docs]"