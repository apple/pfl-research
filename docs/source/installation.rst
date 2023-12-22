.. _Installation:

Installation
============

Install pfl from PyPi
---------------------

By default, ``pfl`` does not install any deep learning framework.
To install the latest compatible version of the deep learning framework you plan to use with ``pfl``, you can specify install extras in brackets:

.. code-block:: default

    # Install pfl with TensorFlow and Keras.
    pip install 'pfl[tf]'

    # Install pfl with PyTorch.
    pip install 'pfl[pytorch]'

    # Install pfl for GBDTs
    pip install 'pfl[trees]'

If you want to access the official benchmarks, clone the `pfl-research`_ repository from Github and go to ``./benchmarks``.
This is also a good starting point when doing your own research. 

.. code-block:: default

    git clone https://github.com/apple/pfl-research.git
    cd pfl-research/benchmarks

Follow instructions in each ``README.md`` of ``pfl-research/benchmarks`` to get started running the default setups.


Install pfl from source
-----------------------

Local development environment for ``pfl`` is managed with `Poetry`_. See :ref:`Contributing` on how to setup ``pfl`` for development and testing.

.. _pfl-research: https://github.com/apple/pfl-research
.. _poetry: https://python-poetry.org
