.. pfl documentation master file, created by
   sphinx-quickstart on Thu Aug 31 12:41:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pfl: Python framework for Private Federated Learning simulations
================================================================

``pfl`` is a Python framework developed at Apple to enable researchers to `run efficient simulations with privacy-preserving federated learning (FL)` and `disseminate the results of their research in FL`.
The framework is `not` intended to be used for third-party FL deployments but the results of the simulations can be tremendously useful in actual FL deployments.
We hope that ``pfl`` will promote open research in FL and its effective dissemination.

``pfl`` provides several useful features, including the following:

* Get started quickly trying out PFL for your use case with your existing model and data.
* Iterate quickly with fast simulations utilizing multiple levels of distributed training (multiple processes, GPUs and machines).
* Flexibility and expressiveness - when a researcher has a PFL idea to try, ``pfl`` has flexible APIs to express these ideas and promote their dissemination (e.g. models, algorithms, federated datasets, privacy mechanisms).
* Fast, scalable simulations for large experiments with state-of-the-art algorithms and models.
* Support of both PyTorch and TensorFlow. This is great for groups that use both, e.g. other large companies.
* Unified benchmarks for datasets that has been vetted for both TensorFlow and PyTorch. Current FL benchmarks are made for one or the other.
* Support of other models in addition to neural networks, e.g. GBDTs. Switching between types of models while keeping the remaining setup fixed is seamless.
* Tight integration with privacy features, including common mechanisms for local and central differential privacy.

Researchers are invited to contribute to the framework. Please, see :doc:`support/contributing` for more details.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   guides/fl_introduction
   guides/simulation_distributed

.. toctree::
   :caption: Support

   installation
   support/contributing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/algorithm
   reference/aggregate
   reference/callback
   reference/common_types
   reference/context
   reference/data
   reference/exception
   reference/hyperparam
   reference/metrics
   reference/model
   reference/postprocessor
   reference/privacy
   reference/stats
   reference/tree
   reference/environment_variables
   reference/internal/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
