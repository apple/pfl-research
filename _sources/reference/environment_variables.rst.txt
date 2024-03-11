Environment variables
---------------------

These are useful environment variables for usage and development in ``pfl``:

* ``PFL_NUMPY_DISTRIBUTE_METHOD`` - When using Horovod with a NumPy-based model, use this environment variable to specify if Horovod should be used with TF or PyTorch as distributed communication library.
  Valid values are ``{'tensorflow', 'pytorch'}``.
* ``PFL_PYTORCH_DEVICE`` - Manually override default device for Torch tensors.
  Valid values are ``{'cpu', 'cuda', 'mps'}``. 
  Prioritizes other devices than ``cpu`` by default.
* ``PFL_WORKER_RANK``, ``PFL_WORKER_ADDRESSES`` - used in distributed simulations with native TF/PyTorch distributed communication libraries, see :ref:`simulation_distributed`.
