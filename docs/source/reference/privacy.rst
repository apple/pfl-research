.. _privacy:

Differential privacy
====================

Abstract base classes
---------------------

.. automodule:: pfl.privacy.privacy_mechanism
   :members:
   :exclude-members: NoPrivacy, NormClippingOnly

Privacy mechanisms
------------------

.. autoclass:: pfl.privacy.NoPrivacy
   :members:

.. autoclass:: pfl.privacy.NormClippingOnly
   :members:

.. automodule:: pfl.privacy.laplace_mechanism
   :members:

.. automodule:: pfl.privacy.gaussian_mechanism
   :members:

.. automodule:: pfl.privacy.ftrl_mechanism
   :members:

.. automodule:: pfl.privacy.joint_mechanism
   :members:

Privacy accountants
-------------------

.. autoclass:: pfl.privacy.PrivacyAccountant
   :members:

.. autoclass:: pfl.privacy.PLDPrivacyAccountant
   :members:

.. autoclass:: pfl.privacy.PRVPrivacyAccountant
   :members:

.. autoclass:: pfl.privacy.RDPPrivacyAccountant
   :members:

Joint Privacy accountants
-------------------

.. autoclass:: pfl.privacy.JointPrivacyAccountant
   :members:

DP with adaptive clipping
-------------------------

.. automodule:: pfl.privacy.adaptive_clipping
   :members:

Approximate local DP with central DP
------------------------------------

.. automodule:: pfl.privacy.approximate_mechanism
   :members:

DP metrics
----------

.. automodule:: pfl.privacy.privacy_snr
   :members:

DP utilities
------------

.. automodule:: pfl.privacy.compute_parameters
   :members:
