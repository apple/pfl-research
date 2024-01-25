# Copyright © 2023-2024 Apple Inc.
"""
Test privacy_guarantee.py.
"""

from pfl.internal.privacy_loss_bound import ApproximatePrivacyLossBound, PrivacyLossBound


class TestPrivacyLossBound:

    def test_comparison(self):
        """
        Check that `lt` represents a strict weak order.
        """
        dp_0 = PrivacyLossBound(0.)
        dp_1 = PrivacyLossBound(1.)
        dp_2 = PrivacyLossBound(2.)

        assert dp_0.epsilon == 0
        assert dp_1.epsilon == 1
        assert dp_2.epsilon == 2

        assert not (dp_0 < dp_0)  # pylint: disable=comparison-with-itself
        assert dp_0 < dp_1
        assert dp_0 < dp_2

        assert not (dp_1 < dp_0)
        assert not (dp_1 < dp_1)  # pylint: disable=comparison-with-itself
        assert dp_1 < dp_2

        assert not (dp_2 < dp_0)
        assert not (dp_2 < dp_1)
        assert not (dp_2 < dp_2)  # pylint: disable=comparison-with-itself


class TestApproximatePrivacyLossBound:

    def test_comparison(self):
        """
        Check that `lt` represents a partial order.
        If epsilon is less but delta is greater, then the elements are
        incomparable.
        """
        dp_0 = ApproximatePrivacyLossBound(0., 0)
        dp_1 = ApproximatePrivacyLossBound(1., 0)
        dp_2 = ApproximatePrivacyLossBound(2., 0)
        adp_1_e_6 = ApproximatePrivacyLossBound(1., 1e-6)
        adp_1_e_5 = ApproximatePrivacyLossBound(1., 1e-5)
        adp_2_e_6 = ApproximatePrivacyLossBound(2., 2e-6)
        adp_2_e_5 = ApproximatePrivacyLossBound(2., 2e-5)

        assert dp_0.epsilon == 0
        assert dp_0.delta == 0
        assert dp_1.epsilon == 1
        assert dp_1.delta == 0
        assert dp_2.epsilon == 2
        assert dp_2.delta == 0

        assert adp_1_e_6.epsilon == 1.
        assert adp_1_e_6.delta == 1e-6
        assert adp_1_e_5.epsilon == 1.
        assert adp_1_e_5.delta == 1e-5
        assert adp_2_e_6.epsilon == 2.
        assert adp_2_e_6.delta == 2e-6
        assert adp_2_e_5.epsilon == 2.
        assert adp_2_e_5.delta == 2e-5

        assert not (dp_0 < dp_0)  # pylint: disable=comparison-with-itself
        assert dp_0 < dp_1
        assert dp_0 < dp_2
        assert dp_0 < adp_1_e_6
        assert dp_0 < adp_1_e_5
        assert dp_0 < adp_2_e_6
        assert dp_0 < adp_2_e_5

        assert not (dp_1 < dp_0)
        assert not (dp_1 < dp_1)  # pylint: disable=comparison-with-itself
        assert dp_1 < dp_2
        assert dp_1 < adp_1_e_6
        assert dp_1 < adp_1_e_5
        assert dp_1 < adp_2_e_6
        assert dp_1 < adp_2_e_5

        assert not (dp_2 < dp_0)
        assert not (dp_2 < dp_1)
        assert not (dp_2 < dp_2)  # pylint: disable=comparison-with-itself
        assert not (dp_2 < adp_1_e_6)
        assert not (dp_2 < adp_1_e_5)
        assert dp_2 < adp_2_e_6
        assert dp_2 < adp_2_e_5

        assert not (adp_1_e_6 < dp_0)
        assert not (adp_1_e_6 < dp_1)
        assert not (adp_1_e_6 < dp_2)
        assert not (adp_1_e_6 < adp_1_e_6)  # pylint: disable=comparison-with-itself
        assert adp_1_e_6 < adp_1_e_5
        assert adp_1_e_6 < adp_2_e_6
        assert adp_1_e_6 < adp_2_e_5

        assert not (adp_1_e_5 < dp_0)
        assert not (adp_1_e_5 < dp_1)
        assert not (adp_1_e_5 < dp_2)
        assert not (adp_1_e_5 < adp_1_e_6)
        assert not (adp_1_e_5 < adp_1_e_5)  # pylint: disable=comparison-with-itself
        assert not (adp_1_e_5 < adp_2_e_6)
        assert adp_1_e_5 < adp_2_e_5

        assert not (adp_2_e_6 < dp_0)
        assert not (adp_2_e_6 < dp_1)
        assert not (adp_2_e_6 < dp_2)
        assert not (adp_2_e_6 < adp_1_e_6)
        assert not (adp_2_e_6 < adp_1_e_5)
        assert not (adp_2_e_6 < adp_2_e_6)  # pylint: disable=comparison-with-itself
        assert adp_2_e_6 < adp_2_e_5

        assert not (adp_2_e_5 < dp_0)
        assert not (adp_2_e_5 < dp_1)
        assert not (adp_2_e_5 < dp_2)
        assert not (adp_2_e_5 < adp_1_e_6)
        assert not (adp_2_e_5 < adp_1_e_5)
        assert not (adp_2_e_5 < adp_2_e_6)
        assert not (adp_2_e_5 < adp_2_e_5)  # pylint: disable=comparison-with-itself
