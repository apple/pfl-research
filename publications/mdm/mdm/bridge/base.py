from typing import Any, Dict, Protocol, Tuple, TypeVar

Tensor = TypeVar('Tensor')


class PolyaMixtureFrameworkBridge(Protocol[Tensor]):
    """
    Interface for Polya-Mixture algorithm for a particular Deep Learning
    framework.
    """

    @staticmethod
    def category_counts_polya_mixture(categories: Tensor,
                                      num_categories: int) -> Tensor:
        """
        """
        pass

    @staticmethod
    def category_probabilities_polya_mixture_initialization(
            num_components, num_categories, component, categories) -> Tensor:
        """
        """
        pass

    @staticmethod
    def expectation_step(phi, alphas, category_counts) -> Tensor:
        """
        """
        pass

    @staticmethod
    def maximization_step(posterior_probabilities, category_counts,
                          alphas) -> Tensor:
        """
        """
        pass

    @staticmethod
    def update_alpha(alphas, numerator, denominator) -> Tensor:
        """
        """
        pass
