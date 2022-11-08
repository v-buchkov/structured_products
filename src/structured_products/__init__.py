"""
StructuredProducts
=====

Provides
  1. Pricing & quoting functions for structured products.
  2. Functions for executing structured products at maturity (get resulting payoff).
  3. Predicting mechansims, used in pricer.
  4. Simulations of the structured product behavior under random changes in underlying prices.

----------------------------
The examples assume that `structured_products` has been imported as `sp`::

  >>> import numpy as np

Available subpackages
---------------------
predictors
    Various statistical functions to estimate parameters, required for pricer.
pricer
    Set of classes for definition of a structured product - price, quote, final payoff etc.

"""
__version__ = "1.0.0"

# from .pricer import *
# from .predictors.mean import *
# from .predictors.volatility_ import *
# from .predictors.correlation import *
# from .results_analysis import *
# from src.structured_products.simulations import *
