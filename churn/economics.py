"""Turn churn probabilities into a contact list that maximises expected profit.

Assumptions are explicit and configurable, because they belong to the client:
contacting a customer costs `offer_cost`; a would-be churner who gets the offer
stays with probability `save_rate`; a saved customer is worth `customer_value`.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Economics:
    offer_cost: float = 50.0       # discount + agent time per contacted customer
    save_rate: float = 0.30        # share of contacted churners who stay
    customer_value: float = 800.0  # margin kept by saving one customer

    def expected_profit(self, p):
        """Expected profit of contacting a customer with churn probability p."""
        return np.asarray(p) * self.save_rate * self.customer_value - self.offer_cost

    @property
    def break_even(self):
        return self.offer_cost / (self.save_rate * self.customer_value)


def realised_profit(y, contact, econ: Economics):
    """Profit actually realised on labelled data (expected over save_rate)."""
    y, contact = np.asarray(y), np.asarray(contact, bool)
    return float((y[contact] * econ.save_rate * econ.customer_value).sum() - contact.sum() * econ.offer_cost)
