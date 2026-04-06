"""
policy_engine.py

Defines and evaluates access-control policies. Policies express rules such as
"only users with role=FINANCE may read chunks tagged department=finance" or
"clearance_level >= document.sensitivity_level".

Policies can be loaded from a YAML/JSON config or defined programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Policy:
    """A named access-control rule."""

    name: str
    description: str
    # Serialised rule expression (syntax TBD — e.g. OPA Rego, CEL, or custom DSL)
    rule: str


class PolicyEngine:
    """Loads policies and evaluates them against user context + resource tags."""

    def __init__(self, policies: list[Policy] | None = None) -> None:
        """
        Args:
            policies: Initial set of Policy objects to register.
                      Additional policies can be added via add_policy().
        """
        self.policies: list[Policy] = policies or []

    def add_policy(self, policy: Policy) -> None:
        """Register a new access-control policy.

        Args:
            policy: The Policy to add.
        """
        raise NotImplementedError

    def evaluate(
        self,
        user_context: dict[str, Any],
        resource_tags: dict[str, Any],
    ) -> bool:
        """Evaluate all registered policies for a user/resource pair.

        All policies are evaluated; access is granted only if every applicable
        policy permits it (default-deny model).

        Args:
            user_context:   Attributes of the requesting user (roles, dept, etc.).
            resource_tags:  Permission tags on the target resource (chunk/doc).

        Returns:
            True if access is granted, False if any policy denies it.
        """
        raise NotImplementedError

    def load_from_file(self, path: str) -> None:
        """Load policies from a YAML or JSON file.

        Args:
            path: Filesystem path to the policy definition file.
        """
        raise NotImplementedError
