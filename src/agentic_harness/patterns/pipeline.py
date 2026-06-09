"""Pipeline Pattern — re-export of ClaimWeaver.

The pipeline pattern is the default sequential chain:

    Task -> Signal -> Evidence -> Policy -> Plan -> Verification -> Outcome

Each step has exactly one job and passes a typed claim to the next step.
This is the simplest agentic pattern and the foundation all others build on.

Use cases: customer support, CRM.
"""

from ..pattern import ClaimWeaver as PipelineWeaver

__all__ = ["PipelineWeaver"]
