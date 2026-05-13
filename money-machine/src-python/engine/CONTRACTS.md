# Trading Stage I/O Contracts

This module defines typed contracts used between the pipeline stages:

- `SignalContract`: normalized strategy output.
- `RiskDecisionContract`: normalized risk-gate verdict.
- `OrderIntentContract`: normalized order intent passed to adapters.
- `SignedOrderContract`: canonical signed envelope for remote relays.

`ContractValidator` is the centralized validator used before each transition:

1. Strategy -> Risk (`SignalContract` validation)
2. Risk -> Execution intent (`RiskDecisionContract` validation)
3. Execution intent -> Adapter (`OrderIntentContract` validation)
4. Signed relay envelope (`SignedOrderContract` validation, e.g., MT5)

Location:
- Implementation: `engine/contracts.py`
- Runtime enforcement: `engine/signal_pipeline.py`
- Adapter contract tests: `tests/test_adapter_contracts.py`
