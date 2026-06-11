# Fraud Detection Policy

## Payment Card Fraud

Transactions flagged for card fraud must be blocked immediately if confidence is high.
Card-not-present transactions above $1,000 require additional verification.
Suspected card testing patterns (multiple small charges in rapid succession) must be blocked.
Chargebacks exceeding 1% of monthly transaction volume trigger an automatic audit.

## Account Takeover (ATO)

Any login from a new, unrecognised device must trigger a step-up authentication challenge.
Password resets followed by a high-value transaction within 30 minutes are treated as ATO risk.
Multiple failed login attempts (5+ in 10 minutes) must trigger a temporary account lock.
Sessions must be invalidated immediately on confirmed ATO detection.

## Synthetic Identity Fraud

Accounts less than 30 days old attempting transactions above $1,000 are flagged automatically.
Mismatched identity signals (SSN, address, DOB inconsistency) require analyst review.
Bust-out fraud patterns (rapid credit utilisation to 100%) trigger an account freeze.

## Velocity Abuse

More than 5 transactions per hour from the same account triggers a velocity alert.
More than 3 transactions in any 10-minute window triggers an immediate block.
Bot-driven transaction patterns (uniform amounts, sub-second intervals) are blocked at the gateway.

## Geographic Risk

Transactions from high-risk jurisdictions require additional KYC verification.
Impossible travel (two transactions from locations >500 km apart within 1 hour) triggers an ATO check.
Transactions routed through Tor exit nodes or known proxy services are flagged for review.
VPN-detected sessions originating from sanctioned countries are blocked automatically.

## Response Actions

| Risk Level       | Composite Score | Action              | SLA         |
|------------------|-----------------|---------------------|-------------|
| Clean            | 0–1             | allow               | Real-time   |
| Suspicious       | 2–3             | challenge_user      | <30 seconds |
| High Risk        | 4–5             | flag_for_review     | <15 minutes |
| Confirmed Fraud  | 6–8             | block_transaction   | Real-time   |
| Critical         | 9+              | freeze_account      | Real-time   |

## Compliance Requirements

All fraud decisions must be logged with a full audit trail for regulatory compliance.
Blocked accounts must be notified within 24 hours per consumer protection regulations.
False positive reviews must be completed within 2 business hours to minimise customer impact.
All fraud findings must be reported to the fraud analytics team for ML model retraining.
