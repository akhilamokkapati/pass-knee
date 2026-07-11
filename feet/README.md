# PASS: Feet (FSR) Module

Part of the **PASS** platform (see [`../README.md`](../README.md)).

**Status:** placeholder - awaiting implementation.
**Owner:** TBD

## Scope

Plantar force sensing with force-sensitive resistors (FSRs). Provides the measurements
the knee module deliberately does **not** claim on its own:

- Weight-bearing / loading
- Foot-contact events (heel strike / toe off)
- Gait phases (from the contact events)

These integrate with the knee and hip signals to enable gait-quality analysis on the
full platform.

## Where things go

Put your source code, tests, firmware and docs for the feet subsystem in this folder.
Keep it self-contained (its own `requirements.txt` / setup) so it runs independently.
See the [`knee/`](../knee/) subsystem for a reference structure.
