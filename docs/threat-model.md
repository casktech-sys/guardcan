# Threat model (lightweight TARA)

Scope: the in-vehicle network. Assets: the integrity, authenticity, and availability of CAN/CAN-FD
frames and (Week 3) SOME/IP services. Attacker: a compromised node — an OBD-II dongle, a compromised
infotainment/telematics SoC, or a malicious ECU — able to read and inject on the bus.

Why CAN is attackable: it is broadcast and **unauthenticated**. Any node can emit any arbitration ID;
every node receives every frame. There is no built-in source authentication, so spoofing and injection
are trivial once a node is on the bus.

## Attack classes (CAN layer — covered by the HCRL dataset)

| Class | What the attacker does | Observable signal GuardCAN uses |
|---|---|---|
| **DoS flooding** | Inject a high-priority ID (e.g. `0x000`) at high rate to starve the bus | Extreme message frequency; dominant low ID; broken inter-arrival timing |
| **Fuzzing** | Inject random IDs and payloads | Never-seen ID bit patterns; improbable payload bytes |
| **Spoofing (RPM / gear)** | Inject forged frames of a legitimate ID (masquerade) | Normal ID, abnormal payload and/or doubled message rate |
| **Replay** | Re-send previously captured frames out of context | Valid frames, wrong timing/sequence |

## Backbone layer (Week 3 — synthetic)

| Class | What the attacker does |
|---|---|
| **SOA service abuse** | Spoofed SOME/IP service discovery, unauthorized subscribe, service flooding, malformed serialization |

## Detection strategy

- **Unsupervised autoencoder** on normal traffic → catches deviations, including novel attacks, via
  reconstruction error.
- **Supervised classifier** → names the attack type for triage and reporting.
- **Windowed per-ID frame counts** (100 ms windows, K-of-N alert rule) → injection at a higher-than-normal rate on
  any ID, including spoofing with plausible payloads, raised as one event per attack episode
  ([windowed-detection.md](windowed-detection.md)). Blind to a masquerade at the normal rate.

## Limitations (honest)

- Per-message features in Week 1 catch frequency/payload/ID anomalies well; stealthy, timing-only
  attacks benefit from the sequence/window features planned for Week 2.
- The backbone-layer data is synthetic; it demonstrates the approach, not field performance.
