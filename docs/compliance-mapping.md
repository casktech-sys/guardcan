# Regulation & standard mapping

GuardCAN exists because in-vehicle intrusion detection is becoming a regulatory requirement, not a
nice-to-have. This maps each part of the system to the regulation/standard it supports.

| Regulation / standard | What it requires | How GuardCAN relates |
|---|---|---|
| **UN R155** — Cyber Security Management System (CSMS) | OEMs must detect and respond to cyberattacks on vehicles in the field, across the lifecycle. Mandatory for new vehicle types since 2022 and all new vehicles since 2024 across UNECE WP.29 markets (EU, UK, Japan, Korea, …). | The in-vehicle IDS + the fleet V-SOC **are** the "detect and monitor" capability. |
| **UN R156** — Software Update Management System (SUMS) | Software updates (especially OTA) must be secure, authenticated, and recorded. | The response loop: push a patched model / rule set over the air after a confirmed campaign. (Design-level in this POC.) |
| **ISO/SAE 21434** — Road-vehicle cybersecurity engineering | The engineering process for meeting R155 — threat analysis (TARA), risk assessment, monitoring. | `docs/threat-model.md` is a lightweight TARA; the detector addresses the identified threats. |
| **ISO 26262** — Functional safety | Safety-critical behaviour must preserve safe states; no unsafe driver distraction. | Detection is advisory and bounded-latency; security detail is not shown to the driver (at most a generic service telltale); the model does not autonomously alter vehicle behaviour. |
| **ISO/IEC 42001** — AI management system (AIMS) | Govern AI responsibly: data lineage, human oversight, drift monitoring, auditability. | The detection model is governed: documented training data, SOC analyst in the loop (not auto-blocking), monitoring for drift and false positives. Composes with the separate ISO 42001 governance agent. |

**The stack in one line:** R155/21434 = *secure* · ISO 26262 = *safe* · ISO 42001 = *responsibly-governed AI* · R156 = *updated properly*.
