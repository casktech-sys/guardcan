# Architecture

GuardCAN is a two-tier intrusion-detection system. Detection runs **in the vehicle**; only compact
events go to a **backend V-SOC** that monitors the whole fleet.

```
 Backbone (high speed):  R-Car ── Automotive Ethernet · SOME/IP · DDS ── Qualcomm ── Orin
                            │        (iceoryx2 zero-copy within a host)
                       Zonal Gateway
                            │
 Edge (classic buses):  Engine ECU ── CAN / CAN-FD (broadcast, unauthenticated) ── ABS ── Body
                            │
                     [ zero-copy tap ] → GuardCAN GPU IDS (CUDA streams)
                            │
          compact security events → OTA → V-SOC (fleet correlation) → OTA patch (UN R156)
```

## Why detection is at the edge

A single vehicle's bus traffic is millions of frames per second — gigabytes per hour. Streaming raw
traffic to the cloud for an entire fleet is infeasible on bandwidth, cost, and latency. So the model
runs locally on the vehicle's existing central/zonal compute; only **events** (attack detected, type,
severity, affected IDs) are sent up — kilobytes, not gigabytes.

## Why two network layers

- **Edge (CAN/CAN-FD):** connects the small ECUs. Broadcast, unauthenticated — the classic attack
  surface, and where public datasets exist. **Week 1 targets this layer.**
- **Backbone (Automotive Ethernet + SOME/IP / DDS, iceoryx2 zero-copy within a host):** connects the
  high-compute SoCs and runs the SDV applications. Monitoring this service-oriented layer (spoofed
  service discovery, unauthorized subscribe, flooding, malformed serialization) is the forward-looking,
  less-solved part. **Week 3 targets this layer with synthetic traffic.**

## The zero-copy tap

The IDS must observe traffic without adding latency to the monitored path. It mirrors frames into GPU
memory via pinned / zero-copy buffers and processes them with CUDA streams. This is the HPC core
(Week 2) and directly reuses the zero-copy techniques from the SDV cross-partition POC.

## Tier 2 — the V-SOC

Receives events from the fleet, correlates them (one vehicle vs. a coordinated campaign across many),
and drives the response: an OTA update of the detection model or rule set, governed by UN R156.
