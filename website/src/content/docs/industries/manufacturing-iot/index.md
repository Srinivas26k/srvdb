---
title: "Manufacturing & IoT"
description: "Five SrvDB projects for manufacturing & iot, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Factories and field sites need answers at the machine, often on edge hardware with poor connectivity.

## The challenge

- Technicians search thick manuals for the right procedure.
- Past fault reports that match today's symptoms are hard to find.
- Edge devices have little memory and unreliable networks.

## Why SrvDB fits

- **Manual and fault-report search** on the device itself.
- **Small storage:** SQ8 stores vectors 4× smaller, PQ up to 32× smaller.
- **Linux aarch64 wheels** for ARM edge hardware; works with no connectivity.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Maintenance Assistant](/srvdb/industries/manufacturing-iot/maintenance-assistant/) | Technicians describe a symptom and get the right manual section and past fixes. |
| 2 · Intermediate | [Defect Matcher](/srvdb/industries/manufacturing-iot/defect-matcher/) | Matches new defects to past ones, from text and images. |
| 3 · Advanced | [Sensor Anomaly Detector](/srvdb/industries/manufacturing-iot/sensor-anomaly-detector/) | Flags sensor windows far from normal operation. |
| 4 · Expert | [Plant Knowledge Agent](/srvdb/industries/manufacturing-iot/plant-knowledge-agent/) | An agent with shift memory across lines, manuals and quality records. |
| 5 · Frontier | [Predictive Maintenance at the Edge](/srvdb/industries/manufacturing-iot/predictive-maintenance-edge/) | Finds machines whose behaviour resembles past pre-failure signatures. |

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
