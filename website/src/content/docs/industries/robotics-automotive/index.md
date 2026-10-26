---
title: "Robotics & automotive"
description: "Five SrvDB projects for robotics & automotive, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Robots, vehicles and devices increasingly carry AI models that need memory, without a cloud round-trip.

## The challenge

- Cloud lookups add latency and fail without connectivity.
- Devices lose power unexpectedly.
- Storage and memory are limited.

## Why SrvDB fits

- **In-process search:** no network hop.
- **Crash-safe:** after `persist()`, data survives power loss.
- **Compact storage** with SQ8 or PQ; ARM Linux wheels; usable from Rust.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [In-car Manual Assistant](/srvdb/industries/robotics-automotive/in-car-manual-assistant/) | Answers drivers' questions from the owner's manual, offline. |
| 2 · Intermediate | [Diagnostic Assistant](/srvdb/industries/robotics-automotive/diagnostic-assistant/) | Technicians match symptoms and trouble codes to past repairs. |
| 3 · Advanced | [Scenario Miner](/srvdb/industries/robotics-automotive/scenario-miner/) | Finds driving or robot episodes similar to a scenario of interest. |
| 4 · Expert | [Robot Episodic Memory](/srvdb/industries/robotics-automotive/robot-episodic-memory/) | Robots remember places, objects and outcomes. |
| 5 · Frontier | [Autonomous Fleet Memory](/srvdb/industries/robotics-automotive/autonomous-fleet-memory/) | Share learned experience across a fleet with database snapshots. |

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
