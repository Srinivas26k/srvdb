---
title: "Quantum computing"
description: "Five SrvDB projects for quantum computing, from beginner to frontier."
sidebar:
  label: Overview
  order: 0
---

Quantum teams produce papers, circuits, calibration data and experiment logs, much of it proprietary research.

## The challenge

- Results from earlier runs are hard to find again.
- Calibration drift is noticed late.
- Group knowledge is scattered across notebooks and chats.

## Why SrvDB fits

- **Lab-notebook search and Q&A** over the group's own material.
- **Calibration and run similarity:** compare snapshots stored as vectors.
- **Local and private** for unpublished research.

## Five projects, five levels

Each project has its own page: the problem, how SrvDB powers it, what changes, architecture, use cases, core code and limits.

| Level | Project | What it does |
|---|---|---|
| 1 · Beginner | [Lab Notebook Search](/srvdb/industries/quantum-computing/lab-notebook-search/) | Search the group's notes, papers and run summaries. |
| 2 · Intermediate | [Lab Q&A Assistant](/srvdb/industries/quantum-computing/lab-qa-assistant/) | Answers questions from the group's own knowledge with citations. |
| 3 · Advanced | [Calibration Drift Detector](/srvdb/industries/quantum-computing/calibration-drift-detector/) | Compares calibration snapshots stored as vectors to catch drift. |
| 4 · Expert | [Experiment Memory Agent](/srvdb/industries/quantum-computing/experiment-memory-agent/) | An agent remembering runs, plots and decisions across experiments. |
| 5 · Frontier | [Circuit Similarity Research](/srvdb/industries/quantum-computing/circuit-similarity-research/) | Research platform for circuit embeddings and similarity search. |

:::caution
SrvDB is classical software: it stores ordinary vectors describing experiments and runs on regular computers, not on quantum hardware.
:::

:::note[Coming soon]
Runnable reference implementations of these projects are in progress.
:::
