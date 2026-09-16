---
layout: project
title: "Quasar: Linearizable Replicated Key-Value Store"
description: "Raft replicated state machine with majority-quorum commit, crash-durable WAL, snapshot compaction, and linearizable reads."
summary: A Raft replicated state machine with term-based leader election, log-matching replication, majority-quorum commit, plus crash-durable WAL, snapshot compaction, and InstallSnapshot catch-up under crash and partition. Linearizable reads go through a majority quorum check so an isolated leader cannot return stale state, with an interactive lab on a live three-node cluster.
importance: 1
category: Engineering
year: 2026
tech:
  - Raft
  - SMR
  - WAL
  - FastAPI
  - Python
github: https://github.com/avisekksarma/nebula/tree/main/projects/quasar
---

A linearizable replicated key-value store implemented as a Raft state machine. Nodes elect a leader by term, replicate an ordered log under log matching, and commit writes only after a majority quorum agrees. Only committed operations are applied to the KV map.

Durability follows the same protocol: a crash-durable write-ahead log, snapshot compaction of committed state, and InstallSnapshot catch-up when a follower has fallen behind the leader's retained log. The cluster is meant to keep this invariant under leader crash, follower restart, and network partition.

Reads are not a local map lookup. The leader confirms its term with a majority before serving a `GET`, so an isolated or former leader cannot return stale state. An interactive lab drives the **same** three-node processes, not a separate simulator, to step through election, partition, crash, and stale-leader recovery.

Source: [projects/quasar](https://github.com/avisekksarma/nebula/tree/main/projects/quasar) in [nebula](https://github.com/avisekksarma/nebula).
