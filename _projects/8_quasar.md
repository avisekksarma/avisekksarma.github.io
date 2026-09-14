---
layout: project
title: Quasar
description: Linearizable replicated KV store: Raft state-machine replication, majority-quorum commit, WAL, snapshot compaction, InstallSnapshot catch-up, and majority-checked reads that refuse stale state.
summary: A linearizable replicated key-value store built as a Raft state machine. Nodes elect a leader, replicate an ordered log, and commit writes only after a majority agrees—then recover from crashes, partitions, and lag via WAL, snapshot compaction, and InstallSnapshot catch-up, with linearizable reads after a majority quorum check so an isolated leader cannot return stale state.
importance: 1
category: Engineering
year: 2026
venue: Distributed systems · Raft
tech:
  - Raft
  - FastAPI
  - WAL
  - Snapshots
  - Python
github: https://github.com/avisekksarma/nebula/tree/main/projects/quasar
---

Quasar is a distributed, replicated key-value store built to show how a Raft cluster behaves in steady state and under failure. Several nodes elect a leader and keep a replicated, ordered log of state-changing operations. Writes go through the leader, are copied to followers, and become committed only once a majority has the entry; only committed operations are applied to the KV map.

It is the first project in [nebula](https://github.com/avisekksarma/nebula), a workspace for building distributed systems from scratch. Source: [projects/quasar](https://github.com/avisekksarma/nebula/tree/main/projects/quasar).

### Consensus and replication

Roles are follower, candidate, and leader. Terms and voting decide leadership; heartbeats keep it. A higher term forces an old leader to step down. The leader’s log is the first acknowledgement of a write. Followers accept an append only if they share the same prefix (`prev_log_index` / `prev_log_term`); on conflict, an uncommitted suffix is replaced. Committed entries are never deleted. Followers apply in index order once `commit_index` advances.

### Faults, durability, and recovery

The cluster is meant to survive the cases that actually break naive replication: leader crashes, follower crashes and restarts, and network partitions. Each node persists a write-ahead log plus Raft term and vote, so a restart reloads metadata, an optional snapshot, then leftover WAL. Snapshot-covered keys are not replayed; newer WAL waits for the leader’s commit index.

Snapshots compact committed state and drop that prefix from the log. A restarted node rebuilds from snapshot plus remaining log. A follower that has fallen behind the leader’s retained log receives a snapshot install, then continues with ordinary appends.

### Linearizable reads

`GET` is not a local map lookup. Followers redirect. The leader asks the others to confirm its term, counts itself as one, and only then reads the committed map. If it cannot reach a majority—an isolated or former leader—it rejects the read rather than returning a stale value.

Together this is state-machine replication: consensus on an ordered log, majority commit, durable recovery, and a consistency model that does not silently serve old state.

### Lab

`quasar-lab` drives the **same** three-node processes, not a separate Raft simulator. Election, partition, crash, catch-up, and snapshot install can be stepped through against the real cluster.
