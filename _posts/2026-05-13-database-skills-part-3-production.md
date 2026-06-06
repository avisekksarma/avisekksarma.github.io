---
layout: post
title: "Database Skills Every Software Engineer Must Have — Part 3: Production-Grade Engineering"
subtitle: "MVCC internals, partitioning, zero-downtime migrations, cursor pagination, replication, and monitoring. The advanced layer that separates engineers who run databases from engineers who own them."
date: 2026-05-13
categories: [Tech, databases, backend, series]
tags: [postgresql, sql, performance, partitioning, replication, migrations, backend-engineering]
series: "Database Engineering: From Good to Great"
series_part: 3
reading_time: 22
---

[Part 1](/tech/databases/backend/series/2026/05/13/database-skills-part-1-foundations.html) gave you the relational foundation. [Part 2](/tech/databases/backend/series/2026/05/13/database-skills-part-2-internals.html) gave you the internals — indexes, query planning, transactions, locks. Both of those posts make you a competent database engineer.

This post is about something different.

Production databases are not development databases. They have hundreds of millions of rows, live traffic that never stops, schema changes that can't afford downtime, replicas that drift, autovacuum processes that fall behind, and query patterns you didn't anticipate when you designed the schema. The gap between "works in development" and "works in production at scale" is where most database incidents live.

This post covers the advanced topics that engineers learn either from deep study or from being paged at 2am:

- **MVCC** — how Postgres versions data, and what table bloat actually is
- **Table partitioning** — splitting very large tables and why it matters
- **Zero-downtime migrations** — schema changes on live production tables
- **Cursor-based pagination** — why OFFSET fails at scale
- **Read replicas and replication lag** — scaling reads and the consistency traps
- **Covering indexes and index-only scans** — the optimization most engineers miss
- **Production monitoring** — the queries that tell you your database is sick before it fails

---

## How Postgres Actually Stores Data: The Heap and MVCC

To understand bloat, vacuuming, and why reads don't block writes in Postgres, you need to understand how data is physically stored.

Every table in Postgres is stored as a **heap file** — a collection of 8KB pages on disk. Rows are written to pages roughly in insertion order, with no inherent sorting. A sequential scan reads these pages from start to end. An index scan jumps to specific pages via pointers.

What makes Postgres different from most databases is **MVCC — Multi-Version Concurrency Control**. Instead of locking a row when updating it, Postgres keeps multiple versions of the row simultaneously.

```
Before UPDATE:
[id=1, balance=1000, xmin=100, xmax=null]   ← live row

After UPDATE by transaction 200:
[id=1, balance=1000, xmin=100, xmax=200]    ← dead version (updated by txn 200)
[id=1, balance=500,  xmin=200, xmax=null]   ← live version
```

Every row version carries two hidden system columns:
- **`xmin`** — the transaction ID that created this row version
- **`xmax`** — the transaction ID that deleted or superseded it (null means still live)

When Transaction A reads a row while Transaction B is in the middle of updating it, Transaction A sees the version that was current when A started — the old `xmin`. Transaction B's new version has a higher `xmin` and is invisible to A until B commits. **No locking required. No waiting. Reads never block writes.**

This is why Postgres handles concurrent reads so gracefully. It's one of the most important architectural decisions in the system.

### The Cost: Bloat and VACUUM

Dead row versions don't disappear on their own. They accumulate on disk, taking up space in the heap and slowing down sequential scans — Postgres reads past them even though they're invisible to queries. This accumulation is called **table bloat**.

Postgres has a background process called **autovacuum** that periodically cleans dead row versions and reclaims space. On most tables it runs transparently and you never think about it. But on high-write tables — tables with constant `UPDATE` and `DELETE` traffic — autovacuum can fall behind, leading to significant bloat and degraded performance.

The more severe consequence of neglected vacuuming is **transaction ID wraparound**. Postgres uses 32-bit transaction IDs, which wrap around after about 2 billion transactions. To prevent data from becoming invisible after a wraparound, Postgres will begin refusing all writes to the database and emit warnings like:

```
WARNING: database "production" must be vacuumed within 11,225,000 transactions
```

At that point you have a serious production emergency on your hands. Autovacuum is supposed to prevent this automatically, but aggressive write loads and misconfigured autovacuum settings can let it happen.

**Practical monitoring for bloat:**

```sql
SELECT
    tablename,
    n_live_tup,
    n_dead_tup,
    ROUND(
        100 * n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0),
    2) AS dead_pct,
    last_autovacuum,
    last_autoanalyze
FROM pg_stat_user_tables
ORDER BY dead_pct DESC;
```

If `dead_pct` is above 10–20% on a large table and `last_autovacuum` is old, autovacuum isn't keeping up. You can trigger a manual vacuum without blocking reads:

```sql
VACUUM ANALYZE orders;        -- reclaims dead rows, refreshes statistics
VACUUM FULL orders;           -- full rewrite, reclaims disk space — BLOCKS ALL ACCESS
```

Use `VACUUM FULL` only during maintenance windows. Regular `VACUUM` (without FULL) is safe to run on a live table.

---

## Covering Indexes and Index-Only Scans

This is the optimization most engineers don't reach for, but it's one of the highest-leverage improvements you can make on read-heavy query paths.

A standard index lookup works in two steps:
1. Traverse the B-tree to find matching keys and their `ctid` pointers
2. Follow each pointer to the heap to fetch the actual row

Step 2 is a random I/O operation. If you're fetching thousands of rows, thousands of random page reads add up — especially when the data isn't cached.

A **covering index** includes all the columns a query needs, so Postgres never has to touch the heap at all:

```sql
-- Query that needs email and full_name
SELECT email, full_name FROM users WHERE email = 'alice@example.com';

-- Regular index — finds the row, then hits the heap for full_name
CREATE INDEX idx_users_email ON users(email);

-- Covering index — everything needed is in the index itself
CREATE INDEX idx_users_email_covering ON users(email, full_name);
```

With the covering index, EXPLAIN shows `Index Only Scan` instead of `Index Scan` — the heap is never touched. For queries on hot read paths with stable access patterns, this can cut query time significantly.

### The INCLUDE Clause

Postgres 11 introduced `INCLUDE` for covering indexes — a cleaner way to add payload columns without affecting the index sort order:

```sql
-- email is the search key; full_name and created_at are just carried along
CREATE INDEX idx_users_email ON users(email)
INCLUDE (full_name, created_at);
```

The difference from putting all columns in the key: `INCLUDE` columns are stored only in leaf nodes and don't affect index ordering. This keeps the index smaller and the tree shallower while still enabling index-only scans for queries that need those columns.

Design covering indexes for your most performance-critical read paths — the queries that run thousands of times per minute on your hottest tables.

---

## Table Partitioning

When a table grows into the hundreds of millions of rows, the economics change. Even well-indexed queries slow down because indexes get larger, sequential scans cover more ground, vacuuming takes longer, and query planning itself gets slower.

**Partitioning** splits a large logical table into smaller physical sub-tables called partitions. From your application's perspective, you query a single table. Under the hood, Postgres routes reads and writes to the appropriate partition automatically.

### Range Partitioning

The most common pattern — partition by a date or numeric range:

```sql
CREATE TABLE orders (
    id          BIGSERIAL,
    user_id     INT          NOT NULL,
    total_cents INT          NOT NULL,
    status      VARCHAR(50)  NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2023_q4 PARTITION OF orders
    FOR VALUES FROM ('2023-10-01') TO ('2024-01-01');

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

When you query with a date filter, Postgres uses **partition pruning** — it identifies which partitions could contain matching rows and skips the rest entirely:

```sql
-- Only touches orders_2024_q1 — the other partitions are not scanned
SELECT * FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31';
```

Without partitioning, this query scans an index (or the table) across all historical data. With partitioning, it only ever touches one quarter's worth of data.

### List and Hash Partitioning

```sql
-- List partitioning — split by discrete values
CREATE TABLE events (
    id          BIGSERIAL,
    region      VARCHAR(50) NOT NULL,
    event_data  JSONB
) PARTITION BY LIST (region);

CREATE TABLE events_us   PARTITION OF events FOR VALUES IN ('us-east', 'us-west');
CREATE TABLE events_eu   PARTITION OF events FOR VALUES IN ('eu-west', 'eu-central');
CREATE TABLE events_apac PARTITION OF events FOR VALUES IN ('ap-southeast', 'ap-northeast');

-- Hash partitioning — distribute rows evenly, no natural range key needed
CREATE TABLE sessions (
    id      BIGSERIAL,
    user_id INT NOT NULL
) PARTITION BY HASH (user_id);

CREATE TABLE sessions_0 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE sessions_1 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE sessions_2 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE sessions_3 PARTITION OF sessions FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### What Partitioning Actually Solves

**Query performance** — smaller partitions mean less data scanned, smaller indexes to traverse, and faster vacuum cycles.

**Data lifecycle management** — this is arguably the biggest practical win. Dropping old data on a large table with `DELETE` is catastrophically slow — it locks rows, generates enormous WAL, and triggers a massive vacuum. Dropping a partition is instant:

```sql
-- Slow, disruptive — deletes millions of rows one by one
DELETE FROM orders WHERE created_at < '2022-01-01';

-- Instant — drops the physical partition file
DROP TABLE orders_2021_q4;
-- Or detach it for archiving instead of deleting
ALTER TABLE orders DETACH PARTITION orders_2021_q4;
```

**Maintenance isolation** — you can `VACUUM`, `ANALYZE`, or reindex a single partition without touching the rest of the table.

### What Partitioning Doesn't Solve

Partitioning is not a substitute for indexes. If a query doesn't filter on the partition key, Postgres scans all partitions — potentially worse than a single unpartitioned table. Partition on columns you *consistently filter on* in your heaviest queries.

Cross-partition queries (aggregations, sorts, joins that span multiple partitions) have coordination overhead. Partitioning works best when most queries naturally stay within a single partition.

---

## Zero-Downtime Migrations

This is one of the most important practical skills for any engineer who operates a production database. The problem is straightforward: almost every DDL operation in Postgres acquires a lock that blocks reads and writes. On a table that serves live traffic, even a one-second lock can cause a visible outage.

The good news: every dangerous migration has a safe alternative pattern. You just have to know them.

### The Lock Hierarchy

Understanding which operations block what is the starting point:

```sql
-- These acquire ACCESS EXCLUSIVE — blocks ALL reads and writes
ALTER TABLE orders ADD COLUMN notes TEXT NOT NULL DEFAULT 'x';
ALTER TABLE orders DROP COLUMN notes;
ALTER TABLE orders ALTER COLUMN status TYPE INT;
CREATE INDEX idx_orders_status ON orders(status);

-- These are safe on a live table
CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status);
ALTER TABLE orders ADD COLUMN notes TEXT;  -- nullable, no default
```

### Adding a Column Safely

```sql
-- ❌ Dangerous — rewrites entire table with lock held
ALTER TABLE orders ADD COLUMN notes TEXT NOT NULL DEFAULT '';

-- ✅ Safe — three steps
-- Step 1: Add nullable column (near-instant, no table rewrite)
ALTER TABLE orders ADD COLUMN notes TEXT;

-- Step 2: Backfill in batches from application code
-- Never do one giant UPDATE — it locks rows and generates huge WAL
-- Do small batches with short sleeps between them
UPDATE orders SET notes = '' WHERE id BETWEEN 1      AND 10000;
UPDATE orders SET notes = '' WHERE id BETWEEN 10001  AND 20000;
-- continue until all rows are backfilled...

-- Step 3: Add the NOT NULL constraint
-- Postgres 12+: uses a table scan to verify but doesn't rewrite
ALTER TABLE orders ALTER COLUMN notes SET NOT NULL;
```

Why does adding a column with a non-null default cause a full table rewrite in older Postgres versions? Because Postgres has to physically write the default value into every existing row. From Postgres 11 onwards, stored defaults are handled without a full rewrite for most types — but the nullable-then-backfill pattern remains safest for large tables.

### Adding an Index Safely

```sql
-- ❌ Blocks all writes for the duration of the index build
-- On a 100M row table, this could be minutes
CREATE INDEX idx_orders_status ON orders(status);

-- ✅ Builds the index without blocking writes
-- Takes longer (multiple passes), cannot run inside a transaction
CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status);
```

`CREATE INDEX CONCURRENTLY` is one of Postgres's most important production operations features. It builds the index in multiple passes while the table remains fully readable and writable. The cost: it takes 2–3x longer and cannot be wrapped in a transaction (so if it fails partway through, you need to clean up the partial index manually with `DROP INDEX CONCURRENTLY`).

Always use `CONCURRENTLY` for index creation on production tables.

### Changing a Column Type

This is the most dangerous migration and there is no shortcut. `ALTER COLUMN TYPE` rewrites the entire table with an exclusive lock. On a large production table, this is not an option.

The safe pattern is **expand-contract**:

```sql
-- Goal: change orders.status from VARCHAR to an ENUM type

-- Phase 1: Expand — add new column alongside the old
ALTER TABLE orders ADD COLUMN status_new order_status_enum;

-- Phase 2: Dual-write — deploy application code that writes to both columns
-- Backfill the new column from the old
UPDATE orders SET status_new = status::order_status_enum
WHERE id BETWEEN 1 AND 10000;
-- ... batch through all rows

-- Phase 3: Switch reads — deploy code that reads from the new column
-- Keep writing to both during this phase for safe rollback

-- Phase 4: Contract — once confident, drop the old column
ALTER TABLE orders DROP COLUMN status;
ALTER TABLE orders RENAME COLUMN status_new TO status;
```

This takes multiple deployments and careful coordination. It's the right answer. Trying to do it in one `ALTER TABLE` on a live table with significant traffic is how you cause an outage.

### The Expand-Contract Mental Model

Every breaking schema change in production should follow this pattern:

```
1. Expand:   add new thing alongside old (new column, new table, new index)
2. Migrate:  move data over gradually; dual-write during transition
3. Contract: remove the old thing once nothing depends on it
```

Never remove or rename a column in the same deployment that stops using it. Code deploys and schema changes are not atomic across your application fleet — at the moment of deploy, some servers are running the old code and some the new. If you remove a column at the same time your application stops reading it, the servers still running the old code will start throwing errors.

Always deprecate in one deploy, remove in a subsequent one.

---

## Cursor-Based Pagination

`OFFSET`-based pagination is the default because ORMs make it trivial. It's also fundamentally broken at scale, and knowing why — and what to use instead — is a real differentiator.

### Why OFFSET Fails

```sql
-- Page 1 — fast
SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 0;

-- Page 500 — scans and discards 9,980 rows to return 20
SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 9980;
```

`OFFSET N` doesn't mean "jump to position N efficiently." It means **scan the first N rows and throw them away**, then return the next page. The work grows linearly with page depth. At page 500, your database is doing 99.8% wasted work for every query.

There's also a correctness problem: if new rows are inserted between a user's page 1 and page 2 requests, all rows shift by one — the user sees duplicates or skips items entirely.

### Cursor-Based Pagination

Instead of "give me rows at position N," cursor pagination asks "give me rows after this specific row":

```sql
-- First page — no cursor, just take the first batch
SELECT id, user_id, total_cents, created_at
FROM orders
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Client receives last row: created_at='2024-03-15 14:22:00', id=98234
-- Next page — use those values as the cursor
SELECT id, user_id, total_cents, created_at
FROM orders
WHERE (created_at, id) < ('2024-03-15 14:22:00', 98234)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

The `WHERE` clause filters to rows that come after the cursor. Combined with an index on `(created_at DESC, id DESC)`, Postgres jumps directly to the right position in the index — **O(log n)** regardless of how deep you are, compared to OFFSET's O(n).

Why the composite cursor `(created_at, id)` rather than just `created_at`? Because timestamps are not unique — multiple orders can have the same `created_at`. Without the `id` as a tiebreaker, the cursor is ambiguous and you can miss or duplicate rows at timestamp boundaries.

The tradeoff: cursor pagination doesn't support random page access ("jump to page 47") or total page counts efficiently. For most product use cases — infinite scroll, API pagination, feeds — that's fine. For admin interfaces where users need to jump to arbitrary pages, `OFFSET` is acceptable because usage is low-volume.

### Implementing It Cleanly

The pattern in your API:

```python
def get_orders(cursor: dict | None, limit: int = 20):
    query = db.query(Order).order_by(
        Order.created_at.desc(),
        Order.id.desc()
    )

    if cursor:
        query = query.filter(
            or_(
                Order.created_at < cursor['created_at'],
                and_(
                    Order.created_at == cursor['created_at'],
                    Order.id < cursor['id']
                )
            )
        )

    rows = query.limit(limit + 1).all()  # fetch one extra to check if next page exists
    has_more = len(rows) > limit
    rows = rows[:limit]

    next_cursor = None
    if has_more:
        last = rows[-1]
        next_cursor = {'created_at': last.created_at.isoformat(), 'id': last.id}

    return rows, next_cursor
```

Fetching `limit + 1` rows is a clean way to determine if a next page exists without running a separate `COUNT` query.

---

## Read Replicas and Replication Lag

Most production Postgres setups run at minimum a **primary** (write node) and one **replica** (read-only copy). Understanding how replication works — and where it breaks — is essential for designing correct systems.

### How Streaming Replication Works

Postgres's primary continuously writes all changes to the **Write-Ahead Log (WAL)**. Replicas connect to the primary and stream the WAL, applying changes in order:

```
Primary:  [writes] → WAL → committed to disk
                             ↓
Replica:  ← streams WAL ← applies changes → available for reads
```

Replicas are read-only — they reject writes. From a query perspective, they look exactly like the primary, just with a slight delay and potentially stale data.

### The Routing Pattern

```python
# Route by operation type
def get_db_session(operation: str = 'read'):
    if operation == 'write':
        return primary_session
    return replica_session

# Writes always go to primary
with get_db_session('write') as db:
    db.add(new_order)
    db.commit()

# Reads that tolerate staleness go to replica
with get_db_session('read') as db:
    orders = db.query(Order).filter_by(status='pending').all()
```

The replica offloads read traffic from the primary — critical for analytics queries, reporting, exports, and any read-heavy operations that would otherwise compete with writes for primary resources.

### Replication Lag: Where It Bites You

Replication lag is the delay between a change committing on the primary and becoming visible on a replica. Usually milliseconds. Under high write load, it can grow to seconds.

The failure mode is **reading your own write** from a replica before replication catches up:

```python
# ❌ Classic bug
def register_user(email: str):
    # Write goes to primary
    user = create_user(email=email)
    db_primary.commit()

    # Read hits replica — which may not have the user yet
    profile = db_replica.query(User).filter_by(email=email).first()
    # profile is None — replication lag
    send_welcome_email(profile)  # NullPointerException in production
```

The rules:
- **Writes always go to the primary**
- **Reads that must see data just written go to the primary**
- **Reads that tolerate staleness go to replicas**

The third category is larger than you might think: dashboards, analytics, search, recommendation feeds, most list views. The second category is narrower: the immediate response after a write, anything in the critical path of a just-completed transaction.

### Monitoring Replication Lag

```sql
-- Run on the primary — shows lag per replica
SELECT
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    (sent_lsn - replay_lsn) AS replication_lag_bytes
FROM pg_stat_replication;
```

Alert when lag exceeds your application's tolerance. For most applications, anything over a few seconds of lag warrants investigation.

---

## The Production Monitoring Queries

These are the queries you bookmark, put in your runbook, and run when something is wrong. Knowing them separates engineers who react to incidents from engineers who prevent them.

### Slowest Queries — Where to Start Every Performance Investigation

```sql
SELECT
    LEFT(query, 80)              AS query_preview,
    calls,
    ROUND(total_exec_time::numeric / calls, 2) AS avg_ms,
    ROUND(total_exec_time::numeric, 2)         AS total_ms,
    ROUND(rows::numeric / calls, 2)            AS avg_rows
FROM pg_stat_statements
ORDER BY avg_ms DESC
LIMIT 20;
```

`pg_stat_statements` is an extension — enable it with `CREATE EXTENSION pg_stat_statements` and add it to `shared_preload_libraries` in `postgresql.conf`. It's standard in managed Postgres services (RDS, Cloud SQL, Neon, Supabase). This view gives you a ranked list of slowest queries by average execution time, aggregated since the last stats reset. Start every performance investigation here.

### Tables Getting Too Many Sequential Scans

```sql
SELECT
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan,
    seq_scan - idx_scan            AS excess_seq_scans,
    pg_size_pretty(pg_total_relation_size(relid)) AS table_size
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan
  AND pg_total_relation_size(relid) > 1024 * 1024 * 10  -- ignore small tables
ORDER BY excess_seq_scans DESC;
```

Tables appearing here are candidates for new indexes. A large table with high `seq_scan` and low `idx_scan` means queries are scanning the whole table repeatedly — one of the most impactful fixes you can make.

### Indexes That Are Never Used

```sql
SELECT
    indexrelname                                    AS index_name,
    relname                                         AS table_name,
    idx_scan                                        AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid))    AS index_size
FROM pg_stat_user_indexes
JOIN pg_index USING (indexrelid)
WHERE idx_scan = 0
  AND NOT indisprimary
  AND NOT indisunique
ORDER BY pg_relation_size(indexrelid) DESC;
```

Every unused index is dead weight — it costs write performance and storage while providing zero read benefit. Drop these after confirming they're genuinely unused (reset stats with `SELECT pg_stat_reset()` and let the system run for a few days first).

### Active Queries and Lock Contention

```sql
-- What's running right now, and how long has it been running
SELECT
    pid,
    now() - pg_stat_activity.query_start AS duration,
    query,
    state,
    wait_event_type,
    wait_event
FROM pg_stat_activity
WHERE state != 'idle'
  AND query_start < now() - INTERVAL '5 seconds'
ORDER BY duration DESC;
```

```sql
-- Who is blocking whom
SELECT
    blocked.pid                AS blocked_pid,
    blocked.query              AS blocked_query,
    blocking.pid               AS blocking_pid,
    blocking.query             AS blocking_query,
    now() - blocked.query_start AS wait_duration
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.wait_event_type = 'Lock'
ORDER BY wait_duration DESC;
```

Run these when the application slows suddenly. The second query immediately shows you which transaction is holding a lock that others are waiting for, and what query it's running. The fix is almost always to find the long-running transaction and terminate it:

```sql
SELECT pg_terminate_backend(blocking_pid);
```

### Table and Index Sizes

```sql
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(relid))     AS total_size,
    pg_size_pretty(pg_relation_size(relid))            AS table_size,
    pg_size_pretty(
        pg_total_relation_size(relid) - pg_relation_size(relid)
    )                                                  AS index_size
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(relid) DESC
LIMIT 20;
```

Know your largest tables and their index overhead. A table where index size exceeds table size isn't necessarily wrong, but it's worth understanding — you may have indexes you don't need.

### Cache Hit Ratio

```sql
SELECT
    SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0) AS cache_hit_ratio
FROM pg_statio_user_tables;
```

Postgres caches table data in `shared_buffers`. A cache hit ratio below 95% on a production system suggests either `shared_buffers` is too small, your working set is too large for the available memory, or you have queries doing large sequential scans that evict hot data from cache. This is a signal to investigate, not a number to chase blindly.

---

## The Production Readiness Checklist

Before calling a database configuration production-ready at scale:

**Schema and queries**
- [ ] Indexes exist on all foreign key columns
- [ ] Composite indexes designed for actual query access patterns
- [ ] Covering indexes on hottest read paths
- [ ] No N+1 patterns in application code
- [ ] Pagination uses cursors, not OFFSET, for large or growing datasets

**Migrations**
- [ ] All index creation uses `CREATE INDEX CONCURRENTLY`
- [ ] No `ALTER TABLE ... NOT NULL DEFAULT value` on large tables without batched backfill
- [ ] Schema changes follow expand-contract for breaking changes
- [ ] Migrations tested for lock duration before running on production

**Operations**
- [ ] `pg_stat_statements` enabled
- [ ] Autovacuum settings tuned for high-write tables
- [ ] Replication lag monitored with alerting
- [ ] Read/write routing separates primary and replica traffic correctly
- [ ] Connection pool configured with `pool_pre_ping=True`

**Incident readiness**
- [ ] Slow query monitoring dashboard exists
- [ ] Lock contention query bookmarked and accessible
- [ ] Process for identifying and terminating blocking queries
- [ ] Runbook for common failure modes (bloat, lag, connection exhaustion)

---

## Closing the Series

Across these three posts, we've built a complete picture of what it means to genuinely own a relational database layer:

**Part 1 — Foundations:** The relational model, schema design, data types, constraints, keys, joins, aggregations, normalization. The bedrock everything else builds on.

**Part 2 — Internals:** How B-tree indexes work, reading query plans with EXPLAIN ANALYZE, ACID and what each guarantee means in practice, isolation levels and concurrency anomalies, window functions, N+1, connection pooling, locks and deadlocks. The layer that makes you effective in a production codebase.

**Part 3 — Production:** MVCC and table bloat, covering indexes, partitioning, zero-downtime migrations, cursor-based pagination, replication and lag, and the monitoring queries that tell you your database is healthy. The layer that distinguishes engineers who operate production systems from engineers who just write code that runs against them.

None of this is exotic. Every concept in this series is something you'll encounter — or already have encountered — in a real backend system. The difference is whether you encounter it reactively (in an incident, in a slow query that's been hurting you for months, in a migration that took down production) or proactively, because you understood the system before the problem surfaced.

That's the goal. Not to memorize syntax, but to understand the system well enough that the right answer is obvious when the situation calls for it.

---

*This post is part of the **Database Engineering: From Good to Great** series. [Part 1](/tech/databases/backend/series/2026/05/13/database-skills-part-1-foundations.html) covers relational fundamentals and schema design. [Part 2](/tech/databases/backend/series/2026/05/13/database-skills-part-2-internals.html) covers indexes, query planning, transactions, and concurrency.*
