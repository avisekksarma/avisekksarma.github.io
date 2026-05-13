---
layout: post
title: "Database Skills Every Software Engineer Must Have — Part 2: Internals That Actually Matter"
subtitle: "Indexes, query planning, transactions, window functions, N+1, connection pooling, and locks. The intermediate layer that separates engineers who write queries from engineers who own them."
date: 2026-05-13
categories: [Tech, databases, backend, series]
tags: [postgresql, sql, indexes, transactions, performance, backend-engineering]
series: "Database Engineering: From Good to Great"
series_part: 2
reading_time: 18
---

In [Part 1](/tech/databases/backend/series/2026/05/13/database-skills-part-1-foundations.html), we covered the foundation: the relational model, schema design, keys, constraints, joins, and aggregations. That's the bedrock.

This post is where things get interesting.

The topics here are the ones that show up in production incidents, in senior engineering interviews, and in the gap between applications that scale and applications that don't. You can build a working backend without understanding indexes deeply. You can ship features without knowing what a deadlock is. But you can't *own* your database layer — can't debug it under pressure, can't make it fast, can't reason about correctness under concurrency — without what's in this post.

Here's what we're covering:

- **Indexes** — how B-trees work, composite and partial indexes, the write cost tradeoff
- **EXPLAIN ANALYZE** — reading query plans like an engineer, not guessing
- **Transactions & ACID** — what the guarantees actually mean, not the textbook definitions
- **Isolation levels** — the concurrency bugs each level protects against
- **Window functions** — one of the most powerful and underused SQL features
- **The N+1 problem** — the most common real-world performance bug
- **Connection pooling** — why it exists and what breaks without it
- **Locks & deadlocks** — how to debug production incidents and prevent them

---

## Indexes: The Most Impactful Optimization You Control

An index is a separate data structure that Postgres maintains alongside your table, optimized for fast lookups. Without one, every filtered query requires a **sequential scan** — reading every row in the table to find matches. On a million-row table, that's a million row reads for a query that should touch one.

The tradeoff is explicit: indexes cost write performance and storage in exchange for dramatically faster reads.

### How a B-Tree Works

The default index type in Postgres is a **B-tree** (balanced tree). When you index a column, Postgres builds a tree where:

- Each internal node contains sorted key values and pointers to child nodes
- Leaf nodes contain the indexed value and a pointer (`ctid`) to the actual row in the table
- The tree stays balanced — every leaf is at the same depth

A lookup for a specific value traverses the tree in **O(log n)** steps. On a 10-million-row table, that's roughly 23 comparisons instead of 10,000,000. The difference between a 2ms query and a 45-second query is often exactly this.

### Creating Indexes

```sql
-- Basic index — speeds up filtering and sorting on this column
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Unique index — enforces uniqueness AND enables fast lookups
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- Composite index — covers queries that filter on multiple columns
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

-- Partial index — only indexes rows matching a condition
-- Smaller, faster, and perfectly suited for filtering on subsets
CREATE INDEX idx_orders_pending ON orders(created_at)
WHERE status = 'pending';
```

### The Rules That Actually Matter

**Foreign key columns are almost never auto-indexed — add them yourself.**

Postgres automatically creates indexes on primary keys and unique constraints. It does *not* create indexes on foreign key columns. This means every join that goes from a parent table to a child table does a full sequential scan on the child, unless you add the index manually.

```sql
-- orders.user_id references users.id
-- Without this, every JOIN from users → orders scans the entire orders table
CREATE INDEX idx_orders_user_id ON orders(user_id);
```

This is one of the most common missed optimizations in production schemas. Check your foreign keys.

**Composite index column order is not arbitrary.**

A composite index on `(user_id, status)` is usable *only* if your query filters on `user_id`, or on `user_id` AND `status`. It cannot be used to filter on `status` alone.

```sql
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

WHERE user_id = 5                          -- ✅ uses index
WHERE user_id = 5 AND status = 'pending'   -- ✅ uses both columns
WHERE status = 'pending'                   -- ❌ cannot use this index
```

The rule: a composite index is usable left-to-right, and you cannot skip the leftmost column. Design composite indexes with the column you filter on most frequently — or most selectively — first.

**Indexes have a real write cost.**

Every `INSERT`, `UPDATE`, and `DELETE` must update every index on that table. A table with 10 indexes pays roughly 10x the write overhead compared to a table with none. Don't add indexes speculatively — add them for specific, known query patterns.

**Low-cardinality columns are often poor index candidates.**

If a column has 3 possible values across 1 million rows, an index scan returns ~333,000 rows. At that point, Postgres's query planner may correctly decide that a sequential scan is faster than following 333,000 individual heap pointers. The exception is composite indexes and partial indexes, where a low-cardinality column narrows a large result set in combination with other filters.

---

## EXPLAIN ANALYZE: Reading the Query Plan

This is the most powerful diagnostic tool in your Postgres toolkit and the one most backend engineers have never opened. `EXPLAIN ANALYZE` tells you exactly how Postgres executed your query — every operation, every row count, every millisecond.

```sql
EXPLAIN ANALYZE
SELECT u.email, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.email;
```

Sample output:

```
HashAggregate  (cost=58.50..60.50 rows=200 width=36) (actual time=2.341..2.387 rows=200 loops=1)
  Group Key: u.id, u.email
  ->  Hash Left Join  (cost=15.50..53.50 rows=1000 width=12) (actual time=0.412..1.893 rows=1000 loops=1)
        Hash Cond: (o.user_id = u.id)
        ->  Seq Scan on orders  (cost=0.00..21.00 rows=1000 width=8) (actual time=0.018..0.385 rows=1000 loops=1)
        ->  Hash  (cost=13.00..13.00 rows=200 width=36) (actual time=0.368..0.368 rows=200 loops=1)
              ->  Seq Scan on users  (cost=0.00..13.00 rows=200 width=36) (actual time=0.010..0.166 rows=200 loops=1)
Planning Time: 0.312 ms
Execution Time: 2.501 ms
```

### How to Read It

Read the output **bottom-up** — the innermost, most-indented nodes execute first. Each node shows:

- **Node type** — the operation: `Seq Scan`, `Index Scan`, `Hash Join`, `Nested Loop`, etc.
- **`cost=X..Y`** — the planner's estimate in abstract units. X is startup cost (before the first row), Y is total cost.
- **`actual time=X..Y`** — real execution time in milliseconds.
- **`rows=N`** — the planner's row estimate vs actual rows returned.

### The Red Flags

**Seq Scan on a large table when you expect an index.**
If you see a sequential scan on a table with millions of rows and your query filters on a column you thought was indexed, that's your problem. Either the index doesn't exist, the query isn't written in a way that allows index use, or the planner decided the index wasn't worth using (check cardinality).

**Estimated rows wildly different from actual rows.**
If the planner estimated 100 rows and got 80,000, it's making bad decisions based on stale statistics. Run `ANALYZE tablename;` to refresh the planner's statistics, or investigate if your data distribution is unusual.

```
-- Red flag: planner is operating blind
(cost=... rows=100 ...) (actual ... rows=82,341 ...)
```

**Nested Loop on large tables.**
A nested loop join is O(n×m). Fine for small tables. Devastating when both sides have thousands of rows. You want `Hash Join` or `Merge Join` for large datasets.

### The Debugging Workflow

```sql
-- Safe on production — shows the plan without running the query
EXPLAIN SELECT ...;

-- Runs the query and shows actual timings
EXPLAIN ANALYZE SELECT ...;

-- Shows cache hits vs disk reads — critical for understanding I/O behavior
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;
```

The `BUFFERS` option shows how many data blocks were served from Postgres's shared buffer cache (`shared hit`) versus read from disk (`shared read`). A slow query with all cache hits has a different solution than one doing heavy disk I/O. This distinction matters when diagnosing performance issues.

---

## Transactions & ACID: What the Guarantees Actually Mean

A transaction is a sequence of operations the database treats as a single unit. Either everything commits, or nothing does.

```sql
BEGIN;

UPDATE accounts SET balance = balance - 500 WHERE id = 1;
UPDATE accounts SET balance = balance + 500 WHERE id = 2;

COMMIT;
```

Without the transaction, a crash between the two updates loses $500. With it, the partial state never exists in the committed database. This is the foundation of correctness in any system that handles meaningful data.

### ACID, Demystified

**Atomicity** means all-or-nothing. If any operation in the transaction fails, the entire transaction rolls back. There is no partial commit. If your server crashes mid-transaction, Postgres replays or discards it on restart — the committed state is always complete.

**Consistency** means a transaction takes the database from one valid state to another. All constraints — NOT NULL, CHECK, UNIQUE, foreign keys — are enforced at commit time. The database will refuse to commit a transaction that would violate any declared rule.

**Isolation** means concurrent transactions don't interfere with each other's in-progress work. The degree of isolation is configurable — more on this below.

**Durability** means once `COMMIT` returns, the data survives even an immediate server crash. Postgres achieves this through the **Write-Ahead Log (WAL)**: every committed change is written to a sequential log on disk before the commit acknowledgment is sent. On crash recovery, Postgres replays the WAL to reconstruct committed state.

### Isolation Levels and Concurrency Anomalies

Isolation is not binary. The SQL standard defines four levels, each protecting against different classes of bugs:

| Anomaly | Read Committed | Repeatable Read | Serializable |
|---|---|---|---|
| Dirty Read | ✅ safe | ✅ safe | ✅ safe |
| Non-repeatable Read | ❌ possible | ✅ safe | ✅ safe |
| Phantom Read | ❌ possible | ✅ safe* | ✅ safe |
| Serialization Anomaly | ❌ possible | ❌ possible | ✅ safe |

*Postgres's Repeatable Read prevents phantom reads — stronger than the SQL standard requires.*

Postgres defaults to **Read Committed**. Here's what each anomaly means in practice:

**Non-repeatable Read:** You read a row within a transaction, another transaction modifies and commits that row, you read it again and get a different value.

```sql
-- Transaction A at Read Committed
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- returns 1000
-- Transaction B commits, sets balance = 500
SELECT balance FROM accounts WHERE id = 1;  -- now returns 500
COMMIT;
-- Within the same transaction, the same row returned two different values
```

**Phantom Read:** You query a set of rows, another transaction inserts or deletes rows that match your condition, you query again and the set is different.

**Serialization Anomaly (write skew):** Two transactions each read data and make decisions based on what they read, and their combined effect produces a state that neither transaction intended and neither would have allowed individually. Classic example: two doctors both check "is anyone on call?" — both see yes — both go off call — nobody is on call. Each transaction was individually valid; together they broke an invariant.

```sql
-- Use higher isolation when reads must be consistent across multiple statements
BEGIN ISOLATION LEVEL REPEATABLE READ;

-- Use serializable when the correctness of a transaction depends
-- on nothing changing in between its reads and writes
BEGIN ISOLATION LEVEL SERIALIZABLE;
```

For most CRUD operations, Read Committed is correct and efficient. For financial transfers, inventory management, any "check-then-act" logic, or reporting queries that must see a consistent snapshot — step up to Repeatable Read or Serializable.

---

## Window Functions: SQL's Most Underused Power Feature

Window functions perform calculations across a set of rows related to the current row — without collapsing the result the way GROUP BY does.

```sql
-- GROUP BY: collapses rows, loses individual data
SELECT user_id, SUM(total_cents) FROM orders GROUP BY user_id;

-- Window function: keeps all rows AND adds the aggregate
SELECT
    id,
    user_id,
    total_cents,
    SUM(total_cents) OVER (PARTITION BY user_id) AS user_lifetime_total
FROM orders;
```

Output:

```
id | user_id | total_cents | user_lifetime_total
---|---------|-------------|---------------------
1  | 1       | 5000        | 12000
2  | 1       | 7000        | 12000
3  | 2       | 3000        | 3000
```

Every row is preserved. Each row knows both its own value and its contribution to the group. This is impossible with GROUP BY alone.

### The Syntax

```sql
function_name() OVER (
    PARTITION BY column   -- defines the group (doesn't collapse rows)
    ORDER BY column       -- order within the partition
    ROWS BETWEEN ...      -- optional frame definition
)
```

`PARTITION BY` is like `GROUP BY` for the window — it defines the scope of the calculation. `ORDER BY` inside `OVER` defines ordering within that scope (different from the query's final `ORDER BY`).

### Ranking Functions

The most interview-tested window functions. Know the difference between all three:

```sql
SELECT
    id,
    user_id,
    total_cents,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY total_cents DESC) AS rn,
    RANK()       OVER (PARTITION BY user_id ORDER BY total_cents DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY user_id ORDER BY total_cents DESC) AS dense_rnk
FROM orders;
```

```
total_cents | ROW_NUMBER | RANK | DENSE_RANK
------------|------------|------|------------
9000        | 1          | 1    | 1
7000        | 2          | 2    | 2
7000        | 3          | 2    | 2    ← tie: RANK skips next number, DENSE_RANK doesn't
3000        | 4          | 4    | 3
```

`ROW_NUMBER` — always unique, arbitrary tiebreaking. `RANK` — ties get the same rank, next rank skips. `DENSE_RANK` — ties get the same rank, next rank doesn't skip.

**The pattern that appears constantly in real work:** get the most recent record per group.

```sql
-- Most recent order per user
SELECT * FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM orders
) ranked
WHERE rn = 1;
```

Window function in a subquery, filter on the rank. This pattern shows up in analytics, reporting, and data pipelines constantly.

### LAG and LEAD

Access the value from the previous or next row within the partition:

```sql
-- Month-over-month revenue change
SELECT
    DATE_TRUNC('month', created_at)   AS month,
    SUM(total_cents)                   AS revenue,
    LAG(SUM(total_cents)) OVER (
        ORDER BY DATE_TRUNC('month', created_at)
    )                                  AS prev_month_revenue,
    SUM(total_cents) - LAG(SUM(total_cents)) OVER (
        ORDER BY DATE_TRUNC('month', created_at)
    )                                  AS month_delta
FROM orders
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;
```

### Running Totals

```sql
SELECT
    id,
    created_at,
    total_cents,
    SUM(total_cents) OVER (ORDER BY created_at) AS running_total
FROM orders
ORDER BY created_at;
```

The `ORDER BY` inside `OVER` without a `PARTITION BY` creates a running total across the entire table, in chronological order. Each row's `running_total` is the sum of all orders up to and including that row.

---

## The N+1 Problem: The Most Common Performance Bug

The N+1 problem is what happens when your application fires one query to fetch N records, then fires one additional query per record to fetch related data. The result: N+1 total database round trips where 1 or 2 would suffice.

### What it looks like

```python
# The bug — extremely common with ORMs
orders = db.query(Order).all()       # Query 1: SELECT * FROM orders

for order in orders:
    print(order.user.email)          # Queries 2 through N+1: SELECT * FROM users WHERE id = ?
```

With 500 orders, this fires 501 queries. Each has network round-trip latency, connection acquisition time, query parsing overhead, and execution time. What should take 5ms takes 2 seconds.

At the SQL level, it looks like this in your query log:

```sql
SELECT * FROM orders;
SELECT * FROM users WHERE id = 1;
SELECT * FROM users WHERE id = 2;
SELECT * FROM users WHERE id = 1;   -- same user fetched again — no caching
SELECT * FROM users WHERE id = 3;
-- ... 496 more identical-looking queries
```

### The fix

The correct approach is to fetch everything you need in the fewest queries possible:

```sql
-- One query — always the right answer
SELECT o.id, o.total_cents, o.status, u.email, u.full_name
FROM orders o
INNER JOIN users u ON u.id = o.user_id;
```

In SQLAlchemy, the ORM equivalent uses eager loading:

```python
# joinedload — generates a JOIN, fetches everything in one query
orders = db.query(Order).options(joinedload(Order.user)).all()

# selectinload — generates a second query using IN (...), often better for collections
orders = db.query(Order).options(selectinload(Order.items)).all()
```

### How to detect it in production

Enable slow query logging and look for bursts of near-identical queries with different ID values. That signature — same query structure, sequential IDs, fired in rapid succession — is always N+1.

```sql
-- In postgresql.conf
log_min_duration_statement = 100   -- log any query taking > 100ms
```

Tools like PgBadger, Datadog APM, or even reading raw Postgres logs will surface this immediately. The fix is almost always the same: replace the loop with a join or an `IN (...)` query.

The deeper lesson is about mindset: **always think in terms of how many queries your code generates, not just whether the code looks clean.** An ORM makes it trivially easy to write N+1 bugs that look perfectly reasonable in application code.

---

## Connection Pooling: Why It Exists and What Breaks Without It

Every serious backend application uses connection pooling. Most engineers know they need it. Fewer can explain why, or know what to do when the pool is the bottleneck.

### Why connections are expensive

Postgres spawns a **dedicated OS process** per connection — not a thread, a full process. Each connection consumes:

- 5–10MB of memory on the Postgres server
- TCP handshake and TLS negotiation time on establishment (~10–50ms)
- Authentication overhead

If your application opens a fresh connection per request and you're serving 500 concurrent requests, you need 500 Postgres processes. At 5MB each, that's 2.5GB of RAM before a single byte of actual data is processed. Most Postgres instances would fall over well before that.

### What a connection pool does

A pool maintains a fixed set of long-lived connections and lends them to application threads on demand:

```
500 concurrent requests
        ↓
  [Connection Pool: 10 connections]
        ↓
  Postgres: 10 processes
```

Requests that can't immediately get a connection wait briefly in a queue. The vast majority of requests complete fast enough that the queue stays short. This trades a small amount of latency (queue wait) for massive reduction in database server load.

### Key configuration parameters

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # persistent connections always maintained
    max_overflow=20,       # additional connections allowed during spikes
    pool_timeout=30,       # seconds to wait for a connection before raising an error
    pool_pre_ping=True,    # validate connections before use (prevents stale connection errors)
    pool_recycle=3600,     # replace connections older than 1 hour (prevents TCP staleness)
)
```

**`pool_size`** is your baseline. A common starting point is 2× the number of CPU cores on your application server, but this should be tuned based on observed queue wait times and Postgres load.

**`pool_pre_ping=True`** is non-negotiable in production. Without it, a connection that went stale due to a network blip or Postgres restart will be handed to your application code, which then gets a cryptic connection error. Pre-ping adds a lightweight `SELECT 1` health check before each connection is handed out.

**`max_overflow`** handles traffic spikes. These connections are created on demand and closed immediately when released, unlike the base pool connections which stay open. Setting this too high defeats the purpose of the pool.

### PgBouncer for high-throughput systems

For applications under serious load, a dedicated connection pooler like **PgBouncer** sits between your application servers and Postgres. It operates at the protocol level and can multiplex thousands of client connections onto a small number of actual server connections.

The most efficient mode is **transaction pooling**: a server connection is held only for the duration of a transaction, then returned to the pool. A single Postgres connection can serve many clients as long as they're not all in a transaction simultaneously.

The rule of thumb: application-side pooling (SQLAlchemy's pool) is sufficient for most applications. When you're running multiple application server instances and your total connection count across all of them starts pushing against Postgres's `max_connections` limit, add PgBouncer.

---

## Locks and Deadlocks: Debugging Production Incidents

Locks are how Postgres enforces isolation between concurrent transactions. Understanding them is what separates engineers who can debug production incidents from engineers who just restart the service and hope.

### How row-level locking works

When a transaction modifies a row, Postgres places an exclusive lock on that row. Other transactions that try to modify the same row must wait. Transactions that only read the row (under Read Committed) see the last committed version and don't wait at all — this is **MVCC** (Multi-Version Concurrency Control), and it's a major reason Postgres handles concurrent reads so well.

You can acquire explicit row locks when you need to read a row and then update it, preventing another transaction from modifying it in between:

```sql
BEGIN;
-- Lock the row now, before we compute the new value
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;
-- No other transaction can modify this row until we commit
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
```

Without `FOR UPDATE`, another transaction could read the same balance, compute its own update, and commit between your SELECT and UPDATE — resulting in a lost update.

### Table-level locks and schema changes

DDL operations acquire table-level locks that block all reads and writes:

```sql
-- This blocks every query touching the orders table until it completes
ALTER TABLE orders ADD COLUMN notes TEXT;
```

On a high-traffic production table, even a fast migration can cause visible downtime because all queries queue up waiting for the lock. Part 3 covers zero-downtime migration strategies — this is one of the most practically important topics for production systems.

### Deadlocks

A deadlock occurs when two transactions each hold a lock the other needs, creating a circular wait that neither can escape:

```sql
-- Transaction A                        -- Transaction B (concurrent)
BEGIN;                                  BEGIN;
UPDATE accounts                         UPDATE accounts
  SET balance = balance - 100             SET balance = balance - 100
  WHERE id = 1;  -- locks row 1           WHERE id = 2;  -- locks row 2

UPDATE accounts                         UPDATE accounts
  SET balance = balance + 100             SET balance = balance + 100
  WHERE id = 2;  -- waits for row 2       WHERE id = 1;  -- waits for row 1
                                       -- DEADLOCK: circular wait
```

Postgres detects deadlocks automatically within a second or two and terminates one of the transactions with:

```
ERROR: deadlock detected
DETAIL: Process 1234 waits for ShareLock on transaction 5678;
        blocked by process 5678.
HINT:  See server log for query details.
```

The killed transaction gets an error; your application should catch it and retry.

### The universal deadlock prevention rule

Always acquire locks in a **consistent order** across all transactions. If every transaction that touches both accounts always locks the lower ID first, circular waits become structurally impossible:

```sql
-- Both transactions lock in the same order: lower id first
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = LEAST(1, 2);
UPDATE accounts SET balance = balance + 100 WHERE id = GREATEST(1, 2);
COMMIT;
```

### Diagnosing lock contention in production

When your application suddenly slows and you suspect lock contention, this query tells you exactly what's blocked and what's blocking it:

```sql
SELECT
    blocked.pid                    AS blocked_pid,
    blocked.query                  AS blocked_query,
    blocking.pid                   AS blocking_pid,
    blocking.query                 AS blocking_query,
    blocking.state                 AS blocking_state
FROM pg_stat_activity blocked
JOIN pg_stat_activity blocking
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.wait_event_type = 'Lock';
```

Bookmark this query. Run it the next time your application slows unexpectedly — you'll immediately see if a long-running transaction is holding locks that dozens of other queries are waiting for. The fix is almost always to find the long-running transaction (often a forgotten `BEGIN` with no `COMMIT`) and terminate it:

```sql
SELECT pg_terminate_backend(blocking_pid);
```

---

## Putting It Together: The Mental Model

These topics aren't independent — they form a single system:

**Indexes** change which rows Postgres physically touches to execute a query. **EXPLAIN ANALYZE** shows you what Postgres actually did, and whether your indexes are being used. **Transactions** group operations into atomic units and give you ACID guarantees. **Isolation levels** control what concurrent transactions see of each other's work. **Window functions** let you express complex analytical queries as single SQL statements instead of application-level loops. **N+1** is what happens when you fetch related data lazily instead of joining. **Connection pooling** is how you make database connections sustainable at scale. **Locks** are how Postgres enforces isolation, and **deadlocks** are what happen when lock acquisition order is inconsistent.

Master these, and you have a genuine intermediate-level foundation. You can write fast queries, understand why they're fast, reason about correctness under concurrency, debug production incidents from first principles, and configure your infrastructure to handle real load.

---

## The Intermediate Checklist

Before calling any schema or query pattern production-ready:

- [ ] Foreign key columns have indexes
- [ ] EXPLAIN ANALYZE shows Index Scans on large tables, not Seq Scans
- [ ] Planner row estimates are within ~2x of actuals (statistics are fresh)
- [ ] Transactions wrap all operations that must succeed or fail together
- [ ] Isolation level matches the correctness requirements of the operation
- [ ] No N+1 patterns — relationships are loaded with joins or eager loading
- [ ] Connection pool is configured with `pool_pre_ping=True` and sensible sizes
- [ ] Lock acquisition order is consistent across transactions that touch multiple rows
- [ ] Lock contention query is bookmarked and ready for production incidents

---

## What's Next

Part 3 goes to the production-grade engineering that distinguishes senior database engineers:

- **Table partitioning** — managing very large tables by splitting them physically
- **Zero-downtime migrations** — schema changes on live production tables without locking
- **Replication** — primary/replica setups, read scaling, failover
- **Vacuuming and bloat** — Postgres's MVCC cleanup mechanism and what happens when it falls behind
- **Advanced indexing** — GIN indexes for full-text search and JSONB, covering indexes, index-only scans
- **Pagination patterns** — why `OFFSET` breaks at scale and what to use instead
- **Monitoring** — the queries and metrics that tell you your database is healthy before it isn't

These are the topics that live in runbooks, postmortems, and the institutional knowledge of engineers who've been paged at 2am. Part 3 is where theory meets production reality.

---

*This post is part of the **Database Engineering: From Good to Great** series. [Part 1](/tech/databases/backend/series/2026/05/13/database-skills-part-1-foundations.html) covers the relational model, schema design, and SQL fundamentals. Part 3 covers production-grade database engineering.*
