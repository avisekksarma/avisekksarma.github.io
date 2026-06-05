---
layout: post
title: "ACID Properties Demystified: What Every Backend Engineer Must Know"
subtitle: "Atomicity, Consistency, Isolation, Durability — explained clearly with real examples, all four anomalies, and a practical guide on which isolation level to actually use."
date: 2026-05-01
categories: [Tech, databases, backend]
tags: [postgresql, sql, acid, transactions, isolation-levels, backend-engineering]
reading_time: 14
---

If you've worked with databases for more than a week, you've heard the acronym ACID. Most engineers know it stands for Atomicity, Consistency, Isolation, and Durability — and can roughly explain what each word means. Fewer can tell you what isolation levels actually protect against, what write skew is, or why the default isolation level in PostgreSQL isn't always enough.

This post covers all of it. We'll go through each property clearly, dig into all four database anomalies, and end with a practical decision guide for which isolation level to reach for in real scenarios.

---

## What is ACID?

ACID is a set of guarantees that a database gives you about how transactions behave. Think of it as a contract:

- **Atomicity** — all operations in a transaction succeed, or none do
- **Consistency** — data follows all declared rules before and after every transaction
- **Isolation** — concurrent transactions don't interfere with each other's in-progress work
- **Durability** — once a transaction commits, that data survives even a server crash

Each property sounds simple on the surface. The depth is in understanding what they actually guarantee — and what they deliberately don't.

---

## Atomicity: All or Nothing

A transaction is indivisible. Either **all** changes commit, or **none** do. There is no partial commit.

The canonical example is a bank transfer:

```sql
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE name = 'Alice';
UPDATE accounts SET balance = balance + 100 WHERE name = 'Bob';
COMMIT;
```

Atomicity guarantees that if the second `UPDATE` fails for any reason, the first is rolled back. If `COMMIT` itself fails partway through writing, everything is undone. Alice never loses $100 without Bob receiving it.

PostgreSQL implements this through the **Write-Ahead Log (WAL)**. Every change is written to the WAL before it's applied to the actual data files. If the server crashes mid-transaction, PostgreSQL replays the WAL on restart and either completes or rolls back any incomplete transactions — leaving the database in a consistent committed state.

---

## Consistency: Keeping Data Valid

Consistency means a transaction takes the database from one valid state to another. All declared constraints must be satisfied at commit time:

```sql
CREATE TABLE users (
    id         INT  PRIMARY KEY,
    email      TEXT UNIQUE,
    age        INT  CHECK (age >= 0),
    country_id INT  REFERENCES countries(id)
);
```

Every `INSERT` or `UPDATE` that touches this table must satisfy all four constraints before `COMMIT` succeeds. If any constraint is violated, the transaction is rejected entirely.

One useful feature here is **deferrable constraints** — you can tell PostgreSQL to check constraints at `COMMIT` time rather than after each individual statement. This is valuable when you have circular dependencies between rows that need to be inserted together:

```sql
BEGIN;
SET CONSTRAINTS ALL DEFERRED;
INSERT INTO accounts (id, type) VALUES (1, 'savings');
INSERT INTO accounts (id, type) VALUES (2, 'checking');
-- Foreign key constraints checked here at COMMIT, not after each INSERT
COMMIT;
```

One important clarification: consistency only enforces what you explicitly declare. If your business rule is "account balances should never go negative" but you haven't written a `CHECK (balance >= 0)` constraint, the database won't enforce it. The database enforces the rules you give it — no more.

---

## Durability: Data That Survives Crashes

Once `COMMIT` returns successfully, your data is permanent. Even if the server loses power the millisecond after your commit acknowledgment arrives, your change survives.

PostgreSQL achieves this through the same WAL mechanism that enables atomicity. The commit sequence works like this:

```
1. Transaction sends changes
2. PostgreSQL writes changes to WAL and forces a physical disk flush
3. COMMIT returns successfully to your application
4. Background writer eventually flushes changes to data files
5. If a crash happens between steps 3 and 4, WAL replays the change on restart
```

The key is step 2 — the WAL is physically written to disk before `COMMIT` returns. This is what makes the durability guarantee real rather than theoretical.

There's a tuning knob here worth knowing about:

```sql
-- Default — full durability guarantee
ALTER SYSTEM SET synchronous_commit = 'on';

-- Higher throughput, reduced durability
ALTER SYSTEM SET synchronous_commit = 'off';
```

With `synchronous_commit = 'off'`, commits return faster because PostgreSQL doesn't wait for the WAL flush. The tradeoff: up to roughly 500ms of committed transactions could be lost on a crash. For some use cases (session data, analytics events) this tradeoff is acceptable. For financial data or anything user-critical, keep it on.

---

## Isolation: The Deep One

Isolation is where things get interesting. It's the property that governs what concurrent transactions see of each other's in-progress work — and it's the most nuanced of the four, because it's not binary. It comes in levels, each protecting against different classes of bugs called **anomalies**.

### The Four Anomalies

These are the concrete bugs that can happen when transactions run concurrently without sufficient isolation.

---

**Dirty Read — reading data that was never committed**

A transaction reads another transaction's uncommitted changes. If that other transaction rolls back, the first transaction acted on data that never actually existed.

```sql
-- Transaction A modifies a row but hasn't committed
BEGIN;
UPDATE products SET price = 500 WHERE id = 1;
-- Not committed yet

-- Transaction B reads the row right now
SELECT price FROM products WHERE id = 1;  -- sees 500

-- Transaction A changes its mind
ROLLBACK;  -- price is still 1000

-- Transaction B saw a price that never existed
```

PostgreSQL never allows dirty reads, even at its lowest isolation level.

---

**Non-repeatable Read — the same row returns different values within one transaction**

A transaction reads a row, another transaction modifies and commits that row, and the first transaction reads the same row again and gets a different value.

```sql
-- Transaction A
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- returns 1000

-- Transaction B commits a change
UPDATE accounts SET balance = 500 WHERE id = 1;
COMMIT;

-- Transaction A reads the same row again
SELECT balance FROM accounts WHERE id = 1;  -- returns 500
COMMIT;
```

If Transaction A is generating a report that sums several values, these mid-transaction changes make the totals unreliable.

---

**Phantom Read — the same query returns a different set of rows**

A transaction runs a query, another transaction inserts or deletes rows that match the query's condition, and the first transaction runs the same query again and gets a different count.

```sql
-- Transaction A
BEGIN;
SELECT COUNT(*) FROM users WHERE status = 'active';  -- returns 10

-- Transaction B inserts a new active user and commits
INSERT INTO users (name, status) VALUES ('Alice', 'active');
COMMIT;

-- Transaction A counts again
SELECT COUNT(*) FROM users WHERE status = 'active';  -- returns 11
COMMIT;
```

The new row is a "phantom" — it appeared between two identical queries within the same transaction.

---

**Serialization Anomaly (Write Skew) — the most subtle one**

Two transactions each read data, make decisions based on what they read, and their combined writes produce a state that violates a business rule — even though each transaction was individually correct.

The classic example: a hospital requires at least one doctor to be on call at all times.

```sql
-- Current state: Alice = ON CALL, Bob = ON CALL

-- Transaction A: Alice wants to go off call
BEGIN;
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- sees 2, safe to proceed
UPDATE doctors SET on_call = false WHERE name = 'Alice';
COMMIT;

-- Transaction B: Bob wants to go off call (running at the same time)
BEGIN;
SELECT COUNT(*) FROM doctors WHERE on_call = true;  -- also sees 2, also safe
UPDATE doctors SET on_call = false WHERE name = 'Bob';
COMMIT;

-- Result: 0 doctors on call. Rule violated.
```

Both transactions read the same state, both concluded it was safe to proceed, both committed successfully — and together they broke an invariant that neither transaction alone would have broken. This is write skew, and it's only prevented by the highest isolation level.

---

### The Four Isolation Levels

Each level is defined by which anomalies it prevents:

| Isolation Level | Dirty Read | Non-repeatable Read | Phantom Read | Serialization Anomaly |
|---|---|---|---|---|
| Read Uncommitted | ❌ | ❌ | ❌ | ❌ |
| Read Committed | ✅ | ❌ | ❌ | ❌ |
| Repeatable Read | ✅ | ✅ | ✅ * | ❌ |
| Serializable | ✅ | ✅ | ✅ | ✅ |

*PostgreSQL's Repeatable Read also prevents phantom reads — stronger than the SQL standard requires.*

---

**Read Committed** — PostgreSQL's default

Every statement sees the latest committed data at the moment it runs. Two reads of the same row within one transaction can return different values if another transaction committed a change in between.

```sql
BEGIN ISOLATION LEVEL READ COMMITTED;
SELECT price FROM products WHERE id = 1;  -- returns 1000
-- Another transaction commits, sets price to 500
SELECT price FROM products WHERE id = 1;  -- returns 500
COMMIT;
```

This is correct behavior for most web applications. You always see current data. The tradeoff is that within a single transaction, data can shift under you.

---

**Repeatable Read** — snapshot isolation

The transaction takes a snapshot of the database at the moment it begins and reads only from that snapshot, regardless of what other transactions commit during its lifetime.

```sql
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT price FROM products WHERE id = 1;  -- returns 1000
-- Another transaction commits, sets price to 500
SELECT price FROM products WHERE id = 1;  -- still returns 1000
COMMIT;
```

The data is frozen from the transaction's perspective. Other transactions' commits are invisible until after this transaction ends. PostgreSQL's implementation also prevents phantom reads at this level — stronger than the standard requires.

What it doesn't prevent: write skew. The hospital scenario above would still occur at Repeatable Read, because each transaction can still make decisions based on a snapshot that the other transaction's writes have already invalidated.

---

**Serializable** — the strongest guarantee

Serializable makes concurrent transactions behave as if they ran one at a time in some sequential order. If their combined effect would produce an inconsistent result, one of them fails with a serialization error rather than committing bad data.

```sql
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM doctors WHERE on_call = true;
UPDATE doctors SET on_call = false WHERE name = 'Alice';
COMMIT;  -- May fail with: ERROR: could not serialize access due to concurrent update
```

When a serialization failure occurs, your application must catch the error and retry the transaction:

```python
def go_off_call(doctor_name: str):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
                cur.execute(
                    "SELECT COUNT(*) FROM doctors WHERE on_call = true"
                )
                count = cur.fetchone()[0]
                if count > 1:
                    cur.execute(
                        "UPDATE doctors SET on_call = false WHERE name = %s",
                        [doctor_name]
                    )
                    cur.execute("COMMIT")
                    return True
                else:
                    cur.execute("ROLLBACK")
                    return False  # Can't go off call, only one doctor remaining
        except Exception as e:
            cur.execute("ROLLBACK")
            if "could not serialize" in str(e) and attempt < max_retries - 1:
                continue  # Retry
            raise
    return False
```

Retries are not optional with Serializable — they're part of the contract.

---

### How Each Level Prevents Anomalies Visually

**Non-repeatable Read:**

```
Time →  ─────────────────────────────────────────────────>

Tx A:  |──[SELECT price=1000]──────[SELECT price=?]──COMMIT|
                  │                       │
Tx B:             │──[UPDATE to 500]──COMMIT

Read Committed:   1000 ──────────────────> 500  (changed)
Repeatable Read:  1000 ──────────────────> 1000 (consistent)
```

**Write Skew:**

```
Time →  ─────────────────────────────────────────────────>

Tx A:  |──[READ: 2 doctors]──[UPDATE Alice OFF]──COMMIT|
                │                      │
Tx B:           │──[READ: 2 doctors]──[UPDATE Bob OFF]──COMMIT|

Repeatable Read:  Both succeed → 0 doctors on call (rule violated)
Serializable:     One fails   → 1 doctor remains   (rule preserved)
```

---

## Which Isolation Level Should You Use?

The honest answer: it depends on what your transaction is doing. Here's a practical decision framework.

**Use Read Committed (default) when:**

Your transaction is doing straightforward reads or writes where seeing the freshest committed data is correct behavior — most web application endpoints, simple lookups, single-row updates.

```sql
-- Incrementing a view count — Read Committed is fine
BEGIN;
UPDATE posts SET views = views + 1 WHERE id = 456;
COMMIT;

-- Inserting a comment — Read Committed is fine
BEGIN;
INSERT INTO comments (post_id, user_id, body) VALUES (1, 42, 'Great post!');
UPDATE posts SET comment_count = comment_count + 1 WHERE id = 1;
COMMIT;
```

**Use Repeatable Read when:**

Your transaction reads data multiple times and those reads must be consistent with each other — generating reports, computing aggregates across multiple queries, any analytics work that spans several statements.

```sql
-- Monthly report — all three queries must see the same snapshot
BEGIN ISOLATION LEVEL REPEATABLE READ;
SELECT SUM(total_cents) FROM orders WHERE DATE_TRUNC('month', created_at) = '2024-01-01';
SELECT COUNT(*)         FROM orders WHERE DATE_TRUNC('month', created_at) = '2024-01-01';
SELECT AVG(total_cents) FROM orders WHERE DATE_TRUNC('month', created_at) = '2024-01-01';
COMMIT;
```

Without Repeatable Read, a concurrent insert between your three queries would mean your sum, count, and average are computed from slightly different datasets.

**Use Serializable when:**

Your transaction follows the pattern "check a condition, then act based on it" — and correctness depends on that condition still being true at commit time. Banking, inventory management, booking systems, any domain where two concurrent transactions making the same check would violate a business rule.

```sql
-- Booking a seat — must prevent two users booking the same seat
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT COUNT(*) FROM seats WHERE id = 123 AND taken = false;
-- If available:
UPDATE seats SET taken = true, user_id = 456 WHERE id = 123;
COMMIT;  -- Serialization failure here is correct behavior, not a bug

-- Decrementing inventory — must prevent selling below zero
BEGIN ISOLATION LEVEL SERIALIZABLE;
SELECT quantity FROM products WHERE id = 789;
-- If quantity > 0:
UPDATE products SET quantity = quantity - 1 WHERE id = 789;
COMMIT;
```

**Quick reference:**

| Scenario | Isolation Level | Reason |
|---|---|---|
| Blog, social feed, most CRUD | Read Committed | Fast, correct for independent operations |
| Reports, analytics, dashboards | Repeatable Read | Consistent snapshot across multiple reads |
| Payments, bookings, inventory | Serializable | Prevents write skew on check-then-act logic |
| High-throughput counters | Read Committed | Use atomic `UPDATE counter = counter + 1` instead |

---

## Checking and Setting Isolation Levels

```sql
-- See current default
SHOW default_transaction_isolation;  -- "read committed"

-- Set for a single transaction
BEGIN ISOLATION LEVEL REPEATABLE READ;

-- Set for the current session
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

## The Mental Model to Keep

ACID is not four independent features — it's a system. Atomicity ensures partial failures don't corrupt data. Durability ensures committed state survives hardware failure. Consistency ensures the database enforces your declared rules. And isolation ensures that the correctness guarantees you get in a single-user system hold up under concurrency.

The one engineers most frequently get wrong is isolation — specifically, assuming Read Committed (the default) is sufficient for all scenarios. For most application code, it is. But the moment your transaction reads data and makes a decision based on it, you need to think carefully about whether another transaction could have changed the relevant data between your read and your write.

If the answer is yes, and correctness depends on it — step up to Repeatable Read or Serializable. The performance cost is real but usually smaller than you'd expect, and the bugs you avoid are the kind that only appear in production under load, are extremely difficult to reproduce, and cause real data corruption.

---

*Enjoyed this post? This is a standalone deep-dive, but if you want the broader context, check out the [Database Engineering series](/posts/) — start with [Part 1: Foundations](/tech/databases/backend/series/2026/05/13/database-skills-part-1-foundations.html).*
