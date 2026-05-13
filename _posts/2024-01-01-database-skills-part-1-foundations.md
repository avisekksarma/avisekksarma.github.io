---
layout: post
title: "The Database Skills Every Backend Engineer Must Have — Part 1: Foundations"
subtitle: "A practical, no-fluff guide to relational databases — the kind of depth that actually shows up in senior engineering interviews."
date: 2024-01-01
categories: [Tech, databases, backend, series]
tags: [postgresql, sql, relational-databases, backend-engineering, database-design]
series: "Database Engineering: From Good to Great"
series_part: 1
reading_time: 12
---

There's a pattern I've noticed in engineering teams. Backend engineers who "know databases" usually know just enough to not break things — they write queries that work, add a foreign key when the PR reviewer asks, maybe throw an index on a column and call it a day.

But database engineering done *well* is a different skill set entirely. It's the difference between a schema that holds up at 10x traffic and one that becomes a liability. Between a query that scans 2 million rows and one that touches 300. Between data that's trustworthy and data that *might* be right.

This is Part 1 of a three-part series:

- **Part 1 (this post):** Relational foundations, schema design, joins, constraints — the bedrock
- **Part 2:** Indexes, query planning, transactions, ACID, concurrency — the internals that matter
- **Part 3:** Advanced patterns, performance tuning, partitioning, replication — production-grade engineering

We're using PostgreSQL throughout, because that's what the industry mostly runs on, and because Postgres rewards engineers who understand it deeply.

---

## Why Relational Databases Are Still the Default

NoSQL had its moment. Document stores, wide-column databases, graph databases — they all have legitimate use cases. But for most backend systems, the relational model remains the dominant choice for good reason:

**Structured data with relationships is the default reality.** Users have orders. Orders have items. Items belong to products. This is a graph of related entities — and the relational model is purpose-built for it.

**ACID guarantees are hard to give up.** More on this in Part 2, but when correctness matters — financial data, inventory, user records — the transactional guarantees of a relational DB are invaluable.

**SQL is a universal skill.** No matter what ORM, language, or framework you use, understanding SQL means you can work with any relational system.

---

## The Relational Model: What It Actually Means

A relational database stores data in **tables** (formally called *relations*). Each table has:

- **Columns** — the schema, defined once (name, data type, constraints)
- **Rows** — individual records

The "relational" in the name comes from the mathematical concept of a *relation* from set theory — a set of tuples with defined attributes. The key philosophical insight of the model: **data is linked by values, not by pointers**.

In a file system or object graph, relationships are pointers — one object literally holds a reference to another. In a relational database, table `orders` references table `users` by *storing the user's ID as a value*. The database engine enforces that this value is valid. This is simultaneously simpler and more powerful than pointer-based systems.

---

## Building a Real Schema

Let's build something concrete. A minimal e-commerce backend: users, orders, order items.

```sql
CREATE TABLE users (
    id          SERIAL PRIMARY KEY,
    email       VARCHAR(255) NOT NULL UNIQUE,
    full_name   VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    user_id     INT          NOT NULL REFERENCES users(id),
    status      VARCHAR(50)  NOT NULL DEFAULT 'pending',
    total_cents INT          NOT NULL CHECK (total_cents >= 0),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    id           SERIAL PRIMARY KEY,
    order_id     INT          NOT NULL REFERENCES orders(id),
    product_name VARCHAR(255) NOT NULL,
    quantity     INT          NOT NULL CHECK (quantity > 0),
    price_cents  INT          NOT NULL CHECK (price_cents >= 0)
);
```

This looks simple, but every decision is deliberate. Let's unpack each one.

---

## The Decisions That Matter in Schema Design

### Never Store Money as a Float

```sql
-- ❌ Wrong
price FLOAT

-- ✅ Correct
price_cents INT
```

Floating point numbers cannot represent most decimal fractions exactly. `0.1 + 0.2` in IEEE 754 floating point equals `0.30000000000000004`. For money, this is unacceptable. Store prices as integers in the smallest currency unit (cents, pence, etc.) and convert to decimals only in your presentation layer.

If you need arbitrary precision (for exchange rates, for example), PostgreSQL's `NUMERIC(precision, scale)` type is exact — but slower than integers. For most money use cases, cents as `INT` or `BIGINT` is the right call.

### Always Use TIMESTAMPTZ, Never TIMESTAMP

```sql
-- ❌ Wrong — stores no timezone information
created_at TIMESTAMP

-- ✅ Correct — stores as UTC, displays in session timezone
created_at TIMESTAMPTZ
```

`TIMESTAMP` stores a naive datetime with no timezone context. If your servers are in different regions, or your team changes deployment regions, or daylight saving time enters the picture, naive timestamps become ambiguous.

`TIMESTAMPTZ` (timestamp with time zone) stores values as UTC internally and converts to the session's timezone on retrieval. Always use it. You will thank yourself when dealing with international users or server migrations.

### NOT NULL Everywhere Reasonable

`NULL` in SQL is a three-valued logic trap. `NULL = NULL` is `NULL` (not `TRUE`). `NULL != NULL` is also `NULL`. Aggregate functions silently skip NULLs. Joins on nullable columns have surprising behavior.

Apply `NOT NULL` to every column unless the absence of a value is genuinely meaningful for your domain. If you're unsure, lean toward `NOT NULL` and use a sentinel value (empty string, 0, or a dedicated status column) to represent "not applicable."

### Constraints Are Your Safety Net

```sql
CHECK (total_cents >= 0)
CHECK (quantity > 0)
UNIQUE (email)
REFERENCES users(id)
```

Every constraint you add is a bug your application *cannot* introduce, regardless of how broken your code gets. Constraints are enforced by the database engine on every write — no exceptions, no race conditions, no way to bypass them through a misconfigured service.

This is the key mindset shift: **the database is not just a dumb store. It is a co-enforcer of your domain invariants.**

---

## Keys and Relationships

### Primary Keys

A primary key uniquely identifies every row. No two rows can share a PK value, and it cannot be NULL.

```sql
-- Single-column PK (most common)
id SERIAL PRIMARY KEY

-- Composite PK — useful for junction tables
PRIMARY KEY (user_id, role_id)
```

`SERIAL` in Postgres is syntactic sugar for an auto-incrementing integer sequence. For new projects, consider `BIGSERIAL` (8-byte integer) instead of `SERIAL` (4-byte) — you're unlikely to exceed 2 billion rows, but it costs nothing to be safe.

A growing school of thought prefers **UUIDs** as primary keys (`gen_random_uuid()` in Postgres 13+) for distributed systems, where auto-increment sequences would require coordination across nodes. The tradeoff: UUIDs are larger (16 bytes vs 4-8), and random UUIDs cause index fragmentation due to non-sequential inserts. ULIDs and UUIDv7 (time-ordered) are worth exploring if you go this route.

### Foreign Keys and Referential Integrity

```sql
user_id INT NOT NULL REFERENCES users(id)
```

This foreign key constraint tells Postgres:
1. **On insert/update:** reject any `user_id` value that doesn't exist in `users.id`
2. **On delete of a user:** reject the delete (default behavior) unless you specify otherwise

```sql
-- Options for what happens when the referenced row is deleted:
REFERENCES users(id) ON DELETE RESTRICT    -- default; reject the delete
REFERENCES users(id) ON DELETE CASCADE     -- delete orders too
REFERENCES users(id) ON DELETE SET NULL    -- set user_id to NULL
REFERENCES users(id) ON DELETE SET DEFAULT -- set to column default
```

`CASCADE` feels convenient but use it carefully. Cascading deletes can propagate through multiple tables and delete far more data than intended. For most production systems, prefer `RESTRICT` and handle deletions explicitly in your application (soft deletes, archiving, etc.).

---

## SQL Fundamentals: Beyond Basic CRUD

### The Mental Model for Joins

Joins are where most engineers plateau. The key mental model: **start with a set of rows from the first table. For each row, find matching rows in the second table based on the condition. Combine them.**

What you do with non-matching rows distinguishes join types:

```
INNER JOIN  — keep only rows that matched on both sides
LEFT JOIN   — keep all rows from left; NULLs where right didn't match
RIGHT JOIN  — keep all rows from right (rarely used; just flip the table order)
FULL JOIN   — keep all rows from both sides
```

```sql
-- INNER JOIN: only users who have at least one order
SELECT u.email, o.id AS order_id, o.status
FROM users u
INNER JOIN orders o ON o.user_id = u.id;

-- LEFT JOIN: ALL users, with order data if it exists
SELECT u.email, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.email;
```

### Multi-Table Joins

```sql
-- Full order details with user info
SELECT
    u.email,
    o.id           AS order_id,
    o.status,
    o.created_at   AS ordered_at,
    oi.product_name,
    oi.quantity,
    oi.price_cents
FROM users u
INNER JOIN orders      o  ON o.user_id  = u.id
INNER JOIN order_items oi ON oi.order_id = o.id
WHERE u.email = 'alice@example.com'
ORDER BY o.created_at DESC;
```

Read this left to right: start with users, join to their orders, join to the items in those orders. Each join narrows or expands the working set.

### Aggregations Done Right

```sql
-- Revenue breakdown per user
SELECT
    u.email,
    COUNT(DISTINCT o.id)             AS total_orders,
    SUM(oi.price_cents * oi.quantity) AS total_spent_cents,
    AVG(o.total_cents)               AS avg_order_value_cents
FROM users u
LEFT JOIN orders      o  ON o.user_id  = u.id
LEFT JOIN order_items oi ON oi.order_id = o.id
GROUP BY u.id, u.email
HAVING SUM(oi.price_cents * oi.quantity) > 0
ORDER BY total_spent_cents DESC;
```

Key rules that trip people up:

- **Every column in SELECT must be in GROUP BY or be an aggregate.** Postgres enforces this strictly (unlike MySQL, which will silently return arbitrary values for non-grouped columns).
- **WHERE filters rows before grouping. HAVING filters groups after aggregation.** You cannot use a column alias defined in SELECT inside a HAVING clause in standard SQL.
- **`COUNT(column)` vs `COUNT(*)`:** `COUNT(column)` ignores NULL values. `COUNT(*)` counts all rows including those with NULLs. This distinction matters when counting optional relationships.
- **`COUNT(DISTINCT column)`:** counts unique non-null values. Essential when your join produces duplicate rows.

---

## NULL: The Billion-Dollar Mistake

Tony Hoare, who invented the null reference, called it his "billion-dollar mistake." SQL NULL is similarly problematic if you don't understand it.

```sql
-- NULL comparisons don't work as expected
SELECT * FROM users WHERE full_name = NULL;    -- ❌ returns nothing
SELECT * FROM users WHERE full_name IS NULL;   -- ✅ correct

-- NULL in boolean logic
SELECT NULL = NULL;   -- NULL (not TRUE)
SELECT NULL != NULL;  -- NULL (not TRUE)
SELECT NULL OR TRUE;  -- TRUE
SELECT NULL AND TRUE; -- NULL
SELECT NULL AND FALSE; -- FALSE
```

NULL represents *unknown*, not *absent* or *false*. This is three-valued logic (TRUE, FALSE, UNKNOWN), and it behaves consistently once you internalize it — but it will catch you off guard repeatedly until you do.

Practical takeaway: use `IS NULL` / `IS NOT NULL` for null checks, never `= NULL`. And again — declare columns `NOT NULL` wherever the value should always exist.

---

## Data Type Discipline

Choosing the right data type isn't pedantry — it affects storage, performance, and correctness.

| Use Case | Recommended Type | Avoid |
|---|---|---|
| Auto-increment ID | `BIGSERIAL` | `SERIAL` for large tables |
| Money | `INT` (cents) or `NUMERIC(12,2)` | `FLOAT`, `REAL` |
| Datetime | `TIMESTAMPTZ` | `TIMESTAMP` |
| Date only | `DATE` | `TIMESTAMP` |
| Short text | `VARCHAR(n)` | `TEXT` for bounded strings |
| Unlimited text | `TEXT` | `VARCHAR` without bound |
| True/false | `BOOLEAN` | `INT` (0/1) |
| Enum-like | `VARCHAR` + CHECK or Postgres `ENUM` | Magic integers |
| JSON data | `JSONB` | `JSON` (JSONB is binary, indexed) |
| IP addresses | `INET` | `VARCHAR` |
| UUIDs | `UUID` | `VARCHAR(36)` |

A note on `JSONB`: Postgres's `JSONB` type is genuinely powerful — it supports indexing, querying, and operators. It's appropriate for truly dynamic or semi-structured data. But don't reach for it just because your ORM makes it easy. If data has known structure and you'll query on its fields, model it as proper columns. JSONB is an escape hatch, not a default.

---

## Normalization: The Principle Behind Schema Design

Normalization is the process of organizing a schema to reduce redundancy and improve integrity. There are formal forms (1NF, 2NF, 3NF, BCNF), but the engineering intuition is:

**Every fact should be stored in exactly one place.**

If a user's email appears in 5 places, you have 5 places to update it and 5 chances to get them out of sync.

A quick practical guide:

**First Normal Form (1NF):** No repeating groups; every column holds atomic values.
```sql
-- ❌ Violates 1NF — multiple values in one column
phone_numbers VARCHAR -- "555-1234, 555-5678"

-- ✅ Correct — separate table
CREATE TABLE user_phones (
    user_id INT REFERENCES users(id),
    phone   VARCHAR(20) NOT NULL
);
```

**Second Normal Form (2NF):** Every non-key column depends on the *whole* primary key (relevant for composite keys).

**Third Normal Form (3NF):** No non-key column depends on another non-key column (eliminate transitive dependencies).

```sql
-- ❌ Violates 3NF — zip_code determines city; city is in the orders table
CREATE TABLE orders (
    id       SERIAL PRIMARY KEY,
    zip_code VARCHAR(10),
    city     VARCHAR(100)  -- derived from zip_code; store it separately
);

-- ✅ Better — city lives with zip_code in its own table
CREATE TABLE zip_codes (
    zip_code VARCHAR(10) PRIMARY KEY,
    city     VARCHAR(100) NOT NULL,
    state    VARCHAR(50)  NOT NULL
);
```

The practical heuristic: **if you find yourself updating the same fact in multiple rows, you've probably violated normalization.** That's when bugs creep in.

> **Note:** Normalization is not always the final answer. At scale, *denormalization* — intentionally duplicating data to avoid expensive joins — is a valid performance strategy. But you denormalize consciously and deliberately, after profiling. The default should always be normalized.

---

## The Schema Review Checklist

Before shipping any new table or migration, run through this:

- [ ] Every column that should always have a value is `NOT NULL`
- [ ] Money is stored as integers (cents), not floats
- [ ] Timestamps use `TIMESTAMPTZ`, not `TIMESTAMP`
- [ ] Foreign keys are declared and indexed
- [ ] `CHECK` constraints exist for any column with business rules (positive quantity, valid status values, etc.)
- [ ] No repeated facts — each piece of information lives in exactly one place
- [ ] IDs use `BIGSERIAL` or `UUID`, not `SERIAL`
- [ ] Column names are consistent across the schema (`user_id` everywhere, not `userId` sometimes and `user_id` other times)

---

## Practice: Build This Schema

The best way to internalize this is to build and query it yourself. Spin up a local Postgres instance (or use the free tier at [neon.tech](https://neon.tech)) and:

1. Create the `users`, `orders`, and `order_items` tables from this post
2. Insert 5 users, give 3 of them 1-3 orders each, populate the order items
3. Write these queries without looking at examples:
   - All orders placed in the last 30 days, with the user's email
   - Total revenue per user, ordered highest to lowest, including users with zero orders
   - The single most expensive order item ever placed
   - Count of orders per status (`pending`, `shipped`, etc.)
4. Try inserting an order with a `user_id` that doesn't exist and observe the error

Deliberately breaking constraints teaches you what they actually protect against.

---

## What's Next

We've covered the foundation: the relational model, schema design principles, data types, keys, constraints, joins, and aggregations. This is the bedrock everything else builds on.

In **Part 2**, we go deeper into the internals that separate good engineers from great ones:

- **Indexes:** how B-trees work, when indexes help (and when they hurt), partial and composite indexes, the `EXPLAIN ANALYZE` command
- **Transactions & ACID:** what atomicity, consistency, isolation, and durability actually guarantee — and what they don't
- **Concurrency:** locks, deadlocks, isolation levels, and the bugs each level protects against
- **Query planning:** how Postgres decides to execute your query, and how to steer it when it gets it wrong

These are the topics that show up in senior engineering interviews and production incidents. Part 2 is where the real leverage is.

---

*This post is part of the **Database Engineering: From Good to Great** series. Part 2 covers indexes, query planning, and transactions. Part 3 covers advanced production patterns.*
----
