# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | Yes |
| < 0.3   | No (please upgrade) |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately, either through GitHub's
[private vulnerability reporting](https://github.com/Srinivas26k/srvdb/security/advisories/new)
or by email to **srinivasvarma764@gmail.com**. Include:

- the SrvDB version, OS and Python or Rust version,
- what an attacker can do and how (a minimal reproduction, or a crafted
  database file, is ideal),
- any suggested fix.

You will get an acknowledgement within 3 working days and a status update
at least weekly until the issue is resolved. Fixes are released as a patch
version and credited in the changelog unless you prefer to stay anonymous.

## Scope

SrvDB is a library that runs inside your application with your
application's permissions. In scope: memory-safety bugs, panics or
crashes from crafted input or database files, data corruption, and
problems in SrvDB's dependencies or release pipeline. Out of scope:
attacks that require write access to the database directory you already
grant SrvDB, unless they escalate beyond it.
