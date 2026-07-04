# Security Policy

## Supported Versions

Only the latest release on [PyPI](https://pypi.org/project/memorylens-bench/) receives security fixes.

## Reporting a Vulnerability

Please **do not open a public issue** for security problems.

Report privately via [GitHub Security Advisories](https://github.com/Neal006/memorylens/security/advisories/new)
or email builtbyneal@gmail.com.

You can expect an acknowledgement within 72 hours and a fix or mitigation plan within 14 days
for confirmed issues.

## Scope notes

MemoryLens is a local benchmark tool. The most security-relevant surfaces are:

- The optional FastAPI server (`memorylens.api`) — intended for local use, ships with no auth.
  Do not expose it to the public internet without a reverse proxy handling auth and rate limits.
- LLM provider keys read from `.env` — never committed, never logged.
