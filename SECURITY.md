# Security policy

Report vulnerabilities through the repository's private GitHub security reporting channel when available.
If that channel is unavailable, contact the maintainer through the GitHub profile.
Do not publish credentials, personal records, or exploit details in public issues.

Keep tokens and input records outside the checkout.
Tests use synthetic records and mocked external services.
Review file output and API write commands before execution.
Dependency updates require review and passing checks.

## Runtime dependency hold

The 2026.10.05 runtime scan reported advisory records in five packages.
The legacy enhancement dependency pins prevent a routine upgrade of that graph.
This branch remains held for dependency review.
See [the dependency scan](security/dependency-audit.json) for package versions, advisory IDs, and scan limits.
The portable unit-test environment does not install the enhancement dependency graph.
