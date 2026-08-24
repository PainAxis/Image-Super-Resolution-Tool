# Security Policy

Please report suspected vulnerabilities privately through GitHub's
**Security advisories → Report a vulnerability** flow. Do not include secrets,
private images, or exploit details in a public issue.

The supported line is the current `master` branch. Reports should include the
commit, operating system, Python version, reproduction steps, and impact.

Image decoding is treated as untrusted input. The application enforces input,
output, and estimated-memory limits before large allocations; see the
environment overrides documented in the README when intentionally processing
larger files.
