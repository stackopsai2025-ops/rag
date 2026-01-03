# Authentication

This system uses a simple token-based authentication flow.

## Overview
1. A user authenticates with a username and password.
2. If valid, the system issues a signed token.
3. Protected actions require token verification.

## Token verification
Token verification checks:
- signature validity (signed using `APP_SECRET`)
- expiry timestamp based on `TOKEN_TTL_SECONDS`

If verification fails, the request is rejected.
