# API Overview

This app exposes two core actions:

## Login
- Validates credentials
- Issues a token

## Protected endpoint
- Requires a token
- Verifies token signature and expiry before returning data
