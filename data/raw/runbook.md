# Service X Restart Runbook

## Overview
Service X is a core backend service handling payment authorization.

## Symptoms
- 5xx errors
- High latency
- Alerts: SERVICE_X_DOWN

## Steps to Restart

1. SSH into production server
2. Run:
   systemctl stop service-x
3. Wait 10 seconds
4. Run:
   systemctl start service-x
5. Verify health:
   curl http://localhost:8080/health

## Rollback
If restart fails, redeploy previous stable version.

## Owner
payments-sre@company.com
