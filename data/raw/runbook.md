# Service X Restart Runbook

## Purpose
This runbook describes how to safely restart Service X, which handles payment authorization, during incidents such as high latency, elevated error rates, or service outages.

---

## When to Use This Runbook
Use this runbook if any of the following conditions are observed:
- Sustained high latency (p95 > 3 seconds)
- Elevated 5xx error rates
- Alert: SERVICE_X_DOWN
- Alert: SERVICE_X_HIGH_LATENCY
- Out-of-memory (OOM) events

---

## Preconditions
Before restarting the service:
- Confirm production access is available
- Verify the affected service is Service X
- Check for active incidents or change freezes

---
# Service X Restart Runbook

## Restart Steps (Single Block)

To restart Service X during incidents such as high latency, elevated error rates, or service outages, perform the following steps in order:

1. SSH into the production host running Service X:
   `ssh prod-service-x-01`

2. Stop the Service X process using systemd:
   `systemctl stop service-x`

3. Verify that the service has stopped successfully:
   `systemctl status service-x`

4. Wait 10 seconds to allow open connections to close, memory to be released, and dependent services to stabilize.

5. Start Service X again:
   `systemctl start service-x`

6. Verify that the service is running:
   `systemctl status service-x`

7. Perform a health check to confirm service availability:
   `curl http://localhost:8080/health`

8. Confirm the expected healthy response:
   `{"status":"ok"}`

9. Monitor the service for at least 10 minutes to ensure:
   - Latency returns to baseline
   - Error rates normalize
   - No new alerts are triggered
   - Memory usage remains stable

If the service fails to start or issues persist after restart, redeploy the previous stable version and escalate to the Payments SRE team.

## Escalation
- Slack: #payments-oncall
- Email: payments-sre@company.com
- Incident Commander: On-call SRE

## Ownership
- Service Owner: Payments Team
- Runbook Maintainer: Payments SRE
- Last Updated: 2023-07-20
