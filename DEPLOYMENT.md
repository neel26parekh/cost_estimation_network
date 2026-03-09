# Deployment Handoff

This repository is code-complete and locally validated.

## What Is Already Done

- FastAPI backend is implemented and tested.
- Streamlit app is wired to the same production model artifacts.
- Next.js frontend is implemented, linted, and production-buildable.
- Dockerfiles exist for the API and frontend.
- Render blueprint exists in `render.yaml`.
- GitHub Actions workflows exist for CI and deploy triggering.

## Local Release Gate

Run this before pushing or cutting a release:

```bash
make release-check
```

It executes:

- `make preflight`
- `make test`
- `make frontend-lint`
- `make frontend-build`

## External Steps Still Required

These steps cannot be completed from this local sandbox because they require access to your GitHub and Render accounts.

### GitHub Secrets

Add these repository secrets:

- `RENDER_API_DEPLOY_HOOK_URL`
- `RENDER_FRONTEND_DEPLOY_HOOK_URL`

Optional if you later expand automation:

- `RENDER_API_KEY`

### Render Services

Create or sync services from `render.yaml`:

- `cost-estimation-api`
- `cost-estimation-frontend`
- `cost-estimation-streamlit`

### Render Environment Variables

Set these on the backend service:

- `API_KEY`
- `API_HOST=0.0.0.0`
- `RATE_LIMIT_REQUESTS=60`
- `RATE_LIMIT_WINDOW_SECONDS=60`
- `RECENT_PREDICTIONS_LIMIT=20`
- `MIN_DRIFT_ALERT_SAMPLE_SIZE=25`

Set these on the frontend service:

- `API_BASE_URL=https://cost-estimation-api.onrender.com`
- `API_KEY=<same backend API key>`

Set this on the Streamlit service if needed:

- `PYTHONPATH=src`

## Deploy Flow After Secrets Exist

1. Push the repository to GitHub.
2. Ensure the default branch is `main`.
3. Trigger the `deploy` workflow manually, or push to `main`.
4. GitHub Actions builds both images and calls the Render deploy hooks.

## Post-Deploy Smoke Checks

After deployment, verify:

1. `GET /health` returns `200` on the API.
2. `GET /ready` returns `200` on the API.
3. Frontend home page loads.
4. A prediction request completes through the frontend.
5. `GET /monitoring/summary` returns `200` with the configured API key.

## Current Limitation In This Environment

Local Docker validation was not run here because Docker is not available in the current machine environment.