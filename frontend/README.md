# Frontend

This directory contains the Next.js product frontend for the laptop price prediction system.

It provides:

- a guided multi-step estimation flow
- server-side proxy routes to the FastAPI backend
- backend API key protection without exposing secrets in the browser
- a standalone production build suitable for container deployment

## Local Development

Install dependencies:

```bash
npm ci
```

Create a local environment file:

```bash
cp .env.example .env.local
```

Start the app:

```bash
npm run dev
```

The app runs on `http://127.0.0.1:3000` by default.

## Required Environment Variables

- `API_BASE_URL`: FastAPI base URL, for example `http://127.0.0.1:8000`
- `API_KEY`: API key forwarded by the server-side proxy when the backend is protected

## Production Build

Build the app:

```bash
npm run build
```

Start the production server:

```bash
npm run start
```

This project is configured with `output: "standalone"`, so the Dockerfile can run the generated standalone server.

## Docker

Build the frontend image from the repository root:

```bash
docker build -f frontend/Dockerfile -t laptop-price-frontend ./frontend
```

Run it:

```bash
docker run --rm -p 3000:3000 \
	-e API_BASE_URL=http://host.docker.internal:8000 \
	-e API_KEY=replace-with-a-secret \
	laptop-price-frontend
```

## Notes

- API requests are proxied through `src/app/api/*` routes.
- The frontend supports both versioned and unversioned backend routes.
- If local development becomes unstable because of stale cache state, remove `.next/` and rebuild.
