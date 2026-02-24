# Login + Access Control (User vs Admin)

## Goal
Add a login so:
- **Logged-out** users cannot access the app.
- **Standard users** can access **only**:
  - Main game cards page (`/` and any required supporting endpoints)
  - Recommendations page (`/recommendations` and any required supporting endpoints)
- **Admins** have full access to all endpoints (optionally excluding ingest endpoints if those remain token-only).

This is intended to protect the UI and API routes that power it, while preserving an admin workflow for diagnostics/ops.

## Current State (Repo Observations)
- App is **Flask** in a single large module ([app.py](../app.py)).
- There is already a centralized `@app.before_request` gate pattern for **SNAPSHOT_ONLY** mode (`_render_mode_gate`) that uses an allowlist of paths.
- There is already a lightweight auth/token pattern for some ingest endpoints.

## Proposed Approaches

### Option A — “Fastest” (No User Backend)
**HTTP Basic Auth** (or a shared password) in `@app.before_request`.

- Effort: ~1–2 hours.
- Pros: no DB, no migrations, simplest to deploy.
- Cons: not per-user, no password resets, hard to manage multiple users, limited auditing.

Good if this is only to keep the site private for a small set of trusted people.

### Option B — Minimal Real Login + Roles (Recommended)
Session-based auth with a small user table.

- Effort: ~1–2 days for a clean minimal version.
- Building blocks:
  - DB (Render Postgres recommended; SQLite acceptable for local/dev only)
  - User table: `id`, `email/username`, `password_hash`, `role` (`user|admin`), `is_active`, timestamps
  - Session auth: `Flask-Login`
  - Password hashing: `werkzeug.security` (or `bcrypt`/`argon2`)
  - Routes: `/login`, `/logout`
  - Admin user creation: simplest is a CLI script/command (or manual DB insert)

#### Access Control Model
Implement a **single centralized** `before_request` gate (similar to snapshot gate) that:
- Always allows: `/login`, `/logout`, `/static/*`
- If not authenticated → redirect to `/login` (for HTML), return `401` (for `/api/*`)
- If authenticated and role == `user` → allowlist only “cards + recommendations” pages and the API endpoints they require.
- If authenticated and role == `admin` → allow everything.

Important: locking down just `/` and `/recommendations` is not enough; users also need the **API endpoints those pages call**.

### Option C — Full User System
Adds signup/invite flows, password reset email, MFA, rate limiting, audit logs, admin UI, etc.

- Effort: 1–2+ weeks depending on requirements.

## What Must Be Identified Before Implementation
1. **User creation model**: invite-only vs self-signup.
2. Public monitoring endpoints: should `/api/health` and `/api/status` remain public?
3. Ingest endpoints: should admins bypass ingest tokens, or should ingest remain token-only?

## Implementation Checklist (Option B)
- [ ] Add dependencies (`Flask-Login`, DB driver, ORM/migrations if used).
- [ ] Configure `SECRET_KEY` and session cookie settings.
- [ ] Create `users` table + migration.
- [ ] Add password hashing + login verification.
- [ ] Add `/login` (GET renders form, POST authenticates) and `/logout`.
- [ ] Add `before_request` gate enforcing:
  - logged-out → redirect/401
  - role-based allowlist for `user`
  - full access for `admin`
- [ ] Confirm which API endpoints are needed by cards/recommendations UI.
- [ ] Add a secure admin bootstrap path (env-provided initial admin or a one-time CLI command).
- [ ] Deploy with DB + environment variables.

## Notes / Risks
- If the app is deployed behind caches/CDN, ensure authenticated pages are `no-store` (you already set aggressive no-cache headers for non-static routes).
- Be explicit about cookie security on Render (e.g., `Secure`, `HttpOnly`, `SameSite`).

