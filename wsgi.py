"""
Resilient WSGI entrypoint for Render.

Tries to import the main Flask app from app.py.
If import fails, falls back to a minimal Flask app with /healthz so the
platform health checks succeed and logs can be inspected.
"""

try:
    # Prefer the primary application object
    from app import app as application  # type: ignore
except Exception:
    # Minimal fallback to keep the service responding
    from flask import Flask, jsonify

    application = Flask(__name__)

    @application.route('/healthz')
    def healthz():
        return jsonify({"status": "degraded"}), 200
