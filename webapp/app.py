"""Small local HTTP application for the STATRL interactive playground."""

from __future__ import annotations

import json
import sys
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from webapp.catalog import Catalog, constructor_signature, import_symbol
from webapp.core import InteractiveSession
from webapp.adapters.bandit import BanditAdapter
from webapp.custom_environment import CustomEnvironment


ROOT = Path(__file__).parent
STATIC = ROOT / "static"
#SETTINGS_ROOT = Path(__file__).resolve().parents[1] / "src/statrl/settings"
PROJECT_ROOT = ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SETTINGS_ROOT = SRC_ROOT / "statrl" / "settings"

class Playground:
    def __init__(self):
        print(f"PATH: {SETTINGS_ROOT}")
        self.catalog = Catalog(SETTINGS_ROOT)
        self.sessions = {}

    def create_session(self, payload):
        setting = payload["setting"]
        env_spec = payload["environment"]

        entries = self.catalog.environments(setting)
        entry = next((x for x in entries if str(x["id"]) == str(env_spec)), None)
        if entry is None:
            raise ValueError(f"Unknown environment: {env_spec}")

        # Use STATRL's native environment catalogue and constructor.
        # This keeps the playground independent of the YAML schema.
        env = self.catalog.make_environment(setting, str(env_spec))

        # First implementation: bandit adapter. Other settings get their own
        # adapter without changing the browser/session protocol.
        adapter = BanditAdapter()
        session = InteractiveSession(env, adapter)

        token = str(len(self.sessions) + 1)
        self.sessions[token] = session
        return token, self._session_json(session, entry)

    def attach_learner(self, token, payload):
        session = self.sessions[token]
        spec = payload["learner"]
        params = payload.get("parameters") or {}

        entries = self.catalog.learners(payload["setting"])
        entry = next((x for x in entries if str(x["id"]) == str(spec)), None)
        if entry is None:
            raise ValueError(f"Unknown learner: {spec}")

        factory_spec = entry.get("factory") or entry.get("class")
        if not factory_spec:
            raise ValueError(
                "Learner entry has no `factory`/`class` field."
            )

        factory = import_symbol(factory_spec)
        signature = constructor_signature(factory)
        if "nbArms" in signature and "nbArms" not in params:
            number_arms = getattr(session.environment, "number_arms", None)
            if number_arms is None:
                raise ValueError(
                    f"Learner {spec!r} requires `nbArms`, but the selected "
                    "environment does not expose `number_arms`."
                )
            params["nbArms"] = number_arms

        session.attach_learner(factory, params)
        return self._session_json(session, None)

    def recommend(self, token):
        session = self.sessions[token]
        return {"action": session.recommended_action()}

    def step(self, token, payload):
        session = self.sessions[token]
        action = payload["action"]
        outcome = None
        if payload.get("custom", False):
            outcome = (
                payload.get("next_state"),
                payload.get("reward"),
                bool(payload.get("done", False)),
                payload.get("info") or {},
            )
        transition = session.step(action, outcome)
        return {
            "transition": transition.__dict__,
            "session": session.snapshot(),
        }

    @staticmethod
    def _session_json(session, entry):
        return {
            "environment": entry,
            "session": session.snapshot(),
        }


PLAYGROUND = Playground()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, value):
        body = json.dumps(value, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/settings":
                return self._json(200, PLAYGROUND.catalog.settings())
            if path == "/api/settings/tree":
                return self._json(200, PLAYGROUND.catalog.settings_tree())
            if path.startswith("/api/environments/"):
                setting = path.split("/", 3)[3]
                return self._json(200, PLAYGROUND.catalog.environments(setting))
            if path.startswith("/api/learners/"):
                setting = path.split("/", 3)[3]
                return self._json(200, PLAYGROUND.catalog.learners(setting))

            if path in ("/", "/index.html"):
                body = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self._json(404, {"error": "Not found"})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._body()

            if path == "/api/session":
                token, data = PLAYGROUND.create_session(payload)
                data["token"] = token
                return self._json(200, data)

            if path.startswith("/api/session/") and path.endswith("/learner"):
                token = path.split("/")[3]
                return self._json(
                    200,
                    PLAYGROUND.attach_learner(token, payload),
                )

            if path.startswith("/api/session/") and path.endswith("/step"):
                token = path.split("/")[3]
                return self._json(200, PLAYGROUND.step(token, payload))

            if path.startswith("/api/session/") and path.endswith("/recommend"):
                token = path.split("/")[3]
                return self._json(200, PLAYGROUND.recommend(token))

            self._json(404, {"error": "Not found"})
        except Exception as exc:
            self._json(400, {"error": str(exc)})


def main(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"STATRL playground: http://{host}:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
