"""Interactive web UI.

A tiny standard-library HTTP server that serves the static report bundle and
lets you run a new scenario from the browser.  It is intentionally dependency
free (``http.server`` + JSON), because the tool must run on a plain Python 3
install.

Endpoints
---------
``GET  /``                 the report page from the loaded bundle
``GET  /api/bundle``       the current results bundle (JSON)
``POST /api/run``          run one scenario; body
                           ``{"service": "...", "traffic": "...",
                             "objective": {...}, "day_hours": 6,
                             "no_tuning": false}`` -> scenario dict
``GET  /api/services``     available services, traffic curves, defaults

The server keeps results in memory and writes updated bundles under
``reports/`` so they survive a restart.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import __version__

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"bundle": None, "results_path": None}


def _load(results: str) -> dict:
    if _STATE["bundle"] is None:
        if results and os.path.exists(results):
            with open(results) as f:
                _STATE["bundle"] = json.load(f)
        else:
            _STATE["bundle"] = {"generated": "", "scenarios": [], "charts": {}}
        _STATE["results_path"] = results
    return _STATE["bundle"]


def _scenario_options() -> dict:
    from .config import ServiceProfile
    services = [n for n in dir(ServiceProfile)
                if not n.startswith("_") and callable(getattr(ServiceProfile, n))
                and n in ("trading_front_end", "session_heavy", "cache_heavy",
                          "short_lived")]
    traffic = ["trading_day", "batch_heavy", "flat"]
    return {
        "services": services,
        "traffic": traffic,
        "default_young_mb": 512,
        "version": __version__,
    }


def _run_scenario(req: dict) -> dict:
    from .acceptance import scenario, _scenario_dict
    from .tuner import Objective
    obj = req.get("objective") or {}
    objective = Objective(
        max_pause_ms=float(obj.get("max_pause_ms", 50.0)),
        min_throughput=float(obj.get("min_throughput", 0.99)),
    )
    res = scenario(
        service_name=req.get("service", "trading_front_end"),
        traffic_name=req.get("traffic", "trading_day"),
        objective=objective,
        seed=int(req.get("seed", 12345)),
        cohort_seconds=float(req.get("cohort_seconds", 1.0)),
        outdir=req.get("outdir", "reports"),
        workers=req.get("workers"),
        do_calibration=bool(req.get("calibration", True)),
        day_hours=req.get("day_hours"),
    )
    return _scenario_dict(res)


def _html_index() -> str:
    from .page import render_page
    bundle = _STATE["bundle"] or {"scenarios": [], "charts": {}}
    page = render_page(bundle)
    # inject a small control panel before </main>
    controls = """
<div class="panel"><h3>Run a scenario</h3>
<div class="row">
  <label>service <select id="svc"></select></label>
  <label>traffic <select id="trf"></select></label>
  <label>day hours <input id="hrs" type="number" value="6" min="0.5" step="0.5"></label>
  <label>max pause ms <input id="mp" type="number" value="50"></label>
  <label>min throughput <input id="thr" type="number" value="0.99" step="0.001"></label>
  <button id="go">Run</button>
</div>
<p id="status" class="note"></p>
<pre id="err" class="bad"></pre>
</div>
<style>
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:end}
.row label{display:flex;flex-direction:column;font-size:12px;color:var(--mut)}
.row input,.row select{background:#0e1117;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 8px;min-width:120px}
button{background:var(--acc);color:#04121f;border:0;border-radius:6px;
padding:8px 16px;font-weight:600;cursor:pointer}
</style>
<script>
(async function(){
  const opt=await (await fetch('/api/services')).json();
  const svc=document.getElementById('svc'), trf=document.getElementById('trf');
  for(const s of opt.services) svc.add(new Option(s,s));
  for(const t of opt.traffic) trf.add(new Option(t,t));
  document.getElementById('go').onclick=async()=>{
    const status=document.getElementById('status'), err=document.getElementById('err');
    err.textContent=''; status.textContent='running…';
    const body={service:svc.value,traffic:trf.value,
      day_hours:parseFloat(document.getElementById('hrs').value),
      objective:{max_pause_ms:parseFloat(document.getElementById('mp').value),
                 min_throughput:parseFloat(document.getElementById('thr').value)}};
    try{
      const r=await fetch('/api/run',{method:'POST',body:JSON.stringify(body)});
      if(!r.ok) throw new Error(await r.text());
      const s=await r.json();
      status.textContent='done; reloading…'; setTimeout(()=>location.reload(),300);
    }catch(e){status.textContent='failed'; err.textContent=e.message;}
  };
})();
</script>
"""
    return page.replace("</main>", controls + "</main>")


class Handler(BaseHTTPRequestHandler):
    server_version = f"gcsim/{__version__}"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send(200, _html_index().encode(), "text/html; charset=utf-8")
            elif self.path == "/api/bundle":
                self._json(200, _STATE["bundle"])
            elif self.path == "/api/services":
                self._json(200, _scenario_options())
            else:
                self._send(404, b"not found", "text/plain")
        except Exception:
            self._json(500, {"error": traceback.format_exc()})

    def do_POST(self):
        if self.path != "/api/run":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(400, {"error": f"bad request: {e}"})
            return
        try:
            with _LOCK:
                sc = _run_scenario(req)
                bundle = _STATE["bundle"] or {"scenarios": [], "charts": {}}
                # replace same service/traffic or append
                bundle["scenarios"] = [
                    s for s in bundle["scenarios"]
                    if not (s["service"] == sc["service"]
                            and s["traffic"] == sc["traffic"])]
                bundle["scenarios"].append(sc)
                bundle["charts"].update(sc.get("charts", {}))
                _STATE["bundle"] = bundle
                rp = _STATE.get("results_path")
                if rp:
                    try:
                        from .report import write_json
                        write_json(rp, bundle)
                    except Exception:
                        pass
            self._json(200, sc)
        except Exception:
            self._json(500, {"error": traceback.format_exc()})


def serve(host: str = "127.0.0.1", port: int = 8000,
          results: Optional[str] = "reports/results.json") -> None:
    _load(results)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"gcsim web UI on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
