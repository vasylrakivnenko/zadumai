"""contract_tool as a local HTTP service for an agent harness (2026-10-05, LAB pilot): POST JSON {docs_dir, action, query,
document, against, offset} -> {"text": the tool's answer}. One ContractTool per documents folder, kept in memory; the
embedding model loads once. Bound to 127.0.0.1 only.
usage: dspy_venv/bin/python -m contract_tool.server [PORT] [--warm DOCS_DIR ...]   (default port 18090)"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .tool import ContractTool, model

_TOOLS, _LOCK = {}, threading.Lock()


def tool_for(docs_dir):
    with _LOCK:
        if docs_dir not in _TOOLS: _TOOLS[docs_dir] = (threading.Lock(), None)
        lock, _ = _TOOLS[docs_dir]
    with lock:  # build once per folder, even under concurrent calls
        t = _TOOLS[docs_dir][1]
        if t is None:
            t = ContractTool(docs_dir); t.emb(); _TOOLS[docs_dir] = (lock, t)
        return t


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        t0 = time.time()
        try:
            a = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = tool_for(a["docs_dir"]).call(a.get("action") or "", a.get("query") or "", a.get("document") or "", a.get("against") or "", a.get("offset") or 0)
        except Exception as e:
            text = f"Error: contract_tool failed: {type(e).__name__}: {e}"
        body = json.dumps({"text": text}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
        print(f"{time.strftime('%H:%M:%S')} {a.get('action')} {str(a.get('query'))[:60]!r} {a.get('document')} -> {len(text)} chars, {time.time() - t0:.2f}s", flush=True)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    import torch; torch.set_num_threads(4)  # leave cores for the agents' sandboxes and the live router
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 18090
    model()
    if "--warm" in sys.argv:
        for d in sys.argv[sys.argv.index("--warm") + 1:]:
            t0 = time.time(); tool_for(d); print(f"warm {d}: {time.time() - t0:.0f}s", flush=True)
    print(f"contract_tool server on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
