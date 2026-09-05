"""Sustained agentic-style soak at concurrency 8, with a validated detector."""
import json, time, urllib.request, concurrent.futures as cf, uuid, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # detect.py lives beside this file
from detect import find_loop

URL = "http://localhost:8001/v1/chat/completions"
CODE = """
sh.fragmentShader = sh.fragmentShader.replace('#include <map_fragment>', `
  #include <map_fragment>
  #ifdef USE_INSTANCING_COLOR
    diffuseColor.rgb *= vInstanceColor;
  #endif
`);
const mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.7 });
""" * 60  # ~7k tokens of code context per request

TASKS = [
    "Reason through every possible cause of this GLSL compile error, step by step, rejecting each hypothesis until you find the root cause.",
    "Explain what preprocessor defines three.js injects for WebGL2 and why texture() vs texture2D() matters.",
    "Debug this shader pipeline and propose a concrete fix with code.",
    "Trace how the map_fragment chunk is assembled and where an override could break it.",
    "Analyse this material setup for correctness and list every risk.",
    "Write a detailed post-mortem of this shader failure.",
    "Compare GLSL ES 1.00 and 3.00 semantics as they apply to this code.",
    "Propose a minimal reproduction for this bug and justify each line.",
]


def one(i):
    q = f"[{uuid.uuid4()}] Project source:\n{CODE}\n\nTask: {TASKS[i % len(TASKS)]}"
    body = json.dumps({
        "model": "qwen38", "messages": [{"role": "user", "content": q}],
        "max_tokens": 1500, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
    }).encode()
    r = urllib.request.Request(URL, data=body,
                               headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    m = d["choices"][0]["message"]
    txt = (m.get("reasoning") or "") + " " + (m.get("content") or "")
    return txt, d["usage"]["completion_tokens"]


DURATION = 300
t0 = time.time()
rnd = 0
total = 0
loops = 0
while time.time() - t0 < DURATION:
    rnd += 1
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, range(8)))
    total += len(res)
    hits = []
    for j, (txt, tok) in enumerate(res, 1):
        reps, unit = find_loop(txt)
        if reps:
            loops += 1
            hits.append(f"req{j} reps={reps} unit={unit[:12]!r}")
    el = int(time.time() - t0)
    print(f"[{el:4d}s] round {rnd:2d}  reqs={total:3d}  loops={loops}"
          + ("  << " + "; ".join(hits) if hits else ""), flush=True)
print(f"\nSOAK DONE: {total} requests, {loops} degenerate loops")
