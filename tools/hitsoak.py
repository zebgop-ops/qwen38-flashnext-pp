"""Conversation-shaped HIT-PATH soak: multi-turn chats whose growing history is
re-sent every turn, so most prefill is served from the prefix cache (the path the
UUID soak deliberately defeats). Each conversation plants a code in turn 1 and is
asked to recall it in turns 3, 5, 7 -> 3 recall probes per conversation. Every
turn's output runs through the validated loop detector.

usage: hitsoak.py [concurrency=8] [conversations=16]
"""
import json, os, sys, time, uuid, urllib.request, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import find_loop

URL = "http://localhost:8001/v1/chat/completions"
METRICS = "http://localhost:8001/metrics"
W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NCONV = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TURNS = 8
RECALL_TURNS = {3, 5, 7}

FILLER = [
    "Explain how three.js assembles a ShaderMaterial from chunks, in detail.",
    "What are the trade-offs between instanced rendering and merged geometry?",
    "Walk through how a depth prepass interacts with transparent materials.",
    "Describe how WebGL2 uniform buffer objects change material design.",
    "List the ways a fragment shader can be made cheaper without visible loss.",
]
CODE_CTX = """
const geo = new THREE.InstancedBufferGeometry().copy(new THREE.BoxGeometry());
geo.instanceCount = N;
geo.setAttribute('instanceColor', new THREE.InstancedBufferAttribute(colors, 3));
const mat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.1 });
mat.onBeforeCompile = (sh) => { sh.vertexShader = sh.vertexShader.replace('#include <color_vertex>', '#include <color_vertex>\\n vInstanceColor = instanceColor;'); };
""" * 110  # ~12k tokens of shared context: align-mode mamba caching reuses whole 1600-token
           # blocks (and the MTP path drops the last complete one), so a prefix must span
           # several blocks before any hit registers


def metrics():
    out = {}
    try:
        for l in urllib.request.urlopen(METRICS, timeout=10).read().decode().splitlines():
            for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"):
                if l.startswith(k + "{"):
                    out[k] = out.get(k, 0.0) + float(l.split()[-1])
    except Exception:
        pass
    return out


def chat(messages, max_tokens):
    body = json.dumps({"model": "qwen38", "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.7, "top_p": 0.95}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    m = d["choices"][0]["message"]
    return (m.get("reasoning") or ""), (m.get("content") or ""), d["usage"]["completion_tokens"]


def conversation(i):
    code = f"{(i * 7919 + 10403) % 90000 + 10000}"
    msgs = [{"role": "user", "content":
             f"[conversation {uuid.uuid4()}] Project source:\n{CODE_CTX}\n\n"
             f"Remember this: the deployment vault code for this project is {code}. "
             f"Now, briefly explain what the instanceColor attribute does here."}]
    loops = recall_ok = recall_n = toks = 0
    for t in range(1, TURNS + 1):
        if t > 1:
            if t in RECALL_TURNS:
                q = "What was the deployment vault code I gave you at the start? Reply with the digits only."
            else:
                q = FILLER[(i + t) % len(FILLER)]
            msgs.append({"role": "user", "content": q})
        reasoning, content, n = chat(msgs, 120 if t in RECALL_TURNS else 400)
        toks += n
        reps, unit = find_loop(reasoning + " " + content)
        if reps:
            loops += 1
        if t in RECALL_TURNS:
            recall_n += 1
            recall_ok += int(code in content)
        msgs.append({"role": "assistant", "content": content})
    return loops, recall_ok, recall_n, toks


t0 = time.time()
m0 = metrics()
done = loops = ok = n = toks = 0
with cf.ThreadPoolExecutor(W) as ex:
    for l, r_ok, r_n, tk in ex.map(conversation, range(NCONV)):
        done += 1; loops += l; ok += r_ok; n += r_n; toks += tk
        print(f"[{int(time.time()-t0):5d}s] convs={done}/{NCONV} loops={loops} recall={ok}/{n} toks={toks}", flush=True)
m1 = metrics()
q = m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0)
h = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
hit = f"{h/q*100:.1f}%" if q else "n/a"
print(f"HITSOAK DONE: {NCONV} convs x {TURNS} turns, loops={loops}, recall={ok}/{n}, prefix-cache hit rate {hit} ({h:,.0f} of {q:,.0f} queried tokens)")
