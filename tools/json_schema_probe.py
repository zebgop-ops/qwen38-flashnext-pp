"""Strict json_schema probe with streaming, so a killed request shows how far it got.
usage: schema_probe.py <concurrency> <requests> <short|long>"""
import json, sys, time, uuid, urllib.request, concurrent.futures as cf
URL = "http://localhost:8001/v1/chat/completions"
C, N, KIND = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
CITY = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
        "landmarks": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}},
        "required": ["city", "population", "landmarks"], "additionalProperties": False}
SCHEMA = CITY if KIND == "short" else {"type": "object", "properties": {"cities": {"type": "array", "items": CITY, "minItems": 12, "maxItems": 12}},
                                        "required": ["cities"], "additionalProperties": False}
ASK = "one city" if KIND == "short" else "12 different cities under the key cities"
def one(i):
    body = {"model": "qwen38", "max_tokens": 2500, "temperature": 0.7, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA, "strict": True}},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4()}] Describe {ASK} as JSON with city, population (integer), landmarks (3 strings)."}]}
    txt, chunks, fin, err = "", 0, None, None
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if "error" in d: err = str(d["error"])[:80]; break
            ch = d["choices"][0]; delta = ch.get("delta", {}).get("content") or ""
            if delta: txt += delta; chunks += 1
            fin = ch.get("finish_reason") or fin
    except Exception as e: err = f"{type(e).__name__}: {e}"[:80]
    ok = False
    if not err and fin == "stop":
        try: json.loads(txt); ok = True
        except Exception: err = "unparseable"
    return ok, fin, err, chunks, txt
t0 = time.time()
with cf.ThreadPoolExecutor(C) as ex: res = list(ex.map(one, range(N)))
bad = [r for r in res if not r[0]]
print(f"{KIND} schema, {C}-way, {N} requests: {N-len(bad)}/{N} ok, wall {time.time()-t0:.0f}s")
for ok, fin, err, chunks, txt in bad[:6]:
    print(f"   FAIL fin={fin} err={err} after {chunks} chunks / {len(txt)} chars; tail={txt[-70:]!r}")
