"""JSON-mode (response_format json_object) failure reproducer for the PP+MTP server.
usage: repro157.py [concurrency=4] [requests=12] [json|plain]"""
import json, sys, time, uuid, urllib.request, urllib.error, concurrent.futures as cf
URL = "http://localhost:8001/v1/chat/completions"
C = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MODE = sys.argv[3] if len(sys.argv) > 3 else "json"
TOPICS = ["european capitals", "programming languages", "planets and moons", "famous bridges",
          "chemical elements", "board games", "musical instruments", "ocean currents"]
def one(i):
    body = {"model": "qwen38", "max_tokens": 700, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                f"[{uuid.uuid4()}] Return a JSON object with a key \"items\": a list of 12 objects about "
                f"{TOPICS[i % len(TOPICS)]}, each with \"title\", \"year\" (number), \"tags\" (list of strings) "
                f"and \"notes\" (one sentence)."}]}
    if MODE == "json":
        body["response_format"] = {"type": "json_object"}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=600).read())
        txt = d["choices"][0]["message"].get("content") or ""
        try: json.loads(txt); ok = "valid"
        except Exception: ok = "200-but-unparseable" if d["choices"][0]["finish_reason"] != "length" else "valid(truncated)"
        return 200, ok, d["usage"]["completion_tokens"]
    except urllib.error.HTTPError as e:
        return e.code, "http-error", 0
t0 = time.time()
with cf.ThreadPoolExecutor(C) as ex: res = list(ex.map(one, range(N)))
n500 = sum(1 for r in res if r[0] != 200)
kinds = {}
for r in res: kinds[r[1]] = kinds.get(r[1], 0) + 1
print(f"mode={MODE} concurrency={C} requests={N}: HTTP 500s = {n500}/{N}  {kinds}  tokens={sum(r[2] for r in res)}  wall={time.time()-t0:.0f}s")
