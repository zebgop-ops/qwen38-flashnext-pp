import sys, types, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("h", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class Ev:   # stand-in for the CUDA event
    def synchronize(self): pass
    def record(self): pass
def mk():
    h = object.__new__(m.DraftTokensHandler)
    h.copy_event = Ev(); h.req_ids = []; h.draft_tokens_np = None; h.num_draft_tokens = 0
    if hasattr(m.DraftTokensHandler, "_fold_pending"):
        h._latest = {}; h._pending = []
    return h
def set_batch(h, req_ids, rows, structured=True):   # what set_draft_tokens does, minus the GPU copy
    h.req_ids = req_ids; h.num_draft_tokens = 3
    h.draft_tokens_np = np.array(rows) if structured else None
    if structured and hasattr(h, "_pending"):
        h._pending.append((req_ids, h.draft_tokens_np))
        if len(h._pending) > 32: h._fold_pending()      # same cap as the real set_draft_tokens
NEW = hasattr(m.DraftTokensHandler, "_fold_pending")
h = mk()
# PP4, four JSON requests in four microbatch phases: sample(t-4)=R1, t-3=R2, t-2=R3, t-1=R4,
# then the scheduler asks for drafts to build R1's bitmask for step t.
for i, r in enumerate(["R1", "R2", "R3", "R4"]): set_batch(h, [r], [[100+i, 200+i, 300+i]])
d = h.get_draft_tokens(); got = dict(zip(d.req_ids, d.draft_token_ids))
print(("NEW" if NEW else "ORIG"), "handler -> drafts returned for:", sorted(got), "| R1's own drafts delivered:", got.get("R1") == [100, 200, 300])
if NEW:
    assert got == {"R1": [100,200,300], "R2": [101,201,301], "R3": [102,202,302], "R4": [103,203,303]}
    set_batch(h, ["R1"], [[7, 8, 9]]); assert dict(zip(*[(x.req_ids, x.draft_token_ids) for x in [h.get_draft_tokens()]][0]))["R1"] == [7, 8, 9]   # newer overwrites
    set_batch(h, ["R2"], [[5, 5, 5]]); h.discard("R2"); d = h.get_draft_tokens(); assert "R2" not in d.req_ids   # finished while its rows were pending
    h.discard("R3"); assert "R3" not in h.get_draft_tokens().req_ids
    set_batch(h, ["N1", "N2"], None, structured=False); d = h.get_draft_tokens(); g = dict(zip(d.req_ids, d.draft_token_ids))
    assert g["N1"] == [-1, -1, -1] and g["R1"] == [7, 8, 9]      # non-structured latest batch still gets placeholders
    h2 = mk(); set_batch(h2, ["A"], None, structured=False); d = h2.get_draft_tokens(); assert d.req_ids == ["A"] and d.draft_token_ids == [[-1,-1,-1]]   # legacy behaviour intact
    for k in range(40): set_batch(h, [f"X{k}"], [[k, k, k]])
    assert len(h._pending) <= 33; h.get_draft_tokens(); assert not h._pending
    # THE REGRESSION CASE: the runner discards a NEW request's id before it has drafts,
    # while older batches are still pending; its first drafts must survive.
    h3 = mk(); set_batch(h3, ["OLD"], [[1, 1, 1]])            # leftover unfolded batch
    h3.discard("NEW")                                          # add_requests -> _remove_request(NEW)
    set_batch(h3, ["NEW"], [[198, 220, 328]])                  # NEW's prefill step proposes drafts
    d = h3.get_draft_tokens(); g = dict(zip(d.req_ids, d.draft_token_ids))
    assert g.get("NEW") == [198, 220, 328], g
    h3.discard("OLD"); assert "OLD" not in h3.get_draft_tokens().req_ids
    print("NEW handler: all assertions passed (own drafts, overwrite, discard, discard-before-first-drafts, placeholder fallback, bounded pending)")
