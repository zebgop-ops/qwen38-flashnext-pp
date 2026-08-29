"""Degenerate-repetition detector, validated against benign formatting.

A real loop ("ductductduct...") is a SHORT unit, containing alphanumerics,
with NO newline, repeated many times consecutively. Benign repetition in
code/markdown is either whitespace, punctuation rules ("---"), or repeated
LINES (which contain a newline). Keying on those three properties separates
them; a naive longest-run metric does not.
"""


def find_loop(s, min_reps=8, max_unit=14):
    """Return (reps, unit) of the worst degenerate run, or (0, '')."""
    best_reps, best_unit = 0, ""
    n = len(s)
    for size in range(2, max_unit + 1):
        i = 0
        while i < n - size:
            unit = s[i:i + size]
            if "\n" in unit or not any(c.isalnum() for c in unit):
                i += 1
                continue
            reps = 1
            while s[i + reps * size:i + (reps + 1) * size] == unit:
                reps += 1
            if reps >= min_reps and reps > best_reps:
                best_reps, best_unit = reps, unit
            i += max(1, reps * size)
    return best_reps, best_unit


if __name__ == "__main__":
    cases = {
        "REAL ductduct loop": "the answer is " + "duct" * 120,
        "REAL single-char":   "hmm " + "a" * 200,
        "REAL word loop":     "so " + "the same " * 40,
        "benign indented code": "def f():\n" + "    x = 1\n" * 40,
        "benign md rule":     "text\n" + "-" * 80 + "\ntext",
        "benign prose":       "The model routes tokens to experts. " * 6,
        "benign numbered":    "".join(f"{i}. item\n" for i in range(40)),
        "benign json":        '{"a":1,"b":2,"c":3}' * 3,
    }
    for name, txt in cases.items():
        reps, unit = find_loop(txt)
        verdict = "LOOP" if reps else "clean"
        print(f"{name:22s} reps={reps:4d} unit={unit[:12]!r:16s} -> {verdict}")
