"""The Norwegian drift patch: W <- W + strength * outer(push, key) in layer 16's MLP down_proj.

  capture  teacher-forced runs of the unedited model on the 310 FLORES+ build sentences, at the reply start and the
           first reply tokens: Norwegian and Danish translation requests, each with the reply forced to the Norwegian
           and to the Danish reference; Swedish and German requests; Norwegian->English and Danish->English requests
  build    push = mean state after layer 16, Norwegian request with the Norwegian reply minus the same request with
           the Danish reply. key = ridge reader on layer 16's neuron activations, 1 for the Norwegian requests (either
           reply) and 0 for every other request. Prints the key's mean score on the held-out fifth of each group, then
           writes models/norwegian_s<strength> for each strength.

usage: norwegian_patch.py capture | build [STRENGTH,...]       (the post's patch is strength 4, the default)
"""
import json
import os
import sys

import torch
from common import DATA, MODELS, RES, base_model, chat, flores, lang_name, load_base, load_model, ridge, rows_of, write_model

TARGET, NEIGHBOR, OTHERS, L = "nob_Latn", "dan_Latn", ["swe_Latn", "deu_Latn"], 16
KEY = f"model.language_model.layers.{L}.mlp.down_proj.weight"
ROWS = os.path.join(RES, "norwegian_rows.pt")


def capture():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model())
    build = json.load(open(os.path.join(DATA, "flores_dev_build.json")))["build"]
    eng, X, S = flores("eng_Latn"), flores(TARGET), flores(NEIGHBOR)
    tr = lambda lang, i: chat(tok, f"Translate the following text into {lang_name(lang)}.\n\n{eng[i]}")
    groups = {f"{req}:{o}": [(tr(req, i), out[i]) for i in build] for req in (TARGET, NEIGHBOR) for o, out in (("X", X), ("S", S))}
    for lang in OTHERS:
        out = flores(lang)
        groups[f"{lang}:own"] = [(tr(lang, i), out[i]) for i in build]
    for lang, src in ((TARGET, X), (NEIGHBOR, S)):
        groups[f"{lang}:x2en"] = [(chat(tok, f"Translate the following text into English.\n\n{src[i]}"), eng[i]) for i in build]
    m, lm = load_model()
    rows = {}
    for g, pairs in groups.items():
        rows[g] = rows_of(lm, tok, pairs, L)
        print("captured", g, flush=True)
    torch.save(rows, ROWS)


def build(strengths):
    rows = torch.load(ROWS)
    acts = lambda g: rows[g][0].float()
    R = lambda g: rows[g][1][:, 1:].float()  # reply tokens only
    u = R(f"{TARGET}:X").mean((0, 1)) - R(f"{TARGET}:S").mean((0, 1))
    pos = [f"{TARGET}:X", f"{TARGET}:S"]
    neg = [f"{NEIGHBOR}:X", f"{NEIGHBOR}:S"] + [f"{l}:own" for l in OTHERS] + [f"{TARGET}:x2en", f"{NEIGHBOR}:x2en"]
    n = rows[pos[0]][0].shape[0]
    k = int(0.8 * n)
    flat = lambda gs, sl: torch.cat([acts(g)[sl].reshape(-1, acts(g).shape[-1]) for g in gs])

    def fit(sl):
        Xp, Xn = flat(pos, sl), flat(neg, sl)
        y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))])
        return ridge(torch.cat([Xp, Xn]), y, lam=1e-3)

    w_ho = fit(slice(0, k))
    print(f"|push| {float(u.norm()):.3f}; key score on the held-out fifth (1 = Norwegian request):")
    for g in pos + neg:
        print(f"  {g:14} {float((acts(g)[k:] @ w_ho).mean()):+.2f}")
    w = fit(slice(0, n))
    W = load_base()[KEY].float()
    for s in strengths:
        out = os.path.join(MODELS, f"norwegian_s{s:g}")
        rel = write_model(out, {KEY: W + s * torch.outer(u, w)}, {"source": f"norwegian_patch.py strength {s:g}"})
        print(out, rel)


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture()
    else:
        build([float(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "4").split(",")])
