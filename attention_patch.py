"""The attention patch: ten weight rows in layers 7 and 11 that make the start of the reply read the system prompt.

In a full-attention layer, one head dimension d that carries no position code is rewritten. The key row of d reads
"this token is system-prompt content" (ridge reader r on the layer input), the query rows of d read "this position is
the reply start" (ridge reader t, at the nine chat-template positions before the reply). After the query and key norms
their product adds about `strength` to the attention scores from the reply start to the system prompt, and about
nothing elsewhere. The four query heads that share a key head all lose dimension d: the chosen heads get the reader,
the others a zero row, so their old query does not multiply the new key.

  capture        layer inputs of the six full-attention layers on the fit prompts of groups(): Gram matrices per
                 position class for the readers, and a sample of activations for the checks and the row scales
  fit            the readers (ridge strength chosen on wordings and languages left out of the Grams), the dimension
                 per layer and the calibration samples -> results/attention_fit.pt
  write [NAME [SPEC]]   the patch as a checkpoint models/NAME (default system_prompt, SPEC 7:2:4:t,11:457:4:t)

SPEC: comma-separated layer:heads:strength:reader (heads as digits; reader t = template positions, tr = template and
reply positions).
Position classes: 0 system-prompt content, 1 role frames, 2 user content, 3-11 the nine template positions
(<|im_end|> ... </think>\\n\\n), 12 teacher-forced reply tokens.
"""
import json
import os
import random
import sys

import torch
from common import (DATA, MODELS, RES, base_model, chat, device, flores, load_base, load_belebele, load_model,
                    write_model)
from drift_program import BUILD_TOPICS, UNIVERSAL, name

FULL = [3, 7, 11, 15, 19, 23]  # the full-attention layers
NT, NH, NREP, NCLS, HD, PASS = 9, 8, 8, 13, 256, 64  # PASS: the first 64 dims of a head carry the rotary position code
PROBE = range(20, 30)  # passages of the probing questions, kept out of the fit
LAMS = (1e-4, 1e-3, 1e-2, 1e-1)
TEMPLATE, REPLY = list(range(3, 3 + NT)), 12
QPOS = {"t": TEMPLATE, "tr": TEMPLATE + [REPLY]}
SPEC = "7:2:4:t,11:457:4:t"
GRAMS, ACTS, FIT = (os.path.join(RES, f"attention_{x}.pt") for x in ("grams", "acts", "fit"))
ENG = "eng_Latn"
LANGS = {"eng_Latn": "English", "uzn_Latn": "Uzbek", "khk_Cyrl": "Mongolian", "tgk_Cyrl": "Tajik",
         "ukr_Cyrl": "Ukrainian", "jav_Latn": "Javanese", "sun_Latn": "Sundanese", "cat_Latn": "Catalan",
         "bul_Cyrl": "Bulgarian", "ind_Latn": "Indonesian", "spa_Latn": "Spanish", "ita_Latn": "Italian",
         "kaz_Cyrl": "Kazakh", "hye_Armn": "Armenian", "kat_Geor": "Georgian", "als_Latn": "Albanian",
         "ell_Grek": "Greek", "kir_Cyrl": "Kyrgyz", "azj_Latn": "Azerbaijani", "nob_Latn": "Norwegian Bokmål",
         "mkd_Cyrl": "Macedonian", "rus_Cyrl": "Russian", "dan_Latn": "Danish", "deu_Latn": "German",
         "tur_Latn": "Turkish", "ron_Latn": "Romanian"}
WORDINGS = {"sysW1": "Always respond in {}.",
            "sysW2": "You are an assistant that replies only in {}, even when the user writes in another language.",
            "sysW3": "Language: {}",
            "sysW4": "You are a helpful assistant. The user's preferred language is {}."}
LANG8 = ["deu_Latn", "rus_Cyrl", "tur_Latn", "kat_Geor", "hye_Armn", "kaz_Cyrl", "ita_Latn", "ind_Latn"]
FMT = {"short": "Answer in one short sentence.",
       "caps": "Write your whole answer in capital letters.",
       "json": 'Reply only with a JSON object of the form {"answer": "..."}.',
       "prefix": 'Begin your answer with the word "Certainly".',
       "nomd": "Do not use any Markdown formatting: no bold text, no headings and no bullet lists.",
       "bullets3": "Answer with at most three short bullet points and nothing else.",
       "long": "Give a long, detailed answer of several paragraphs.",
       "helpful": "You are a helpful assistant."}
BUILD_W = {"B2": "Your answers must be written in {}.", "B3": "Output language = {}",
           "B4": "The person you are helping reads only {}."}
NATIVE = {"hye_Armn": "Թարգմանիր հետևյալ տեքստը հայերեն։", "kat_Geor": "თარგმნე შემდეგი ტექსტი ქართულად.",
          "kaz_Cyrl": "Келесі мәтінді қазақ тіліне аударыңыз.", "kir_Cyrl": "Төмөнкү текстти кыргыз тилине которуңуз.",
          "azj_Latn": "Aşağıdakı mətni Azərbaycan dilinə tərcümə edin.",
          "uzn_Latn": "Quyidagi matnni o'zbek tiliga tarjima qiling.",
          "tgk_Cyrl": "Матни зеринро ба забони тоҷикӣ тарҷума кунед.",
          "khk_Cyrl": "Дараах текстийг монгол хэл рүү орчуулна уу.", "als_Latn": "Përktheni tekstin e mëposhtëm në shqip.",
          "mkd_Cyrl": "Преведи го следниов текст на македонски јазик.", "nob_Latn": "Oversett følgende tekst til norsk bokmål.",
          "rus_Cyrl": "Переведите следующий текст на русский язык.", "tur_Latn": "Aşağıdaki metni Türkçeye çevirin."}
att = lambda L: f"model.language_model.layers.{L}.self_attn."


# ---------------------------------------------------------------- fit prompts

def discovery_rows():
    """English Belebele probing rows outside the 128 tuning passages."""
    tune = set(json.load(open(os.path.join(DATA, "flores_dev_build.json")))["tune_links"])
    return [r for r in load_belebele(ENG, "discovery") if r["link"] not in tune]


def qa_rows():
    seen, rows = set(), []
    for r in discovery_rows():
        if r["link"] not in seen:
            seen.add(r["link"])
            rows.append(r)
    return rows[:50]


def probe_passages():
    """The 40 probing passages with two or more questions; PROBE of them hold the probing questions."""
    by_link = {}
    for r in discovery_rows():
        by_link.setdefault(r["link"], []).append(r)
    return [v[0] for v in by_link.values() if len(v) >= 2][:40]


def groups(tok):
    """{group: (kind, [(prompt, teacher-forced reply)])}; kind S = system "Always respond in X.", B = system prompt
    in another wording, O = other, X = left out of the Grams (sets the ridge strength)."""
    rows = qa_rows()
    users = ([f"Read the passage below and answer the question.\n\nPassage: {r['flores_passage']}\n\n"
              f"Question: {r['question']}" for r in rows] + [f"Explain {t}." for t in BUILD_TOPICS])
    bidx = json.load(open(os.path.join(DATA, "flores_dev_build.json")))["build"]
    text = {}

    def reply(lang, i):
        if lang not in text:
            text[lang] = flores(lang)
        return text[lang][bidx[i % len(bidx)]]

    chunks = lambda lang: (lambda s: [" ".join(s[i] for i in bidx[3 * k:3 * k + 3]) for k in range(30)])(flores(lang))
    chat_langs = [l for l in LANGS if l != ENG]
    nat = {l: {r["key"]: r for r in load_belebele(l, "discovery")} for l in chat_langs}
    natq = lambda l, r: f"{nat[l][r['key']]['flores_passage']}\n\n{nat[l][r['key']]['question']}"
    W1 = WORDINGS["sysW1"]
    G = {}
    for l in LANG8 + [ENG]:
        w = W1.format(LANGS[l])
        G[f"S_{l}"] = ("S", [(chat(tok, u, w), reply(l if i % 2 == 0 else ENG, i)) for i, u in enumerate(users)])
        G[f"U_{l}"] = ("O", [(chat(tok, f"{u}\n\n{w}"), reply(l, i)) for i, u in enumerate(users)])
    G["plain"] = ("O", [(chat(tok, u), reply(ENG, i)) for i, u in enumerate(users)])
    for k, l in enumerate(chat_langs):
        nm, sub = LANGS[l], [rows[(3 * k + i) % len(rows)] for i in range(30)]
        G[f"V_{l}"] = ("O", [(chat(tok, f"Read the passage below and answer the question. Write your answer in {nm}."
                                        f"\n\nPassage: {r['flores_passage']}\n\nQuestion: {r['question']}"), reply(l, i))
                             for i, r in enumerate(sub[:20])]
                       + [(chat(tok, f"Explain {t}. Write your answer in {nm}."), reply(l, i))
                          for i, t in enumerate(BUILD_TOPICS[k % 3::3])])
        G[f"nat_{l}"] = ("O", [(chat(tok, natq(l, r)), reply(l, i)) for i, r in enumerate(sub)])
    for k, (key, ins) in enumerate(FMT.items()):
        ls = [chat_langs[(k + 3 * i) % len(chat_langs)] for i in range(20)]
        G[f"fmt_{key}"] = ("O", [(chat(tok, u, ins), reply(ENG, i)) for i, u in enumerate(users[k % 4::4])]
                           + [(chat(tok, natq(l, rows[i]), ins), reply(l, i)) for i, l in enumerate(ls)])
    eng = chunks(ENG)[:20]
    for l in [x for x in UNIVERSAL if x != ENG]:
        x = chunks(l)[:20]
        G[f"en2x_{l}"] = ("O", [(chat(tok, f"Translate the following text into {name(l)}.\n\n{e}"), t)
                                for e, t in zip(eng, x)])
        G[f"x2en_{l}"] = ("O", [(chat(tok, f"Translate the following text into English.\n\n{t}"), e)
                                for e, t in zip(eng, x)])
        if l in NATIVE:
            G[f"natx_{l}"] = ("O", [(chat(tok, f"{NATIVE[l]}\n\n{e}"), t) for e, t in zip(eng, x)])
    for b, wb in BUILD_W.items():
        for l in LANG8 + [ENG]:
            G[f"S{b}_{l}"] = ("B", [(chat(tok, u, wb.format(LANGS[l])), reply(l if i % 2 == 0 else ENG, i))
                                    for i, u in enumerate(users)])
    for l in chat_langs:
        if l not in LANG8:
            G[f"xS_{l}"] = ("X", [(chat(tok, u, W1.format(LANGS[l])), reply(l, i)) for i, u in enumerate(users[:20])])
    for wk in ("sysW2", "sysW3", "sysW4"):
        for l in LANG8:
            G[f"x{wk[3:]}_{l}"] = ("X", [(chat(tok, u, WORDINGS[wk].format(LANGS[l])), reply(l, i))
                                         for i, u in enumerate(users[:10])])
    return G


def classes(tok, ids):
    """Position class of every prompt token (see the module docstring)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    n, c = len(ids), [2] * len(ids)
    c[:3] = [1] * 3
    if tok.decode(ids[1:2]) == "system":
        e = ids.index(im_end)
        c[3:e] = [0] * (e - 3)
        c[e:e + 5] = [1] * 5
    c[n - NT:] = list(range(3, 3 + NT))
    return c


@torch.no_grad()
def capture():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model())
    G = groups(tok)
    held = {probe_passages()[i]["flores_passage"] for i in PROBE}
    m, lm = load_model()
    got, dev = {}, device()
    for L in FULL:
        lm.layers[L].self_attn.register_forward_pre_hook(
            lambda mod, args, kwargs, L=L: got.__setitem__(L, kwargs["hidden_states"][0].double()), with_kwargs=True)
    gram = torch.zeros(len(FULL), NCLS, 1024, 1024, dtype=torch.float64, device=dev)
    ssum = torch.zeros(len(FULL), NCLS, 1024, dtype=torch.float64, device=dev)
    cnt = torch.zeros(NCLS, dtype=torch.float64, device=dev)
    rng, X, meta, skipped = random.Random(0), [], [], 0
    for g, (kind, pairs) in G.items():
        for j, (prompt, reply) in enumerate(pairs):
            if any(h in prompt for h in held):  # passages of the probing questions stay out of the fit
                skipped += 1
                continue
            ids = tok.encode(prompt, add_special_tokens=False)
            rep = tok.encode(reply, add_special_tokens=False)[:NREP]
            cls = torch.tensor(classes(tok, ids) + [12] * len(rep), device=dev)
            lm(input_ids=torch.tensor([ids + rep], device=dev))
            H = torch.stack([got[L] for L in FULL])  # [6, T, d]
            if kind != "X":
                for c in cls.unique().tolist():
                    Hc = H[:, cls == c]
                    gram[:, c] += Hc.transpose(1, 2) @ Hc
                    ssum[:, c] += Hc.sum(1)
                    cnt[c] += Hc.shape[1]
            if kind == "X" or j % 4 == 0:
                user = [p for p in range(len(ids)) if cls[p] == 2]
                keep = [p for p in range(len(cls)) if cls[p] != 2] + rng.sample(user, min(16, len(user)))
                X.append(H[:, keep].transpose(0, 1).half().cpu())
                meta += [(g, kind, j, int(cls[p]), p) for p in keep]
        print(f"capture: {g} done, {int(cnt.sum())} fit positions", flush=True)
    torch.save({"gram": gram.cpu(), "sum": ssum.cpu(), "cnt": cnt.cpu(), "skipped": skipped}, GRAMS)
    torch.save({"X": torch.cat(X), "meta": meta}, ACTS)
    print("capture: fit positions per class", [int(x) for x in cnt.tolist()], "| skipped prompts", skipped,
          "| sample rows", len(meta), flush=True)


# ---------------------------------------------------------------- fit

def ridge(G, s, n, pos, lam):
    """Reader with target 1 on the classes `pos` and 0 on the others, positives and negatives at equal total weight.
    No intercept: the projections have no bias."""
    w = torch.zeros(len(n), dtype=G.dtype)
    isp = torch.zeros(len(n), dtype=torch.bool)
    isp[pos] = True
    w[isp], w[~isp] = .5 / n[isp].sum(), .5 / n[~isp].sum()
    A = (w[:, None, None] * G).sum(0)
    A += lam * A.diagonal().mean() * torch.eye(A.shape[0], dtype=G.dtype)
    return torch.linalg.solve(A, (w[isp, None] * s[isp]).sum(0))


def dprime(y, pos, neg):
    return float((y[pos].mean() - y[neg].mean()) / y[neg].std())


def stat(y, m):
    return f"{float(y[m].mean()):+.2f}±{float(y[m].std()):.2f}" if int(m.sum()) > 1 else "  --  "


def fit():
    g, a = torch.load(GRAMS), torch.load(ACTS)
    G, s, n = g["gram"], g["sum"], g["cnt"]
    meta = a["meta"]
    cls = torch.tensor([m[3] for m in meta])
    isx = torch.tensor([m[1] == "X" for m in meta])
    word = [m[0].split("_")[0] for m in meta]
    sysk = torch.tensor([m[1] in ("S", "B", "X") or m[0].startswith("fmt") for m in meta])  # prompts with a system turn
    sd = load_base()
    out = {"cnt": n}
    print("fit positions per class", [int(x) for x in n.tolist()], "| sample rows", len(meta), flush=True)
    for li, L in enumerate(FULL):
        X = a["X"][:, li].double()
        WQ, WK = sd[att(L) + "q_proj.weight"].double(), sd[att(L) + "k_proj.weight"].double()
        eq, ek = 1 + sd[att(L) + "q_norm.weight"].double(), 1 + sd[att(L) + "k_norm.weight"].double()
        # readers, the ridge strength chosen on the sample of the unseen wordings and languages (never in the Grams)
        rd = {}
        for nm, pos, negc in (("r", [0], 2), ("t", TEMPLATE, 2), ("tr", TEMPLATE + [REPLY], 2)):
            ispos = torch.isin(cls, torch.tensor(pos))
            best = max(LAMS, key=lambda lam: dprime(X @ ridge(G[li], s[li], n, pos, lam), ispos & isx,
                                                    (cls == negc) & isx))
            rd[nm] = ridge(G[li], s[li], n, pos, best)
            y = X @ rd[nm]
            line = f"L{L} reader {nm:2} lam {best:g} d' on unseen {dprime(y, ispos & isx, (cls == negc) & isx):5.1f} |"
            if nm == "r":
                line += (f" system content: fit {stat(y, (cls == 0) & ~isx)}"
                         + "".join(f" {w} {stat(y, (cls == 0) & torch.tensor([x == w for x in word]))}"
                                   for w in ("xS", "xW2", "xW3", "xW4", "fmt"))
                         + f" | frames {stat(y, cls == 1)} user {stat(y, cls == 2)} template {stat(y, (cls >= 3) & (cls < 12))}"
                           f" reply {stat(y, cls == 12)}")
            else:
                line += (" template " + " ".join(f"{float(y[cls == c].mean()):.2f}" for c in TEMPLATE)
                         + f" | reply {stat(y, cls == 12)} system {stat(y, cls == 0)} frames {stat(y, cls == 1)}"
                           f" user {stat(y, cls == 2)}")
            print(line, flush=True)
        # the dim: among the position-independent dims whose old logit term (rms over template queries x sampled keys,
        # largest of the 8 heads) is in the lowest quarter, the one with the largest norm weights
        tq = (cls >= 3) & (cls < 12)
        Q = (X[tq] @ WQ.T).view(-1, NH, 2 * HD)[:, :, :HD]
        K = (X @ WK.T).view(-1, 2, HD)
        rq = (Q.pow(2) / Q.pow(2).mean(-1, keepdim=True)).mean(0).sqrt()  # [8, 256] rms of the unit-rms query
        rk = (K.pow(2) / K.pow(2).mean(-1, keepdim=True)).mean(0).sqrt()  # [2, 256]
        old = eq * ek * rq * rk.repeat_interleave(4, 0) / 16
        ok = old.max(0).values <= old.max(0).values[PASS:].quantile(.25)
        ok[:PASS] = False
        d = int(torch.where(ok, eq * ek, torch.zeros_like(eq)).argmax())
        lost = old[:, d].tolist()
        # calibration samples: the reader output and the summed squares of the other 255 dims at the positive positions
        cal = {"k": {}, "t": {}, "tr": {}}
        for kh in range(2):
            m = cls == 0
            cal["k"][kh] = (X[m] @ rd["r"], K[m, kh].pow(2).sum(1) - K[m, kh, d] ** 2)
        for h in range(NH):
            q = X @ WQ[h * 2 * HD:h * 2 * HD + HD].T
            for nm in ("t", "tr"):
                m = torch.isin(cls, torch.tensor(QPOS[nm]))
                cal[nm][h] = (X[m] @ rd[nm], q[m].pow(2).sum(1) - q[m][:, d] ** 2)
        out[L] = {"d": d, "eq": float(eq[d]), "ek": float(ek[d]), "r": rd["r"], "t": rd["t"], "tr": rd["tr"], "cal": cal}
        # what the patch adds, in units of strength: normalized key value on the keys of prompts with a system turn
        b = solve(float(ek[d]), *cal["k"][0], 4.0)
        k = X @ WK[:HD].T
        kv = b * (X @ rd["r"])
        yk = float(ek[d]) * kv / ((k.pow(2).sum(1) - k[:, d] ** 2 + kv ** 2) / HD).sqrt() / 4.0
        print(f"L{L} dim {d}: (1+w_q)(1+w_k) = {float(eq[d] * ek[d]):.2f} (largest {float((eq * ek)[PASS:].max()):.2f});"
              f" old logit term of the dim, rms per head: " + " ".join(f"{x:.2f}" for x in lost), flush=True)
        print(f"L{L} added logits per unit strength (key head 0): system content fit {stat(yk, (cls == 0) & ~isx)}"
              + "".join(f" {w} {stat(yk, (cls == 0) & torch.tensor([x == w for x in word]))}"
                        for w in ("xS", "xW2", "xW3", "xW4", "fmt"))
              + f" | user, with a system turn {stat(yk, (cls == 2) & sysk)}, without {stat(yk, (cls == 2) & ~sysk)}"
                f" | template {stat(yk, (cls >= 3) & (cls < 12))} reply {stat(yk, cls == 12)}", flush=True)
    torch.save(out, FIT)


def solve(e, p, S, Y):
    """Row scale a with mean(e * a p / sqrt((S + (a p)^2) / 256)) = Y, by bisection: p = the reader's output at the
    positive positions, S = the summed squares of the other 255 dims there, e = 1 + norm weight of the dim."""
    lo, hi = 0.0, 1e4
    for _ in range(80):
        mid = (lo + hi) / 2
        v = float((e * mid * p / ((S + (mid * p) ** 2) / HD).sqrt()).mean())
        lo, hi = (mid, hi) if v < Y else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- write

def edit(F, spec):
    """{layer: {"d": dim, "q": {head: row}, "k": {key head: row}}} for a SPEC. Query and key share the strength: each
    post-norm value is 4 sqrt(strength), their product over 16 is the strength."""
    out = {}
    for part in spec.split(","):
        L, hs, beta, rd = part.split(":")
        L, hs, beta = int(L), list(range(NH)) if hs == "all" else [int(c) for c in hs], float(beta)
        f, Y = F[L], 4 * beta ** .5
        q = {h: torch.zeros(1024, dtype=torch.float64) for kh in {h // 4 for h in hs} for h in range(4 * kh, 4 * kh + 4)}
        for h in hs:
            q[h] = solve(f["eq"], *f["cal"][rd][h], Y) * f[rd]
        k = {kh: solve(f["ek"], *f["cal"]["k"][kh], Y) * f["r"] for kh in {h // 4 for h in hs}}
        out[L] = {"d": f["d"], "q": q, "k": k}
    return out


def write(name="system_prompt", spec=SPEC):
    sd, out = load_base(), {}
    for L, e in edit(torch.load(FIT), spec).items():
        WQ, WK = sd[att(L) + "q_proj.weight"].float().clone(), sd[att(L) + "k_proj.weight"].float().clone()
        for h, v in e["q"].items():
            WQ[h * 2 * HD + e["d"]] = v.float()  # q_proj holds 512 rows per head: 256 query rows, then 256 gate rows
        for kh, v in e["k"].items():
            WK[kh * HD + e["d"]] = v.float()
        out[att(L) + "q_proj.weight"], out[att(L) + "k_proj.weight"] = WQ, WK
    rel = write_model(os.path.join(MODELS, name), out, {"source": f"attention_patch.py {spec}"})
    print(os.path.join(MODELS, name), {k.split("layers.")[1]: round(v, 4) for k, v in rel.items()})


if __name__ == "__main__":
    {"capture": capture, "fit": fit, "write": write}[sys.argv[1]](*sys.argv[2:])
