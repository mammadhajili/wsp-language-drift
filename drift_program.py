"""The program: finds the languages Qwen3.5-0.8B drifts away from in translation and writes a drift patch for each.

  diagnose  the unedited model translates 120 practice paragraphs into each of the 30 panel languages (and answers 50
            in the "answer in X" frame). Failing = under .90 of the translation replies in the requested language;
            neighbor = the most frequent wrong label that has FLORES+ text (share >= .03), else no patch.
  capture   teacher-forced rows at layer 16, three request formats (translate, answer about a passage, explain a
            topic). Once per language: its own text under its own request (180) and translation from it into English
            (100). Per target: target request with neighbor text and neighbor request with target text (360).
  fit       push u = mean state after layer 16, (target request, target text) - (target request, neighbor text),
            over the translation rows; key w = ridge reader on layer 16's neuron activations, 1 for target requests
            and 0 for every other request; watched languages = the three other panel languages whose own requests
            score highest under the key on the held-out fifth of the fit rows.
  dose      write norm c = strength * |u| in LADDER, ascending, the weight copied into one running engine. Per dose:
            target and neighbor translation (120 paragraphs each), each watched language 50 translation + 50
            "answer in X", 80 FLORES+ dev sentences English->target and target->English (chrF++). Admissible: the
            neighbor and each watched language lose at most .05, neither chrF falls by more than 1. Stop at the first
            inadmissible dose or when the target share is over .05 under the best so far. Pick the admissible dose
            with the highest target share (a smaller dose within .03 wins); no patch if it gains under .10.
  audit     held-out paragraphs, every picked patch copied into one engine after one unedited pass over the union.
            Target and neighbor: 300 paragraphs in three frames; the other panel languages: 60 in two frames;
            100 long replies (320 tokens) for loops; FLORES+ devtest 0-99 in four directions.
  report    paired gains, flags (paired loss > max(.02, |d0| + 2 se), d0 = batch-order noise of the unedited model),
            chrF and loop changes.

usage: drift_program.py diagnose | capture | fit | dose | audit | report        env ONLY=<targets, comma>
Every stage writes results/program_*; a stage (or a target within it) whose file exists is skipped. A new language is
added by putting its Belebele code in LANGS.
"""
import collections
import functools
import glob
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine in this process, so a weight can be copied in

import torch  # noqa: E402
from common import (DATA, EXP, RES, base_model, chat, chrf, engine, expected, flores, lang_name, lid,  # noqa: E402
                    load_belebele, load_model, loop_start, ridge, rows_of, stats)

UNIVERSAL = ["eng_Latn", "rus_Cyrl", "ukr_Cyrl", "bul_Cyrl", "srp_Cyrl", "mkd_Cyrl", "kaz_Cyrl", "kir_Cyrl",
             "tgk_Cyrl", "khk_Cyrl", "hye_Armn", "kat_Geor", "azj_Latn", "uzn_Latn", "tur_Latn", "ell_Grek",
             "ron_Latn", "als_Latn", "ita_Latn", "spa_Latn", "cat_Latn", "fra_Latn", "deu_Latn", "dan_Latn",
             "swe_Latn", "nob_Latn", "ind_Latn", "jav_Latn", "zho_Hans", "arb_Arab"]  # requests every key must ignore
CORE = ["nob_Latn", "dan_Latn", "swe_Latn", "deu_Latn", "mkd_Cyrl", "bul_Cyrl", "srp_Cyrl", "rus_Cyrl", "ukr_Cyrl"]
LANGS = CORE + [l for l in UNIVERSAL + ["sun_Latn"] if l not in CORE and l != "eng_Latn"]  # the 30-language panel
NAMES = {"eng_Latn": "English", "nob_Latn": "Norwegian Bokmål", "zho_Hans": "Chinese", "arb_Arab": "Arabic",
         "uzn_Latn": "Uzbek", "khk_Cyrl": "Mongolian", "azj_Latn": "Azerbaijani", "als_Latn": "Albanian"}
BUILD_TOPICS = ["how rain forms", "the history of the telephone", "how to plant a tree", "why we have seasons",
                "how a refrigerator works", "the rules of football", "how to cook rice", "what a computer virus is",
                "how birds migrate", "why the ocean is salty", "how to ride a horse", "the history of paper money",
                "how glass is made", "what a democracy is", "how coffee is grown", "why leaves change color",
                "how a camera works", "the history of the bicycle", "how to stay safe in a thunderstorm",
                "what DNA is", "how wind turbines work", "the benefits of drinking water", "how cheese is made",
                "why cats purr", "how maps are made", "what a museum does", "how to tie a knot",
                "the history of tea", "how a lighthouse works", "why people dream"]
N_TRANS, N_QA, L = 100, 50, 16
POOL = os.path.join(DATA, "drift_practice_pool.json")  # English Wikipedia paragraphs: 1200 practice, 1200 held out
ROWS = os.path.join(RES, "program_rows")
KEY = f"model.language_model.layers.{L}.mlp.down_proj.weight"
LADDER = [.3, .45, .675, 1.0, 1.5]
FAIL, MIN_SIB, MIN_GAIN = .90, .03, .10
LIM_FID, LIM_CHRF, TIE, OVERSHOOT = .05, 1.0, .03, .05
DOSE_ITEMS, NB_ITEMS, N_FLORES = (1050, 1170), (1050, 1100), 80
AUDIT_N, AUDIT_OTHER, AUDIT_LONG, AUDIT_FLORES = 300, 60, (800, 900), 100
FMT = {"trans": slice(0, N_TRANS), "qa": slice(N_TRANS, N_TRANS + N_QA), "explain": slice(N_TRANS + N_QA, None)}
EPS = 1e-9


def name(lang):
    return NAMES.get(lang, lang_name(lang))


def prompts(tok, lang, build, eng, qa_rows):
    nm = name(lang)
    out = [("trans", chat(tok, f"Translate the following text into {nm}.\n\n{eng[i]}")) for i in build[:N_TRANS]]
    out += [("qa", chat(tok, f"Read the passage below and answer the question. Write your answer in {nm}.\n\n"
                             f"Passage: {r['flores_passage']}\n\nQuestion: {r['question']}")) for r in qa_rows]
    out += [("explain", chat(tok, f"Explain {t}. Write your answer in {nm}.")) for t in BUILD_TOPICS]
    return out


@functools.lru_cache(None)
def text(lang, split="dev"):
    return flores(lang, split)


def code_of(label):
    """Our code for a GlotLID label, if FLORES+ text exists under it."""
    mine = [k for k, v in EXP.items() if v == label]
    for c in [k for k in mine if k in LANGS] + [label] + mine:  # a panel language's own code first
        try:
            text(c)
            return c
        except (KeyError, TypeError, FileNotFoundError):
            pass
    return None


def p_en2(lang, t):
    return f"Translate the following text into {name(lang)}.\n\n{t}"


def p_ask(lang, t):
    return (f"Read the passage below and answer the question. Write your answer in {name(lang)}."
            f"\n\nPassage: {t}\n\nQuestion: What is this passage about?")


def p_sum(lang, t):
    return f"Summarize the text below in one or two sentences. Reply in {name(lang)}.\n\n{t}"


FRAME = {"en2": p_en2, "ask": p_ask, "sum": p_sum, "long": p_en2}


def para(cell, lang, i, rows, cap=64):
    return {"cell": cell, "lang": lang, "i": i, "prompt": FRAME[cell](lang, rows[i]["text"]), "cap": cap}


def flores_jobs(lang, idx, split):
    """English->lang (cell f_en2x) and lang->English (cell f_x2en) over FLORES+ sentences idx, with references."""
    eng, x = text("eng_Latn", split), text(lang, split)
    return ([{"cell": "f_en2x", "lang": lang, "i": i, "prompt": p_en2(lang, eng[i]), "cap": 128, "ref": x[i]} for i in idx]
            + [{"cell": "f_x2en", "lang": lang, "i": i, "prompt": f"Translate the following text into English.\n\n{x[i]}",
                "cap": 128, "ref": eng[i]} for i in idx])


def key(j):
    return (j["cell"], j["lang"], j["i"])


def ok(lang, t):
    return lid(t)[0] == expected(lang)


def put(llm, W):
    """Copy W into the engine's layer-L down_proj weight (cast to the engine's dtype, as a written checkpoint is)."""
    def f(m):
        hit = [p for n, p in m.named_parameters() if n.endswith(f"layers.{L}.mlp.down_proj.weight")]
        assert len(hit) == 1
        with torch.no_grad():
            hit[0].copy_(W.to(hit[0].device, hit[0].dtype))
    llm.apply_model(f)


def gen(llm, tok, jobs):
    from vllm import SamplingParams
    outs = llm.generate([chat(tok, j["prompt"]) for j in jobs],
                        [SamplingParams(max_tokens=j["cap"], temperature=0) for j in jobs], use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def base_weight():
    from safetensors import safe_open
    for f in glob.glob(os.path.join(base_model(), "*.safetensors")):
        with safe_open(f, "pt") as s:
            if KEY in s.keys():
                return s.get_tensor(KEY)
    raise KeyError(KEY)


def dump(path, jobs, texts):
    with open(path, "w") as f:
        for j, t in zip(jobs, texts):
            f.write(json.dumps({"cell": j["cell"], "lang": j["lang"], "i": j["i"], "text": t, "ref": j.get("ref")},
                               ensure_ascii=False) + "\n")


def load(path):
    return {(r["cell"], r["lang"], r["i"]): r for r in map(json.loads, open(path))}


def targets():
    d = json.load(open(os.path.join(RES, "program_diagnose.json")))["langs"]
    only = [t for t in os.environ.get("ONLY", "").split(",") if t]
    return [(t, v["sibling"]) for t, v in d.items() if v["failing"] and v["sibling"] and (not only or t in only)]


# ---------------------------------------------------------------- diagnose

def diagnose():
    out = os.path.join(RES, "program_diagnose.json")
    if os.path.exists(out):
        return show_diagnose()
    rows = json.load(open(POOL))["practice"]
    jobs = [para("en2", l, i, rows) for l in LANGS for i in range(*DOSE_ITEMS)]
    jobs += [para("ask", l, i, rows) for l in LANGS for i in range(*NB_ITEMS)]
    llm = engine()
    texts = gen(llm, llm.get_tokenizer(), jobs)
    dump(os.path.join(RES, "program_diag_raw.jsonl"), jobs, texts)
    labs = collections.defaultdict(collections.Counter)
    for j, t in zip(jobs, texts):
        if j["cell"] == "en2":
            labs[j["lang"]][lid(t)[0]] += 1
    langs = {}
    for l in LANGS:
        n = sum(labs[l].values())
        share = labs[l][expected(l)] / n
        sib = None
        for lab, k in labs[l].most_common():
            if lab == expected(l):
                continue
            if k / n < MIN_SIB:
                break
            c = code_of(lab)
            if c and c != l:
                sib = c
                break
        langs[l] = {"share": share, "failing": share < FAIL, "sibling": sib if share < FAIL else None,
                    "labels": labs[l].most_common(6)}
    json.dump({"langs": langs, "replies": len(jobs)}, open(out, "w"), indent=1)
    show_diagnose()


def show_diagnose():
    d = json.load(open(os.path.join(RES, "program_diagnose.json")))
    for l, v in d["langs"].items():
        print(f"{l[:8]}  {v['share']:.2f}  {'FAILING' if v['failing'] else '       '}  neighbor {str(v['sibling'])[:8]:8}  "
              + " ".join(f"{a[:8]} {k}" for a, k in v["labels"]))
    f = [l for l, v in d["langs"].items() if v["failing"]]
    print(f"{len(f)} failing of {len(d['langs'])}; without a neighbor: {[l for l in f if not d['langs'][l]['sibling']]}; "
          f"replies {d['replies']}", flush=True)


# ---------------------------------------------------------------- capture

def capture():
    from transformers import AutoTokenizer
    os.makedirs(ROWS, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_model())
    fb = json.load(open(os.path.join(DATA, "flores_dev_build.json")))
    build, tune, eng = fb["build"], set(fb["tune_links"]), text("eng_Latn")
    seen, qa_rows = set(), []
    for r in load_belebele("eng_Latn", "discovery"):
        if r["link"] not in tune and r["link"] not in seen:
            seen.add(r["link"])
            qa_rows.append(r)
    qa_rows = qa_rows[:N_QA]
    m, lm = load_model()
    tg = targets()
    passes, t0 = 0, time.time()

    def pairs(req, out_lang):
        o = text(out_lang)
        return [(p, o[build[i % len(build)]]) for i, (_, p) in enumerate(prompts(tok, req, build, eng, qa_rows))]

    for l in UNIVERSAL + sorted({x for ts in tg for x in ts if x not in UNIVERSAL}):
        path = os.path.join(ROWS, f"{l}.pt")
        if os.path.exists(path):
            continue
        A, R = rows_of(lm, tok, pairs(l, l), L)
        x = text(l)
        X, _ = rows_of(lm, tok, [(chat(tok, f"Translate the following text into English.\n\n{x[i]}"), eng[i])
                                 for i in build[:N_TRANS]], L)
        torch.save({"acts": A, "resid": R, "x2en": X}, path)
        passes += len(A) + len(X)
        print("captured", l, f"{time.time() - t0:.0f}s", flush=True)
    for t, s in tg:
        path = os.path.join(ROWS, f"{t}__{s}.pt")
        if os.path.exists(path):
            continue
        A, R = rows_of(lm, tok, pairs(t, s), L)
        B, _ = rows_of(lm, tok, pairs(s, t), L)
        torch.save({"TS_acts": A, "TS_resid": R, "ST_acts": B}, path)
        passes += len(A) + len(B)
        print("captured", t, s, f"{time.time() - t0:.0f}s", flush=True)
    print(f"passes this run {passes}, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- fit

def fit():
    os.makedirs(os.path.join(RES, "program_fit"), exist_ok=True)
    own = functools.lru_cache(None)(lambda l: torch.load(os.path.join(ROWS, f"{l}.pt")))
    rep_path = os.path.join(RES, "program_fit.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    for t, s in targets():
        if t in report and report[t]["sibling"] == s:
            continue
        cross = torch.load(os.path.join(ROWS, f"{t}__{s}.pt"))
        groups = []  # (label, name, acts [n, 7, f]); the last fifth of each is held out of the first fit
        for fmt, sl in FMT.items():
            groups += [(1, f"{t}:T|{fmt}", own(t)["acts"][sl]), (1, f"{t}:S|{fmt}", cross["TS_acts"][sl]),
                       (0, f"{s}:S|{fmt}", own(s)["acts"][sl]), (0, f"{s}:T|{fmt}", cross["ST_acts"][sl])]
            groups += [(0, f"{nl}|{fmt}", own(nl)["acts"][sl]) for nl in UNIVERSAL if nl not in (t, s)]
        groups += [(0, f"x2en_{t}", own(t)["x2en"]), (0, f"x2en_{s}", own(s)["x2en"])]

        def solve(frac):
            Xs, ys = [], []
            for lab, _, A in groups:
                a = A[:int(frac * len(A))].float().reshape(-1, A.shape[-1])
                Xs.append(a)
                ys.append(torch.full((len(a),), float(lab)))
            return ridge(torch.cat(Xs), torch.cat(ys), lam=1e-3)

        w_ho = solve(0.8)
        ho = {nm: (lab, A[int(0.8 * len(A)):].float() @ w_ho) for lab, nm, A in groups}
        leak = {}
        for nl in UNIVERSAL:
            if nl in LANGS and nl not in (t, s):
                leak[nl] = float(torch.cat([ho[f"{nl}|{fmt}"][1].flatten() for fmt in FMT]).mean())
        nb = sorted(leak, key=leak.get, reverse=True)[:3]
        w = solve(1.0)
        tr = FMT["trans"]
        u = own(t)["resid"][tr][:, 1:].float().mean((0, 1)) - cross["TS_resid"][tr][:, 1:].float().mean((0, 1))
        torch.save({"u": u, "w": w}, os.path.join(RES, "program_fit", f"{t}.pt"))
        report[t] = {"sibling": s, "u_norm": float(u.norm()), "w_norm": float(w.norm()), "neighbours": nb,
                     "leak": {k: round(v, 3) for k, v in sorted(leak.items(), key=lambda kv: -kv[1])[:6]},
                     "gate_heldout": {nm: round(float(v.mean()), 3) for nm, (lab, v) in ho.items()
                                      if lab == 1 or nm.startswith("x2en") or nm.startswith(s)}}
        json.dump(report, open(rep_path, "w"), indent=1)
        r = report[t]
        print(f"{t[:3]} neighbor {s[:3]} |u| {r['u_norm']:.2f} watched {[n[:3] for n in nb]} "
              f"leak {list(r['leak'].values())[:3]} key on held-out rows {r['gate_heldout']}", flush=True)


# ---------------------------------------------------------------- dose

def dose():
    diag = load(os.path.join(RES, "program_diag_raw.jsonl"))
    rows = json.load(open(POOL))["practice"]
    fits = json.load(open(os.path.join(RES, "program_fit.json")))
    build = set(json.load(open(os.path.join(DATA, "flores_dev_build.json")))["build"])
    fidx = [i for i in range(len(text("eng_Latn"))) if i not in build][:N_FLORES]
    tg = [(t, s) for t, s in targets() if not os.path.exists(os.path.join(RES, f"program_dose_{t}.json"))]
    if not tg:
        return show_dose()
    llm = engine()
    tok = llm.get_tokenizer()
    W0 = base_weight()
    # unedited replies the diagnose stage does not hold: FLORES+ items of each target, neighbors outside the panel
    bj = [j for t, _ in tg for j in flores_jobs(t, fidx, "dev")]
    bj += [para("en2", s, i, rows) for s in sorted({s for _, s in tg if s not in LANGS}) for i in range(*DOSE_ITEMS)]
    base = {key(j): {"text": x} for j, x in zip(bj, gen(llm, tok, bj))}
    base.update(diag)
    for t, s in tg:
        f = torch.load(os.path.join(RES, "program_fit", f"{t}.pt"))
        u, w, nb = f["u"], f["w"], fits[t]["neighbours"]
        jobs = [para("en2", l, i, rows) for l in (t, s) for i in range(*DOSE_ITEMS)]
        jobs += [para(c, l, i, rows) for l in nb for c in ("en2", "ask") for i in range(*NB_ITEMS)]
        jobs += flores_jobs(t, fidx, "dev")

        def measure(texts):
            share = lambda sel, src: (lambda v: sum(v) / len(v))([ok(j["lang"], src(j, x)) for j, x in zip(jobs, texts) if sel(j)])
            edit, unedited = (lambda j, x: x), (lambda j, x: base[key(j)]["text"])
            fl = lambda cell, src: chrf(*zip(*[(src(j, x), j["ref"]) for j, x in zip(jobs, texts) if j["cell"] == cell]))
            is_t = lambda j: j["lang"] == t and j["cell"] == "en2"
            is_s = lambda j: j["lang"] == s and j["cell"] == "en2"
            m = {"target": share(is_t, edit), "target_base": share(is_t, unedited),
                 "sib_loss": share(is_s, unedited) - share(is_s, edit),
                 "nb_loss": {l: share(lambda j, l=l: j["lang"] == l and j["cell"] in ("en2", "ask"), unedited)
                             - share(lambda j, l=l: j["lang"] == l and j["cell"] in ("en2", "ask"), edit) for l in nb},
                 "chrf_en2t": fl("f_en2x", edit), "chrf_en2t_base": fl("f_en2x", unedited),
                 "chrf_t2en": fl("f_x2en", edit), "chrf_t2en_base": fl("f_x2en", unedited)}
            m["target_labels"] = collections.Counter(lid(x)[0] for j, x in zip(jobs, texts) if is_t(j)).most_common(4)
            m["admissible"] = bool(m["sib_loss"] <= LIM_FID + EPS and all(v <= LIM_FID + EPS for v in m["nb_loss"].values())
                                   and m["chrf_en2t_base"] - m["chrf_en2t"] <= LIM_CHRF
                                   and m["chrf_t2en_base"] - m["chrf_t2en"] <= LIM_CHRF)
            return m

        visited, best, why = [], None, "ladder exhausted"
        with open(os.path.join(RES, f"program_dose_raw_{t}.jsonl"), "w") as raw:
            for c in LADDER:
                beta = c / float(u.norm())
                put(llm, W0.float() + beta * torch.outer(u, w))
                texts = gen(llm, tok, jobs)
                for j, x in zip(jobs, texts):
                    raw.write(json.dumps({"c": c, "cell": j["cell"], "lang": j["lang"], "i": j["i"], "text": x},
                                         ensure_ascii=False) + "\n")
                m = dict(c=c, beta=beta, **measure(texts))
                visited.append(m)
                print(f"{t[:3]} c {c:<5} strength {beta:5.2f} target {m['target_base']:.2f} -> {m['target']:.2f}  "
                      f"neighbor loss {m['sib_loss']:+.3f}  watched " + " ".join(f"{l[:3]} {v:+.2f}" for l, v in m["nb_loss"].items())
                      + f"  chrF en->T {m['chrf_en2t_base']:.1f} -> {m['chrf_en2t']:.1f}  T->en {m['chrf_t2en_base']:.1f}"
                      f" -> {m['chrf_t2en']:.1f}  {'ok' if m['admissible'] else 'INADMISSIBLE'}  {m['target_labels']}",
                      flush=True)
                if not m["admissible"]:
                    why = f"dose {c} inadmissible"
                    break
                if best is not None and m["target"] < best - OVERSHOOT - EPS:
                    why = f"target share at dose {c} over {OVERSHOOT} under the best"
                    break
                best = m["target"] if best is None else max(best, m["target"])
        put(llm, W0)
        adm = [m for m in visited if m["admissible"]]
        pick, reason = None, "no admissible dose"
        if adm:
            top = max(m["target"] for m in adm)
            cand = min((m for m in adm if m["target"] >= top - TIE - EPS), key=lambda m: m["c"])
            if cand["target"] - cand["target_base"] >= MIN_GAIN - EPS:
                pick, reason = cand, "picked"
            else:
                reason = f"best admissible gain {cand['target'] - cand['target_base']:+.3f} is under {MIN_GAIN}"
        json.dump({"target": t, "sibling": s, "neighbours": nb, "u_norm": float(u.norm()), "doses": visited,
                   "walk_end": why, "pick": pick and pick["c"], "beta": pick and pick["beta"], "reason": reason,
                   "replies": len(jobs) * len(visited), "replies_unedited": 2 * len(fidx) + (len(range(*DOSE_ITEMS)) if s not in LANGS else 0)},
                  open(os.path.join(RES, f"program_dose_{t}.json"), "w"), indent=1)
        print(f"{t[:3]}: {reason}" + (f" c {pick['c']} strength {pick['beta']:.2f}" if pick else "") + f" ({why})", flush=True)
    show_dose()


def show_dose():
    for t, s in targets():
        p = os.path.join(RES, f"program_dose_{t}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"{t[:3]} -> neighbor {s[:3]}  watched {[n[:3] for n in d['neighbours']]}  doses "
                  + " ".join(f"{m['c']}:{m['target']:.2f}{'' if m['admissible'] else '!'}" for m in d["doses"])
                  + f"  base {d['doses'][0]['target_base']:.2f}  pick {d['pick']} (strength {d['beta'] and round(d['beta'], 2)})"
                  f"  {d['reason']}; {d['walk_end']}; replies {d['replies']}")


# ---------------------------------------------------------------- audit

def picked():
    out = []
    for t, s in targets():
        p = os.path.join(RES, f"program_dose_{t}.json")
        if os.path.exists(p) and json.load(open(p))["pick"]:
            out.append((t, s, json.load(open(p))["beta"]))
    return out


def audit_jobs(t, s):
    held = json.load(open(POOL))["heldout"]
    js = [para(c, l, i, held) for l in (t, s) for c in ("en2", "ask", "sum") for i in range(AUDIT_N)]
    js += [para(c, l, i, held) for l in LANGS if l not in (t, s) for c in ("en2", "ask") for i in range(AUDIT_OTHER)]
    js += [para("long", t, i, held, cap=320) for i in range(*AUDIT_LONG)]
    for l in (t, s):
        if l != "eng_Latn":
            js += flores_jobs(l, range(AUDIT_FLORES), "devtest")
    return js


def audit():
    pk = picked()
    llm = engine()
    tok = llm.get_tokenizer()
    W0 = base_weight()
    bp = os.path.join(RES, "program_audit_base.jsonl")
    union = list({key(j): j for t, s, _ in pk for j in audit_jobs(t, s)}.values())
    have = load(bp) if os.path.exists(bp) else {}
    if any(key(j) not in have for j in union):
        t0 = time.time()
        dump(bp, union, gen(llm, tok, union))
        print(f"unedited pass: {len(union)} replies, {time.time() - t0:.0f}s", flush=True)
    for t, s, beta in pk:
        path = os.path.join(RES, f"program_audit_{t}.jsonl")
        if os.path.exists(path):
            continue
        f = torch.load(os.path.join(RES, "program_fit", f"{t}.pt"))
        put(llm, W0.float() + beta * torch.outer(f["u"], f["w"]))
        js, t0 = audit_jobs(t, s), time.time()
        dump(path, js, gen(llm, tok, js))
        put(llm, W0)
        print(f"audited {t} (strength {beta:.2f}): {len(js)} replies, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- report

def report():
    # paired difference between two batch orders of the unedited model on the practice paragraphs, per cell
    d0 = {tuple(k.split("|")): v for k, v in json.load(open(os.path.join(DATA, "batch_noise.json"))).items()}
    base = load(os.path.join(RES, "program_audit_base.jsonl"))
    lab = functools.lru_cache(None)(lambda x: lid(x)[0])
    diag = json.load(open(os.path.join(RES, "program_diagnose.json")))["langs"]
    out, lines = {}, []
    for t, s, beta in picked():
        path = os.path.join(RES, f"program_audit_{t}.jsonl")
        if not os.path.exists(path):
            continue
        ed = load(path)
        g, labels = collections.defaultdict(list), collections.defaultdict(lambda: [collections.Counter(), collections.Counter()])
        for k, r in ed.items():
            cell, lang, i = k
            b = base[k]["text"]
            if cell in ("en2", "ask", "sum"):
                g[(cell, lang)].append((lab(b) == expected(lang), lab(r["text"]) == expected(lang), i, lang))
                labels[(cell, lang)][0][lab(b)] += 1
                labels[(cell, lang)][1][lab(r["text"])] += 1
            elif cell == "long":
                g[("noloop", lang)].append((loop_start(b) == len(b), loop_start(r["text"]) == len(r["text"]), i, lang))
        cells, flags = {}, []
        for (cell, lang), v in g.items():
            st = stats(v)
            # share cells: loss over max(.02, |d0| + 2 se); loops: rise over max(.02, 2 se)
            margin = max(.02, (abs(d0.get((cell, lang), 0.0)) if cell != "noloop" else 0.0) + 2 * st["se_item"])
            cells[f"{cell}|{lang}"] = {k: st[k] for k in ("n", "base", "edit", "gain", "se_item", "won", "lost")}
            if cell != "noloop":
                cells[f"{cell}|{lang}"]["labels"] = [c.most_common(4) for c in labels[(cell, lang)]]
            if -st["gain"] > margin:
                flags.append((cell, lang[:3], round(st["gain"], 3), round(margin, 3)))
        cf = {}
        for l in (t, s):
            for cell in ("f_en2x", "f_x2en"):
                ks = [k for k in ed if k[0] == cell and k[1] == l]
                if ks:
                    refs = [ed[k]["ref"] for k in ks]
                    cf[f"{cell}|{l}"] = (chrf([base[k]["text"] for k in ks], refs), chrf([ed[k]["text"] for k in ks], refs))
        cflags = [(k, round(e - b, 1)) for k, (b, e) in cf.items() if b - e > LIM_CHRF]
        tt = cells[f"en2|{t}"]
        other = [c for k, c in cells.items() if k.split("|")[1] not in (t, s) and not k.startswith("noloop")]
        moved = sorted(((k, round(c["gain"], 3)) for k, c in cells.items()
                        if k.split("|")[1] not in (t, s) and abs(c["gain"]) >= .10), key=lambda kv: -abs(kv[1]))
        res = {"sibling": s, "beta": beta, "cells": cells, "flags": flags, "chrf": cf, "chrf_flags": cflags,
               "hit": bool(tt["gain"] >= MIN_GAIN - EPS and tt["gain"] > 2 * tt["se_item"]),
               "clean": not flags and not cflags, "fixed": bool(tt["edit"] >= FAIL - EPS),
               "others_mean_gain": sum(c["gain"] * c["n"] for c in other) / sum(c["n"] for c in other),
               "others_won_lost": [sum(c["won"] for c in other), sum(c["lost"] for c in other), sum(c["n"] for c in other)],
               "others_moved": moved, "replies": len(ed)}
        out[t] = res
        f = lambda k: (lambda c: f"{c['base']:.2f}→{c['edit']:.2f} ({c['gain']:+.3f}±{c['se_item']:.3f})")(cells[k])
        short = lambda cs: " ".join(f"{a[:3]} {n}" for a, n in cs)
        lines.append(
            f"{t[:3]}←{s[:3]} strength {beta:.2f} | {'HIT' if res['hit'] else 'no hit'}{' CLEAN' if res['clean'] else ''}"
            f"{' FIXED' if res['fixed'] else ''}\n   target   en2 {f(f'en2|{t}')}  ask {f(f'ask|{t}')}  sum {f(f'sum|{t}')}  no-loop {f(f'noloop|{t}')}"
            f"\n   neighbor en2 {f(f'en2|{s}')}  ask {f(f'ask|{s}')}  sum {f(f'sum|{s}')}"
            "\n   chrF " + "  ".join(f"{k.replace('f_en2x|', 'en→').replace('f_x2en|', 'en←')[:6]} {b:.1f}→{e:.1f}" for k, (b, e) in cf.items())
            + f"\n   others pooled {res['others_mean_gain']:+.4f} (won {res['others_won_lost'][0]}, lost {res['others_won_lost'][1]}"
            f" of {res['others_won_lost'][2]})  moved ≥.10: {[(k.replace('|', ' ')[:7], v) for k, v in moved]}"
            f"\n   flags {flags} chrF {cflags}"
            f"\n   target en2 labels: {short(cells[f'en2|{t}']['labels'][0])} → {short(cells[f'en2|{t}']['labels'][1])}")
    json.dump(out, open(os.path.join(RES, "program_report.json"), "w"), indent=1)
    print("\n".join(lines))
    failing = [l for l, v in diag.items() if v["failing"]]
    print(f"\nfailing {len(failing)}; patched {len(out)}; hits {sum(r['hit'] for r in out.values())}; clean hits "
          f"{sum(r['hit'] and r['clean'] for r in out.values())}; clean {sum(r['clean'] for r in out.values())}; "
          f"fixed {sum(r['fixed'] for r in out.values())}; audit replies {sum(r['replies'] for r in out.values())} + {len(base)} unedited")


if __name__ == "__main__":
    {"diagnose": diagnose, "capture": capture, "fit": fit, "dose": dose, "audit": audit, "report": report}[sys.argv[1]]()
