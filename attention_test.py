"""Held out test of the attention patch: 36 Belebele test questions on passages no probing or fitting run used, 25
languages besides English, greedy, 512 tokens. Right language = GlotLID label of the language the cell asks for.

Cells per language:
  sysW1..sysW4  the system prompt names the language in one of four wordings, English user turn (the post's test)
  user          no system prompt, "Write your answer in X." in the user turn
  native        passage and question written in X, no system prompt                          right language: X
  p1_native     system "You are a helpful assistant." + the native user turn                 right language: X
  p3_native     system = a deployment prompt with no language instruction + native turn      right language: X
  en_native     system "Always respond in English." + the native user turn                   right language: English
  mention       English user turn that says "The passage was translated from X."            right language: English
English: plain, sysW1 ("Always respond in English."), p1_eng, p3_eng, and xsys (system "You are a helpful assistant."
written in German, Russian or Spanish + the English user turn; prints the reply language).

usage: attention_test.py run MODEL_DIR TAG | score BASE_TAG TAG       (MODEL_DIR "base": unedited)
"""
import collections
import json
import os
import sys

from attention_patch import LANGS, WORDINGS
from common import RES, base_model, engine, expected, lid, load_belebele, loop_start, stats

P1 = "You are a helpful assistant."
P3 = "You are the assistant of a public library. Be polite and concise. Do not give legal or medical advice."
XSYS = {"deu_Latn": "Du bist ein hilfreicher Assistent.", "rus_Cyrl": "Ты полезный помощник.",
        "spa_Latn": "Eres un asistente útil."}
RIGHT_EN = ["en_native", "mention"]


def items():
    """One question per passage from row 140 of the English test split on, skipping the passages of the first 140 rows
    and the first 80 such questions (both used by earlier tests)."""
    t = load_belebele("eng_Latn", "test")
    used = {r["link"] for r in t[:140]}
    seen, rows = set(), []
    for r in t[140:]:
        if r["link"] not in used and r["link"] not in seen:
            seen.add(r["link"])
            rows.append(r)
    return rows[80:]


def jobs():
    """(cell, language, passage link, messages), in a fixed order."""
    eng = items()
    nat = {l: {r["key"]: r for r in load_belebele(l, "test")} for l in LANGS}
    qa = lambda r: f"Read the passage below and answer the question.\n\nPassage: {r['flores_passage']}\n\nQuestion: {r['question']}"
    sy = lambda s, u: [{"role": "system", "content": s}, {"role": "user", "content": u}]
    out = []
    for lang, name in LANGS.items():
        if lang == "eng_Latn":
            out += [("plain", lang, r["link"], [{"role": "user", "content": qa(r)}]) for r in eng]
            out += [("sysW1", lang, r["link"], sy(WORDINGS["sysW1"].format("English"), qa(r))) for r in eng]
            out += [("p1_eng", lang, r["link"], sy(P1, qa(r))) for r in eng]
            out += [("p3_eng", lang, r["link"], sy(P3, qa(r))) for r in eng]
            out += [(f"xsys:{x}", lang, r["link"], sy(s, qa(r))) for x, s in XSYS.items() for r in eng]
            continue
        nt = lambda r: f"{nat[lang][r['key']]['flores_passage']}\n\n{nat[lang][r['key']]['question']}"
        for w, s in WORDINGS.items():
            out += [(w, lang, r["link"], sy(s.format(name), qa(r))) for r in eng]
        out += [("user", lang, r["link"],
                 [{"role": "user", "content": f"Read the passage below and answer the question. Write your answer in {name}."
                                              f"\n\nPassage: {r['flores_passage']}\n\nQuestion: {r['question']}"}]) for r in eng]
        out += [("native", lang, r["link"], [{"role": "user", "content": nt(r)}]) for r in eng]
        out += [("p1_native", lang, r["link"], sy(P1, nt(r))) for r in eng]
        out += [("p3_native", lang, r["link"], sy(P3, nt(r))) for r in eng]
        out += [("en_native", lang, r["link"], sy(WORDINGS["sysW1"].format("English"), nt(r))) for r in eng]
        out += [("mention", lang, r["link"],
                 [{"role": "user", "content": f"Read the passage below and answer the question. The passage was translated from {name}."
                                              f"\n\nPassage: {r['flores_passage']}\n\nQuestion: {r['question']}"}]) for r in eng]
    return out


def run(model, tag):
    from vllm import SamplingParams
    llm = engine(base_model() if model == "base" else model)
    tok = llm.get_tokenizer()
    js = jobs()
    prompts = [tok.apply_chat_template(j[3], tokenize=False, add_generation_prompt=True, enable_thinking=False) for j in js]
    outs = llm.generate(prompts, SamplingParams(max_tokens=512, temperature=0))
    with open(os.path.join(RES, f"attention_{tag}.jsonl"), "w") as f:
        for j, o in zip(js, outs):
            f.write(json.dumps({"task": j[0], "lang": j[1], "link": j[2], "text": o.outputs[0].text,
                                "finish": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")


def labelled(tag):
    """per reply: cell, language, passage link, GlotLID label, whether it loops"""
    return [(r["task"], r["lang"], r["link"], lid(r["text"])[0], loop_start(r["text"]) < len(r["text"]))
            for r in map(json.loads, open(os.path.join(RES, f"attention_{tag}.jsonl")))]


def score(base, tag):
    b, m = labelled(base), labelled(tag)
    assert [x[:3] for x in b] == [x[:3] for x in m]
    g, xs, loops = collections.defaultdict(list), collections.defaultdict(collections.Counter), collections.defaultdict(list)
    for (task, lang, link, lb, loopb), (_, _, _, lm, loopm) in zip(b, m):
        if task.startswith("xsys"):
            xs[task][lm] += 1
            continue
        w = "eng_Latn" if task in RIGHT_EN else expected(lang)
        g[(task, lang != "eng_Latn")].append((lb == w, lm == w, link, lang))
        loops[(task, lang != "eng_Latn")].append((loopb, loopm))
    print(f"{tag} against {base}: replies in the right language, unedited -> patched (paired gain ± se); replies that loop")
    s = stats([r for w in WORDINGS for r in g[(w, True)]])
    print(f"  {'4 wordings':20} {s['base']:.3f} -> {s['edit']:.3f}  gain {s['gain']:+.3f} ± {s['se']:.3f}  languages up "
          f"{s['langs_up']}/{s['langs']}, down {s['langs_down']}")
    for k, rows in g.items():
        s, lp = stats(rows), loops[k]
        print(f"  {k[0] + ('' if k[1] else ' (English)'):20} {s['base']:.3f} -> {s['edit']:.3f}  gain {s['gain']:+.3f} ± {s['se']:.3f}"
              f"  n {s['n']:4}  loops {sum(a for a, _ in lp) / len(lp):.2f} -> {sum(b for _, b in lp) / len(lp):.2f}")
    for task, c in xs.items():
        print(f"  {task}: reply language {dict(c.most_common(4))}")


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]](*sys.argv[2:])
