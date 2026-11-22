"""Test of the Norwegian patch on Belebele passages: English->Norwegian, ->Danish and ->Swedish, and Norwegian->English.
Greedy, 768 tokens. Scores: share of replies GlotLID labels as the requested language, and chrF++ against FLORES+.

  test  the 234 held out passages (the post's test)
  tune  64 of the probing passages (used to pick the strength)

usage: norwegian_test.py run MODEL_DIR TAG [test|tune] | score TAG [TAG...] [test|tune]       (MODEL_DIR "base": unedited)
"""
import collections
import json
import os
import sys

from common import DATA, RES, base_model, chat, chrf, engine, expected, lang_name, lid, load_belebele, passages

TARGET, LANGS = "nob_Latn", ["nob_Latn", "dan_Latn", "swe_Latn"]


def items(lang, split):
    if split == "tune":
        tune = set(json.load(open(os.path.join(DATA, "flores_dev_build.json")))["tune_links"])
        return [p for p in passages(load_belebele(lang, "discovery")) if p["link"] in tune][:64]
    return passages(load_belebele(lang, "test"))


def run(model, tag, split):
    from vllm import SamplingParams
    llm = engine(base_model() if model == "base" else model)
    tok = llm.get_tokenizer()
    eng = {r["key"]: r["flores_passage"] for r in load_belebele("eng_Latn", "discovery" if split == "tune" else "test")}
    jobs = []
    for lang in LANGS:
        for p in items(lang, split):
            jobs.append((f"en2{lang}", p["flores_passage"], chat(tok, f"Translate the following text into {lang_name(lang)}.\n\n{eng[p['key']]}")))
            if lang == TARGET:
                jobs.append((f"{lang}2en", eng[p["key"]], chat(tok, f"Translate the following text into English.\n\n{p['flores_passage']}")))
    outs = llm.generate([j[2] for j in jobs], SamplingParams(max_tokens=768, temperature=0))
    res = collections.defaultdict(list)
    for j, o in zip(jobs, outs):
        res[j[0]].append((o.outputs[0].text, j[1], o.outputs[0].finish_reason))
    json.dump(res, open(os.path.join(RES, f"norwegian_{split}_{tag}.json"), "w"), ensure_ascii=False)


def score(tags, split):
    for tag in tags:
        for cond, rows in json.load(open(os.path.join(RES, f"norwegian_{split}_{tag}.json"))).items():
            lang = cond[3:] if cond.startswith("en2") else "eng_Latn"
            texts, refs = [r[0] for r in rows], [r[1] for r in rows]
            right = sum(lid(t)[0] == expected(lang) for t in texts) / len(texts)
            print(f"{tag:12} {cond:14} n {len(rows)}  in the requested language {right:.3f}  chrF++ {chrf(texts, refs):5.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "test")
    else:
        split = sys.argv[-1] if sys.argv[-1] in ("test", "tune") else "test"
        score([t for t in sys.argv[2:] if t not in ("test", "tune")], split)
