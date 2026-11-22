"""Shared by every script: the pinned model and datasets, prompts, language ID, ridge fits and checkpoint writing."""
import functools
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, RES, MODELS = (os.path.join(ROOT, d) for d in ("data", "results", "models"))
os.makedirs(RES, exist_ok=True)

# everything downloaded from Hugging Face, at the revisions the post used
QWEN = ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17")
BELEBELE = ("facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2")
FLORES = ("openlanguagedata/flores_plus", "1147e2dc59f37fbb5c22fd8478770de02cd399f9")
GLOTLID = ("cis-lmu/glotlid", "e9592e0a9bc3b6f2a946de94faaa188d1dd6f522")

LANG_MAP = json.load(open(os.path.join(DATA, "languages.json")))  # Belebele code -> FLORES+ code (null: none)
FLORES_CODES = {v for v in LANG_MAP.values() if v}
EXP = json.load(open(os.path.join(DATA, "expected_lid.json")))  # the GlotLID label of each language's own text
SHARD = "model.safetensors-00001-of-00001.safetensors"


@functools.cache
def snapshot(repo, rev, patterns=None, kind="model"):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, repo_type=kind, revision=rev, allow_patterns=patterns and list(patterns))


@functools.cache
def base_model():
    return os.environ.get("QWEN_BASE") or snapshot(*QWEN)


@functools.cache
def device():
    import torch
    return os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data

def split_of(link):
    """Belebele passages are split in two by a hash of their link: discovery (probing) and test (held out)."""
    return "discovery" if int(hashlib.md5(link.encode()).hexdigest(), 16) % 2 == 0 else "test"


def load_belebele(lang, split="discovery"):
    d = os.path.join(snapshot(*BELEBELE, ("data/*.jsonl",), "dataset"), "data")
    rows = [json.loads(l) for l in open(os.path.join(d, f"{lang}.jsonl"))]
    for r in rows:
        r["key"] = f'{r["link"]}#{r["question_number"]}'
    return [r for r in rows if split_of(r["link"]) == split]


def passages(rows):
    seen, out = set(), []
    for r in rows:
        if r["link"] not in seen:
            seen.add(r["link"])
            out.append(r)
    return out


def flores(lang, split="dev"):
    """FLORES+ sentences of a Belebele language, dev or devtest."""
    code = LANG_MAP.get(lang, lang)
    if code not in FLORES_CODES:
        raise KeyError(lang)
    d = snapshot(*FLORES, ("dev/*.jsonl", "devtest/*.jsonl"), "dataset")
    return [json.loads(l)["text"] for l in open(os.path.join(d, split, f"{code}.jsonl"))]


def lang_name(code):
    import langcodes
    return langcodes.Language.get(code.split("_")[0][:3]).display_name().replace(" (individual language)", "")


def chat(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


# ---------------------------------------------------------------- scoring

_lid = None


def lid(text):
    """(label, prob) of the top GlotLID label; numpy-2-safe call into the fasttext core."""
    global _lid
    if _lid is None:
        import fasttext
        _lid = fasttext.load_model(os.path.join(snapshot(*GLOTLID, ("model.bin",)), "model.bin"))
    t = " ".join(text.split())
    if not t:
        return "empty", 1.0
    p, lab = _lid.f.predict(t + "\n", 1, 0.0, "strict")[0]
    return lab.replace("__label__", ""), float(p)


def expected(lang):
    return EXP.get(lang, lang)


def chrf(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_chrf(list(hyps), [list(refs)], word_order=2).score


def loop_start(text):
    """Character offset where a repetition loop starts (a 4-word n-gram that occurs 3+ times), else len(text)."""
    w = text.split()
    s = " ".join(w)
    for i in range(len(w) - 12):
        if s.count(" ".join(w[i:i + 4])) >= 3:
            return len(" ".join(w[:i]))
    return len(text)


def se(x):
    n, m = len(x), sum(x) / len(x)
    return (sum((v - m) ** 2 for v in x) / (n - 1) / n) ** .5 if n > 1 else float("nan")


def stats(rows):
    """Paired gain of rows (base ok, edit ok, passage id, language), with standard errors by item, passage, language."""
    import collections
    d = [e - b for b, e, _, _ in rows]
    by_p, by_l = collections.defaultdict(list), collections.defaultdict(list)
    for (b, e, p, l) in rows:
        by_p[p].append(e - b)
        by_l[l].append(e - b)
    out = {"n": len(rows), "base": sum(b for b, _, _, _ in rows) / len(rows), "edit": sum(e for _, e, _, _ in rows) / len(rows),
           "gain": sum(d) / len(d), "won": sum(v > 0 for v in d), "lost": sum(v < 0 for v in d), "se_item": se(d),
           "se_p": se([sum(v) / len(v) for v in by_p.values()]), "passages": len(by_p)}
    if len(by_l) > 1:
        gl = [sum(v) / len(v) for v in by_l.values()]
        out.update(se_l=se(gl), langs=len(by_l), langs_up=sum(g > 0 for g in gl), langs_down=sum(g < 0 for g in gl))
    out["se"] = max(out["se_p"], out.get("se_l", 0))
    return out


# ---------------------------------------------------------------- models

def engine(model=None):
    """A vLLM engine for greedy generation; KV_GIB and GPU_UTIL size it."""
    from vllm import LLM
    return LLM(model or base_model(), tokenizer=base_model(), kv_cache_memory_bytes=int(float(os.environ.get("KV_GIB", "16")) * 2**30),
               gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.3")), max_model_len=8192,
               language_model_only=True, seed=0, enable_prefix_caching=False)


def load_model():
    """The unedited model in transformers, for teacher-forced captures."""
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(base_model(), dtype=torch.bfloat16).to(device()).eval()
    return m, m.model.language_model


POS = [1, 2, 3, 5, 8, 12]  # reply-token offsets captured, besides the reply start


def rows_of(lm, tok, pairs, L):
    """Teacher-forced (prompt, reply) pairs: layer L's neuron activations (input of down_proj) [n, 7, 3584] and the
    state after layer L [n, 7, 1024], at the reply start and the POS reply tokens, fp16."""
    import torch
    now = {}
    h = lm.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: now.__setitem__("a", inp[0][0].detach()))
    A, R = [], []
    with torch.no_grad():
        for prompt, out in pairs:
            p_ids = tok.encode(prompt)
            ids = torch.tensor([p_ids + tok.encode(out)[:max(POS)]], device=device())
            hs = lm(input_ids=ids, output_hidden_states=True).hidden_states
            idx = [min(i, ids.shape[1] - 1) for i in [len(p_ids) - 1] + [len(p_ids) - 1 + k for k in POS]]
            A.append(now["a"][idx].half().cpu())
            R.append(hs[L + 1][0, idx].half().cpu())
    h.remove()
    return torch.stack(A), torch.stack(R)


def ridge(X, y, lam):
    import torch
    X = X.float()
    return torch.linalg.solve(X.T @ X + lam * X.shape[0] * torch.eye(X.shape[1]), X.T @ y)


def load_base():
    from safetensors.torch import load_file
    return load_file(os.path.join(base_model(), SHARD))


def write_model(out_dir, replacements, info=None):
    """Copy of the base model dir (other files symlinked) with `replacements` {name: tensor} swapped in as bf16."""
    import torch
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(base_model()):
        if f != SHARD and not os.path.exists(os.path.join(out_dir, f)):
            os.symlink(os.path.realpath(os.path.join(base_model(), f)), os.path.join(out_dir, f))
    sd, rel = load_file(os.path.join(base_model(), SHARD)), {}
    for name, t in replacements.items():
        assert name in sd and sd[name].shape == t.shape, name
        new = t.to(torch.bfloat16).contiguous()
        rel[name] = float((new.float() - sd[name].float()).norm() / sd[name].float().norm())
        sd[name] = new
    save_file(sd, os.path.join(out_dir, SHARD), metadata={"format": "pt"})
    json.dump({"relative_change": rel, **(info or {})}, open(os.path.join(out_dir, "edits.json"), "w"), indent=1)
    return rel
