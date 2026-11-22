# wsp-language-drift

Code for the note [Weight-space programming against language confusion in a small model](https://mhajili.com/notes/weight-space-programming/). Each patch is a
closed-form edit of Qwen3.5-0.8B's weights, computed from recorded activations with no gradient training. The patches
themselves are on Hugging Face at [hajili/qwen3.5-0.8b-language-patches](https://huggingface.co/hajili/qwen3.5-0.8b-language-patches).

| File | What it does |
| --- | --- |
| `norwegian_patch.py` | builds the Norwegian drift patch: one rank-one edit of layer 16's MLP `down_proj` |
| `norwegian_test.py` | the held out test on 234 Belebele passages (translations into Norwegian 47% to 94%) |
| `drift_program.py` | the program that finds the languages that drift and builds, doses and audits a drift patch for each |
| `attention_patch.py` | builds the attention patch: ten query and key rows in layers 7 and 11 |
| `attention_test.py` | the held out test of a language set in the system prompt (5% to 54% over four wordings) |
| `common.py` | model and dataset loading, prompts, GlotLID, ridge fits, checkpoint writing |

## Setup

```bash
pip install -r requirements.txt
hf auth login
```

The model, Belebele, FLORES+ and GlotLID download from Hugging Face at the revisions the note used. FLORES+ is gated,
so accept its terms on its dataset page first. Generation uses vLLM on a GPU. Captures and fits run in transformers
on the GPU, or on the CPU with `DEVICE=cpu`. Results go to `results/` and patched checkpoints to `models/`.

## Norwegian patch

```bash
python norwegian_patch.py capture
python norwegian_patch.py build 1,2,4,8
python norwegian_test.py run base base tune
python norwegian_test.py run models/norwegian_s4 s4 tune
python norwegian_test.py score base s4 tune
python norwegian_test.py run base base
python norwegian_test.py run models/norwegian_s4 s4
python norwegian_test.py score base s4
```

`tune` scores 64 probing passages, which is where strength 4 was picked. Without it the test runs on the 234 held out
passages.

## The program

```bash
python drift_program.py diagnose
python drift_program.py capture
python drift_program.py fit
python drift_program.py dose
python drift_program.py audit
python drift_program.py report
```

The docstring of `drift_program.py` states every rule the program follows. To try a new language, add its Belebele
code to `LANGS` in `drift_program.py`.

## Attention patch

```bash
python attention_patch.py capture
python attention_patch.py fit
python attention_patch.py write
python attention_test.py run base base
python attention_test.py run models/system_prompt patched
python attention_test.py score base patched
```

## Data

- `flores_dev_build.json`: the 310 FLORES+ dev sentences the patches are fitted on, and the 128 Belebele passages
  used for tuning.
- `drift_practice_pool.json`: 2400 English Wikipedia introduction paragraphs from the HotpotQA corpus (CC BY-SA 4.0),
  1200 for practice and 1200 held out.
- `expected_lid.json`: the GlotLID label of each language's own Belebele text.
- `languages.json`: the FLORES+ code of each Belebele language.
- `batch_noise.json`: how much the unedited model's language shares move when only the batch order changes, used by
  the program's report.

Belebele passages are split into probing and held out halves by a hash of their link (`split_of` in `common.py`).

## License

Apache 2.0 for the code. `drift_practice_pool.json` holds Wikipedia text under CC BY-SA 4.0.
