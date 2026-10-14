# SysContract

Stop–Check–Act runtime for LLM agents that control computer systems.

Code, logged results, and paper for *SysContract: Stop–Check–Act Runtimes for Agents that Control Computer Systems* (NeurIPS 2026 Workshop: ML for Systems).

Israel Olanrewaju Odeajo, WorldQuant University.

## Setup

```bash
cd syscontract
python3 -m pip install -r requirements.txt
```

The eval agent talks to any OpenAI-compatible API. Table 1 in the paper was run with GPT-4o-mini; the logged runs are in `results/`. The providers below work as substitutes, but their numbers will differ from Table 1.

| Provider | Env | Example `--model` |
|---|---|---|
| Groq (best GPT-4o-mini stand-in) | `GROQ_API_KEY` | `llama-3.3-70b-versatile` or `qwen/qwen3-32b` |
| Hugging Face Inference | `HF_TOKEN` | `Qwen/Qwen3-32B` |
| Local llama.cpp | `OPENAI_BASE_URL=http://127.0.0.1:8080/v1` `OPENAI_API_KEY=no-key` | `Qwen3-4B` |

Local on an 8 GB Intel Mac: `llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M -c 4096` — smoke-test only.

## Reproduce Table 1

```bash
# baselines
for env in cluster serving sqlite; do
  python3 scripts/run_eval.py --env $env --agent default --seeds 5 --out results/${env}_default.json
  python3 scripts/run_eval.py --env $env --agent random --seeds 5 --out results/${env}_random.json
done

# GPT-4o-mini (unconstrained + SysContract + replay)
for env in cluster serving sqlite; do
  python3 scripts/run_eval.py --env $env --agent llm --seeds 5 --steps 6 --model gpt-4o-mini --out results/${env}_llm.json
done

# GPT-aware: invariants in the prompt, no gate
for env in cluster serving sqlite; do
  python3 scripts/run_eval.py --env $env --agent llm --aware --unconstrained-only \
    --seeds 5 --steps 6 --model gpt-4o-mini --out results/${env}_llm_aware.json
done

python3 scripts/paper_table.py results/*.json
```

## Paper

```bash
cd paper && pdflatex syscontract_camera_ready.tex && bibtex syscontract_camera_ready && pdflatex syscontract_camera_ready.tex && pdflatex syscontract_camera_ready.tex
```

Paper PDF: `paper/syscontract_camera_ready.pdf`.
