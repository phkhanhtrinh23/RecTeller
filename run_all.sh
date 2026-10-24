#!/usr/bin/env bash
# End-to-end pipeline. Every step skips work whose result file already exists (use --force on a
# script to recompute). The "llm" and "sec" stages can run concurrently on one 24 GB GPU.
#
#   bash run_all.sh data   # datasets, SASRec victims and shadows, GRO, need texts
#   bash run_all.sh llm    # projectors, utility, trade-off, ablations, budget, violations
#   bash run_all.sh sec    # floods, extraction, membership inference, repetition, long attacks
#   bash run_all.sh gpt    # closed client, GPT-4o-mini at temperature 0 (requires OPENAI_API_KEY)
#   python scripts/make_report.py
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
P=python
LLAMA=${LLAMA:-$HOME/recteller_work/hf/Llama-3.1-8B-Instruct}
MAIN="block web mcp iagent firewall gro ours ours_text einf"

case "${1:-}" in
data)
  for d in games ind ml; do $P -c "import sys;sys.path.insert(0,'.');from recteller import data;print(data.stats(data.load('$d')))"; done
  for d in ml games ind; do $P scripts/train_sasrec.py --dataset $d --epochs 400; done
  for d in ml games ind; do $P scripts/train_sasrec.py --dataset $d --which shadow --epochs 400; done
  for d in games ind ml; do $P scripts/train_gro.py --dataset $d; done
  $P scripts/make_needs.py --datasets games ind ml --synth 8000
  ;;
llm)
  # RQ1: utility and cost
  for d in games ind ml; do $P scripts/train_projector.py --dataset $d; done
  $P scripts/run_utility.py --dataset games --modes $MAIN pub --seeds 3
  $P scripts/run_utility.py --dataset ind --modes $MAIN --seeds 3
  $P scripts/run_utility.py --dataset ml --modes $MAIN --seeds 3
  # RQ3: budget selection on validation users, trade-off on test users, random replacement, cap-only control
  $P scripts/run_utility.py --dataset games --split val --modes ours@0.5 ours@1 ours@2 ours ours@8 einf --seeds 3
  $P scripts/run_utility.py --dataset games --modes ours@0.5 ours@1 ours@2 ours@8 rr0.1 rr0.3 rr0.5 capo2 capo4 --seeds 3
  # RQ4: sensitivity and ablations (A2 no parser, A3 shuffled attributes, minimal = A2 + A4 + no levels)
  $P scripts/run_utility.py --dataset games --modes kp1 kp5 nolev ours_shuf ours_a2 min --seeds 3
  $P scripts/run_budget.py --dataset games --modes ours capo2 capo4
  $P scripts/run_security.py violations --dataset games
  # A5: projector of another receiver (Llama-3.1-8B, same 4096-d input embeddings) and its control
  if [ -d "$LLAMA" ]; then
    $P scripts/train_projector.py --dataset games --receiver "$LLAMA"
    ln -sf "projector_games_Llama-3.1-8B-Instruct.pt" "${RECTELLER_WORK:-$HOME/recteller_work}/checkpoints/projector_games_other.pt"
    $P scripts/run_utility.py --dataset games --modes ours_xproj --seeds 3
    $P scripts/run_utility.py --dataset games --modes ours --seeds 3 --receiver "$LLAMA" --tag _llama
  fi
  ;;
sec)
  S=5   # attack seeds
  ALL="pub ours einf mcp firewall gro"
  $P scripts/run_security.py flood --dataset games --modes ours nocap mcp firewall gro
  $P scripts/run_security.py extraction --dataset games --modes $ALL --accounts 50 1 --seeds $S
  $P scripts/run_security.py extraction --dataset ml --modes $ALL --accounts 50 --seeds $S
  $P scripts/run_security.py mia --dataset games --modes $ALL
  $P scripts/run_security.py mia --dataset ml --modes $ALL
  $P scripts/run_security.py extraction --dataset games --modes ours@0.5 ours@1 ours@2 ours@8 rr0.1 rr0.3 rr0.5 capo2 capo4 kp1 kp5 nolev nocap neither --accounts 50 --seeds $S
  $P scripts/run_security.py repetition --dataset games --seeds $S
  $P scripts/run_security.py horizon --dataset games --seeds $S
  ;;
gpt)
  export CUDA_VISIBLE_DEVICES=""
  for d in games ind ml; do
    $P scripts/run_utility.py --dataset $d --client gpt --modes block web mcp iagent firewall gro ours_text
  done
  ;;
*) echo "usage: $0 {data|llm|sec|gpt}"; exit 1;;
esac
