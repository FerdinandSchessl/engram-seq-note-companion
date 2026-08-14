#!/usr/bin/env python3
"""
Paper 2 — P4b Trajektorien-Placebo (PRE_REG_PAPER2.md Nachtrag 09.07. spaet, eingefroren).

3 Placebo-Zustaende (CUT_KEYS x alpha=0.6, Qwen2-0.5B-Instruct): Engram-Schnitt
reproduzieren -> layer_dists (tied-sicher) -> dosis-gematchtes Gauss-Rauschen auf
frischer Kopie (Seeds 2000+idx) -> Block-2-Instrument unveraendert (p2_dialogs.json,
greedy, Var(sem_velocity_ai)/E_ai). Baseline + Engram-Vergleich kommen aus
results_p2_trajectories.json (deterministisch, wird NICHT neu gemessen).
Checkpoint nach jeder Zelle.
"""
import os, sys, json, copy, gc, time, argparse

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
sys.path.insert(0, os.path.expanduser("~/ai-engram-seq"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import apply_engram
from seq_c4_test import extract
from p2_placebo import layer_dists, bias_energy_share, apply_placebo
from p2_trajectories import MODEL_ID, D, MAX_NEW, W, measure_state

DIR = os.path.dirname(os.path.abspath(__file__))
CUT_KEYS = ["E_insulin", "F_python", "A_eiffel"]
ALPHA = 0.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(16)
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    fu = D["_meta"]["followups"]
    dialogs = [[op] + fu for op in D["openers"]]
    if args.smoke:
        dialogs = [d[:3] for d in dialogs[:2]]

    ref = json.load(open(os.path.join(DIR, "results_p2_trajectories.json")))
    assert ref["model"] == MODEL_ID and not ref["smoke"], "Referenz-Baseline passt nicht"

    ckpt_path = os.path.join(DIR, f"results_p2_traj_placebo{'_smoke' if args.smoke else ''}_ckpt.json")
    if os.path.exists(ckpt_path):
        res = json.load(open(ckpt_path))
        print(f"  Resume: {len(res['cells'])} Zellen fertig", flush=True)
    else:
        res = {"model": MODEL_ID, "smoke": bool(args.smoke), "alpha": ALPHA,
               "dialogs_version": D["_meta"]["version"], "max_new_tokens": MAX_NEW, "W": W,
               "baseline_source": "results_p2_trajectories.json (wiederverwendet)",
               "cells": []}

    print(f"[{time.strftime('%H:%M:%S')}] Lade {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    done = {c["cut"] for c in res["cells"]}
    cells = ([("E_insulin",)] if args.smoke else [(k,) for k in CUT_KEYS])
    for idx, (key,) in enumerate(cells):
        if key in done:
            continue
        t0 = time.time()
        m_e = copy.deepcopy(m0)
        eng = extract(m_e, tok, key)
        apply_engram(m_e, eng, alpha=ALPHA, inplace=True)
        layer_names = list(eng.layers.keys())
        del eng
        gc.collect()
        dists = layer_dists(m0, m_e, layer_names)
        b_share = bias_energy_share(m0, m_e, layer_names)
        del m_e
        gc.collect()
        seed = 2000 + idx
        m_p = copy.deepcopy(m0)
        apply_placebo(m_p, dists, seed)
        cell = {"cut": key, "alpha": ALPHA, "placebo_seed": seed,
                "n_layers": len(dists), "bias_energy_share": b_share,
                "dialogs": measure_state(m_p, tok, st_model, dialogs),
                "sec": round(time.time() - t0, 1)}
        del m_p
        gc.collect()
        res["cells"].append(cell)
        json.dump(res, open(ckpt_path, "w"), indent=1)
        print(f"  Placebo-Zelle {key}: {cell['sec']}s (n_layers={cell['n_layers']})", flush=True)

    out = os.path.join(DIR, f"results_p2_traj_placebo{'_smoke' if args.smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] -> {out}", flush=True)


if __name__ == "__main__":
    main()
