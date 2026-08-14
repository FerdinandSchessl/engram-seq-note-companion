#!/usr/bin/env python3
"""
P4b-Replikation auf Qwen3-0.6B (Chat): Baseline + Engram(alpha=0.6) + Placebo(alpha=0.6)
für die 3 Konzepte in EINEM Lauf (Block-2-Trajektorien-Instrument, unverändert).
Qwen3-safe: gen_dialog mit enable_thinking=False (bei Qwen2 No-Op, hier nötig).
Metrik-Funktionen (sem_velocity/user_ai_sep) aus p2_trajectories importiert (reviewt).
Dosis-Matching layer_dists/apply_placebo aus p2_placebo (tied-sicher). Checkpoint/Resume.
"""
import os, sys, json, copy, gc, time, argparse

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
sys.path.insert(0, os.path.expanduser("~/ai-engram-seq"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import apply_engram
from seq_c4_test import extract
from p2_trajectories import sem_velocity, user_ai_sep, D, MAX_NEW, W
from p2_placebo import layer_dists, apply_placebo, bias_energy_share

MODEL_ID = "Qwen/Qwen3-0.6B"
DIR = os.path.dirname(os.path.abspath(__file__))
CUT_KEYS = ["E_insulin", "F_python", "A_eiffel"]
ALPHA = 0.6


def gen_dialog(model, tok, user_turns):
    """Greedy, deterministisch; enable_thinking=False (Qwen3-Template)."""
    msgs, ai_texts = [], []
    for u in user_turns:
        msgs.append({"role": "user", "content": u})
        enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True,
                                      return_dict=True, enable_thinking=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ai_texts.append(text if text else "(empty)")
        msgs.append({"role": "assistant", "content": text})
    return ai_texts


def measure_state(model, tok, st_model, dialogs):
    out = []
    for user_turns in dialogs:
        ai_texts = gen_dialog(model, tok, user_turns)
        ai_emb = st_model.encode(ai_texts, normalize_embeddings=True)
        u_emb = st_model.encode(user_turns, normalize_embeddings=True)
        sv = sem_velocity(np.asarray(ai_emb))
        ua = user_ai_sep(np.asarray(u_emb), np.asarray(ai_emb))
        ea = ua["E_ai"]
        out.append({
            "var_sv_ai": float(np.var(sv[1:])) if len(sv) > 1 else float("nan"),
            "E_ai_mean": (float(np.nanmean(ea)) if np.any(~np.isnan(ea)) else float("nan")),
            "epsilon_ai_mean": (float(np.nanmean(ua["epsilon_ai"]))
                                if np.any(~np.isnan(ua["epsilon_ai"])) else float("nan")),
            "ai_texts": ai_texts,
        })
    return out


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

    ckpt = os.path.join(DIR, f"results_p2_traj_qwen3{'_smoke' if args.smoke else ''}_ckpt.json")
    if os.path.exists(ckpt):
        res = json.load(open(ckpt))
        print(f"  Resume: baseline={'baseline' in res}, {len(res.get('cells', []))} Zellen", flush=True)
    else:
        res = {"model": MODEL_ID, "smoke": bool(args.smoke), "alpha": ALPHA,
               "dialogs_version": D["_meta"]["version"], "max_new_tokens": MAX_NEW, "W": W,
               "cells": []}

    def save():
        json.dump(res, open(ckpt, "w"), indent=1)

    print(f"[{time.strftime('%H:%M:%S')}] Lade {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    if "baseline" not in res:
        t0 = time.time()
        res["baseline"] = measure_state(m0, tok, st_model, dialogs)
        print(f"  Baseline: {round(time.time()-t0,1)}s", flush=True)
        save()

    done = {(c["cut"], c["arm"]) for c in res["cells"]}
    keys = ["E_insulin"] if args.smoke else CUT_KEYS
    for idx, key in enumerate(keys):
        # Engram-Arm
        if (key, "engram") not in done:
            t0 = time.time()
            m = copy.deepcopy(m0)
            eng = extract(m, tok, key)
            apply_engram(m, eng, alpha=ALPHA, inplace=True)
            layer_names = list(eng.layers.keys())
            del eng; gc.collect()
            res["cells"].append({"cut": key, "arm": "engram",
                                 "dialogs": measure_state(m, tok, st_model, dialogs)})
            # Dosis fuer Placebo: aus diesem Engram gegen m0
            m_ref = copy.deepcopy(m0)
            dists = layer_dists(m_ref, m, layer_names)
            b_share = bias_energy_share(m_ref, m, layer_names)
            res.setdefault("_dose", {})[key] = {"dists_keys": len(dists),
                                                "bias_energy_share": b_share,
                                                "_dists": {k: v for k, v in dists.items()}}
            del m, m_ref; gc.collect()
            save()
            print(f"  Engram {key}: {round(time.time()-t0,1)}s", flush=True)
        # Placebo-Arm (dosisgleich)
        if (key, "placebo") not in done:
            t0 = time.time()
            dists = res["_dose"][key]["_dists"]
            m = copy.deepcopy(m0)
            apply_placebo(m, dists, seed=4000 + idx)
            res["cells"].append({"cut": key, "arm": "placebo", "placebo_seed": 4000 + idx,
                                 "dialogs": measure_state(m, tok, st_model, dialogs)})
            del m; gc.collect()
            save()
            print(f"  Placebo {key}: {round(time.time()-t0,1)}s", flush=True)

    out = os.path.join(DIR, f"results_p2_traj_qwen3{'_smoke' if args.smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] -> {out}", flush=True)


if __name__ == "__main__":
    main()
