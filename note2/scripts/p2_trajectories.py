#!/usr/bin/env python3
"""
Paper 2 "Post-Edit Qualification" — Messblock 2: Trajektorien (P2).
Instrument-Freeze 2026-07-08 (PRE_REG_PAPER2.md, Block-2-Nachtrag).

Editiert vs. uneditiert auf 10 fixen RETAIN-Dialogen (p2_dialogs.json, 8 User-Turns),
greedy/deterministisch, Qwen2-0.5B-Instruct. Metriken je Dialog:
  - Var(sem_velocity_ai): compute_sem_velocity (L2) EXAKT wie
    layer_mapping_validation.py:765, ueber Assistant-Turn-Embeddings.
  - E_ai (mean ueber berechenbare t): compute_user_ai_separation_metrics EXAKT wie
    layer_mapping_validation.py:880 (W=5, Kosinus-Distanz), User-Turns fix
    -> sigma_user zwischen Zustaenden identisch, Delta-E_ai traegt nur epsilon_ai.
Embeddings: sentence-transformers/all-MiniLM-L6-v2, normalize_embeddings=True.
"""
import os, sys, json, copy, gc, time, argparse

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
sys.path.insert(0, os.path.expanduser("~/ai-engram-seq"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import apply_engram
from seq_c4_test import extract

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DIR = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(DIR, "p2_dialogs.json")))
CUT_KEYS = ["E_insulin", "F_python", "A_eiffel"]
ALPHAS = [0.3, 0.6]
MAX_NEW = 100
W = 5


def cos_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sem_velocity(emb):
    """EXAKT layer_mapping_validation.compute_sem_velocity: L2-Norm konsekutiver Embeddings."""
    T = len(emb)
    sv = np.zeros(T)
    for t in range(1, T):
        sv[t] = np.linalg.norm(emb[t] - emb[t - 1])
    return sv


def user_ai_sep(user_emb, ai_emb, w=W):
    """EXAKT layer_mapping_validation.compute_user_ai_separation_metrics (relevante Teile)."""
    T = len(user_emb)
    sigma_user = np.full(T, np.nan)
    epsilon_ai = np.full(T, np.nan)
    E_ai = np.full(T, np.nan)
    if T < w:
        return {"sigma_user": sigma_user, "epsilon_ai": epsilon_ai, "E_ai": E_ai}
    C_a_base = np.mean(ai_emb[:w], axis=0)
    for t in range(w, T):
        C_a_recent = np.mean(ai_emb[max(0, t - w):t], axis=0)
        base_dist = cos_dist(user_emb[t], C_a_recent)
        dir_p = 1.0
        if t >= 2:
            s_prev = user_emb[t - 1] - user_emb[t - 2]
            s_curr = user_emb[t] - user_emb[t - 1]
            n1, n2 = np.linalg.norm(s_prev), np.linalg.norm(s_curr)
            if n1 > 1e-9 and n2 > 1e-9:
                dir_p = max(0.0, 1.0 + float(np.dot(s_prev, s_curr) / (n1 * n2)))
        sigma_user[t] = base_dist * dir_p
        epsilon_ai[t] = cos_dist(ai_emb[t], C_a_base)
        if epsilon_ai[t] > 1e-8:
            E_ai[t] = sigma_user[t] / epsilon_ai[t]
    return {"sigma_user": sigma_user, "epsilon_ai": epsilon_ai, "E_ai": E_ai}


def gen_dialog(model, tok, user_turns):
    """Greedy, deterministisch; History waechst mit."""
    msgs, ai_texts = [], []
    for u in user_turns:
        msgs.append({"role": "user", "content": u})
        enc = tok.apply_chat_template(msgs, return_tensors="pt",
                                      add_generation_prompt=True, return_dict=True)
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
            "sv_ai": [round(float(x), 5) for x in sv],
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

    print(f"[{time.strftime('%H:%M:%S')}] Lade {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    ckpt_path = os.path.join(DIR, f"results_p2_trajectories{'_smoke' if args.smoke else ''}_ckpt.json")
    if os.path.exists(ckpt_path):
        res = json.load(open(ckpt_path))
        print(f"  Resume aus Checkpoint: baseline={'baseline' in res}, "
              f"{len(res.get('cells', []))} Zellen fertig", flush=True)
    else:
        res = {"model": MODEL_ID, "smoke": bool(args.smoke),
               "dialogs_version": D["_meta"]["version"], "max_new_tokens": MAX_NEW, "W": W,
               "cells": []}

    def ckpt():
        json.dump(res, open(ckpt_path, "w"), indent=1)

    if "baseline" not in res:
        t0 = time.time()
        res["baseline"] = measure_state(m0, tok, st_model, dialogs)
        print(f"  Baseline: {round(time.time()-t0,1)}s", flush=True)
        ckpt()

    done = {(c["cut"], c["alpha"]) for c in res["cells"]}
    cells = [("E_insulin", 0.6)] if args.smoke else [(k, a) for k in CUT_KEYS for a in ALPHAS]
    cells = [c for c in cells if c not in done]
    for key, alpha in cells:
        t0 = time.time()
        m = copy.deepcopy(m0)
        eng = extract(m, tok, key)
        apply_engram(m, eng, alpha=alpha, inplace=True)
        del eng
        gc.collect()
        cell = {"cut": key, "alpha": alpha,
                "dialogs": measure_state(m, tok, st_model, dialogs),
                "sec": round(time.time() - t0, 1)}
        res["cells"].append(cell)
        ckpt()
        print(f"  Zelle {key} a={alpha}: {cell['sec']}s", flush=True)
        del m
        gc.collect()

    out = os.path.join(DIR, f"results_p2_trajectories{'_smoke' if args.smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] -> {out}", flush=True)


if __name__ == "__main__":
    main()
