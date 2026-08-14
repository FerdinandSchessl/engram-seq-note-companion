#!/usr/bin/env python3
"""
Entlastungs-Test (Paper-3-Vorarbeit), Kriterien vor der Messung eingefroren.
Hält der intakte Nachbar das geschnittene Ziel am Leben? Nähe-strukturierte Wiederbelebung
via Folgeschnitt, mit dosisgleichem Placebo (edit-spezifisch vs. Dosis) + Negativ-Ziel.

Nur NLL(Ziel) — keine Marker/Generierung. Nutzt seq_c4 (cut/nll/concept_stats/extract)
und p2_placebo (layer_dists/apply_placebo, tied-sicher). CPU/FP32, alpha=0.6.
"""
import os, sys, json, copy, gc, time, argparse

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
sys.path.insert(0, os.path.expanduser("~/ai-engram-seq"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from seq_c4_test import CONCEPTS, nll, concept_stats, cut, ALPHA
from p2_placebo import layer_dists, apply_placebo

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
TARGETS = ["A_eiffel", "E_insulin"]  # Haupt-Ziel (nah=B_louvre) + Kontroll-Ziel (kein Nachbar)


def target_nll(model, tok, key):
    p, a = CONCEPTS[key]["probe"]
    return round(nll(model, tok, p, a), 4)


def sigma_cos(sa, sb):
    """Mittlerer Kosinus ueber Layer zwischen geflatteten Sigma_c (Traeger-Ueberlappung)."""
    vals = []
    for k in sa:
        if k in sb:
            a, b = sa[k].float().flatten(), sb[k].float().flatten()
            vals.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
    return float(np.mean(vals)) if vals else float("nan")


def run(smoke):
    torch.set_num_threads(16)
    print(f"[{time.strftime('%H:%M:%S')}] Lade {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m0 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()

    # Naehe: Sigma_c aller Konzepte auf M0, einmal
    print("  concept_stats (Naehe-Basis) ...", flush=True)
    keys = list(CONCEPTS.keys())
    stats0 = {k: concept_stats(m0, tok, CONCEPTS[k]["forget"]) for k in keys}
    nll_base = {k: target_nll(m0, tok, k) for k in keys}

    res = {"model": MODEL_ID, "alpha": ALPHA, "smoke": bool(smoke),
           "nll_baseline": nll_base, "targets": {}}
    gidx = 0
    tgts = TARGETS[:1] if smoke else TARGETS
    for T in tgts:
        print(f"[{time.strftime('%H:%M:%S')}] Ziel {T}: cut ...", flush=True)
        m_T = copy.deepcopy(m0)
        cut(m_T, tok, T)  # re-kalibriert auf M0, alpha=0.6
        nll_solo = target_nll(m_T, tok, T)
        cands = [k for k in keys if k != T]
        if smoke:
            cands = ["B_louvre", "F_python"]
        arms = []
        for X in cands:
            t0 = time.time()
            # Real: Folgeschnitt X auf m_T-Zustand
            m_TX = copy.deepcopy(m_T)
            layers = cut(m_TX, tok, X)
            nll_TX = target_nll(m_TX, tok, T)
            dists = layer_dists(m_T, m_TX, list(layers))
            del m_TX; gc.collect()
            # Placebo: dosisgleiches Rauschen auf X's Layern
            m_pl = copy.deepcopy(m_T)
            apply_placebo(m_pl, dists, seed=3000 + gidx)
            nll_TX_pl = target_nll(m_pl, tok, T)
            del m_pl; gc.collect()
            gidx += 1
            arm = {"X": X, "nll_TX": nll_TX, "nll_TX_placebo": nll_TX_pl,
                   "dNLL_real": round(nll_TX - nll_solo, 4),
                   "dNLL_placebo": round(nll_TX_pl - nll_solo, 4),
                   "naehe_cos": round(sigma_cos(stats0[T], stats0[X]), 5),
                   "dose_mean_rel_frob": round(sum(dists.values()) / len(dists), 6),
                   "sec": round(time.time() - t0, 1)}
            arms.append(arm)
            print(f"  X={X:13s} naehe={arm['naehe_cos']:+.3f} "
                  f"dNLL real={arm['dNLL_real']:+.3f} placebo={arm['dNLL_placebo']:+.3f} "
                  f"(dose={arm['dose_mean_rel_frob']:.4f}, {arm['sec']}s)", flush=True)
        res["targets"][T] = {"nll_solo": nll_solo, "nll_baseline": nll_base[T],
                             "cut_effective": round(nll_solo - nll_base[T], 4), "arms": arms}
        del m_T; gc.collect()

    out = os.path.join(DIR, f"results_entlastung{'_smoke' if smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args().smoke)
