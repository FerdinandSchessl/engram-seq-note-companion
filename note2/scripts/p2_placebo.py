#!/usr/bin/env python3
"""
Paper 2 — P4 Placebo-Kontrollarm (PRE_REG_PAPER2.md Nachtrag 09.07.2026, eingefroren).

Je Zelle (Konzept x alpha x Charge):
  1. Engram-Schnitt deterministisch reproduzieren (frische Kopie von M0),
     pro editiertem Layer d_l = ||W_edit - W_0||_F / ||W_0||_F messen.
  2. PLACEBO: auf zweiter frischer Kopie Gauss-Rauschen je Layer exakt auf d_l
     skalieren (gleiche Layer, gleiche Dosis; seed = 1000 + Zellindex).
  3. Beide Zustaende messen: eingefrorenes Marker-Set + NLL-Panel + Retain-NLL
     + NEU beidseitig: mittlere Antwortlaenge (12 geradzahlige neutrale Prompts,
     greedy, max_new_tokens=120).
Engram-Marker werden hier MITGEMESSEN (identische Zahlen wie Block 1 erwartet,
dient zugleich als Reproduktions-Check); Antwortlaenge ist fuer beide Arme neu.
"""
import os, sys, json, copy, gc, time, argparse

sys.path.insert(0, os.path.expanduser("~/ai-engram-seq/src"))
sys.path.insert(0, os.path.expanduser("~/ai-engram-seq"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import apply_engram
from seq_c4_test import extract, nll_all
from p2_markers import (P, measure_state, retain_nll, build_ctx, is_instruct_model,
                        CUT_KEYS, ALPHAS)

DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-0.5B"]
GEN_PROMPTS = [P["neutral"][i] for i in range(0, 24, 2)]  # 12 fixe Prompts (frozen)
MAX_NEW = 120


def mean_answer_len(model, tok, is_instruct, smoke=False):
    """Mittlere Antwortlaenge (Zeichen) auf den 12 fixen Prompts, greedy."""
    prompts = GEN_PROMPTS[:3] if smoke else GEN_PROMPTS
    total = 0
    for prompt in prompts:
        ctx, sp = build_ctx(tok, prompt, is_instruct)
        enc = tok(ctx, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        total += len(text.strip())
    return round(total / len(prompts), 1)


def layer_dists(m0, m_edit, layer_names):
    """Relative Frobenius-Distanz je editiertem Layer (weight).
    Tied-sicher via named_modules: named_parameters() wuerde lm_head bei
    tie_word_embeddings deduplizieren und still fallen lassen (Review-BLOCK 09.07.)."""
    mods0, mods1 = dict(m0.named_modules()), dict(m_edit.named_modules())
    d = {}
    for name in layer_names:
        a_mod, b_mod = mods0.get(name), mods1.get(name)
        if a_mod is None or getattr(a_mod, "weight", None) is None:
            continue
        a, b = a_mod.weight.float(), b_mod.weight.float()
        d[name] = float((b - a).norm() / (a.norm() + 1e-12))
    return d


def bias_energy_share(m0, m_edit, layer_names):
    """Anteil der Bias-Deltas an der Edit-Energie ||Delta||_F^2 (Bericht, kein Match):
    Engram editiert q/k/v-Biases mit (absorb_bias=True), Dosis-Match ist gewichts-only."""
    mods0, mods1 = dict(m0.named_modules()), dict(m_edit.named_modules())
    e_w = e_b = 0.0
    for name in layer_names:
        a, b = mods0.get(name), mods1.get(name)
        if a is None:
            continue
        if getattr(a, "weight", None) is not None:
            e_w += float((b.weight.float() - a.weight.float()).norm() ** 2)
        if getattr(a, "bias", None) is not None:
            e_b += float((b.bias.float() - a.bias.float()).norm() ** 2)
    return round(e_b / (e_w + e_b + 1e-12), 6)


def apply_placebo(m, dists, seed):
    """Gauss-Rauschen je Layer, exakt auf relative Frobenius-Distanz skaliert (tied-sicher)."""
    gen = torch.Generator().manual_seed(seed)
    mods = dict(m.named_modules())
    with torch.no_grad():
        for name, d_rel in dists.items():
            W = mods[name].weight
            noise = torch.randn(W.shape, generator=gen, dtype=torch.float32)
            target = d_rel * W.float().norm()
            noise = noise * (target / (noise.norm() + 1e-12))
            W.add_(noise.to(W.dtype))


def state_measurements(m, tok, is_instruct, smoke):
    return {"markers": measure_state(m, tok, is_instruct, smoke),
            "nll": nll_all(m, tok), "retain_nll": retain_nll(m, tok),
            "mean_answer_len": mean_answer_len(m, tok, is_instruct, smoke)}


def run_model(model_id, smoke, cell_offset):
    is_instruct = is_instruct_model(model_id)
    slug = model_id.split("/")[-1].replace(".", "_")
    print(f"[{time.strftime('%H:%M:%S')}] Lade {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    m0 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    res = {"model": model_id, "is_instruct": is_instruct, "smoke": smoke,
           "prompts_version": P["_meta"]["version"], "gen_max_new_tokens": MAX_NEW}
    t0 = time.time()
    res["baseline"] = state_measurements(m0, tok, is_instruct, smoke)
    print(f"  Baseline ({round(time.time()-t0,1)}s): len={res['baseline']['mean_answer_len']}", flush=True)

    cells = [("E_insulin", 0.6)] if smoke else [(k, a) for k in CUT_KEYS for a in ALPHAS]
    res["cells"] = []
    for idx, (key, alpha) in enumerate(cells):
        t0 = time.time()
        # --- Engram-Arm (Reproduktion + Laengen-Neumessung)
        m_e = copy.deepcopy(m0)
        eng = extract(m_e, tok, key)
        apply_engram(m_e, eng, alpha=alpha, inplace=True)
        layer_names = list(eng.layers.keys())
        del eng
        gc.collect()
        dists = layer_dists(m0, m_e, layer_names)
        bias_energy_share_val = bias_energy_share(m0, m_e, layer_names)
        engram_meas = state_measurements(m_e, tok, is_instruct, smoke)
        del m_e
        gc.collect()
        # --- Placebo-Arm (gematchte Dosis)
        seed = 1000 + cell_offset + idx
        m_p = copy.deepcopy(m0)
        apply_placebo(m_p, dists, seed)
        placebo_meas = state_measurements(m_p, tok, is_instruct, smoke)
        del m_p
        gc.collect()
        cell = {"cut": key, "alpha": alpha, "placebo_seed": seed,
                "n_layers": len(dists),
                "dose_mean_rel_frob": round(sum(dists.values()) / len(dists), 6),
                "bias_energy_share": bias_energy_share_val,
                "engram": engram_meas, "placebo": placebo_meas,
                "sec": round(time.time() - t0, 1)}
        res["cells"].append(cell)
        b = res["baseline"]
        print(f"  Zelle {key} a={alpha}: Engram dNLL(Ziel)="
              f"{engram_meas['nll'][key]-b['nll'][key]:+.3f} len={engram_meas['mean_answer_len']} | "
              f"Placebo dNLL(Ziel)={placebo_meas['nll'][key]-b['nll'][key]:+.3f} "
              f"len={placebo_meas['mean_answer_len']} | retain E/P "
              f"{engram_meas['retain_nll']-b['retain_nll']:+.3f}/"
              f"{placebo_meas['retain_nll']-b['retain_nll']:+.3f} ({cell['sec']}s)", flush=True)

    out = os.path.join(DIR, f"results_p2_placebo_{slug}{'_smoke' if smoke else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="both")
    args = ap.parse_args()
    torch.set_num_threads(16)
    models = MODELS if args.model == "both" else [args.model]
    for i, mid in enumerate(models):
        run_model(mid, args.smoke, cell_offset=100 * i)
    print("FERTIG.", flush=True)


if __name__ == "__main__":
    main()
