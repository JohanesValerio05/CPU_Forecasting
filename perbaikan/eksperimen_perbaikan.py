"""Eksperimen perbaikan model forecasting CPU 1 jam (kandidat v4).

Prinsip:
- Data, pembersihan, fitur, origin, fold, dan metrik diambil LANGSUNG dari sel skripsi_forecast_v3.ipynb
  (dieksekusi apa adanya), sehingga pembanding "sebelum" identik dengan seleksi beku v3.
- Semua keputusan (kapasitas model, fitur tambahan, model per pita horizon, half-life, koreksi bias)
  diambil HANYA dari periode validation dengan aturan adopsi yang ditetapkan di bawah sebelum dijalankan.
- Seleksi v4 dibekukan (JSON + sha256) sebelum test dihitung; test dijalankan sekali.
- Folder outputs_v3_baru hanya dibaca (cache). Pra-registrasi v3 tidak disentuh. Data >= 1 Nov 2026 tidak dibaca.

Pemakaian: python perbaikan/eksperimen_perbaikan.py
"""
import os, sys, json, time, pickle, hashlib
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ["PYTHONUTF8"] = "1"
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
OUT = "outputs_perbaikan"; os.makedirs(f"{OUT}/cache", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
LOGF = f"{OUT}/log_eksperimen.txt"
def log(*a):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a); print(s, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f: f.write(s + "\n")

# ---------------------------------------------------------------- 1. pipeline v3 apa adanya
nb = json.load(open("skripsi_forecast_v3.ipynb", encoding="utf8"))
G = {"__name__": "nbv3"}
def run_cell(i): exec(compile("".join(nb["cells"][i]["source"]), f"v3_sel_{i}", "exec"), G)
run_cell(3)
G["show"] = lambda *a, **k: None; G["display"] = lambda *a, **k: None
for i in (5, 7, 9, 11, 14, 17, 18, 19): run_cell(i)
assert G["how_v"] == "cache", "cache validation v3 tidak terpakai (versi pustaka/perangkat berbeda?)"
np, pd, lgb, xgb, stats = G["np"], G["pd"], G["lgb"], G["xgb"], G["stats"]
from sklearn.ensemble import HistGradientBoostingRegressor
(FEAT_DF, TR, FEATS, FALLBACK, H, HIGH, SPLIT_ALL, TARGET, cpu_raw, PV, BEST, FINAL, NAME, SEED, N_JOBS, LR, XGB_DEV, LGB_DEV) = (G[k] for k in
    ("FEAT_DF", "TR", "FEATS", "FALLBACK", "H", "HIGH", "SPLIT_ALL", "TARGET", "cpu_raw", "PV", "BEST", "FINAL", "NAME", "SEED", "N_JOBS", "LR", "XGB_DEV", "LGB_DEV"))
build_rows, phase_origins, recency_w, spec_key_v3, sha_file, sha_obj = (G[k] for k in ("build_rows", "phase_origins", "recency_w", "spec_key", "sha_file", "sha_obj"))
log(f"pipeline v3 dimuat | final v3 = {NAME[FINAL]} | baris training {len(TR):,} | versi {G['VERSIONS']}")

# ---------------------------------------------------------------- 2. aturan (ditetapkan sebelum eksperimen)
MIN_GAIN, MIN_FOLDS = 0.005, 3      # adopsi bila MAE validation turun >= 0,5 % DAN lebih baik di >= 3 dari 4 fold bulanan
CAPS = {"s": {"leaves": 15, "depth": 4}, "m": {"leaves": 31, "depth": 5}, "l": {"leaves": 63, "depth": 6}}
BANDS = [(0, 6, 1, 12), (6, 24, 4, 36), (24, 72, 18, 84), (72, 120, 60, 120)]   # (pita prediksi (lo, hi]; rentang h untuk training [tlo, thi])
BAND_EDGES = [0, 6, 24, 72, H]
KINDS = ("hgb", "lgbm", "xgb")

# ---------------------------------------------------------------- 3. fitur tambahan (kausal: hanya CPU <= origin)
cp = FEAT_DF[TARGET].to_numpy(float).copy(); IDX = FEAT_DF.index; N = len(cp)
def extra_features(X):
    pos = IDX.get_indexer(X.origin); h = X.h.to_numpy().astype(int); assert (pos >= 0).all()
    back = (24 - h % 24) % 24                                  # jam mundur dari origin ke kemunculan terakhir "jam yang sama dengan target"
    src = np.stack([pos - back - 24 * j for j in range(7)], axis=1); assert (src <= pos[:, None]).all(), "sumber setelah origin"
    v = np.where(src >= 0, cp[np.clip(src, 0, N - 1)], np.nan)
    with np.errstate(all="ignore"):
        m7 = np.nanmean(v, axis=1); n4 = X[[f"naif_{k}minggu" for k in (1, 2, 3, 4)]].to_numpy(float)
        out = {"x_jamsama_1hari": v[:, 0], "x_jamsama_mean7": m7, "x_jamsama_std7": np.nanstd(v, axis=1), "x_jamsama_max7": np.nanmax(v, axis=1),
               "x_geser_7hari_vs_4minggu": m7 - X["rata_4minggu"].to_numpy(float),
               "x_naif_std4": np.nanstd(n4, axis=1), "x_naif_max4": np.nanmax(n4, axis=1), "x_naif_min4": np.nanmin(n4, axis=1), "x_naif_median4": np.nanmedian(n4, axis=1)}
    return pd.DataFrame(out, index=X.index).astype(np.float32)
import warnings; warnings.filterwarnings("ignore")
XTR = extra_features(TR); XCOLS = list(XTR.columns); TRX = pd.concat([TR, XTR], axis=1)
FEATSX = {fs: f + XCOLS for fs, f in FEATS.items()}
# uji kebocoran fitur tambahan: acak semua CPU setelah origin -> fitur tidak boleh berubah
_t = pd.Timestamp("2026-07-15 12:00"); _r = build_rows(FEAT_DF, G["OF"], G["PROF"], pd.DatetimeIndex([_t]), cpu_raw); _a = extra_features(_r)
_cp0 = cp.copy(); cp[IDX > _t] = np.random.default_rng(0).uniform(1, 100, int((IDX > _t).sum())); _b = extra_features(_r); cp[:] = _cp0
assert np.allclose(_a.to_numpy(), _b.to_numpy(), equal_nan=True), "fitur tambahan bocor"
log(f"fitur tambahan: {len(XCOLS)} kolom, uji kebocoran lolos")

# ---------------------------------------------------------------- 4. fold, model, cache
_FOLDS = {}
def folds(ph, weekly=False):
    """weekly=False: refit bulanan seperti v3. weekly=True: refit tiap 7 hari (origin dan target yang dinilai tetap sama)."""
    if (ph, weekly) not in _FOLDS:
        sp = SPLIT_ALL[ph]; end = pd.Timestamp(sp["end"])
        refits = ([pd.Timestamp(r) for r in sp["refits"]] if not weekly else list(pd.date_range(sp["refits"][0], end - pd.Timedelta(hours=1), freq="7D"))) + [end]; out = []
        for a, b in zip(refits[:-1], refits[1:]):
            so = phase_origins(ph, a, b)
            if not so: continue
            te = build_rows(FEAT_DF, G["OF"], G["PROF"], pd.DatetimeIndex(so), cpu_raw); te = te[te.y.notna()].reset_index(drop=True)
            te = pd.concat([te, extra_features(te)], axis=1)
            trm = (TRX.origin + pd.Timedelta(hours=H) <= a).to_numpy(); assert TRX.target_time[trm].max() <= a
            if len(te): out.append((a, trm, te))
        _FOLDS[(ph, weekly)] = out
    return _FOLDS[(ph, weekly)]
def base_frame(ph):
    parts = []
    for a, _, te in folds(ph):
        b = te[["origin", "target_time", "h", "y"]].copy(); b["refit"] = a; b["phase"] = ph
        b["snaive"] = te.naif_1minggu.to_numpy(float); b["mean4w"] = te.rata_4minggu.to_numpy(float); parts.append(b)
    return pd.concat(parts, ignore_index=True)

def key(sp): return f"{sp['kind']}|{sp['fs']}|{sp['target']}|{sp['hl']}|n{sp['n_iter']}|c{sp['cap']}|x{int(sp['extra'])}|b{int(sp['bands'])}"
def make_model(sp):
    c = CAPS[sp["cap"]]; k = sp["kind"]
    if k == "hgb": return HistGradientBoostingRegressor(loss="absolute_error", max_iter=sp["n_iter"], learning_rate=LR, max_leaf_nodes=c["leaves"], min_samples_leaf=50,
                                                        early_stopping=False, random_state=SEED)
    if k == "lgbm": return lgb.LGBMRegressor(objective="l1", n_estimators=sp["n_iter"], learning_rate=LR, num_leaves=c["leaves"], min_child_samples=50, subsample=.8, subsample_freq=1,
                                             colsample_bytree=.8, n_jobs=N_JOBS, random_state=SEED, verbose=-1, **LGB_DEV)
    if k == "xgb": return xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=sp["n_iter"], learning_rate=LR, max_depth=c["depth"], min_child_weight=50, subsample=.8,
                                           colsample_bytree=.8, tree_method="hist", n_jobs=N_JOBS, random_state=SEED, **XGB_DEV)
def fit_predict(sp, trm, te, start):
    f = (FEATSX if sp["extra"] else FEATS)[sp["fs"]]; resid = sp["target"] == "resid"; tr = TRX[trm]
    Xtr, Xte = tr[f].to_numpy(np.float32), te[f].to_numpy(np.float32)
    ytr = (tr.y - tr.rata_4minggu).to_numpy(float) if resid else tr.y.to_numpy(float)
    w = recency_w(tr, start, sp["hl"]); htr, hte = tr.h.to_numpy(), te.h.to_numpy(); pred = np.full(len(te), np.nan)
    parts = [((htr >= tlo) & (htr <= thi), (hte > lo) & (hte <= hi)) for lo, hi, tlo, thi in BANDS] if sp["bands"] else [(np.ones(len(tr), bool), np.ones(len(te), bool))]
    for mtr, mte in parts:
        m = make_model(sp); m.fit(Xtr[mtr], ytr[mtr], sample_weight=None if w is None else w[mtr]); pred[mte] = m.predict(Xte[mte])
    assert np.isfinite(pred).all()
    return np.clip(pred + (te.rata_4minggu.to_numpy(float) if resid else 0.0), 0, 100)
def raw_pred(ph, sp, weekly=False):
    path = f"{OUT}/cache/{ph}_{hashlib.sha256((key(sp) + ('|mingguan' if weekly else '')).encode()).hexdigest()[:16]}.pkl"
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    t0 = time.time(); out = np.concatenate([fit_predict(sp, trm, te, a) for a, trm, te in folds(ph, weekly)])
    if weekly:
        o = pd.concat([te[["origin", "h"]] for _, _, te in folds(ph, True)], ignore_index=True); b = base_frame(ph)
        assert len(o) == len(b) and (o.origin.to_numpy() == b.origin.to_numpy()).all() and np.allclose(o.h, b.h), "urutan baris refit mingguan berbeda"
    pickle.dump(out, open(path, "wb")); log(f"  dilatih {ph:10s} {key(sp)}{' refit mingguan' if weekly else ''}  {time.time() - t0:.0f} dtk"); return out
def pred(ph, sp, weekly=False):
    """Prediksi dengan aturan fallback C3 v3: origin tanpa telemetri lengkap memakai kembaran set fitur CPU + kalender."""
    p = raw_pred(ph, sp, weekly)
    if sp["fs"] == "tele":
        fb = FALLBACK.reindex(base_frame(ph).origin).to_numpy()
        if fb.any(): p = np.where(fb, raw_pred(ph, dict(sp, fs="cpu"), weekly), p)
    return p

# ---------------------------------------------------------------- 5. metrik & uji (sama dengan v3)
def met(g, c):
    e = g[c] - g.y; ep = e[g.y >= HIGH]; base = (g.mean4w - g.y).abs().mean()
    return {"MAE": e.abs().mean(), "MAE relatif": e.abs().mean() / base, "RMSE": np.sqrt((e ** 2).mean()), "Bias": e.mean(), "MAE puncak": ep.abs().mean(), "Bias puncak": ep.mean()}
def dm_fixed_m(d):
    d = np.asarray(d, float); T = len(d); m = max(1, int(np.floor(T ** (1 / 3)))); u = d - d.mean(); t = np.arange(1, T + 1)
    lam = 2 * np.pi * np.arange(1, m + 1) / T
    I = np.abs(np.exp(-1j * np.outer(lam, t)) @ u) ** 2 / (2 * np.pi * T)
    se = np.sqrt(2 * np.pi * I.mean() / T); df_ = 2 * m; q = stats.t.ppf(.975, df_)
    return {"T": T, "m": m, "selisih": d.mean(), "ci_lo": d.mean() - q * se, "ci_hi": d.mean() + q * se, "p": 2 * stats.t.sf(abs(d.mean() / se), df_)}
def compare(g, a, b):
    d = ((g[a] - g.y).abs() - (g[b] - g.y).abs()).groupby(g.origin).mean().sort_index()
    r = dm_fixed_m(d.to_numpy()); r["Kesimpulan"] = "lebih baik" if r["ci_hi"] < 0 else "lebih buruk" if r["ci_lo"] > 0 else "tidak berbeda nyata"; return r

BV = base_frame("validation")
assert len(BV) == len(PV) and (BV.origin.to_numpy() == PV.origin.to_numpy()).all() and np.allclose(BV.h, PV.h), "urutan baris berbeda dari cache v3"
BV["v3_final"] = PV[FINAL].to_numpy()
for k_ in KINDS: BV[f"v3_{k_}"] = PV[spec_key_v3(BEST[(k_, "tele")])].to_numpy()
def vmae(col): return float((BV[col] - BV.y).abs().mean())
def fold_mae(col): return (BV[col] - BV.y).abs().groupby(BV.refit).mean()
SCREEN = []
def judge(name, col, ref, stage):
    """Aturan adopsi: turun >= MIN_GAIN relatif dan lebih baik di >= MIN_FOLDS fold."""
    a, b = vmae(col), vmae(ref); fw = int((fold_mae(col) < fold_mae(ref)).sum()); ok = bool(a <= b * (1 - MIN_GAIN) and fw >= MIN_FOLDS)
    SCREEN.append({"Tahap": stage, "Kandidat": name, "MAE validation": a, "Acuan": ref, "MAE acuan": b, "Perubahan (%)": 100 * (a / b - 1), "Fold lebih baik (dari 4)": fw,
                   "Diadopsi": "ya" if ok else "tidak", **{f"MAE fold {str(k.date())}": v for k, v in fold_mae(col).items()}})
    log(f"  [{stage}] {name}: MAE val {a:.4f} vs acuan {b:.4f} ({100 * (a / b - 1):+.2f} %), fold lebih baik {fw}/4 -> {'ADOPSI' if ok else 'tolak'}"); return ok

# ---------------------------------------------------------------- 6. koreksi bias sekuensial (galat yang sudah matang)
def bias_shift(stream, col, eval_origins, window, group, shrink, min_n=30):
    """Untuk tiap origin t: median (aktual - prediksi) dari galat yang SUDAH MATANG (target_time <= t) milik origin dalam `window` hari terakhir,
    per pita horizon (group='pita') atau per pita horizon x jam target (group='pita_jam'). Prediksi dikoreksi sebesar shrink x median itu."""
    st = stream[stream.y.notna()]; o, tt = st.origin.to_numpy(), st.target_time.to_numpy()
    r = (st.y - st[col]).to_numpy(); gid = (st.band_i * 24 + (st.jam if group == "pita_jam" else 0)).to_numpy().astype(int)
    W = np.timedelta64(window, "D"); res = {}
    for t in np.sort(np.asarray(pd.DatetimeIndex(eval_origins), dtype="datetime64[ns]")):
        m = (o >= t - W) & (tt <= t); s = pd.Series(r[m]).groupby(gid[m]).agg(["median", "size"])
        res[pd.Timestamp(t)] = (s["median"] * shrink).where(s["size"] >= min_n, 0.0)
    return res
def apply_bias(frame, stream, col, window, group, shrink):
    sh = bias_shift(stream, col, frame.origin.unique(), window, group, shrink)
    gid = (frame.band_i * 24 + (frame.jam if group == "pita_jam" else 0)).to_numpy().astype(int); c = np.zeros(len(frame))
    for t, idx in frame.groupby("origin").indices.items(): c[idx] = sh[t].reindex(gid[idx]).fillna(0.0).to_numpy()
    return np.clip(frame[col].to_numpy() + c, 0, 100)
def add_groups(f):
    f["band_i"] = pd.cut(f.h, BAND_EDGES, labels=False).astype(int); f["jam"] = (f.target_time - pd.Timedelta(hours=1)).dt.hour; return f
BIAS_GRID = [(w, g, s) for w in (14, 28) for g in ("pita", "pita_jam") for s in (0.5, 1.0)]

# prediksi v3 dari cache (kalibrasi/validation; test hanya dipanggil di tahap 2)
exec("".join(nb["cells"][37]["source"]).split("def stream_part")[0], G)
run_phase, POINT_SPECS = G["run_phase"], G["POINT_SPECS"]
_V3P = {}
def v3_preds(ph):
    if ph not in _V3P:
        if ph == "validation": P = PV
        else:
            specs = list(BEST.values()) if ph == "test" else POINT_SPECS + G["QSPECS"]
            (P, _), how = run_phase(ph, specs); assert how == "cache", f"cache v3 {ph} tidak ada"
            if ph == "test": P = G["add_ens"](G["apply_fallback"](P, specs))
        _V3P[ph] = P
    return _V3P[ph]
def v3_spec(k):
    b = BEST[(k, "tele")]; return dict(kind=k, fs="tele", target=b["target"], hl=b["hl"], n_iter=300, cap="s", extra=False, bands=False)
def frame(ph, newfam, weekly=False):
    """Bingkai fase: prediksi final v3 (cache) dan prediksi titik v4 (rata-rata 3 keluarga; keluarga tanpa spesifikasi baru memakai konfigurasi beku v3)."""
    b = add_groups(base_frame(ph)); P = v3_preds(ph)
    assert len(P) == len(b) and (P.origin.to_numpy() == b.origin.to_numpy()).all() and np.allclose(P.h, b.h), f"urutan baris {ph} berbeda dari cache v3"
    b["v3_final"] = P[FINAL].to_numpy() if FINAL in P else P[[spec_key_v3(s) for s in POINT_SPECS]].mean(axis=1).to_numpy()
    b["v4_titik"] = np.mean([pred(ph, newfam[k]) if k in newfam else P[spec_key_v3(BEST[(k, "tele")])].to_numpy() for k in KINDS], axis=0) if newfam else b["v3_final"]
    if weekly: b["v4_titik"] = np.mean([pred(ph, newfam.get(k) or v3_spec(k), True) for k in KINDS], axis=0)
    return b

# ================================================================= TAHAP 1: seleksi di validation
SELP = f"{OUT}/seleksi_v4_beku.json"
if not os.path.exists(SELP):
    assert not os.path.exists(f"{OUT}/test_v4_sudah_dijalankan.flag")
    v3x = BEST[("xgb", "tele")]; cur = dict(kind="xgb", fs="tele", target=v3x["target"], hl=v3x["hl"], n_iter=300, cap="s", extra=False, bands=False)
    BV["s0"] = pred("validation", cur)
    log(f"S0 reproduksi XGBoost v3: MAE val {vmae('s0'):.4f} (cache v3 {vmae('v3_xgb'):.4f})"); ref = "s0"
    # S1 kapasitas (jumlah pohon x kedalaman)
    best = None
    for n_iter in (300, 600, 1000):
        for cap in CAPS:
            if (n_iter, cap) == (300, "s"): continue
            sp = dict(cur, n_iter=n_iter, cap=cap); c = f"s1_{n_iter}_{cap}"; BV[c] = pred("validation", sp)
            if judge(f"XGBoost {n_iter} pohon, kedalaman {CAPS[cap]['depth']}", c, ref, "S1 kapasitas") and (best is None or vmae(c) < vmae(best[0])): best = (c, sp)
    if best: ref, cur = best
    # S2 fitur tambahan
    sp = dict(cur, extra=True); BV["s2"] = pred("validation", sp)
    if judge("+ fitur jam-sama 7 hari & statistik 4 minggu", "s2", ref, "S2 fitur"): ref, cur = "s2", sp
    # S3 half-life bobot waktu
    best = None
    for hl in (None, 45, 90, 180, 365):
        if hl == cur["hl"]: continue
        sp = dict(cur, hl=hl); c = f"s3_{hl}"; BV[c] = pred("validation", sp)
        if judge(f"half-life {hl} hari" if hl else "tanpa bobot waktu", c, ref, "S3 half-life") and (best is None or vmae(c) < vmae(best[0])): best = (c, sp)
    hl_changed = best is not None
    if best: ref, cur = best
    # S4 model per pita horizon
    sp = dict(cur, bands=True); BV["s4"] = pred("validation", sp)
    if judge("model terpisah per pita horizon (4 pita)", "s4", ref, "S4 pita horizon"): ref, cur = "s4", sp
    adopted = {k: cur[k] for k in ("n_iter", "cap", "extra", "bands")}
    changed = hl_changed or adopted != {"n_iter": 300, "cap": "s", "extra": False, "bands": False}
    log(f"pengaturan lolos screening XGBoost: {adopted}, half-life {'-> ' + str(cur['hl']) if hl_changed else 'tetap'}")
    # S5 per keluarga: pengaturan baru dipakai hanya bila keluarga itu sendiri membaik di validation
    newfam = {}
    if changed:
        for k in KINDS:
            b = BEST[(k, "tele")]; new = dict(kind=k, fs="tele", target=b["target"], hl=cur["hl"] if hl_changed else b["hl"], **adopted)
            BV[f"s5_{k}"] = pred("validation", new)
            if judge(f"{G['FAM'][k]} dengan pengaturan baru", f"s5_{k}", f"v3_{k}", "S5 per keluarga"): newfam[k] = new
    # S6 ensemble
    use_ens = False
    if newfam:
        BV["ens_v4"] = np.mean([BV[f"s5_{k}"] if k in newfam else BV[f"v3_{k}"] for k in KINDS], axis=0)
        use_ens = judge("Ensemble GBDT v4 (rata-rata 3 keluarga)", "ens_v4", "v3_final", "S6 ensemble")
    if not use_ens: newfam = {}
    # S7 koreksi bias sekuensial pada prediksi titik (riwayat galat awal: April = fase kalibrasi, tidak dinilai)
    FK, FV = frame("kalibrasi", newfam), frame("validation", newfam)
    assert np.allclose(FV.v3_final, BV.v3_final) and (not use_ens or np.allclose(FV.v4_titik, BV.ens_v4))
    BV["titik"] = FV.v4_titik.to_numpy(); add_groups(BV); STREAM = pd.concat([FK, FV], ignore_index=True)
    bbest = None
    for w, g, s in BIAS_GRID:
        c = f"s7_{w}_{g}_{s}"; BV[c] = apply_bias(FV, STREAM, "v4_titik", w, g, s)
        if judge(f"koreksi bias: jendela {w} hari, per {'pita x jam target' if g == 'pita_jam' else 'pita horizon'}, shrink {s}", c, "titik", "S7 koreksi bias") and \
           (bbest is None or vmae(c) < vmae(bbest[0])): bbest = (c, (w, g, s))
    final_col = bbest[0] if bbest else "titik"
    sel = {"dibekukan": time.strftime("%Y-%m-%d %H:%M:%S"),
           "aturan": f"adopsi bila MAE validation turun >= {100 * MIN_GAIN:g} % relatif dan lebih baik di >= {MIN_FOLDS} dari 4 fold bulanan; urutan tahap S1-S7 tetap",
           "ada_perubahan": bool(use_ens or bbest), "ensemble_baru": use_ens, "spesifikasi_keluarga_baru": newfam,
           "keluarga_tetap_v3": [k for k in KINDS if k not in newfam],
           "koreksi_bias": dict(zip(("jendela_hari", "kelompok", "shrink"), bbest[1])) if bbest else None,
           "mae_validation": {"v3_final": vmae("v3_final"), "v4_tanpa_koreksi_bias": vmae("titik"), "v4_final": vmae(final_col), "rata_4minggu": vmae("mean4w")},
           "kapasitas": CAPS, "fitur_tambahan": XCOLS, "pita_training": BANDS, "seleksi_v3_sha256": open("outputs_v3_baru/seleksi_beku.json.sha256").read().strip(),
           "data_sha256": G["DATA_SHA"], "data_cutoff": G["DATA_CUTOFF"], "versi_pustaka": G["VERSIONS"]}
    pd.DataFrame(SCREEN).to_csv(f"{OUT}/seleksi_validation.csv", index=False)
    json.dump(sel, open(SELP, "w"), indent=2, default=str); open(SELP + ".sha256", "w").write(sha_file(SELP))
    log(f"SELEKSI DIBEKUKAN: ensemble baru={use_ens} {list(newfam)}, koreksi bias={sel['koreksi_bias']}, MAE val {sel['mae_validation']}")
assert sha_file(SELP) == open(SELP + ".sha256").read().strip(), "seleksi v4 berubah setelah dibekukan"
sel = json.load(open(SELP))

# ================================================================= PUTARAN 2 (sebelum test): refit mingguan, aturan adopsi sama
# Diagnosis putaran 1: yang membantu adalah penyesuaian terhadap pergeseran pola terkini, bukan kapasitas/fitur. Latih ulang lebih sering adalah
# penanganan baku concept drift (Bayram dkk., 2022). Kandidat: ensemble yang sama dilatih ulang tiap 7 hari + koreksi bias beku putaran 1.
# Acuan: hasil putaran 1. Parameter koreksi bias TIDAK dipilih ulang.
SELP2 = f"{OUT}/seleksi_v4_putaran2_beku.json"
if not os.path.exists(SELP2):
    assert not os.path.exists(f"{OUT}/test_v4_sudah_dijalankan.flag")
    bc, nf = sel["koreksi_bias"], sel["spesifikasi_keluarga_baru"]; add_groups(BV)
    def final_of(weekly):
        FK, FV = frame("kalibrasi", nf, weekly), frame("validation", nf, weekly)
        return FV.v4_titik.to_numpy(), (apply_bias(FV, pd.concat([FK, FV], ignore_index=True), "v4_titik", bc["jendela_hari"], bc["kelompok"], bc["shrink"]) if bc else FV.v4_titik.to_numpy())
    BV["r1_titik"], BV["r1_final"] = final_of(False); BV["r2_titik"], BV["r2_final"] = final_of(True)
    n0 = len(SCREEN)
    judge("refit mingguan, tanpa koreksi bias (informasi)", "r2_titik", "r1_titik", "P2 refit mingguan")
    ok2 = judge("refit mingguan + koreksi bias beku putaran 1", "r2_final", "r1_final", "P2 refit mingguan")
    sel2 = {"dibekukan": time.strftime("%Y-%m-%d %H:%M:%S"), "aturan": sel["aturan"] + "; acuan = hasil putaran 1; parameter koreksi bias tidak dipilih ulang",
            "refit_mingguan": ok2, "mae_validation": {"putaran1_final": vmae("r1_final"), "refit_mingguan_tanpa_koreksi": vmae("r2_titik"), "refit_mingguan_final": vmae("r2_final"),
                                                      "v4_final": vmae("r2_final" if ok2 else "r1_final")}, "seleksi_putaran1_sha256": open(SELP + ".sha256").read().strip()}
    pd.concat([pd.read_csv(f"{OUT}/seleksi_validation.csv"), pd.DataFrame(SCREEN[n0:])], ignore_index=True).to_csv(f"{OUT}/seleksi_validation.csv", index=False)
    json.dump(sel2, open(SELP2, "w"), indent=2, default=str); open(SELP2 + ".sha256", "w").write(sha_file(SELP2))
    log(f"PUTARAN 2 DIBEKUKAN: refit mingguan={ok2}, MAE val {sel2['mae_validation']}")
assert sha_file(SELP2) == open(SELP2 + ".sha256").read().strip(), "seleksi putaran 2 berubah setelah dibekukan"
sel2 = json.load(open(SELP2)); WEEKLY = bool(sel2["refit_mingguan"])
if "--hanya-seleksi" in sys.argv: sys.exit(0)

# ================================================================= TAHAP 2: test sekali dengan seleksi beku
FR = {ph: frame(ph, sel["spesifikasi_keluarga_baru"], WEEKLY) for ph in ("kalibrasi", "validation", "test")}
if not os.path.exists(f"{OUT}/test_v4_sudah_dijalankan.flag"):
    open(f"{OUT}/test_v4_sudah_dijalankan.flag", "w").write(f"test v4 dijalankan {time.strftime('%Y-%m-%d %H:%M:%S')} dengan seleksi sha256 {open(SELP + '.sha256').read()[:12]}")
STREAM = pd.concat([FR[p] for p in ("kalibrasi", "validation", "test")], ignore_index=True)
bc = sel["koreksi_bias"]
for ph in ("validation", "test"):
    f = FR[ph]
    f["v4_final"] = apply_bias(f, STREAM, "v4_titik", bc["jendela_hari"], bc["kelompok"], bc["shrink"]) if bc else f["v4_titik"]
    # ablation (dihitung SETELAH seleksi beku, tidak dipakai untuk memilih apa pun): koreksi bias yang sama di atas model v3
    f["v3_plus_koreksi"] = apply_bias(f, STREAM, "v3_final", bc["jendela_hari"], bc["kelompok"], bc["shrink"]) if bc else f["v3_final"]
ALL = pd.concat([FR["validation"], FR["test"]], ignore_index=True); ALL["bulan"] = ALL.origin.dt.strftime("%Y-%m")
ALL["pita"] = pd.cut(ALL.h, BAND_EDGES, labels=["1–6 jam", "7–24 jam", "25–72 jam", "73–120 jam"])
MODELS = {"snaive": "Naif musiman mingguan", "mean4w": "Rata-rata 4 minggu", "v3_final": "SEBELUM: model final v3 (Ensemble GBDT + telemetri)",
          "v3_plus_koreksi": "Ablasi: v3 + koreksi bias", "v4_titik": "Ablasi: v4 tanpa koreksi bias", "v4_final": "SESUDAH: model v4"}
FASE = {"validation": "Validation", "test": "Test"}
ALL[["origin", "target_time", "h", "y", "phase", "refit"] + list(MODELS)].to_csv(f"{OUT}/prediksi_v4.csv.gz", index=False, float_format="%.4f")
pd.DataFrame([{"Fase": FASE[ph], "Model": n, **met(g, c)} for ph, g in ALL.groupby("phase", sort=False) for c, n in MODELS.items()]).to_csv(f"{OUT}/metrik_sebelum_sesudah.csv", index=False)
rows = []
for ph, g in ALL.groupby("phase", sort=False):
    for a, b in (("v4_final", "v3_final"), ("v4_final", "mean4w"), ("v3_final", "mean4w"), ("v4_final", "snaive"), ("v4_titik", "v3_final"), ("v3_plus_koreksi", "v3_final")):
        r = compare(g, a, b); rows.append({"Fase": FASE[ph], "Model A": MODELS[a], "Dibanding (B)": MODELS[b], "Selisih MAE (A − B)": r["selisih"], "CI 95% bawah": r["ci_lo"],
                                           "CI 95% atas": r["ci_hi"], "p": r["p"], "T origin": r["T"], "m": r["m"], "Kesimpulan": r["Kesimpulan"]})
pd.DataFrame(rows).to_csv(f"{OUT}/uji_dm_v4.csv", index=False)
K3 = ["mean4w", "v3_final", "v4_final"]
t = ALL.groupby(["phase", "pita"], observed=True, sort=False).apply(lambda g: pd.Series({c: (g[c] - g.y).abs().mean() for c in K3})).reset_index()
t.assign(phase=t.phase.map(FASE)).to_csv(f"{OUT}/mae_per_pita.csv", index=False)
rows = []
for bl, g in ALL.groupby("bulan"):
    r = compare(g, "v4_final", "v3_final")
    rows.append({"Bulan origin": bl, "Fase": FASE[g.phase.iloc[0]], "Origin": g.origin.nunique(), **{c: (g[c] - g.y).abs().mean() for c in K3},
                 "Selisih v4 − v3": r["selisih"], "CI 95% bawah": r["ci_lo"], "CI 95% atas": r["ci_hi"], "Kesimpulan": r["Kesimpulan"]})
pd.DataFrame(rows).to_csv(f"{OUT}/mae_per_bulan.csv", index=False)
ALL.assign(jam=ALL.jam).groupby(["phase", "jam"]).apply(lambda g: pd.Series({f"bias_{c}": (g[c] - g.y).mean() for c in K3} | {f"mae_{c}": (g[c] - g.y).abs().mean() for c in K3})) \
   .reset_index().to_csv(f"{OUT}/bias_per_jam.csv", index=False)
ALL.groupby(["phase", "h"]).apply(lambda g: pd.Series({c: (g[c] - g.y).abs().mean() for c in K3})).reset_index().to_csv(f"{OUT}/mae_per_horizon.csv", index=False)
log("SELESAI. Metrik:\n" + pd.read_csv(f"{OUT}/metrik_sebelum_sesudah.csv").round(3).to_string() + "\n" + pd.read_csv(f"{OUT}/uji_dm_v4.csv").round(3).to_string())
