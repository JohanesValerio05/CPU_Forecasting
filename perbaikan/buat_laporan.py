"""Menyusun laporan PDF: hasil run notebook forecasting, evaluasi, dan perbandingan sebelum/sesudah perbaikan.

Pemakaian: python perbaikan/buat_laporan.py   (butuh pandas, matplotlib, reportlab)
Keluaran : laporan_forecasting_sebelum_sesudah.pdf di folder proyek.
"""
import os, re, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
V3, R30, R15, PB, EX = "outputs_v3_baru", "outputs_30m_baru", "outputs_15m_baru", "outputs_perbaikan", "hasil_eksekusi"
PDF = "laporan_forecasting_sebelum_sesudah.pdf"
FIG = f"{PB}/figures"; os.makedirs(FIG, exist_ok=True)
fd = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DV", os.path.join(fd, "DejaVuSans.ttf"))); pdfmetrics.registerFont(TTFont("DVB", os.path.join(fd, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")
INK, INK2, GRIDC, C_BASE, C_V3, C_V4 = "#0b0b0b", "#52514e", "#e6e5e0", "#52514e", "#6250d6", "#eb6834"   # palet notebook v3 (sudah dicek CVD)
S = {"h1": ParagraphStyle("h1", fontName="DVB", fontSize=17, leading=22, spaceAfter=4, textColor=INK),
     "h2": ParagraphStyle("h2", fontName="DVB", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, textColor=INK, keepWithNext=True),
     "h3": ParagraphStyle("h3", fontName="DVB", fontSize=10, leading=13, spaceBefore=7, spaceAfter=3, textColor=INK, keepWithNext=True),
     "p": ParagraphStyle("p", fontName="DV", fontSize=9, leading=13, spaceAfter=5, textColor=INK),
     "s": ParagraphStyle("s", fontName="DV", fontSize=7.6, leading=10.5, textColor=INK2, spaceAfter=4),
     "c": ParagraphStyle("c", fontName="DV", fontSize=7.3, leading=9.2, textColor=INK),
     "ch": ParagraphStyle("ch", fontName="DVB", fontSize=7.3, leading=9.2, textColor=INK),
     "cap": ParagraphStyle("cap", fontName="DVB", fontSize=8.3, leading=11, spaceBefore=6, spaceAfter=3, textColor=INK),
     "b": ParagraphStyle("b", fontName="DV", fontSize=9, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=2, textColor=INK)}
story = []
koma = lambda t: re.sub(r"(?<=\d)\.(?=\d)", ",", t)          # desimal koma di teks naratif dan sel angka
def P(t, st="p"): story.append(Paragraph(koma(t) if st == "p" else t, S[st]))
def bullets(items):
    for t in items: story.append(Paragraph(koma(t), S["b"], bulletText="•"))
def rd(path):
    return pd.read_csv(path) if os.path.exists(path) else None
def table(df, caption=None, fmt=None, widths=None, hl=None, note=None):
    """df -> tabel; fmt: {kolom: format}; hl: fungsi(baris) -> True untuk baris yang ditebalkan."""
    fmt = fmt or {}; W = 17.4 * cm
    def cell(v, c, bold=False):
        if isinstance(v, float) and np.isnan(v): t = "–"
        elif c in fmt and isinstance(v, (int, float, np.floating, np.integer)): t = koma(fmt[c].format(v))
        else: t = str(v)
        return Paragraph(t.replace("&", "&amp;").replace("<", "&lt;"), S["ch" if bold else "c"])
    data = [[Paragraph(str(c), S["ch"]) for c in df.columns]] + [[cell(r[c], c, bool(hl and hl(r))) for c in df.columns] for _, r in df.iterrows()]
    if widths is None:
        ln = [max(len(str(c)), df[c].astype(str).str.len().clip(upper=46).max() if len(df) else 0) for c in df.columns]; widths = [W * max(l, 5) / sum(max(l, 5) for l in ln) for l in ln]
    else: widths = [W * w / sum(widths) for w in widths]
    for name, mn in (("Fase", 1.75 * cm), ("T origin", 1.15 * cm), ("Kesimpulan", 2.3 * cm), ("Diadopsi", 1.45 * cm)):      # kolom sempit jangan terpotong di tengah kata
        if name in df.columns:
            i = list(df.columns).index(name)
            if widths[i] < mn: j = int(np.argmax(widths)); widths[j] -= mn - widths[i]; widths[i] = mn
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), .8, colors.HexColor(INK)), ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor(GRIDC)),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0eb")), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if hl:
        for i, (_, r) in enumerate(df.iterrows(), 1):
            if hl(r): st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fbf3ee")))
    t.setStyle(TableStyle(st)); blk = ([Paragraph(caption, S["cap"])] if caption else []) + [t] + ([Paragraph(note, S["s"])] if note else [])
    story.append(KeepTogether(blk) if len(df) <= 22 else blk[0]);
    if len(df) > 22: story.extend(blk[1:])
    story.append(Spacer(1, 5))
def image(path, caption=None, width=17.4 * cm, note=None):
    if not os.path.exists(path): return
    from PIL import Image as PI
    w, h = PI.open(path).size; hh = width * h / w
    if hh > 21 * cm: width, hh = width * 21 * cm / hh, 21 * cm
    story.append(KeepTogether(([Paragraph(caption, S["cap"])] if caption else []) + [Image(path, width=width, height=hh)] + ([Paragraph(note, S["s"])] if note else [])))
    story.append(Spacer(1, 5))
plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": .6, "font.size": 8.5,
                     "axes.titlesize": 9.5, "axes.titlelocation": "left", "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.axisbelow": True})
FASES = ["Validation", "Test"]

# ------------------------------------------------------------------ data
sel = json.load(open(f"{PB}/seleksi_v4_beku.json")); M = rd(f"{PB}/metrik_sebelum_sesudah.csv"); DM = rd(f"{PB}/uji_dm_v4.csv"); SC = rd(f"{PB}/seleksi_validation.csv")
PITA, BULAN, JAM, HOR = rd(f"{PB}/mae_per_pita.csv"), rd(f"{PB}/mae_per_bulan.csv"), rd(f"{PB}/bias_per_jam.csv"), rd(f"{PB}/mae_per_horizon.csv")
T10, DM3 = rd(f"{V3}/metrik.csv"), rd(f"{V3}/uji_dm.csv"); PROB, STAB, RIDGE, WL = rd(f"{V3}/metrik_probabilistik.csv"), rd(f"{V3}/stabilitas_bulanan.csv"), rd(f"{V3}/metrik_ridge.csv"), rd(f"{V3}/weighted_loss_puncak.csv")
M30, M15, DMR3 = rd(f"{R30}/metrik_per_model.csv"), rd(f"{R15}/metrik_per_model.csv"), rd(f"{R15}/uji_dm_tiga_resolusi.csv")
NB, NA = "SEBELUM: model final v3 (Ensemble GBDT + telemetri)", "SESUDAH: model v4"
def m(fase, model, col): return float(M[(M.Fase == fase) & (M.Model == model)][col].iloc[0])
def dm(fase, a, b): return DM[(DM.Fase == fase) & (DM["Model A"] == a) & (DM["Dibanding (B)"] == b)].iloc[0]
sel2 = json.load(open(f"{PB}/seleksi_v4_putaran2_beku.json")); berubah = bool(sel["ada_perubahan"] or sel2["refit_mingguan"])
for ab in [x for x in M.Model.unique() if x.startswith("Ablasi")]:      # ablasi yang identik dengan baris sebelum/sesudah tidak ditampilkan
    if any(np.allclose(M[M.Model == ab].MAE.to_numpy(), M[M.Model == o].MAE.to_numpy()) for o in (NB, NA)):
        M = M[M.Model != ab].reset_index(drop=True); DM = DM[DM["Model A"] != ab].reset_index(drop=True)
runlog = open(f"{EX}/run_log.txt", encoding="utf-8").read() if os.path.exists(f"{EX}/run_log.txt") else ""
runs = [(a, int(b), float(c)) for a, b, c in re.findall(r"END (\S+) rc=(\d+) ([\d.]+) min", runlog)]

# ------------------------------------------------------------------ gambar
def fig_horizon():
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.1), sharey=True)
    for ax, ph in zip(axs, ("validation", "test")):
        g = HOR[HOR.phase == ph].sort_values("h")
        for c, col, ls, lab in (("mean4w", C_BASE, "--", "Rata-rata 4 minggu"), ("v3_final", C_V3, "-", "Sebelum (v3)"), ("v4_final", C_V4, "-", "Sesudah (v4)")):
            s = g.set_index("h")[c].rolling(6, center=True, min_periods=1).mean(); ax.plot(s.index, s.values, color=col, ls=ls, lw=2 if c == "v4_final" else 1.5, label=lab)
        ax.set_title(ph.capitalize()); ax.set_xlabel("Horizon (jam ke depan)"); ax.set_xlim(1, 120); ax.set_xticks([1, 24, 48, 72, 96, 120])
    axs[0].set_ylabel("MAE (pp, rata-rata bergerak 6 jam)"); axs[0].set_ylim(0, None); axs[0].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3)
    fig.tight_layout(); fig.savefig(f"{FIG}/L1_mae_per_horizon.png"); plt.close(fig)
def fig_jam():
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True)
    for ax, ph in zip(axs, ("validation", "test")):
        g = JAM[JAM.phase == ph].sort_values("jam"); ax.axhline(0, color=INK2, lw=.8)
        for c, col, lab in (("v3_final", C_V3, "Sebelum (v3)"), ("v4_final", C_V4, "Sesudah (v4)")):
            ax.plot(g.jam, g[f"bias_{c}"], color=col, lw=1.8, marker="o", ms=3.5, label=lab)
        ax.set_title(ph.capitalize()); ax.set_xlabel("Jam target (awal interval, WIB)"); ax.set_xticks(range(0, 24, 3))
    axs[0].set_ylabel("Bias = prediksi − aktual (pp)"); axs[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/L2_bias_per_jam.png"); plt.close(fig)
def fig_bulan():
    fig, ax = plt.subplots(figsize=(9.2, 2.9)); x = np.arange(len(BULAN)); w = .26
    for i, (c, col, lab) in enumerate((("mean4w", C_BASE, "Rata-rata 4 minggu"), ("v3_final", C_V3, "Sebelum (v3)"), ("v4_final", C_V4, "Sesudah (v4)"))):
        b = ax.bar(x + (i - 1) * w, BULAN[c], w - .03, color=col, label=lab)
        for xi, v in zip(x + (i - 1) * w, BULAN[c]): ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n({f})" for b, f in zip(BULAN["Bulan origin"], BULAN.Fase)]); ax.set_ylabel("MAE (pp)"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper right"); ax.set_ylim(0, BULAN[["mean4w", "v3_final", "v4_final"]].max().max() * 1.18)
    fig.tight_layout(); fig.savefig(f"{FIG}/L3_mae_per_bulan.png"); plt.close(fig)
fig_horizon(); fig_jam(); fig_bulan()

# ------------------------------------------------------------------ isi
F3 = {"MAE": "{:.2f}", "MAE relatif": "{:.3f}", "RMSE": "{:.2f}", "Bias": "{:+.2f}", "MAE puncak": "{:.2f}", "Bias puncak": "{:+.2f}"}
FD = {"Selisih MAE (A − B)": "{:+.3f}", "CI 95% bawah": "{:+.3f}", "CI 95% atas": "{:+.3f}", "p": "{:.3f}", "T origin": "{:.0f}", "m": "{:.0f}"}
exec(open(os.path.join(ROOT, "perbaikan", "isi_laporan.py"), encoding="utf-8").read())

def footer(c, d):
    c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor(INK2))
    c.drawString(1.8 * cm, 1.1 * cm, "Laporan forecasting CPU 120 jam · sebelum vs sesudah perbaikan"); c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"hal. {d.page}"); c.restoreState()
SimpleDocTemplate(PDF, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                  title="Laporan forecasting CPU: sebelum vs sesudah perbaikan", author="Claude Code").build(story, onFirstPage=footer, onLaterPages=footer)
print("ditulis", os.path.abspath(PDF), f"{os.path.getsize(PDF) / 1024:.0f} KB")
