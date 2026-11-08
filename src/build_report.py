import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_DIRS = ["/System/Library/Fonts/Supplemental", "C:/Windows/Fonts", "/usr/share/fonts/truetype/msttcorefonts"]


def register_fonts():
    for d in FONT_DIRS:
        for reg, bold in (("Arial.ttf", "Arial Bold.ttf"), ("arial.ttf", "arialbd.ttf")):
            if Path(d, reg).exists() and Path(d, bold).exists():
                pdfmetrics.registerFont(TTFont("R", str(Path(d, reg))))
                pdfmetrics.registerFont(TTFont("B", str(Path(d, bold))))
                registerFontFamily("R", normal="R", bold="B", italic="R", boldItalic="B")
                return "R", "B"
    return "Helvetica", "Helvetica-Bold"


def load(d: Path, name: str) -> dict:
    f = d / name
    return json.loads(f.read_text()) if f.exists() else {}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    preview = "--preview" in sys.argv
    sub_dir = Path(args[0]) if args else Path("submissions")
    out = Path(args[1]) if len(args) > 1 else Path("report/raport_tehnic.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    l1, l2, l3 = (load(sub_dir, f"level{i}_cv.json") for i in (1, 2, 3))
    l3e = load(sub_dir, "level3_cv_entropy1.json")

    R, B = register_fonts()
    st = {
        "h0": ParagraphStyle("h0", fontName=B, fontSize=14, leading=17, spaceAfter=2),
        "sub": ParagraphStyle("sub", fontName=R, fontSize=8.5, leading=11, textColor=colors.grey, spaceAfter=6),
        "h": ParagraphStyle("h", fontName=B, fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2),
        "p": ParagraphStyle("p", fontName=R, fontSize=8.8, leading=11.6, spaceAfter=3),
        "warn": ParagraphStyle("w", fontName=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#b00020")),
    }
    P = lambda t, s="p": Paragraph(t, st[s])
    f = lambda v, n=4: f"{v:.{n}f}" if isinstance(v, (int, float)) else "–"

    story = [P("AI Cyber Defense Challenge — Raport tehnic", "h0"),
             P("Echipa: Eunoia · Membri: Drangoi Karina, Secrieru Maxim · ZTS XIII, 8 noiembrie 2026", "sub")]
    if preview:
        story.append(P("PREVIZUALIZARE — cifrele provin dintr-o repetiție pe date publice UNSW-NB15 (nu datele concursului).", "warn"))

    story += [
        P("1. Pregătirea datelor", "h"),
        P("Am folosit toate cele 39 de caracteristici numerice (fără <i>id</i>). Valorile lipsă sau ne-numerice → 0, "
          "valorile infinite → 0. Fiind modele pe arbori, nu am scalat datele pentru clasificare. Am adăugat 8 caracteristici "
          "derivate care exprimă interacțiuni greu de învățat din separări simple: raport octeți src/dst, raport pachete, "
          "octeți per pachet (src și dst), volum total, perechea TTL src·1000+dst (amprenta gazdei generatoare), "
          "indicator „destinația nu răspunde” (dst_bytes = 0) și raportul încărcărilor. "
          "Clasele rare (Worms, Shellcode, Backdoor, Analysis) au primit ponderi per eșantion = (n<sub>max</sub>/n<sub>c</sub>)<super>0.5</super> — "
          "rădăcina pătrată ridică clasele mici fără să distrugă precizia pe cele mari. Pentru componenta kNN de la Nivelul 3 "
          "am folosit log1p + standardizare (distanțele sunt dominate altfel de octeți)."),
        P("2. Modelele pe niveluri", "h"),
        P("<b>Nivelul 1 (binar).</b> LightGBM (≤1200 arbori, lr 0.05, 63 frunze, early stopping), validare încrucișată "
          f"stratificată 5-fold, iar predicția pe test = media celor 5 modele. Pragul de decizie ales pe predicțiile out-of-fold "
          f"pentru F1 maxim (prag {f(l1.get('threshold'), 2)})."),
        P("<b>Nivelul 2 (10 clase).</b> LightGBM multiclasă cu ponderile de clasă de mai sus, 5-fold. Macro F1 tratează "
          "egal fiecare clasă, așa că după antrenare am căutat (coordinate ascent pe predicțiile out-of-fold) un multiplicator "
          "per clasă al probabilităților care maximizează macro F1 — mută frontierele de decizie spre clasele rare."),
        P("<b>Nivelul 3 (open-set).</b> Clasificator LightGBM pe cele 7 clase cunoscute (aceeași rețetă ca la nivelul 2). "
          "<i>unknown_score</i> = medie ponderată de ranguri (normalizate în [0,1]) a patru semnale: (a) 1 − probabilitatea maximă, "
          "(b) entropia distribuției de probabilitate, (c) distanța medie la cei mai apropiați 10 vecini din train în spațiul log-standardizat, "
          "(d) un clasificator <i>train-vs-test</i> (fără etichete): train și test sunt din aceeași distribuție cu excepția claselor noi, "
          "deci fluxurile care „arată a test” sunt candidați la Unknown. Ponderile au fost alese printr-o simulare "
          "<i>leave-one-class-out</i>: pe rând am ascuns fiecare clasă de atac din train, am rărit-o la ~2% în validare (prevalență realistă) "
          "și am ales combinația de ponderi cu AUROC mediu maxim. Pragul pentru eticheta Unknown: marcăm cele mai suspecte "
          f"{f(100 * l3.get('unknown_frac', 0), 1)}% fluxuri din test — fracția estimată din diferența de dimensiune "
          "între train-ul nivelului 3 (~139,3k) și cel al nivelului 2 (~141,7k), adică rândurile claselor scoase. "
          "Simularea a arătat că macro F1 pe 8 etichete e aproape plat între 1% și 2%, deci alegerea e robustă."),
        P("3. Experimente și testare", "h"),
    ]

    rows = [["Nivel", "Validare (out-of-fold)", "Scor pe test" + (" (repetiție)" if preview else ""), "Timp"],
            ["1 — F1 binar", f(l1.get("oof_f1@best")), f(l1.get("mock_score")), f"{l1.get('runtime_s', '–')} s"],
            ["2 — Macro F1 (10)", f"{f(l2.get('oof_macro_f1_raw'))} → {f(l2.get('oof_macro_f1_tuned'))} (scalare)",
             f(l2.get("mock_score")), f"{l2.get('runtime_s', '–')} s"],
            ["3 — 0.6·F1 + 0.4·AUROC", f"AUROC simulat {f(l3.get('loco_blend_auroc'))}",
             f"{f(l3.get('mock_score'))} (blend) / {f(l3e.get('mock_score'))} (entropie)", f"{l3.get('runtime_s', '–')} s"]]
    t = Table(rows, colWidths=[4.2 * cm, 6.1 * cm, 5.0 * cm, 1.6 * cm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), R, 8.3), ("FONT", (0, 0), (-1, 0), B, 8.3),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 4)]

    pc = l2.get("per_class_f1", {})
    weak = ", ".join(f"{k} {v:.2f}" for k, v in sorted(pc.items(), key=lambda kv: kv[1])[:3]) if pc else "–"
    la = l3.get("loco_auroc", {})
    story += [
        P("Validare: 5-fold stratificat pe train pentru toate nivelurile, iar pentru Nivelul 3 simularea leave-one-class-out "
          "descrisă mai sus (AUROC mediu per semnal: " + ", ".join(f"{k} {v:.2f}" for k, v in la.items()) + "). "
          f"Clasele cele mai slabe la nivelul 2: {weak} — fluxurile Analysis/Backdoor/DoS au adesea vectori de caracteristici "
          "identici cu etichete diferite, deci niciun model nu le poate separa complet."),
        P("<b>Ce nu a mers:</b> (1) Regresia logistică din starter kit pe date brute — mult sub arbori, sensibilă la scara octeților. "
          "(2) Numărarea duplicatelor exacte (un vector apare în test mai des decât prezice train-ul → clasă nouă): AUROC ≈ 0.47, "
          "nu ajută. (3) Distanța kNN singură: bună când clasa ascunsă e diferită (Generic, AUROC 0.91), dar sub hazard pentru clasele "
          "noi care seamănă cu DoS/Exploits — de aceea ponderea ei a fost redusă. (4) Ponderile alese de simulare nu s-au transferat "
          "perfect: la Nivelul 3 entropia singură a fost la fel de bună ca blend-ul, așa că am trimis ambele variante."),
        P("4. Concluzii și utilizarea AI", "h"),
        P("Boosting pe arbori + ponderare a claselor + ajustarea pragurilor pe predicții out-of-fold a dat cel mai mult. "
          "La open-set, incertitudinea clasificatorului (entropia) a fost cel mai stabil semnal, iar semnalele de densitate ajută doar "
          "când clasele noi sunt diferite de cele cunoscute."),
        P("<b>Utilizarea AI:</b> Am folosit AI pentru structurarea codului și a raportului."),
    ]
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                      title="AI Cyber Defense Challenge - Raport tehnic").build(story)
    print("wrote", out)


if __name__ == "__main__":
    main()
