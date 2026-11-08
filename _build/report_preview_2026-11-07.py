import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("UTM_REPORT_TOOLS", str(Path.home() / "Projects/utm-report-tools")))
from reportkit.pipeline import run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANUL3 = ROOT.parent.parent
SUB = Path(os.environ.get("SUB", ROOT / "submissions_mock"))
PREVIEW = os.environ.get("PREVIEW", "1") == "1"
OUT = Path(os.environ.get("OUT", ROOT / "Raport_tehnic_ZTS2026_Eunoia"))
FIGS = HERE / "figuri"
TITLE_KEEP = [i for i in range(36) if i != 12]
WITH_FIG = os.environ.get("FIG", "0") == "1"


def load(n):
    f = SUB / n
    return json.loads(f.read_text()) if f.exists() else {}


L1, L2, L3, L3E = load("level1_cv.json"), load("level2_cv.json"), load("level3_cv.json"), load("level3_cv_entropy1.json")


def f(v, n=4):
    return f"{v:.{n}f}".replace(".", ",") if isinstance(v, (int, float)) else "–"


def cover(r):
    r.set_title({"RAPORT": "RAPORT TEHNIC",
                 "la cursul": "la concursul „AI Cyber Defense Challenge” – ZTS, ediția a XIII-a",
                 "Tema:": "Tema: Detectarea atacurilor cibernetice în traficul de rețea prin învățare automată"})
    r.replace_text("A efectuat:", "Echipa „Eunoia”:")
    r.replace_text("St. gr. R-241, Drangoi Karina", "Drangoi Karina, Secrieru Maxim")
    r.replace_text("A verificat:", "Organizator:")
    r.replace_text("Asist. univ. Tutunaru Vladina", "OSACE × RSI × Universitatea din Craiova")
    r.replace_text("Chișinău 2026", "Craiova 2026")
    r.require("Departamentul Ingineria Sistemelor")
    r.title = [r.title[i] for i in TITLE_KEEP]


def build(r):
    cover(r)

    r.h1("INTRODUCERE", new_page=False)
    if PREVIEW:
        r.body_text("**Previzualizare: cifrele provin dintr-o repetiție pe setul public UNSW-NB15 adus la formatul concursului.**")
    r.body_text("**Scopul:** un model care deosebește traficul de rețea normal de atacuri pe trei niveluri: detectarea atacului, "
                "recunoașterea tipului de atac și izolarea tipurilor de atac absente la antrenare. Fiecare rând este un flux de "
                "rețea cu 39 de caracteristici numerice, train ~141 700 rânduri, test ~38 650 rânduri.")
    r.chapter(1)
    r.table([["Nivel", "Sarcina", "Fișier", "Metrica", "Pondere"],
             ["1", "Normal (0) / Atac (1)", "id,label", "F1 binar", "20%"],
             ["2", "10 clase: Normal + 9 atacuri", "id,attack_cat", "Macro F1", "30%"],
             ["3", "7 clase cunoscute + Unknown", "id,attack_cat,unknown_score", "0,6·Macro F1 + 0,4·AUROC", "30%"]],
            [600, 2900, 2800, 2500, 900], "Nivelurile concursului", sz=19,
            col_align=["center", "left", "left", "left", "center"])

    r.h1("1 PREGĂTIREA DATELOR", new_page=False)
    r.p("Au fost folosite toate cele 39 de caracteristici, fără id. Valorile lipsă, ne-numerice sau infinite au devenit 0. "
        "Modelele pe arbori nu cer scalare, de aceea datele nu au fost normalizate pentru clasificare. Au fost adăugate 8 "
        "caracteristici derivate: raportul octeților și al pachetelor sursă/destinație, octeți per pachet pe fiecare direcție, "
        "volumul total, perechea TTL sursă·1000 + destinație (amprenta gazdei), indicatorul „destinația nu răspunde” și raportul "
        "încărcărilor. Clasele rare (Worms, Shellcode, Backdoor, Analysis au de zeci de ori mai puține exemple decât Normal) au "
        "primit ponderi w = (n_max / n_c)^0,5, care le ridică fără să strice precizia pe clasele mari. Pentru semnalul kNN de la "
        "nivelul 3 datele au fost logaritmate (log1p) și standardizate.")

    r.h1("2 MODELELE PE NIVELURI", new_page=False)
    r.p("**Nivelul 1.** LightGBM (până la 1200 de arbori, rata 0,05, 63 de frunze, oprire timpurie) cu validare încrucișată "
        "stratificată în 5 părți. Predicția pe test este media celor cinci modele, iar pragul de decizie a fost ales pe "
        f"predicțiile out-of-fold pentru F1 maxim ({f(L1.get('threshold'), 2)}).")
    r.p("**Nivelul 2.** Același model, multiclasă, cu ponderile de clasă. Pe predicțiile out-of-fold a fost căutat câte un "
        "multiplicator al probabilității fiecărei clase care maximizează macro F1. Multiplicatorii mută frontierele spre clasele "
        f"rare și cresc macro F1 de la {f(L2.get('oof_macro_f1_raw'))} la {f(L2.get('oof_macro_f1_tuned'))}.")
    r.p("**Nivelul 3.** Clasificatorul pentru cele 7 clase cunoscute este construit ca la nivelul 2. Scorul unknown_score este "
        "media ponderată a rangurilor (în [0, 1]) a patru semnale: 1 − probabilitatea maximă, entropia distribuției de "
        "probabilitate, distanța medie la cei mai apropiați 10 vecini din train și un clasificator train-vs-test antrenat fără "
        "etichete (train și test diferă doar prin clasele noi, deci fluxurile care „arată a test” sunt candidați la Unknown). "
        "Ponderile au fost alese printr-o simulare leave-one-class-out: fiecare clasă de atac a fost scoasă pe rând din train, "
        "rărită la ~2% în validare, iar combinația cu AUROC mediu maxim a fost păstrată. Ca Unknown sunt marcate cele mai suspecte "
        f"{f(100 * L3.get('unknown_frac', 0), 1)}% fluxuri din test, fracție estimată din diferența dintre train-ul nivelului 3 "
        "(~139 300 rânduri) și cel al nivelului 2 (~141 700 rânduri).")

    r.h1("3 EXPERIMENTE ȘI TESTARE", new_page=False)
    r.chapter(3)
    r.p("Toate modelele au fost verificate prin validare încrucișată în 5 părți. Înainte de concurs, programul a fost repetat "
        "pe setul public UNSW-NB15 adus la aceleași coloane, iar fișierele au fost punctate local cu metricile platformei.")
    rows = [["Nivel", "Validare (out-of-fold)", "Scor pe test", "Timp, s"],
            ["1 – F1 binar", f(L1.get("oof_f1@best")), f(L1.get("mock_score")), str(L1.get("runtime_s", "–"))],
            ["2 – Macro F1", f"{f(L2.get('oof_macro_f1_raw'))} → {f(L2.get('oof_macro_f1_tuned'))}",
             f(L2.get("mock_score")), str(L2.get("runtime_s", "–"))],
            ["3 – compozit", f"AUROC simulat {f(L3.get('loco_blend_auroc'))}",
             f"{f(L3.get('mock_score'))} blend, {f(L3E.get('mock_score'))} entropie", str(L3.get("runtime_s", "–"))]]
    r.table(rows, [2300, 2700, 3400, 1300], "Rezultatele pe niveluri" + (" (repetiție)" if PREVIEW else ""), sz=19,
            col_align=["left", "center", "center", "center"])
    if WITH_FIG and L3.get("loco_auroc_per_class"):
        r.figure("loco", "AUROC al fiecărui semnal în simularea leave-one-class-out",
                 "Entropia e stabilă pentru toate clasele ascunse, kNN și train-vs-test ajută doar pentru clase foarte diferite (Generic).",
                 maxw=400)
    r.p("La nivelul 2 cele mai slabe clase sunt Backdoor, Analysis și DoS: au adesea vectori de caracteristici identici cu "
        "etichete diferite, deci niciun model nu le separă complet. **Ce nu a funcționat:** regresia logistică din starter kit "
        "(mult sub arbori), numărarea duplicatelor exacte dintre train și test (AUROC ≈ 0,47), distanța kNN singură (sub hazard "
        "pentru clasele noi asemănătoare cu DoS și Exploits, de aceea ponderea ei a fost redusă). Ponderile din simulare nu s-au "
        "transferat perfect, entropia singură fiind la fel de bună, de aceea la nivelul 3 au fost trimise ambele variante.")

    r.h1("CONCLUZII", new_page=False)
    r.p("Cel mai mult au contat boosting-ul pe arbori, ponderarea claselor rare și ajustarea pragurilor pe predicții "
        "out-of-fold, nu pe clasament. La open-set, entropia clasificatorului a fost cel mai stabil semnal. Limita principală vine "
        "din date: fluxurile Analysis, Backdoor și DoS cu caracteristici identice nu pot fi separate, iar clasele noi seamănă "
        "tocmai cu acestea.")
    r.p("**Utilizarea AI:** Am folosit AI-ul Claude de la Anthropic pentru ghidare.")

    r.h1("BIBLIOGRAFIE", new_page=False)
    for it in ["https://hackathon.osace.ro/guide",
               "https://lightgbm.readthedocs.io/en/stable/",
               "[N. Moustafa, J. Slay. UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS, 2015.]",
               "[D. Hendrycks, K. Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples. ICLR, 2017.]"]:
        r.source(it)


run(build, template=ANUL3 / "sem_1" / "AI" / "Lab1_AI_Braccio.docx", out=OUT, figures=FIGS, toc_heading="Cuprins")
