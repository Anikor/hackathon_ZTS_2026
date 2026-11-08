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
PREVIEW = os.environ.get("PREVIEW", "0") == "1"
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


LIVE = {"L1": os.environ.get("L1", "95,88%"), "L2": os.environ.get("L2", "61,17%"), "L3": os.environ.get("L3", "83,94%")}


def build(r):
    cover(r)

    r.h1("INTRODUCERE", new_page=False)
    r.body_text("**Scopul:** un model care deosebește traficul de rețea normal de atacuri pe trei niveluri: detectarea atacului, "
                "recunoașterea tipului de atac și izolarea tipurilor de atac absente la antrenare. Fiecare rând este un flux de "
                "rețea cu 39 de caracteristici numerice (train ~141 700 rânduri, test ~38 650 rânduri pe nivel). Metricile sunt "
                "F1 binar (nivelul 1), macro F1 pe 10 clase (nivelul 2) și 0,6·macro F1 pe 8 etichete + 0,4·AUROC pentru Unknown "
                "(nivelul 3).")

    r.h1("1 PREGĂTIREA DATELOR", new_page=False)
    r.p("Au fost folosite toate cele 39 de caracteristici, fără id, plus 8 derivate: raportul octeților și al pachetelor "
        "sursă/destinație, octeți per pachet pe fiecare direcție, volumul total, perechea TTL sursă·1000 + destinație, "
        "indicatorul „destinația nu răspunde” și raportul încărcărilor. Valorile lipsă sau infinite au devenit 0, iar modelele "
        "pe arbori nu au cerut scalare. Analiza datelor a arătat că nu există niciun rând duplicat, deși octeții și pachetele "
        "sunt numere întregi intacte: perturbate sunt doar durata și rata fluxului. Clasele rare au primit ponderi "
        "w = (n_max / n_c)^p cu p între 0,5 și 1, iar pentru kNN datele au fost logaritmate și standardizate.")

    r.h1("2 MODELELE PE NIVELURI", new_page=False)
    r.p("Toate modelele au fost evaluate prin validare încrucișată stratificată în 5 părți, iar fiecare decizie (combinație, "
        "prag, ponderi pe clase) a fost luată pe predicțiile out-of-fold, nu pe clasament. Antrenarea a rulat pe un calculator "
        "cu 28 de nuclee și o placă RTX 4070, controlat prin SSH: XGBoost și CatBoost pe GPU, restul pe CPU.")
    r.p("**Nivelul 1.** Au fost comparate LightGBM, ExtraTrees, RandomForest, XGBoost, CatBoost și HistGradientBoosting. "
        "Combinația aleasă greedy pe F1 out-of-fold a fost ExtraTrees (două configurații) + LightGBM, cu pragul optimizat "
        "pe aceleași predicții (0,475).")
    r.p("**Nivelul 2.** Modele de bază: trei variante LightGBM (ponderi de clasă 0,5 / 0,75 / 1), ExtraTrees, RandomForest și "
        "XGBoost. Peste probabilitățile lor out-of-fold (70 de valori pe rând, în logaritm) a fost antrenat un al doilea "
        "LightGBM mic (stacking), tot în 5 părți, care învață ce model să creadă pentru fiecare clasă. Predicția finală "
        "este media stacking ×2 + LightGBM ×2 + ExtraTrees, urmată de câte un multiplicator pe clasă ales prin optimizare pe "
        "coordonate pentru macro F1.")
    r.p("**Nivelul 3.** Clasificatorul pentru cele 7 clase cunoscute este construit ca la nivelul 2, cu XGBoost și CatBoost "
        "pe GPU în stacking. Scorul unknown_score combină ranguri ale entropiei probabilităților, distanței kNN până la train și "
        "ale unui clasificator train-vs-test fără etichete, cu ponderi alese printr-o simulare leave-one-class-out (fiecare "
        "clasă de atac scoasă pe rând din train). Pasul decisiv a fost pseudo-etichetarea: cele mai suspecte 12% din fluxurile "
        "de test sunt tratate ca Unknown și separate de train de un model antrenat încrucișat, iar scorul final este media "
        "rangurilor celor două scoruri. Un model cu 8 clase, antrenat cu aceste rânduri ca a opta clasă, contribuie cu 30% la "
        "scorul final. Ca Unknown sunt marcate primele ~1,7% fluxuri (650–657), fracție estimată din diferența dintre "
        "train-ul nivelului 3 (~139 300 rânduri) și cel al nivelului 2 (~141 700 rânduri) și confirmată de ruptura de pantă "
        "a scorului.")

    r.h1("3 EXPERIMENTE ȘI TESTARE", new_page=False)
    r.chapter(3)
    rows = [["Nivel", "Prima variantă", "Varianta finală", "Validare finală", "Clasament"],
            ["1 – F1 binar", "95,76%", "ExtraTrees + LightGBM", "0,9579", LIVE["L1"]],
            ["2 – Macro F1", "59,66%", "stacking + LightGBM + ET", "0,6200", LIVE["L2"]],
            ["3 – compozit", "75,83%", "stacking + pseudo-etichetare", "F1 cunoscute 0,6958", LIVE["L3"]]]
    r.table(rows, [1800, 1700, 3000, 1900, 1300], "Evoluția scorurilor pe niveluri", sz=19,
            col_align=["left", "center", "left", "center", "center"])
    r.p("Diferențele de peste ~0,5 puncte pe validare s-au regăsit pe test, cele sub 0,2 nu. **Ce nu a funcționat:** "
        "caracteristicile de frecvență, mai multe seed-uri pentru același model și combinațiile cu 9–11 modele (câștig sub "
        "zgomot), CatBoost pe CPU (prea lent pe 10 clase), semnăturile pe valori exacte ale octeților (aceleași valori apar la "
        "Exploits, DoS, Analysis și Backdoor, acuratețe leave-one-out 47%) și pseudo-etichetarea în două iterații (82,75% față "
        "de 83,40%, erorile se amplifică).")

    r.h1("CONCLUZII", new_page=False)
    r.p("Cel mai mult au contat stacking-ul peste modele diferite și pseudo-etichetarea pe test pentru atacurile noi, care a "
        "ridicat nivelul 3 cu aproape 8 puncte. Limita principală vine din date: clasele Analysis, Backdoor și DoS au fluxuri "
        "practic identice, deci macro F1 la nivelul 2 rămâne în jur de 0,6 indiferent de model.")
    r.p("**Utilizarea AI:** Am folosit AI pentru structurarea codului și a raportului.")

    r.h1("BIBLIOGRAFIE", new_page=False)
    for it in ["https://hackathon.osace.ro/guide",
               "https://lightgbm.readthedocs.io/en/stable/",
               "[D. H. Wolpert. Stacked generalization. Neural Networks, 5(2), 1992.]",
               "[D. Hendrycks, K. Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples. ICLR, 2017.]"]:
        r.source(it)


run(build, template=ANUL3 / "sem_1" / "AI" / "Lab1_AI_Braccio.docx", out=OUT, figures=FIGS, toc_heading="Cuprins")
