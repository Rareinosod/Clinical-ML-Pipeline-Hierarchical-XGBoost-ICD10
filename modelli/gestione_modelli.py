"""
gestione_modelli.py
=====================
Funzioni condivise per tutti gli script modello. Centralizza:
  - Scelta iniziale in 3 step: Modello, Metrica (F1/F2), K interno.
  - Esecuzione BATCH automatica su tutti gli esperimenti.
  - Creazione rigida di cartelle separate per F1 e F2.
  - Metriche, Grafici diagnostici e Permutation Importance.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, validation_curve, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    classification_report, confusion_matrix, make_scorer,
)
from sklearn.inspection import permutation_importance

import optuna
from optuna.samplers import TPESampler


# =====================================================================
# VARIABILI GLOBALI DI STATO (Impostate all'inizio, poi blindate)
# =====================================================================

N_SPLITS_INTERNO = 3           
SEED = 42                      

BETA_METRICA = 1.0
METRICA_PRIMARIA = "f1_macro"
NOME_METRICA_PRIMARIA = "F1-Score (macro)"
# Usata dall'orchestratore per passare il run corrente allo script modello
RUN_FORZATO: Path | None = None
ATTIVA_FALLBACK_CLASSE_1 = False  # <-- NUOVA VARIABILE: Disattivata di default


# Usata dall'orchestratore per passare il run corrente allo script modello
RUN_FORZATO: Path | None = None


def imposta_metrica_primaria(scelta: str) -> None:
    """Imposta la metrica globalmente per tutto il flusso (Optuna, Grafici, Importance)."""
    global BETA_METRICA, METRICA_PRIMARIA, NOME_METRICA_PRIMARIA
    
    if scelta == "1":
        BETA_METRICA = 1.0
        METRICA_PRIMARIA = "f1_macro"
        NOME_METRICA_PRIMARIA = "F1-Score (macro)"
    elif scelta == "2":
        BETA_METRICA = 2.0
        # sklearn non ha una stringa pronta per f2_macro, creiamo lo scorer
        METRICA_PRIMARIA = make_scorer(fbeta_score, beta=2.0, average="macro", zero_division=0)
        NOME_METRICA_PRIMARIA = "F2-Score (macro)"
    else:
        raise ValueError("Scelta metrica non valida.")
    print(f"\n-> [SISTEMA] Metrica primaria bloccata su: {NOME_METRICA_PRIMARIA}")


def get_cv_interna(n_splits: int = None) -> StratifiedKFold:
    if n_splits is None:
        n_splits = N_SPLITS_INTERNO
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def cartella_output_modello(cartella_run: Path, nome_modello: str, sottocartella: str | None = None) -> Path:
    """Crea cartelle distinte in base alla metrica E alla scelta del fallback.
       Mantiene il nome originale se è Baseline per compatibilità."""
    nome_base = f"Risultati_{nome_modello}_F{int(BETA_METRICA)}"
    
    # Se usiamo il fallback, aggiungiamo il suffisso. Altrimenti lasciamo originale.
    if ATTIVA_FALLBACK_CLASSE_1:
        nome_base += "_Fallback"
        
    percorso = cartella_run / nome_base
    if sottocartella is not None:
        percorso = percorso / sottocartella
    return percorso


# =====================================================================
# SEZIONE 1 - INDIVIDUAZIONE PROGETTO E DATI
# =====================================================================

def trova_cartella_progetto() -> Path:
    return Path(__file__).resolve().parent.parent

def mostra_log_esperimento(cartella_run: Path) -> str | None:
    percorso_log = cartella_run / "log_esperimento.txt"
    print("\n" + "-" * 70)
    print(f"LOG ESPERIMENTO: {cartella_run.name}")
    print("-" * 70)
    if percorso_log.is_file():
        testo = percorso_log.read_text(encoding="utf-8")
        print(testo)
    else:
        testo = None
        print("   (Nessun log_esperimento.txt trovato per questo run)")
    print("-" * 70)
    return testo


@dataclass
class Compartimento:
    nome: str
    cartella_split: Path


def rileva_compartimenti(cartella_run: Path) -> list[Compartimento]:
    """
        Rileva automaticamente la strategia di split ESTERNO usata in questo
        run guardando la struttura di cartelle. Gestisce Holdout, K-Fold e Bootstrap.
        """
    cartelle_fold = sorted([d for d in cartella_run.iterdir() if d.is_dir() and d.name.startswith("Fold_")], key=lambda d: d.name)
    if cartelle_fold:
        print(f"\n-> Strategia rilevata: K-FOLD ESTERNO ({len(cartelle_fold)} compartimenti)")
        return [Compartimento(nome=d.name, cartella_split=d / "split") for d in cartelle_fold]

    cartelle_boot = sorted([d for d in cartella_run.iterdir() if d.is_dir() and d.name.startswith("Bootstrap_")], key=lambda d: d.name)
    if cartelle_boot:
        print(f"\n-> Strategia rilevata: BOOTSTRAP ({len(cartelle_boot)} compartimenti)")
        return [Compartimento(nome=d.name, cartella_split=d / "split") for d in cartelle_boot]

    cartella_split_diretta = cartella_run / "split"
    if cartella_split_diretta.is_dir():
        print("\n-> Strategia rilevata: HOLDOUT CLASSICO (1 compartimento)")
        return [Compartimento(nome="Holdout", cartella_split=cartella_split_diretta)]

    raise FileNotFoundError(f"Nessuna cartella valida trovata in {cartella_run}.")


def carica_dati_compartimento(cartella_split: Path) -> dict:
    """
        Carica X_train/X_test/y_train/y_test da una cartella 'split/', scegliendo
        automaticamente il livello di elaborazione piu' avanzato disponibile:
        X_train_fs -> X_train_ready -> X_train.
        """
    candidati = [("X_train_fs.csv", "X_test_fs.csv"), ("X_train_ready.csv", "X_test_ready.csv"), ("X_train.csv", "X_test.csv")]
    file_train, file_test = None, None
    for nome_train, nome_test in candidati:
        if (cartella_split / nome_train).is_file() and (cartella_split / nome_test).is_file():
            file_train, file_test = nome_train, nome_test
            break

    if file_train is None:
        raise FileNotFoundError(f"Nessun file X_train in {cartella_split}")

    X_train = pd.read_csv(cartella_split / file_train)
    X_test = pd.read_csv(cartella_split / file_test)
    y_train = pd.read_csv(cartella_split / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(cartella_split / "y_test.csv").squeeze("columns")

    print(f"   File X usato: {file_train} / {file_test}")
    print(f"   Dimensioni: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "file_x_usato": file_train}


def prepara_esperimento(nome_modello: str) -> tuple[Path, list[Compartimento]]:
    """Usato all'interno degli script dei modelli. Ora legge solo le variabili globali."""
    if RUN_FORZATO is None:
        raise RuntimeError("Errore: RUN_FORZATO non impostato. Usa la funzione 'avvia_pipeline_completa()' per avviare i modelli.")
    
    print("\n" + "#" * 70)
    print(f"# AVVIO MODELLO: {nome_modello.upper()} su {RUN_FORZATO.name}")
    print("#" * 70)
    print(f"-> Metrica: {NOME_METRICA_PRIMARIA}")
    print(f"-> K CV Interna: {N_SPLITS_INTERNO}")

    mostra_log_esperimento(RUN_FORZATO)
    compartimenti = rileva_compartimenti(RUN_FORZATO)
    return RUN_FORZATO, compartimenti


# =====================================================================
# SEZIONE 2 - MOTORE OPTUNA E VALUTAZIONE
# =====================================================================

def ottimizza_con_optuna(funzione_obiettivo, n_trials: int = 30):
    """
    Motore Optuna generico e condiviso. Non conosce il modello specifico:
    esegue solo la funzione obiettivo che ogni script modello definisce
    per conto proprio (dove viene scelto lo spazio di ricerca e come viene
    costruito/valutato IL PROPRIO modello).

    Usa un sampler con seed fisso per rendere la ricerca riproducibile.
    """
    print(f"\n[OPTUNA] Avvio ottimizzazione con {n_trials} tentativi "
          f"(CV interna: {N_SPLITS_INTERNO} fold, metrica: {NOME_METRICA_PRIMARIA})...")

    # Sopprime i log eccessivi di Optuna per mantenere pulito il terminale
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(funzione_obiettivo, n_trials=n_trials)

    print("\n" + "=" * 50)
    print(" OTTIMIZZAZIONE COMPLETATA")
    print("=" * 50)
    print(f"Miglior {NOME_METRICA_PRIMARIA} (CV): {study.best_value:.4f}")
    print("Parametri ottimali:")
    for chiave, valore in study.best_params.items():
        print(f"   - {chiave}: {valore}")

    return study.best_params, study.best_value

def trova_soglia_fallback_ottimale(modello, X_train, y_train):
    print(f"\n   [!] Ricerca soglia di Fallback ottimale tramite simulazione Out-Of-Fold...")
    probabilita_oof = cross_val_predict(modello, X_train, y_train, cv=get_cv_interna(), method='predict_proba', n_jobs=-1)
    
    miglior_soglia = 0.51
    miglior_score = 0
    
    for soglia in np.linspace(0.35, 0.80, 46):
        y_pred_modificata = []
        for prob in probabilita_oof:
            prob_classe_0 = prob[0]
            prob_classe_2 = prob[2] if len(prob) > 2 else 0
            
            if prob_classe_0 >= soglia and prob_classe_0 > prob_classe_2:
                y_pred_modificata.append(0)
            elif prob_classe_2 >= soglia and prob_classe_2 > prob_classe_0:
                y_pred_modificata.append(2)
            else:
                y_pred_modificata.append(1)
                
        score = fbeta_score(y_train, y_pred_modificata, beta=BETA_METRICA, average='macro', zero_division=0)
        
        if score > miglior_score:
            miglior_score = score
            miglior_soglia = soglia
            
    print(f"   >>> Soglia di Fallback matematicamente ottimale: {miglior_soglia:.2f} (Score OOF: {miglior_score:.4f})")
    return miglior_soglia








import numpy as np

def applica_fallback_classe_1(modello, X_test, y_pred_originale, soglia_sicurezza=0.51):
    if not hasattr(modello, "predict_proba"):
        print("   [ATTENZIONE] Il modello non supporta predict_proba. Fallback ignorato.")
        return y_pred_originale
        
    probabilita = modello.predict_proba(X_test)
    y_pred_modificata = []
    
    for prob in probabilita:
        prob_classe_0 = prob[0]
        prob_classe_2 = prob[2] if len(prob) > 2 else 0
        
        if prob_classe_0 >= soglia_sicurezza and prob_classe_0 > prob_classe_2:
            y_pred_modificata.append(0)
        elif prob_classe_2 >= soglia_sicurezza and prob_classe_2 > prob_classe_0:
            y_pred_modificata.append(2)
        else:
            y_pred_modificata.append(1)
            
    differenze = sum(1 for a, b in zip(y_pred_originale, y_pred_modificata) if a != b)
    if differenze > 0:
         print(f"   [FALLBACK ATTIVO] {differenze} pazienti ambigui reindirizzati forzatamente alla Classe 1.")
         
    return np.array(y_pred_modificata)




def valuta_prestazioni(y_test, y_pred, nome_modello: str, cartella_output: Path | None = None, modello=None, X_test=None, soglia_fallback=0.51) -> dict:
    
    global ATTIVA_FALLBACK_CLASSE_1
    
    # --- LOGICA DI FALLBACK CLINICO ---
    if ATTIVA_FALLBACK_CLASSE_1 and modello is not None and X_test is not None:
        y_pred = applica_fallback_classe_1(modello, X_test, y_pred, soglia_sicurezza=soglia_fallback)

    print(f"\n{'=' * 60}")
    print(f" VALUTAZIONE PRESTAZIONI: {nome_modello}")
    print(f"{'=' * 60}")

    # Calcolo metriche standard globali
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)

    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    # Calcolo metrica primaria (F-beta)
    fbeta_macro = fbeta_score(y_test, y_pred, beta=BETA_METRICA, average='macro', zero_division=0)
    fbeta_per_classe = fbeta_score(y_test, y_pred, beta=BETA_METRICA, average=None, zero_division=0)

    # Stampa a terminale
    print(f"\nAccuracy               : {accuracy:.4f}")
    print(f"{NOME_METRICA_PRIMARIA} [*]  : {fbeta_macro:.4f}   <- metrica primaria (tuning)")
    
    if BETA_METRICA != 1.0:
        print(f"F1-Score  (macro)       : {f1_macro:.4f}   (per confronto storico)")
        
    print(f"Precision (macro)       : {prec_macro:.4f}")
    print(f"Recall    (macro)       : {rec_macro:.4f}")
    print(f"F1-Score  (weighted)    : {f1_weighted:.4f}")
    print(f"Precision (weighted)    : {prec_weighted:.4f}")
    print(f"Recall    (weighted)    : {rec_weighted:.4f}")

    report_testo = classification_report(y_test, y_pred, zero_division=0)
    print("\nReport dettagliato per classe (Standard F1):")
    print(report_testo)

    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    classi_str = [k for k in report_dict.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

    # --- Generazione Matrice di Confusione ---
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice di Confusione - {nome_modello}\n({NOME_METRICA_PRIMARIA}: {fbeta_macro:.3f})')
    plt.xlabel('Classe Predetta')
    plt.ylabel('Classe Reale')
    plt.tight_layout()

    # --- Generazione Barplot DINAMICO per classe (F1 o F2) ---
    fig_bar = plt.figure(figsize=(8, 5))
    sns.barplot(x=classi_str, y=fbeta_per_classe, hue=classi_str, legend=False, palette="viridis")
    plt.axhline(fbeta_macro, color="red", linestyle="--", linewidth=1, label=f"Media {NOME_METRICA_PRIMARIA} ({fbeta_macro:.3f})")
    plt.title(f'{NOME_METRICA_PRIMARIA} per classe - {nome_modello}')
    plt.ylabel(NOME_METRICA_PRIMARIA)
    plt.xlabel('Classe')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    # --- Costruzione stringa report personalizzata ---
    report_extra = f"\n--- {NOME_METRICA_PRIMARIA.upper()} PER CLASSE ---\n"
    for cls_name, fbeta_val in zip(classi_str, fbeta_per_classe):
        report_extra += f"Classe {cls_name}: {fbeta_val:.4f}\n"

    print(report_extra)

    # --- Salvataggio su Disco ---
    if cartella_output is not None:
        cartella_output = Path(cartella_output)
        cartella_output.mkdir(parents=True, exist_ok=True)
        
        fig_cm.savefig(cartella_output / f"matrice_confusione_{nome_modello}.png", dpi=300)
        
        # Rinomina dinamicamente il file del barplot in base alla metrica
        nome_file_bar = f"f2_per_classe_{nome_modello}.png" if BETA_METRICA == 2.0 else f"f1_per_classe_{nome_modello}.png"
        fig_bar.savefig(cartella_output / nome_file_bar, dpi=300)
        
        with open(cartella_output / f"report_{nome_modello}.txt", "w", encoding="utf-8") as f:
            f.write(f"=== REPORT PRESTAZIONI - {nome_modello} ===\n\n")
            f.write(f"Accuracy            : {accuracy:.4f}\n")
            f.write(f"{NOME_METRICA_PRIMARIA}  : {fbeta_macro:.4f}  <- metrica primaria\n")
            f.write(f"F1-Score  (macro)   : {f1_macro:.4f}\n")
            f.write(f"Precision (macro)   : {prec_macro:.4f}\n")
            f.write(f"Recall    (macro)   : {rec_macro:.4f}\n")
            f.write(f"F1-Score  (weighted): {f1_weighted:.4f}\n")
            f.write(f"Precision (weighted): {prec_weighted:.4f}\n")
            f.write(f"Recall    (weighted): {rec_weighted:.4f}\n")
            f.write(report_extra)
            f.write("\n--- CLASSIFICATION REPORT (Standard sklearn) ---\n")
            f.write(report_testo)
            
        print(f"\n-> Salvati: matrice_confusione_{nome_modello}.png, "
              f"{nome_file_bar}, report_{nome_modello}.txt")

    plt.show()
    plt.close(fig_cm)
    plt.close(fig_bar)

    return {
        "Modello": nome_modello,
        "Accuracy": accuracy,
        "F_Target_macro": fbeta_macro,
        "F1_macro": f1_macro,
        "Precision_macro": prec_macro,
        "Recall_macro": rec_macro,
        "F1_weighted": f1_weighted,
        "Precision_weighted": prec_weighted,
        "Recall_weighted": rec_weighted,
    }

# =====================================================================
# SEZIONE 3 - DIAGNOSTICA E IMPORTANZA FEATURE
# =====================================================================



def calcola_permutation_importance(modello, X_test, y_test, n_repeats: int = 5) -> pd.DataFrame:
    """
    Calcola la permutation importance di un modello gia' addestrato, sul
    test set del compartimento corrente. Va richiamata UNA VOLTA per ogni
    compartimento (dentro il ciclo for), poi i risultati dei vari
    compartimenti si aggregano con 'aggrega_permutation_importance'.
    """
    risultato = permutation_importance(
        modello, X_test, y_test,
        n_repeats=n_repeats, random_state=SEED, n_jobs=-1, scoring=METRICA_PRIMARIA,
    )
    return pd.DataFrame({
        "Feature": X_test.columns,
        "Importanza_Media": risultato.importances_mean,
        "Importanza_Std": risultato.importances_std,
    })


def aggrega_permutation_importance(lista_df_importanza: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggrega la permutation importance calcolata su piu' compartimenti:
    media tra i vari Fold_X nel caso K-Fold esterno, passthrough se c'e'
    un solo compartimento (Holdout). Ritorna il risultato ordinato dalla
    feature piu' importante alla meno importante.
    """
    df_concat = pd.concat(lista_df_importanza, ignore_index=True)
    df_media = df_concat.groupby("Feature", as_index=False)["Importanza_Media"].mean()
    return df_media.sort_values("Importanza_Media", ascending=False).reset_index(drop=True)


def plot_feature_importance(df_importanza: pd.DataFrame, nome_modello: str,
                             top_n: int = 15, cartella_output: Path | None = None):
    """
    Barplot delle 'top_n' feature per importanza (permutation importance),
    salvato su disco (PNG + CSV completo) se 'cartella_output' e' fornita.
    """
    df_top = df_importanza.head(top_n)

    fig = plt.figure(figsize=(9, max(5, 0.35 * len(df_top))))
    sns.barplot(x="Importanza_Media", y="Feature", data=df_top, hue="Feature", legend=False, palette="viridis")
    plt.title(f"Top {top_n} Feature - Permutation Importance\n{nome_modello}")
    plt.xlabel(f"Calo medio di {NOME_METRICA_PRIMARIA} quando la feature viene permutata")
    plt.ylabel("Feature")
    plt.tight_layout()

    if cartella_output is not None:
        cartella_output = Path(cartella_output)
        cartella_output.mkdir(parents=True, exist_ok=True)
        fig.savefig(cartella_output / f"feature_importance_{nome_modello}.png", dpi=300)
        df_importanza.to_csv(cartella_output / f"feature_importance_{nome_modello}.csv", index=False)
        print(f"   -> Salvati: feature_importance_{nome_modello}.png / .csv")

    plt.show()
    plt.close(fig)







def plot_curva_boosting(modello_finale, X_train, y_train, X_test, y_test, nome_modello: str, cartella_output: Path | None = None):
    """
    Per modelli di BOOSTING SEQUENZIALE (AdaBoost e simili) che espongono
    'staged_predict': plotta il Macro-F1 su train e su test dopo ogni
    singolo round di boosting. Se la curva di train continua a salire
    mentre quella di test si appiattisce/scende, e' overfitting visibile
    rispetto al numero di round.
    """
    if not hasattr(modello_finale, "staged_predict"):
        print(f"   [plot_curva_boosting] '{nome_modello}' non espone 'staged_predict', grafico saltato.")
        return

    scores_train = [
        fbeta_score(y_train, y_pred_stage, beta=BETA_METRICA,average='macro', zero_division=0)
        for y_pred_stage in modello_finale.staged_predict(X_train)
    ]
    scores_test = [
        fbeta_score(y_test, y_pred_stage, beta=BETA_METRICA, average='macro', zero_division=0)
        for y_pred_stage in modello_finale.staged_predict(X_test)
    ]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores_train) + 1), scores_train, label=f"Train ({NOME_METRICA_PRIMARIA})")
    plt.plot(range(1, len(scores_test) + 1), scores_test, label=f"Test ({NOME_METRICA_PRIMARIA})")
    plt.xlabel("Numero di round di boosting")
    plt.ylabel(NOME_METRICA_PRIMARIA)
    plt.title(f"Curva di Boosting - {nome_modello}")
    plt.legend()
    plt.tight_layout()

    if cartella_output is not None:
        cartella_output = Path(cartella_output)
        cartella_output.mkdir(parents=True, exist_ok=True)
        fig.savefig(cartella_output / f"curva_boosting_{nome_modello}.png", dpi=300)
        print(f"   -> Salvato: curva_boosting_{nome_modello}.png")

    plt.show()
    plt.close(fig)






def plot_validation_curve_condivisa(modello_base, X_train, y_train,
                                     nome_parametro: str, range_valori,
                                     nome_modello: str, cartella_output: Path | None = None):
    """
    Per modelli SENZA round sequenziali (SVM, KNN, Random Forest...): fa
    variare un iperparametro chiave su un range e plotta train-score vs
    validation-score (CV interna condivisa) al variare di quel parametro.
    Stessa logica diagnostica della curva di boosting, ma sull'asse x c'e'
    un iperparametro invece del numero di round.
    """
    print(f"\n[VALIDATION CURVE] Calcolo per '{nome_parametro}' "
          f"su {len(range_valori)} valori (puo' richiedere qualche minuto)...")

    train_scores, test_scores = validation_curve(
        modello_base, X_train, y_train,
        param_name=nome_parametro, param_range=range_valori,
        cv=get_cv_interna(), scoring=METRICA_PRIMARIA, n_jobs=-1,
    )

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range_valori, train_scores.mean(axis=1), 'o-', label="Train")
    plt.plot(range_valori, test_scores.mean(axis=1), 'o-', label="Validation (CV)")
    plt.xlabel(nome_parametro)
    plt.ylabel(METRICA_PRIMARIA)
    plt.title(f"Validation Curve - {nome_modello} ({nome_parametro})")
    plt.legend()
    plt.tight_layout()

    if cartella_output is not None:
        cartella_output = Path(cartella_output)
        cartella_output.mkdir(parents=True, exist_ok=True)
        fig.savefig(cartella_output / f"validation_curve_{nome_modello}_{nome_parametro}.png", dpi=300)
        print(f"   -> Salvato: validation_curve_{nome_modello}_{nome_parametro}.png")

    plt.show()
    plt.close(fig)



def aggrega_risultati_compartimenti(lista_risultati: list[dict], nome_modello: str,
                                     cartella_output: Path | None = None) -> pd.DataFrame:
    """
    Aggrega i dizionari restituiti da 'valuta_prestazioni' provenienti da
    più compartimenti (i vari Fold_X o Bootstrap_X). Se è presente un
    solo compartimento (Holdout), riporta semplicemente quel valore, senza
    calcolare la deviazione standard.
    """
    df = pd.DataFrame(lista_risultati)

    print("\n" + "=" * 60)
    print(f" RISULTATI AGGREGATI - {nome_modello} ({len(df)} compartimento/i)")
    print("=" * 60)

    for colonna in df.columns:
        # Si ignorano le colonne descrittive o fisse nel calcolo delle medie
        if colonna in ["Modello", "Beta"]:
            continue
            
        if len(df) > 1:
            print(f"   {colonna:20s}: {df[colonna].mean():.4f} +/- {df[colonna].std():.4f}")
        else:
            print(f"   {colonna:20s}: {df[colonna].iloc[0]:.4f}")

    if cartella_output is not None:
        cartella_output = Path(cartella_output)
        cartella_output.mkdir(parents=True, exist_ok=True)
        # Salvataggio del DataFrame completo su disco
        percorso_file = cartella_output / f"risultati_per_compartimento_{nome_modello}.csv"
        df.to_csv(percorso_file, index=False)
        print(f"\n-> Salvato: risultati_per_compartimento_{nome_modello}.csv")

    return df




# =====================================================================
# SEZIONE 4 - IL NUOVO ORCHESTRATORE (Le 3 Domande)
# =====================================================================


def avvia_pipeline_completa():
    """
    Sostituisce il vecchio meccanismo di run. 
    L'esecuzione inizia lanciando questa funzione, la quale richiede 
    modello, metrica e K, per poi iterare automaticamente sui run.
    """
    global N_SPLITS_INTERNO, RUN_FORZATO
    
    print("======================================================================")
    print(" ORCHESTRATORE MODELLI - CONFIGURAZIONE INIZIALE")
    print("======================================================================")
    
    cartella_modelli = Path(__file__).resolve().parent
    cartella_risultati = trova_cartella_progetto() / "risultati_finali"
    
    # 1. CERCA I MODELLI DISPONIBILI (Leggendo le sottocartelle)
    sottocartelle = sorted([d for d in cartella_modelli.iterdir() if d.is_dir() and d.name != "__pycache__"])
    
    info_modelli = []
    for cartella in sottocartelle:
        # Cerca script python dentro la cartella, ignorando file di sistema come __init__.py
        candidati = [f for f in cartella.glob("*.py") if not f.name.startswith("__")]
        if candidati:
            info_modelli.append((cartella, candidati[0])) # Associa la cartella al suo script

    if not info_modelli:
        print("Errore: Nessuna cartella modello valida trovata.")
        return

    print("\n[DOMANDA 1] Modelli disponibili trovati:")
    for i, (cartella, script) in enumerate(info_modelli, 1):
        print(f"   [{i}] {cartella.name}  ->  {script.name}")
    
    try:
        scelta_mod = int(input(f"Scegli il modello da eseguire (1-{len(info_modelli)}): ").strip()) - 1
        cartella_scelta, script_scelto = info_modelli[scelta_mod]
        # Usiamo il nome della cartella (es. AdaBoost) come nome ufficiale del modello
        nome_modello = cartella_scelta.name
    except (ValueError, IndexError):
        print("Scelta non valida, chiusura in corso.")
        return

    # 2. SCEGLI LA METRICA
    print("\n[DOMANDA 2] Scegli la Metrica Primaria per il tuning e l'importanza:")
    print("   [1] F1-Score (Bilanciata)")
    print("   [2] F2-Score (Pesa maggiormente il Recall - tolleranza Falsi Positivi)")
    scelta_metrica = input("Scegli (1 o 2) [1]: ").strip()
    if scelta_metrica not in ["1", "2"]: 
        scelta_metrica = "1"
    imposta_metrica_primaria(scelta_metrica)

    # 3. SCEGLI IL K
    print(f"\n[DOMANDA 3] CV Interna (Tuning)")
    scelta_k = input(f"Vuoi usare K={N_SPLITS_INTERNO} di default? [s/n]: ").strip().lower()
    if scelta_k == 'n':
        try:
            N_SPLITS_INTERNO = int(input("Inserisci il valore di K (es. 5): ").strip())
        except ValueError:
            print("Input non valido, mantenuto il valore di default.")
    print(f"-> K interno impostato a: {N_SPLITS_INTERNO}")

    global ATTIVA_FALLBACK_CLASSE_1

    # 4. SCEGLI FALLBACK CLASSE 1
    print("\n[DOMANDA 4] Fallback Euristico Classe 1")
    print("   (Riassegna alla Classe 1 i pazienti per cui il modello è indeciso tra 0 e 2)")
    scelta_fallback = input("Vuoi attivare il Fallback per la Classe 1? [s/n]: ").strip().lower()
    if scelta_fallback == 's':
        ATTIVA_FALLBACK_CLASSE_1 = True
        print("-> Fallback ATTIVATO (Soglia tolleranza: 51%).")
    else:
        ATTIVA_FALLBACK_CLASSE_1 = False
        print("-> Fallback DISATTIVATO. Verranno usate le probabilità standard.")

    # 5. RECAP ESPERIMENTI E CONFERMA
    run_disponibili = sorted([d for d in cartella_risultati.iterdir() if d.is_dir()], key=lambda d: d.name)
    
    if not run_disponibili:
        print("\nNessun esperimento trovato in risultati_finali/.")
        return

    print(f"\n[DOMANDA 5] Trovati {len(run_disponibili)} esperimenti in '{cartella_risultati.name}/':")
    for i, d in enumerate(run_disponibili, 1):
        print(f"   [{i}] {d.name}")
        
    print("\nPuoi avviare l'esecuzione in tre modi:")
    print(" - Premi [INVIO] per eseguirli TUTTI")
    print(" - Inserisci NUMERI (es. 1, 3) per INCLUDERE solo quelli")
    print(" - Inserisci NUMERI COL MENO (es. -2, -4) per ESCLUDERE solo quelli")
    scelta_esperimenti = input(f"Scelta (INVIO per tutti): ").strip()
    
    if not scelta_esperimenti:
        print("-> Esecuzione confermata su TUTTI gli esperimenti.")
        run_da_eseguire = run_disponibili
    elif "-" in scelta_esperimenti:
        # LOGICA DI ESCLUSIONE
        indici_da_escludere = []
        parti = scelta_esperimenti.split(",")
        for p in parti:
            try:
                num = int(p.replace("-", "").strip())
                if 1 <= num <= len(run_disponibili):
                    indici_da_escludere.append(num - 1)
            except ValueError:
                pass
        
        run_da_eseguire = [run for i, run in enumerate(run_disponibili) if i not in indici_da_escludere]
        print(f"-> Ottimo, verranno ESCLUSI {len(indici_da_escludere)} esperimenti. Eseguo i restanti {len(run_da_eseguire)}.")
    else:
        # LOGICA DI INCLUSIONE
        indici_scelti = []
        parti = scelta_esperimenti.split(",")
        for p in parti:
            try:
                num = int(p.strip())
                if 1 <= num <= len(run_disponibili):
                    indici_scelti.append(num - 1)
            except ValueError:
                pass
                
        if not indici_scelti:
            print("-> Formato non valido. Ricado sull'esecuzione di tutti gli esperimenti.")
            run_da_eseguire = run_disponibili
        else:
            run_da_eseguire = [run_disponibili[i] for i in indici_scelti]
            print(f"-> Ottimo, verranno eseguiti solo {len(run_da_eseguire)} esperimento/i selezionato/i.")
        
    conferma = input(f"\nVuoi confermare l'avvio per {nome_modello}? [s/n]: ").strip().lower()
    if conferma != 's':
        print("Operazione annullata dall'utente.")
        return

    # ESECUZIONE DEL BATCH SUI RUN TROVATI

    print("\n" + "=" * 70)
    print(f" AVVIO BATCH AUTOMATICO: {nome_modello} su {len(run_da_eseguire)} esperimenti")
    print("=" * 70)

    riusciti = []
    saltati = []

    for run in tqdm(run_da_eseguire, desc=f"Elaborazione {nome_modello}"):
        RUN_FORZATO = run
        try:
            # Lancia lo script del modello (es. adaboost_training.py)
            runpy.run_path(str(script_scelto), run_name="__main__")
            riusciti.append(run.name)
        except Exception as e:
            # Se esplode (es. per i NaN), cattura l'errore e salva il motivo
            print(f"\n[ERRORE] Fallimento nell'esperimento {run.name}: {e}")
            saltati.append((run.name, str(e)))
        finally:
            RUN_FORZATO = None
            
    # ==========================================
    # RIEPILOGO FINALE A SCHERMO
    # ==========================================
    print("\n" + "=" * 70)
    print(f" RIEPILOGO FINALE BATCH - {nome_modello}")
    print("=" * 70)
    print(f"Esperimenti totali      : {len(run_da_eseguire)}")
    print(f"Completati con successo : {len(riusciti)}")
    print(f"Saltati / Falliti       : {len(saltati)}\n")

    if riusciti:
        print("--- COMPLETATI ---")
        for nome in riusciti:
            print(f"  [OK] {nome}")
            
    if saltati:
        print("\n--- SALTATI ---")
        for nome, errore in saltati:
            print(f"  [SKIP] {nome}")
            print(f"         Motivo: {errore}")

    print("\n>>> PIPELINE COMPLETATA <<<")

# Blocco di avvio autonomo
if __name__ == "__main__":
    avvia_pipeline_completa()