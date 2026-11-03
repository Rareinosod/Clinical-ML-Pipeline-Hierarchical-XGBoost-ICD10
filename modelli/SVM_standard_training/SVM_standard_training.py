"""
=====================================================================
 SCRIPT MODELLO: SVM (SUPPORT VECTOR MACHINE) - STANDARD
=====================================================================
Architettura One-vs-One nativa. Ottimizzata con StandardScaler.
Include funzione SMART SKIP: se gli iperparametri sono già stati
calcolati in una run precedente, salta Optuna e ricarica i valori
storici per risparmiare tempo di calcolo.
=====================================================================
"""

import sys
from pathlib import Path

# Aggiungiamo la cartella progetto a sys.path
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CARTELLA_PROGETTO))

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

from modelli.gestione_modelli import (
    prepara_esperimento,
    carica_dati_compartimento,
    get_cv_interna,
    ottimizza_con_optuna,
    valuta_prestazioni,
    aggrega_risultati_compartimenti,
    calcola_permutation_importance,
    aggrega_permutation_importance,
    plot_feature_importance,
    cartella_output_modello, # <-- IMPORTAZIONE AGGIUNTA
    METRICA_PRIMARIA,
    NOME_METRICA_PRIMARIA,   # <-- IMPORTAZIONE AGGIUNTA
    SEED,
)

NOME_MODELLO = "SVM_Standard"

# =====================================================================
# SEZIONE 1 - SELEZIONE ESPERIMENTO E COMPARTIMENTI
# =====================================================================
cartella_run, compartimenti = prepara_esperimento(NOME_MODELLO)

# ---> MODIFICA CRITICA: Usa la funzione dinamica per la cartella globale <---
cartella_output_globale = cartella_output_modello(cartella_run, NOME_MODELLO)

# --- LOGICA SMART SKIP (Controllo storico pre-ciclo) ---
file_storico = cartella_output_globale / "iperparametri_per_compartimento.csv"
df_storico = None

if file_storico.exists():
    print("\n" + "*" * 70)
    print(f"[SMART SKIP] Trovato storico iperparametri: {file_storico.name}")
    print("-> Ottimizzazione Optuna bypassata. Ricarico i parametri salvati.")
    print("*" * 70)
    df_storico = pd.read_csv(file_storico)


# =====================================================================
# SEZIONE 2 - FUNZIONE OBIETTIVO OPTUNA
# =====================================================================
def costruisci_obiettivo(X_train, y_train):
    def obiettivo(trial):
        c_val = trial.suggest_float("C", 0.01, 100.0, log=True)
        gamma_val = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel_val = trial.suggest_categorical("kernel", ["linear", "rbf"])

        modello = make_pipeline(
            StandardScaler(),
            SVC(
                C=c_val, 
                gamma=gamma_val, 
                kernel=kernel_val, 
                random_state=SEED, 
                class_weight='balanced',
                probability=True  
            )
        )

        punteggi = cross_val_score(
            modello, X_train, y_train,
            cv=get_cv_interna(), scoring=METRICA_PRIMARIA, n_jobs=-1,
        )
        return punteggi.mean()

    return obiettivo


# =====================================================================
# SEZIONE 3 - CICLO SUI COMPARTIMENTI: TUNING, FIT FINALE, VALUTAZIONE
# =====================================================================
risultati_per_compartimento = []
log_iperparametri = []
lista_permutation_importance = []

for compartimento in compartimenti:
    print("\n" + "#" * 70)
    print(f"# COMPARTIMENTO: {compartimento.nome}")
    print("#" * 70)

    dati = carica_dati_compartimento(compartimento.cartella_split)
    X_train, X_test = dati["X_train"], dati["X_test"]
    y_train, y_test = dati["y_train"], dati["y_test"]

    # ---> MODIFICA CRITICA: Usa la funzione dinamica per il nome cartella <---
    cartella_output = cartella_output_modello(cartella_run, NOME_MODELLO, compartimento.nome)

    # --- Tuning Optuna vs Smart Skip ---
    if df_storico is not None:
        # Estrae i parametri esatti per questo compartimento dal file CSV
        riga = df_storico[df_storico["Compartimento"] == compartimento.nome].iloc[0]
        best_params = {
            "C": float(riga["C"]),
            "gamma": riga["gamma"],
            "kernel": riga["kernel"]
        }
        # ---> MODIFICA CRITICA: Estrazione dinamica della metrica per lo smart skip <---
        best_score_cv = float(riga[f"{NOME_METRICA_PRIMARIA}_CV_train"])
        print(f"-> Parametri caricati dal CSV: {best_params}")
    else:
        # Avvia Optuna normalmente
        obiettivo = costruisci_obiettivo(X_train, y_train)
        best_params, best_score_cv = ottimizza_con_optuna(obiettivo, n_trials=20)

    # --- Addestramento finale ---
    print("\n-> Addestramento del modello finale su tutto il train del compartimento...")
    final_model = make_pipeline(
        StandardScaler(),
        SVC(
            C=best_params["C"],
            gamma=best_params["gamma"],
            kernel=best_params["kernel"],
            random_state=SEED,
            class_weight='balanced',
            probability=True  # <-- AGGIUNGI QUI
        )
    )

    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    risultato = valuta_prestazioni(y_test, y_pred, NOME_MODELLO, cartella_output=cartella_output, modello=final_model, X_test=X_test)
    risultati_per_compartimento.append(risultato)

    print("\n-> Calcolo Permutation Importance sul test del compartimento...")
    pi_df = calcola_permutation_importance(final_model, X_test, y_test)
    lista_permutation_importance.append(pi_df)

    log_iperparametri.append({
        "Compartimento": compartimento.nome,
        "C": best_params["C"],
        "gamma": best_params["gamma"],
        "kernel": best_params["kernel"],
        f"{NOME_METRICA_PRIMARIA}_CV_train": best_score_cv, # <-- ORA È DINAMICO!
    })


# =====================================================================
# SEZIONE 4 - AGGREGAZIONE FINALE TRA COMPARTIMENTI
# =====================================================================
df_finale = aggrega_risultati_compartimenti(
    risultati_per_compartimento, NOME_MODELLO, cartella_output=cartella_output_globale
)

pi_aggregata = aggrega_permutation_importance(lista_permutation_importance)
plot_feature_importance(pi_aggregata, NOME_MODELLO, top_n=15, cartella_output=cartella_output_globale)

df_iperparametri = pd.DataFrame(log_iperparametri)
print("\n" + "-" * 60)
print(" IPERPARAMETRI SCELTI PER COMPARTIMENTO")
print("-" * 60)
print(df_iperparametri.to_string(index=False))
df_iperparametri.to_csv(cartella_output_globale / "iperparametri_per_compartimento.csv", index=False)
print("\n-> Salvato: iperparametri_per_compartimento.csv")

print("\n" + "=" * 70)
print(f"SCRIPT {NOME_MODELLO} COMPLETATO")
print(f"Risultati completi in: {cartella_output_globale}")
print("=" * 70)