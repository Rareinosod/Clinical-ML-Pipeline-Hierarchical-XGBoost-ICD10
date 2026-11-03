"""
=====================================================================
 SCRIPT MODELLO: RANDOM FOREST
=====================================================================
Usa il modulo condiviso 'modelli/gestione_modelli.py' per tutta la parte
comune a ogni modello: selezione dell'esperimento, rilevamento della
strategia di split esterno (Holdout o K-Fold), CV interna, motore
Optuna, valutazione delle prestazioni e grafici diagnostici.

Qui resta SOLO la logica specifica della Random Forest:
    - lo spazio di ricerca Optuna (quali iperparametri esplorare)
    - la costruzione del classificatore Random Forest vero e proprio
=====================================================================
"""

import sys
from pathlib import Path

# --- Bootstrap: aggiungiamo la cartella progetto a sys.path, cosi'
#     possiamo importare 'modelli.gestione_modelli' indipendentemente
#     da dove/come viene lanciato questo script.
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CARTELLA_PROGETTO))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

from modelli.gestione_modelli import (
    prepara_esperimento,
    carica_dati_compartimento,
    get_cv_interna,
    ottimizza_con_optuna,
    valuta_prestazioni,
    plot_validation_curve_condivisa,
    aggrega_risultati_compartimenti,
    calcola_permutation_importance,
    aggrega_permutation_importance,
    plot_feature_importance,
    cartella_output_modello, # <-- IMPORTAZIONE AGGIUNTA
    METRICA_PRIMARIA,
    NOME_METRICA_PRIMARIA,   # <-- IMPORTAZIONE AGGIUNTA
    SEED,
)

NOME_MODELLO = "RandomForest"


# =====================================================================
# SEZIONE 1 - SELEZIONE ESPERIMENTO E COMPARTIMENTI (condivisa)
# =====================================================================
# Gestisce da sola sia il caso Holdout (1 compartimento) sia il caso
# K-Fold esterno (un compartimento per ogni Fold_X), stampa il log
# dell'esperimento scelto.
cartella_run, compartimenti = prepara_esperimento(NOME_MODELLO)


# =====================================================================
# SEZIONE 2 - FUNZIONE OBIETTIVO OPTUNA (specifica di Random Forest)
# =====================================================================
def costruisci_obiettivo(X_train, y_train):
    """
    Ritorna la funzione obiettivo Optuna per QUESTI dati di training
    (serve un obiettivo diverso per ogni compartimento, perche' i dati
    su cui fare la CV interna cambiano a ogni Fold_X).
    """

    def obiettivo(trial):
        # Spazio di ricerca per la Random Forest
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

        # Inizializzazione del modello con i parametri proposti
        modello = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=SEED,
            n_jobs=-1
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

    # --- Tuning con Optuna (usa la CV interna condivisa) ---
    obiettivo = costruisci_obiettivo(X_train, y_train)
    best_params, best_score_cv = ottimizza_con_optuna(obiettivo, n_trials=30)

    # --- Addestramento finale con i parametri migliori, su tutto il train del compartimento ---
    print("\n-> Addestramento del modello finale su tutto il train del compartimento...")
    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        class_weight=best_params["class_weight"],
        random_state=SEED,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    # --- Valutazione sul test del compartimento (mai visto durante il tuning) ---
    y_pred = final_model.predict(X_test)
    risultato = valuta_prestazioni(y_test, y_pred, NOME_MODELLO, cartella_output=cartella_output, modello=final_model, X_test=X_test)
    risultati_per_compartimento.append(risultato)

    # --- Permutation importance sul test di questo compartimento ---
    print("\n-> Calcolo Permutation Importance sul test del compartimento...")
    pi_df = calcola_permutation_importance(final_model, X_test, y_test)
    lista_permutation_importance.append(pi_df)

    # --- Grafico diagnostico: validation curve ---
    # Facciamo variare la profondità dell'albero per controllare l'overfitting.
    plot_validation_curve_condivisa(
        modello_base=final_model, 
        X_train=X_train, 
        y_train=y_train,
        nome_parametro="max_depth", 
        range_valori=[3, 5, 10, 15, 20, 25, 30],
        nome_modello=f"{NOME_MODELLO}_{compartimento.nome}", 
        cartella_output=cartella_output
    )

    # --- Teniamo traccia separata degli iperparametri scelti per questo compartimento ---
    log_iperparametri.append({
        "Compartimento": compartimento.nome,
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "class_weight": best_params["class_weight"],
        f"{NOME_METRICA_PRIMARIA}_CV_train": best_score_cv, # <-- ORA È DINAMICO!
    })


# =====================================================================
# SEZIONE 4 - AGGREGAZIONE FINALE TRA COMPARTIMENTI
# =====================================================================
# ---> MODIFICA CRITICA: Usa la funzione dinamica per la cartella globale <---
cartella_output_globale = cartella_output_modello(cartella_run, NOME_MODELLO)

df_finale = aggrega_risultati_compartimenti(
    risultati_per_compartimento, NOME_MODELLO, cartella_output=cartella_output_globale
)

# --- Permutation importance aggregata tra tutti i compartimenti ---
pi_aggregata = aggrega_permutation_importance(lista_permutation_importance)
plot_feature_importance(pi_aggregata, NOME_MODELLO, top_n=15, cartella_output=cartella_output_globale)

# --- Salviamo anche gli iperparametri scelti per compartimento ---
df_iperparametri = pd.DataFrame(log_iperparametri)
print("\n" + "-" * 60)
print(" IPERPARAMETRI SCELTI PER COMPARTIMENTO")
print("-" * 60)
print(df_iperparametri.to_string(index=False))
df_iperparametri.to_csv(cartella_output_globale / "iperparametri_per_compartimento.csv", index=False)
print(f"\n-> Salvato: iperparametri_per_compartimento.csv")

print("\n" + "=" * 70)
print(f"SCRIPT {NOME_MODELLO.upper()} COMPLETATO")
print(f"Risultati completi in: {cartella_output_globale}")
print("=" * 70)