"""
=====================================================================
 SCRIPT MODELLO: ALBERO DECISIONALE (DECISION TREE)
=====================================================================
Usa il modulo condiviso 'modelli/gestione_modelli.py' per tutta la parte
comune a ogni modello: selezione dell'esperimento, rilevamento della
strategia di split esterno, CV interna, motore Optuna, valutazione
delle prestazioni e grafici diagnostici.
=====================================================================
"""

import sys
from pathlib import Path
import pandas as pd

# --- Bootstrap: aggiungiamo la cartella progetto a sys.path
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CARTELLA_PROGETTO))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

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

NOME_MODELLO = "DecisionTree"

# =====================================================================
# SEZIONE 1 - SELEZIONE ESPERIMENTO E COMPARTIMENTI
# =====================================================================
cartella_run, compartimenti = prepara_esperimento(NOME_MODELLO)


# =====================================================================
# SEZIONE 2 - FUNZIONE OBIETTIVO OPTUNA (specifica di Decision Tree)
# =====================================================================
def costruisci_obiettivo(X_train, y_train):
    """
    Ritorna la funzione obiettivo Optuna per esplorare i parametri chiave
    dell'Albero Decisionale (controllo overfitting tramite depth e split/leaf).
    """
    def obiettivo(trial):
        # Spazio di ricerca ricalcato sulla tua precedente GridSearchCV
        max_depth = trial.suggest_categorical("max_depth", [None, 3, 5, 10, 15, 20])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        modello = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=SEED,
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

    # --- Tuning con Optuna ---
    obiettivo = costruisci_obiettivo(X_train, y_train)
    best_params, best_score_cv = ottimizza_con_optuna(obiettivo, n_trials=30)

    # --- Addestramento finale ---
    print("\n-> Addestramento del modello finale su tutto il train del compartimento...")
    final_model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
        random_state=SEED,
    )
    final_model.fit(X_train, y_train)

    # --- Valutazione sul test ---
    y_pred = final_model.predict(X_test)
    risultato = valuta_prestazioni(y_test, y_pred, NOME_MODELLO, cartella_output=cartella_output, modello=final_model, X_test=X_test)
    risultati_per_compartimento.append(risultato)

    # --- Permutation importance ---
    print("\n-> Calcolo Permutation Importance sul test del compartimento...")
    pi_df = calcola_permutation_importance(final_model, X_test, y_test)
    lista_permutation_importance.append(pi_df)

    # --- Tracciamento iperparametri ---
    log_iperparametri.append({
        "Compartimento": compartimento.nome,
        "max_depth": best_params["max_depth"],
        "min_samples_split": best_params["min_samples_split"],
        "min_samples_leaf": best_params["min_samples_leaf"],
        "criterion": best_params["criterion"],
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

pi_aggregata = aggrega_permutation_importance(lista_permutation_importance)
plot_feature_importance(pi_aggregata, NOME_MODELLO, top_n=15, cartella_output=cartella_output_globale)

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