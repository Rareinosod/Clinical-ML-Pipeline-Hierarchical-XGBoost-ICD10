"""
=====================================================================
 SCRIPT MODELLO: ADABOOST
=====================================================================
"""

import sys
from pathlib import Path

# --- Bootstrap ---
CARTELLA_PROGETTO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(CARTELLA_PROGETTO))

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

from modelli.gestione_modelli import (
    prepara_esperimento,
    carica_dati_compartimento,
    get_cv_interna,
    ottimizza_con_optuna,
    valuta_prestazioni,
    plot_curva_boosting,
    aggrega_risultati_compartimenti,
    calcola_permutation_importance,
    aggrega_permutation_importance,
    plot_feature_importance,
    cartella_output_modello, # <-- IMPORTAZIONE AGGIUNTA
    METRICA_PRIMARIA,
    NOME_METRICA_PRIMARIA,
    SEED,
)

NOME_MODELLO = "AdaBoost"

# =====================================================================
# SEZIONE 1 - SELEZIONE ESPERIMENTO E COMPARTIMENTI
# =====================================================================
cartella_run, compartimenti = prepara_esperimento(NOME_MODELLO)


# =====================================================================
# SEZIONE 2 - FUNZIONE OBIETTIVO OPTUNA
# =====================================================================
def costruisci_obiettivo(X_train, y_train):
    def obiettivo(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300, step=25)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)
        profondita_stump = trial.suggest_int("profondita_stump", 1, 3)

        modello = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=profondita_stump),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=SEED,
        )

        punteggi = cross_val_score(
            modello, X_train, y_train,
            cv=get_cv_interna(), scoring=METRICA_PRIMARIA, n_jobs=-1,
        )
        return punteggi.mean()

    return obiettivo


# =====================================================================
# SEZIONE 3 - CICLO SUI COMPARTIMENTI
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

    obiettivo = costruisci_obiettivo(X_train, y_train)
    best_params, best_score_cv = ottimizza_con_optuna(obiettivo, n_trials=30)

    print("\n-> Addestramento del modello finale su tutto il train del compartimento...")
    final_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=best_params["profondita_stump"]),
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        random_state=SEED,
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    risultato = valuta_prestazioni(y_test, y_pred, NOME_MODELLO, cartella_output=cartella_output, modello=final_model, X_test=X_test)
    risultati_per_compartimento.append(risultato)

    print("\n-> Calcolo Permutation Importance sul test del compartimento...")
    pi_df = calcola_permutation_importance(final_model, X_test, y_test)
    lista_permutation_importance.append(pi_df)

    plot_curva_boosting(
        final_model, X_train, y_train, X_test, y_test,
        nome_modello=f"{NOME_MODELLO}_{compartimento.nome}",
        cartella_output=cartella_output,
    )

    log_iperparametri.append({
        "Compartimento": compartimento.nome,
        "n_estimators": best_params["n_estimators"],
        "learning_rate": best_params["learning_rate"],
        "profondita_stump": best_params["profondita_stump"],
        f"{NOME_METRICA_PRIMARIA}_CV_train": best_score_cv, 
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