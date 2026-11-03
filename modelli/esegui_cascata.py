"""
esegui_cascata_universale.py
=====================================================================
Architettura a Cascata (Hierarchical Classification) MULTI-MOTORE IN BATCH.

Esegue la pipeline dicotomica scegliendo il motore (XGBoost o SVM) su 
UNO, MOLTI o TUTTI gli esperimenti in sequenza:
- MODELLO A: Distingue la Classe 0 (Psichiatrica) dal "Resto" (1 e 2).
- MODELLO B: Addestrato SOLO sui pazienti 1 e 2, distingue tra Classe 1 e 2.
=====================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

CARTELLA_RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARTELLA_RADICE))

import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import fbeta_score

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# Importiamo la logica condivisa senza intaccarla
import modelli.gestione_modelli as gm

# =====================================================================
# CLASSE WRAPPER: UNISCE I DUE MODELLI BINARI IN UN UNICO MODELLO
# =====================================================================
class ModelloCascataWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_A, model_B, soglia_B=0.5):
        self.model_A = model_A
        self.model_B = model_B
        self.soglia_B = soglia_B
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        pred_A = self.model_A.predict(X)
        final_preds = np.zeros(len(X), dtype=int)
        
        idx_resto = np.where(pred_A == 1)[0]
        
        if len(idx_resto) > 0:
            X_resto = X.iloc[idx_resto]
            prob_B = self.model_B.predict_proba(X_resto)[:, 1]
            pred_B = (prob_B >= self.soglia_B).astype(int)
            final_preds_resto = np.where(pred_B == 0, 1, 2)
            final_preds[idx_resto] = final_preds_resto
            
        return final_preds

# =====================================================================
# FUNZIONI DI OTTIMIZZAZIONE OPTUNA (XGBOOST E SVM)
# =====================================================================
def ottimizza_xgboost_binario(X, y, nome_fase, scale_pos_weight=1.0):
    print(f"   [!] Lancio Optuna (XGBoost) per {nome_fase} (20 trials)...")
    
    def obiettivo(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": scale_pos_weight,
            "random_state": gm.SEED,
            "n_jobs": -1
        }
        modello = XGBClassifier(**param)
        punteggi = cross_val_score(modello, X, y, cv=gm.get_cv_interna(), scoring=gm.METRICA_PRIMARIA, n_jobs=-1)
        return punteggi.mean()
        
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=gm.SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(obiettivo, n_trials=20)
    
    print(f"   >>> {nome_fase} - Miglior Score CV: {study.best_value:.4f}")
    return study.best_params, study.best_value


def ottimizza_svm_binario(X, y, nome_fase):
    print(f"   [!] Lancio Optuna (SVM) per {nome_fase} (20 trials)...")
    
    def obiettivo(trial):
        c_val = trial.suggest_float("C", 0.01, 100.0, log=True)
        gamma_val = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel_val = trial.suggest_categorical("kernel", ["linear", "rbf"])

        modello = make_pipeline(StandardScaler(), SVC(C=c_val, gamma=gamma_val, kernel=kernel_val, random_state=gm.SEED, class_weight='balanced', probability=True))
        punteggi = cross_val_score(modello, X, y, cv=gm.get_cv_interna(), scoring=gm.METRICA_PRIMARIA, n_jobs=-1)
        return punteggi.mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=gm.SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(obiettivo, n_trials=20)
    
    print(f"   >>> {nome_fase} - Miglior Score CV: {study.best_value:.4f}")
    return study.best_params, study.best_value

# =====================================================================
# RICERCA SOGLIA OTTIMALE (ROC OUT-OF-FOLD)
# =====================================================================
def trova_soglia_ottimale(model_B, X_train_B, y_train_B, nome_metrica_str):
    print(f"   [!] Ricerca soglia ottimale {nome_metrica_str} per Modello B tramite simulazione Out-Of-Fold...")
    y_prob_oof = cross_val_predict(model_B, X_train_B, y_train_B, cv=gm.get_cv_interna(), method='predict_proba', n_jobs=-1)[:, 1]
    
    miglior_soglia = 0.5
    miglior_score = 0
    beta_val = 2 if nome_metrica_str == "F2" else 1
    
    for soglia in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob_oof >= soglia).astype(int)
        score = fbeta_score(y_train_B, y_pred, beta=beta_val, average='macro')
        if score > miglior_score:
            miglior_score = score
            miglior_soglia = soglia
            
    print(f"   >>> Soglia matematica ottimale trovata: {miglior_soglia:.2f} (Score OOF: {miglior_score:.4f})")
    return miglior_soglia

# =====================================================================
# ORCHESTRATORE PRINCIPALE DELLA CASCATA (BATCH)
# =====================================================================
def avvia_cascata():
    print("======================================================================")
    print(" 🚀 AVVIO ARCHITETTURA AVANZATA: HIERARCHICAL CASCADE UNIVERSALE")
    print("======================================================================")
    
    cartella_risultati = CARTELLA_RADICE / "risultati_finali"
    run_disponibili = sorted([d for d in cartella_risultati.iterdir() if d.is_dir()], key=lambda d: d.name)
    
    if not run_disponibili:
        print("\nNessun esperimento trovato in risultati_finali/.")
        return

    # 1. SCELTA ESPERIMENTI (LOGICA BATCH)
    print(f"\n[DOMANDA 1] Trovati {len(run_disponibili)} esperimenti in '{cartella_risultati.name}/':")
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
            
    # 2. SCELTA MOTORE
    print("\n[DOMANDA 2] Quale Motore di Machine Learning vuoi usare per la Cascata?")
    print("   [1] XGBoost")
    print("   [2] SVM Standard")
    scelta_motore_input = input("Scegli (1 o 2) [1]: ").strip()
    motore = "SVM" if scelta_motore_input == "2" else "XGBoost"
    NOME_MODELLO_CASCATA = f"{motore}_Cascata"
    print(f"-> Motore selezionato: {motore}")

    # 3. SCELTA METRICA
    print("\n[DOMANDA 3] Scegli la Metrica Primaria (per Optuna e Valutazione Finale):")
    print("   [1] F1-Score (Bilanciata)")
    print("   [2] F2-Score (Recall pesato)")
    scelta_metrica_input = input("Scegli (1 o 2) [1]: ").strip()
    if scelta_metrica_input not in ["1", "2"]: scelta_metrica_input = "1"
    
    gm.imposta_metrica_primaria(scelta_metrica_input)
    gm.ATTIVA_FALLBACK_CLASSE_1 = False 
    nome_metrica_str = "F1" if scelta_metrica_input == "1" else "F2"

 

    # =====================================================================
    # ESECUZIONE BATCH
    # =====================================================================
    print("\n" + "=" * 70)
    print(f" AVVIO BATCH CASCATA: {NOME_MODELLO_CASCATA} su {len(run_da_eseguire)} esperimenti")
    print("=" * 70)

    riusciti = []
    saltati = []

    for run_scelto in tqdm(run_da_eseguire, desc=f"Elaborazione {NOME_MODELLO_CASCATA}"):
        gm.RUN_FORZATO = run_scelto
        print("\n" + "*" * 70)
        print(f"*** ELABORAZIONE ESPERIMENTO: {run_scelto.name} ***")
        print("*" * 70)

        try:
            compartimenti = gm.rileva_compartimenti(run_scelto)
            
            risultati_per_compartimento = []
            lista_permutation_importance = []
            log_iperparametri = []

            for comp in compartimenti:
                print("\n" + "#" * 60)
                print(f"# COMPARTIMENTO: {comp.nome}")
                print("#" * 60)

                dati = gm.carica_dati_compartimento(comp.cartella_split)
                X_train, X_test = dati["X_train"], dati["X_test"]
                y_train, y_test = dati["y_train"], dati["y_test"]

                cartella_output = run_scelto / f"Risultati_{NOME_MODELLO_CASCATA}_{nome_metrica_str}" / comp.nome

                # FASE A
                print("\n--- FASE A: Addestramento Modello (Classe 0 vs Classe 1+2) ---")
                y_train_A = np.where(y_train == 0, 0, 1)
                
                if motore == "XGBoost":
                    best_params_A, score_A = ottimizza_xgboost_binario(X_train, y_train_A, "MODELLO A")
                    model_A = XGBClassifier(**best_params_A, objective="binary:logistic", random_state=gm.SEED, n_jobs=-1)
                else:
                    best_params_A, score_A = ottimizza_svm_binario(X_train, y_train_A, "MODELLO A")
                    model_A = make_pipeline(StandardScaler(), SVC(**best_params_A, random_state=gm.SEED, class_weight='balanced', probability=True))
                    
                model_A.fit(X_train, y_train_A)

                # FASE B
                print("\n--- FASE B: Addestramento Modello (Classe 1 vs Classe 2) ---")
                mask_B = (y_train == 1) | (y_train == 2)
                X_train_B, y_train_filato = X_train[mask_B], y_train[mask_B]
                y_train_B = np.where(y_train_filato == 1, 0, 1)
                
                if motore == "XGBoost":
                    ratio_bilanciamento = float(np.sum(y_train_B == 0) / np.sum(y_train_B == 1))
                    print(f"   -> Bilanciamento XGBoost: scale_pos_weight = {ratio_bilanciamento:.2f}")
                    best_params_B, score_B = ottimizza_xgboost_binario(X_train_B, y_train_B, "MODELLO B", scale_pos_weight=ratio_bilanciamento)
                    model_B = XGBClassifier(**best_params_B, objective="binary:logistic", scale_pos_weight=ratio_bilanciamento, random_state=gm.SEED, n_jobs=-1)
                else:
                    print("   -> Bilanciamento SVM: class_weight='balanced' attivato")
                    best_params_B, score_B = ottimizza_svm_binario(X_train_B, y_train_B, "MODELLO B")
                    model_B = make_pipeline(StandardScaler(), SVC(**best_params_B, random_state=gm.SEED, class_weight='balanced', probability=True))
                    
                model_B.fit(X_train_B, y_train_B)

                soglia_ottimale = trova_soglia_ottimale(model_B, X_train_B, y_train_B, nome_metrica_str)

                # FASE FINALE
                print("\n--- FASE FINALE: Fusione e Valutazione sul Test Set ---")
                modello_cascata = ModelloCascataWrapper(model_A, model_B, soglia_B=soglia_ottimale)
                y_pred = modello_cascata.predict(X_test)

                risultato = gm.valuta_prestazioni(y_test, y_pred, nome_modello=NOME_MODELLO_CASCATA, cartella_output=cartella_output)
                risultati_per_compartimento.append(risultato)

                print("   -> Calcolo Permutation Importance della Cascata...")
                pi_df = gm.calcola_permutation_importance(modello_cascata, X_test, y_test)
                lista_permutation_importance.append(pi_df)

                dizionario_log = {"Compartimento": comp.nome, "Score_A_CV": score_A, "Score_B_CV": score_B, "Soglia_Ottimale_B": soglia_ottimale}
                for k, v in best_params_A.items(): dizionario_log[f"A_{k}"] = v
                for k, v in best_params_B.items(): dizionario_log[f"B_{k}"] = v
                log_iperparametri.append(dizionario_log)

            # AGGREGAZIONE FINALE GLOBALE PER IL RUN
            cartella_output_globale = run_scelto / f"Risultati_{NOME_MODELLO_CASCATA}_{nome_metrica_str}"
            
            df_finale = gm.aggrega_risultati_compartimenti(risultati_per_compartimento, NOME_MODELLO_CASCATA, cartella_output=cartella_output_globale)
            pi_aggregata = gm.aggrega_permutation_importance(lista_permutation_importance)
            gm.plot_feature_importance(pi_aggregata, NOME_MODELLO_CASCATA, top_n=15, cartella_output=cartella_output_globale)

            df_iperparametri = pd.DataFrame(log_iperparametri)
            df_iperparametri.to_csv(cartella_output_globale / "iperparametri_per_compartimento.csv", index=False)
            
            riusciti.append(run_scelto.name)

        except Exception as e:
            print(f"\n[ERRORE] Fallimento nell'esperimento {run_scelto.name}: {e}")
            saltati.append((run_scelto.name, str(e)))
        finally:
            gm.RUN_FORZATO = None

    # =====================================================================
    # RIEPILOGO FINALE A SCHERMO
    # =====================================================================
    print("\n" + "=" * 70)
    print(f" RIEPILOGO FINALE  - {NOME_MODELLO_CASCATA}")
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

    print("\n>>> PIPELINE  CASCATA COMPLETATA <<<")

if __name__ == "__main__":
    avvia_cascata()