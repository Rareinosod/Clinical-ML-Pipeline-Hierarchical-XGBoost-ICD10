import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

CARTELLA_RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARTELLA_RADICE))

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Import dei modelli supportati per le architetture Standard/Fallback
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import modelli.gestione_modelli as gm

# Importiamo le logiche della Cascata se l'utente sceglie quell'architettura
from modelli.esegui_cascata import (
    ModelloCascataWrapper,
    ottimizza_xgboost_binario,
    ottimizza_svm_binario,
    trova_soglia_ottimale
)

SEED = 42
METRICA_PRIMARIA = make_scorer(fbeta_score, beta=2, average='macro', zero_division=0)

# =====================================================================
# CLASSE WRAPPER PER IL MODELLO CON FALLBACK (Salva la logica nel .pkl)
# =====================================================================
class ModelloFallbackWrapper(BaseEstimator, ClassifierMixin):
    """
    Incapsula un modello multiclasse standard aggiungendo nativamente 
    la regola clinica di Fallback. Utile per l'esportazione in produzione.
    """
    def __init__(self, base_model, soglia_sicurezza=0.51):
        self.base_model = base_model
        self.soglia_sicurezza = soglia_sicurezza
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y=None):
        self.base_model.fit(X, y)
        return self
        
    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Sicurezza: se il modello non supporta predict_proba (es. SVM con kernel linear a volte), predice normalmente
        if not hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict(X)
            
        probabilita = self.base_model.predict_proba(X)
        y_pred_modificata = []
        
        for prob in probabilita:
            prob_classe_0 = prob[0]
            prob_classe_2 = prob[2] if len(prob) > 2 else 0
            
            if prob_classe_0 >= self.soglia_sicurezza and prob_classe_0 > prob_classe_2:
                y_pred_modificata.append(0)
            elif prob_classe_2 >= self.soglia_sicurezza and prob_classe_2 > prob_classe_0:
                y_pred_modificata.append(2)
            else:
                y_pred_modificata.append(1) # Assegnazione clinica conservativa
                
        return np.array(y_pred_modificata)

# =====================================================================
# FUNZIONI DI SUPPORTO E CARICAMENTO
# =====================================================================
def ricostruisci_dataset_completo(cartella_esperimento):
    """Ricostruisce il 100% dei dati a seconda della strategia di split (Holdout, K-Fold, Bootstrap, FullDataset)."""
    cartella_split = cartella_esperimento / "Fold_1" / "split"
    if not cartella_split.exists():
        cartella_split = cartella_esperimento / "split"
        
    candidati = ["X_train_fs.csv", "X_train_ready.csv", "X_train.csv"]
    file_usato = None
    for cand in candidati:
        if (cartella_split / cand).exists():
            file_usato = cand
            break
            
    if not file_usato:
        raise FileNotFoundError("Nessun file X_train trovato per ricostruire il dataset.")
        
    file_test = file_usato.replace("train", "test")
    
    X_train = pd.read_csv(cartella_split / file_usato)
    y_train = pd.read_csv(cartella_split / "y_train.csv").squeeze("columns")
    
    # GESTIONE CASO 1: FullDataset (Il train è già il 100% dei dati)
    if "FullDataset" in cartella_esperimento.name:
        print(f"-> Rilevato 'FullDataset': utilizzo direttamente {file_usato} (nessuna duplicazione)")
        return X_train, y_train
        
    X_test = pd.read_csv(cartella_split / file_test)
    y_test = pd.read_csv(cartella_split / "y_test.csv").squeeze("columns")
    
    # GESTIONE CASO 2: Bootstrap (Train ha rimpiazzi/duplicati, Test è Out-Of-Bag)
    if "Bootstrap" in cartella_esperimento.name:
        print("-> Rilevato 'Bootstrap': unisco Train e Test e rimuovo i duplicati del rimpiazzo...")
        X_tot = pd.concat([X_train, X_test], axis=0)
        y_tot = pd.concat([y_train, y_test], axis=0)
        
        # Rimuove le righe duplicate ricostruendo il set unico originale
        X_tot_unica = X_tot.drop_duplicates()
        y_tot_unica = y_tot.loc[X_tot_unica.index]
        return X_tot_unica.reset_index(drop=True), y_tot_unica.reset_index(drop=True)
    
    # GESTIONE CASO 3: K-Fold o Holdout (Es. 80% Train + 20% Test)
    print(f"-> Ricostruzione 100% del dataset unendo {file_usato} e {file_test}...")
    X_tot = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_tot = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    
    return X_tot, y_tot

def ottieni_modello_da_trial(trial, nome_modello):
    """Costruisce i modelli standard Multiclasse per l'architettura Puro e Fallback."""
    if "SVM" in nome_modello:
        base_svc = make_pipeline(
            StandardScaler(),
            SVC(
                C=trial.suggest_float("C", 0.01, 100.0, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
                random_state=SEED, class_weight='balanced', probability=True
            )
        )
        return base_svc
    elif "AdaBoost" in nome_modello:
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=trial.suggest_int("profondita_stump", 1, 3)),
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            random_state=SEED
        )
    elif "RandomForest" in nome_modello:
        return RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
            random_state=SEED, n_jobs=-1
        )
    elif "XGBoost" in nome_modello:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            objective="multi:softprob", eval_metric="mlogloss", random_state=SEED, n_jobs=-1
        )
    elif "KNN" in nome_modello:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 21, step=2),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"])
            ))
        ])
    elif "DecisionTree" in nome_modello:
        return DecisionTreeClassifier(
            max_depth=trial.suggest_categorical("max_depth", [None, 3, 5, 10, 15, 20]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            random_state=SEED
        )
    else:
        raise NotImplementedError(f"Modello non supportato in questa funzione: {nome_modello}")

# =====================================================================
# ESECUZIONE PRINCIPALE (Menu Universale)
# =====================================================================
def main():
    print("=" * 70)
    print(" 🧠 ESPORTATORE UNIVERSALE MODELLO CLINICO DEFINITIVO (RE-FIT 100%)")
    print("=" * 70)
    
    cartella_risultati = CARTELLA_RADICE / "risultati_finali"
    run_disponibili = sorted([d for d in cartella_risultati.iterdir() if d.is_dir()], key=lambda d: d.name)
    
    if not run_disponibili:
        print("\nNessun esperimento trovato in risultati_finali/.")
        return

    # DOMANDA 1: SCELTA DATASET
    print(f"\n[DOMANDA 1] Seleziona l'esperimento (Dataset) da usare per l'esportazione:")
    for i, d in enumerate(run_disponibili, 1):
        print(f"   [{i}] {d.name}")
        
    scelta_run = input("\nInserisci il numero: ").strip()
    try:
        cartella_exp = run_disponibili[int(scelta_run) - 1]
    except (ValueError, IndexError):
        print("Scelta non valida.")
        return

    # DOMANDA 2: SCELTA ARCHITETTURA LOGICA
    print("\n[DOMANDA 2] Quale Architettura Clinica vuoi esportare?")
    print("   [1] Standard / Puro (Nessuna forzatura, ottimizzazione multiclasse nativa)")
    print("   [2] Con Fallback (Soglia 51% per massima sensibilità sui pazienti di Classe 1)")
    print("   [3] A Cascata (Architettura gerarchica 0 vs Resto, poi 1 vs 2)")
    scelta_arch = input("Scegli (1, 2 o 3) [1]: ").strip()

    # DOMANDA 3: SCELTA ALGORITMO
    print("\n[DOMANDA 3] Quale Motore di Machine Learning vuoi usare?")
    if scelta_arch in ["1", "2"]:
        print("   [1] XGBoost")
        print("   [2] SVM Standard")
        print("   [3] AdaBoost")
        print("   [4] RandomForest")
        print("   [5] DecisionTree")
        print("   [6] KNN")
        scelte_modelli = {"1": "XGBoost", "2": "SVM", "3": "AdaBoost", "4": "RandomForest", "5": "DecisionTree", "6": "KNN"}
        scelta_alg = input("Scegli (1-6) [1]: ").strip()
        motore = scelte_modelli.get(scelta_alg, "XGBoost")
    else:
        print("   [1] XGBoost")
        print("   [2] SVM Standard")
        scelta_alg = input("Scegli (1 o 2) [1]: ").strip()
        motore = "XGBoost" if scelta_alg == "1" else "SVM"
    
    # Preparazione Dati
    gm.imposta_metrica_primaria("2") # Setta F2 come metrica primaria per sicurezza diagnostica
    X_tot, y_tot = ricostruisci_dataset_completo(cartella_exp)
    print(f"\n-> Dimensioni dataset finale (100%): {X_tot.shape[0]} pazienti, {X_tot.shape[1]} features\n")

    # =================================================================
    # FLUSSO 1 e 2: STANDARD o FALLBACK (Multiclasse)
    # =================================================================
    if scelta_arch in ["1", "2"]:
        tipo_str = "Fallback" if scelta_arch == "2" else "Standard"
        print(f"--- Ottimizzazione e Addestramento {motore} ({tipo_str}) sul 100% dei dati ---")
        
        def obiettivo_multiclasse(trial):
            modello_candidato = ottieni_modello_da_trial(trial, motore)
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            return cross_val_score(modello_candidato, X_tot, y_tot, cv=cv_inner, scoring=METRICA_PRIMARIA, n_jobs=-1).mean()

        print(f"   [!] Lancio Optuna per {motore} multiclasse (30 trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
        study.optimize(obiettivo_multiclasse, n_trials=30)
        
        print(f"   ✅ Ricerca completata! Miglior Macro-F2 stimato in CV: {study.best_value:.4f}")
        
        best_trial = optuna.trial.FixedTrial(study.best_params)
        modello_base = ottieni_modello_da_trial(best_trial, motore)
        
        print("   -> Fit del Modello sul 100% del dataset...")
        modello_base.fit(X_tot, y_tot)
        
        # Confezionamento Finale (Wrapper o Diretto)
        if scelta_arch == "2":
            modello_definitivo = ModelloFallbackWrapper(base_model=modello_base, soglia_sicurezza=0.51)
            nome_esportazione = f"Modello_Clinico_Definitivo_{motore}_Fallback"
        else:
            modello_definitivo = modello_base
            nome_esportazione = f"Modello_Clinico_Definitivo_{motore}_Standard"
            
        parametri_da_salvare = study.best_params
        miglior_score_stima = study.best_value

    # =================================================================
    # FLUSSO 3: CASCATA
    # =================================================================
    else:
        print(f"--- FASE A: Ottimizzazione e Addestramento {motore} Cascata (Classe 0 vs Resto) ---")
        y_tot_A = np.where(y_tot == 0, 0, 1)
        
        if motore == "XGBoost":
            best_params_A, score_A = ottimizza_xgboost_binario(X_tot, y_tot_A, "MODELLO A (Produzione)")
            model_A = XGBClassifier(**best_params_A, objective="binary:logistic", random_state=SEED, n_jobs=-1)
        else:
            best_params_A, score_A = ottimizza_svm_binario(X_tot, y_tot_A, "MODELLO A (Produzione)")
            model_A = make_pipeline(StandardScaler(), SVC(**best_params_A, random_state=SEED, class_weight='balanced', probability=True))
            
        model_A.fit(X_tot, y_tot_A)

        print(f"\n--- FASE B: Ottimizzazione e Addestramento {motore} Cascata (Classe 1 vs 2) ---")
        mask_B = (y_tot == 1) | (y_tot == 2)
        X_tot_B, y_tot_filato = X_tot[mask_B], y_tot[mask_B]
        y_tot_B = np.where(y_tot_filato == 1, 0, 1)
        
        if motore == "XGBoost":
            ratio_bilanciamento = float(np.sum(y_tot_B == 0) / np.sum(y_tot_B == 1))
            best_params_B, score_B = ottimizza_xgboost_binario(X_tot_B, y_tot_B, "MODELLO B (Produzione)", scale_pos_weight=ratio_bilanciamento)
            model_B = XGBClassifier(**best_params_B, objective="binary:logistic", scale_pos_weight=ratio_bilanciamento, random_state=SEED, n_jobs=-1)
        else:
            best_params_B, score_B = ottimizza_svm_binario(X_tot_B, y_tot_B, "MODELLO B (Produzione)")
            model_B = make_pipeline(StandardScaler(), SVC(**best_params_B, random_state=SEED, class_weight='balanced', probability=True))

        miglior_soglia = trova_soglia_ottimale(model_B, X_tot_B, y_tot_B, "F2")
        model_B.fit(X_tot_B, y_tot_B)

        # Confezionamento
        modello_definitivo = ModelloCascataWrapper(model_A, model_B, soglia_B=miglior_soglia)
        nome_esportazione = f"Modello_Clinico_Definitivo_{motore}_Cascata"
        parametri_da_salvare = {"Fase_A": best_params_A, "Fase_B": best_params_B, "Soglia_F2": miglior_soglia}
        miglior_score_stima = (score_A + score_B) / 2 # Media indicativa

    # ===============================================================
    # SALVATAGGIO COMUNE A TUTTI I FLUSSI
    # ===============================================================
    out_dir = CARTELLA_RADICE / "MODELLO DEFINITIVO"
    out_dir.mkdir(exist_ok=True)
    
    percorso_export = out_dir / f"{nome_esportazione}.pkl"
    joblib.dump(modello_definitivo, percorso_export)
    
    with open(out_dir / f"Parametri_{nome_esportazione}.txt", "w") as f:
        f.write(f"Iperparametri {nome_esportazione.replace('_', ' ')} (100% Dataset)\n")
        f.write("-" * 50 + "\n")
        f.write(f"STIMA PRESTAZIONI OOF/CV (Macro-F2): {miglior_score_stima:.4f}\n")
        f.write("-" * 50 + "\n")
        if scelta_arch == "3":
            f.write("MODELLO A (Classe 0 vs Resto):\n")
            for k, v in parametri_da_salvare["Fase_A"].items(): f.write(f"  {k}: {v}\n")
            f.write("\nMODELLO B (Classe 1 vs 2):\n")
            for k, v in parametri_da_salvare["Fase_B"].items(): f.write(f"  {k}: {v}\n")
            f.write(f"\nSOGLIA DECISIONALE OTTIMIZZATA F2: {parametri_da_salvare['Soglia_F2']:.2f}\n")
        else:
            for k, v in parametri_da_salvare.items(): f.write(f"  {k}: {v}\n")
            if scelta_arch == "2":
                f.write(f"\nFALLBACK CLINICO ATTIVO SULLA CLASSE 1 (Soglia: >51%)\n")

    print("\n" + "=" * 70)
    print(f" 🏆 MODELLO ESPORTATO E PRONTO PER LA MESSA IN PRODUZIONE / TESI!")
    print(f" File e parametri salvati in: {out_dir.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()