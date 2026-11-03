# =============================================================================
# IMPORT 
# =============================================================================
import os
import sys              
from pathlib import Path 
import re               
import pandas as pd
from datetime import datetime
import seaborn as sn

CARTELLA_RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CARTELLA_RADICE))
# =============================================================================
# IMPORT - MODULI DEL PROGETTO
# =============================================================================

# Split train/test e strategie di validazione
from validation_strategies.validation_strategies import (
    split_holdout,
    get_stratified_kfold_splits,
    split_bootstrap,
    split_full_dataset  # <--- AGGIUNTO
)

# Parte A (pulizia valori impossibili) + Parte B (outlier univariati) + Parte C (DBSCAN)
from outlier_function.outlier_pipeline import (
    clean_raw_measurements,
    clean_raw_bloodtest_lists,
    apply_univariate_outlier_detection,
    apply_dbscan_outlier_detection,
)
from outlier_function.physiological_ranges import (
    PHYSIOLOGICAL_RANGES_TRIAGE,
    PHYSIOLOGICAL_RANGES_BLOOD,
)

# Gestione NaN e imputazione
from nan_and_imputation_function.nan_and_imputation_handling import (
    fit_nan_handler,
    transform_nan,
)

# Elaborazione base dei singoli file sorgente + creazione del matricione
from csv_function.crea_matricione_func import (
    process_ICD,
    process_Demographic,
    process_Triage,
    process_Bloodtest,
    analyze_CRP,
    analyze_blood_statistics,
)
from csv_function.SplitXY_func import split_features_target

# Feature selection
from feature_selection_function.feature_selection import (
    remove_collinear_features,
    select_by_statistical_test,
)

from plotting_function.plotting_utils import genera_grafici_esperimento

# =============================================================================
# PARAMETRI
# =============================================================================

COLONNE_ESAMI = [
    "Hemoglobin", "Hematocrit", "White Blood Cells", "Platelet Count",
    "Red Blood Cells", "Mean Corpuscular Volume", "Bicarbonate",
    "Calcium", "Magnesium", "Phosphate",
]

EXCLUDE_COLUMNS = ["hadm_id", "ICD chapter"]
DBSCAN_FEATURE_SET = "triage"

# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def main():
    print("=" * 70)
    print("PIPELINE ML - INZIO ESECUZIONE PREPROCESSING DATI")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # SETUP INIZIALE CARTELLA E VARIABILI DI OUTPUT GLOBALI
    # -------------------------------------------------------------------------
    os.makedirs("risultati_finali", exist_ok=True)
    
    # Cerca il numero di esperimento più alto tra le cartelle esistenti
    max_esperimento = 0
    for d in os.listdir("risultati_finali"):
        if os.path.isdir(os.path.join("risultati_finali", d)):
            match = re.search(r'_Esperimento(\d+)', d)
            if match:
                num = int(match.group(1))
                if num > max_esperimento:
                    max_esperimento = num
                    
    numero_esperimento = max_esperimento + 1

    timestamp = datetime.now().strftime("%d%b_%H%M")
    cartella_base = os.path.join("risultati_finali", f"Run_{timestamp}_Esperimento{numero_esperimento}")
    os.makedirs(cartella_base, exist_ok=True)
        
    report_global_dir = os.path.join(cartella_base, "report_globali")
    os.makedirs(report_global_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: ELABORAZIONE BASE DEI SINGOLI FILE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[STEP 1] Caricamento ed elaborazione base dei file...")
    os.chdir('Data')
    db_ICD = pd.read_csv("ICD_chapter.csv")
    db_Demographic = pd.read_csv("Demographic.csv")
    db_Triage = pd.read_csv("Triage.csv")
    db_BloodTest = pd.read_csv("BloodTest.csv")
    os.chdir('..')

    processed_ICD = process_ICD(db_ICD)
    processed_Demographic = process_Demographic(db_Demographic)
    processed_Triage = process_Triage(db_Triage)
    processed_BloodTest = process_Bloodtest(db_BloodTest)

    # -------------------------------------------------------------------------
    # STEP 2: MERGE COMPLETO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[STEP 2] Merge di ICD + Demographic + Triage + BloodTest...")
    df_features = pd.merge(processed_ICD, processed_Demographic, on="hadm_id")
    df_features = pd.merge(df_features, processed_Triage, on="hadm_id")
    df_features = pd.merge(df_features, processed_BloodTest, on="hadm_id")

    # -------------------------------------------------------------------------
    # STEP 3 & 4: PULIZIA VALORI IMPOSSIBILI (RANGES) E SALVATAGGIO REPORT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[STEP 3 & 4] Pulizia valori impossibili (Triage e BloodTest)...")
    triage_cleaning_result = clean_raw_measurements(df_features, PHYSIOLOGICAL_RANGES_TRIAGE)
    df_features = triage_cleaning_result.dataframe
    triage_cleaning_result.report.to_csv(os.path.join(report_global_dir, "report_pulizia_triage.csv"), index=False)
    
    blood_cleaning_result = clean_raw_bloodtest_lists(df_features, PHYSIOLOGICAL_RANGES_BLOOD)
    df_features = blood_cleaning_result.dataframe
    blood_cleaning_result.report.to_csv(os.path.join(report_global_dir, "report_pulizia_bloodtest.csv"), index=False)

    # -------------------------------------------------------------------------
    # STEP 5: STATISTICHE SUL BLOODTEST GIÀ PULITO
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[STEP 5] Calcolo statistiche su BloodTest pulito...")
    df_features = analyze_CRP(df_features)
    df_features = analyze_blood_statistics(df_features, COLONNE_ESAMI)

    # -------------------------------------------------------------------------
    # STEP 6: SEPARAZIONE FEATURE E TARGET
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("\n[STEP 6] Separazione Feature e Target...")
    X_full, y_full = split_features_target(df_features)

    # =========================================================================
    # STEP 7: MENU DI CONFIGURAZIONE INTERATTIVA
    # =========================================================================
    print("\n" + "=" * 70)
    print("=== MENU DI CONFIGURAZIONE PIPELINE ===")
    print("=" * 70)

    # --- SPLIT ---
    print("\n[STRATEGIA DI VALIDAZIONE]")
    print("   Scegli come validare i modelli:")
    print("   [1] Hold-out (Singolo Train/Test all'80/20)")
    print("   [2] K-Fold (Genera K set indipendenti per la Nested Cross-Validation)")
    print("   [3] Bootstrap (Train con rimpiazzo, Test Out-Of-Bag)")
    print("   [4] Intero Dataset (Nessuno split, 100% dei dati per modello definitivo)")
    scelta_split = input("   -> Digita 1, 2, 3 o 4 [1]: ").strip()
    
    usa_kfold = (scelta_split == '2')
    usa_bootstrap = (scelta_split == '3')
    usa_intero = (scelta_split == '4')

    if usa_kfold:
        scelta_k = input("   -> Quanti split vuoi per la K-Fold (scrivi il numero) [Premi Invio per 5]: ").strip()
        if scelta_k == "":
            n_splits = 5
        else:
            try:
                n_splits = int(scelta_k)
                if n_splits < 2:
                    print("      [!] Il numero minimo di split per la K-Fold è 2. Imposto a 2.")
                    n_splits = 2
            except ValueError:
                print("      [!] Input non valido (non è un numero). Uso il default: 5.")
                n_splits = 5
    else:
        n_splits = 1

    # --- MAD ---
    print("\n" + "-" * 60)
    print("\n[OUTLIER UNIVARIATI - MAD]")
    fai_mad = input("   -> Vuoi eseguire la rilevazione outlier univariati (MAD)? (s/n) [s]: ").strip().lower() != 'n'
    mad_thresh, min_obs = 3.5, 10
    if fai_mad:
        if input(f"   -> Usare parametri MAD di default (soglia={mad_thresh}, min_osservazioni={min_obs})? (s/n) [s]: ").strip().lower() == 'n':
            try: mad_thresh = float(input("      Inserisci nuova soglia Z-Score (es. 4.0): "))
            except ValueError: pass
            try: min_obs = int(input("      Inserisci nuove min_osservazioni (es. 15): "))
            except ValueError: pass

    # --- DBSCAN ---
    print("\n" + "-" * 60)
    print("\n[OUTLIER MULTIVARIATI - DBSCAN]")
    usa_dbscan = input("   -> Vuoi eseguire l'analisi DBSCAN? (s/n) [s]: ").strip().lower() != 'n'
    eps_val, min_samp_val = 1.5, 10
    set_scelto = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    if usa_dbscan:
        print(f"\n   [Feature per DBSCAN] Set fisso: Triage {set_scelto}")
        print("   -> Scelta motivata: sono le uniche misurazioni prese tutte nello stesso istante,")
        print("      quindi l'unico gruppo per cui 'combinazione anomala' ha un significato clinico coerente.")
        if input(f"\n   -> Usare parametri DBSCAN di default (eps={eps_val}, min_samples={min_samp_val})? (s/n) [s]: ").strip().lower() == 'n':
            try: eps_val = float(input("      Inserisci nuovo raggio epsilon (es. 2.0): "))
            except ValueError: pass
            try: min_samp_val = int(input("      Inserisci nuovo min_samples (es. 5): "))
            except ValueError: pass

    # --- NaN IMPUTATION ---
    print("\n" + "-" * 60)
    print("\n[GESTIONE VALORI MANCANTI - NaN]")
    fai_imputazione = input("   -> Vuoi eseguire l'imputazione dei NaN? (s/n) [s]: ").strip().lower() != 'n'
    strategia_scelta, pool_type, strategia_imputazione, scelta_mask = "nessuna_imputazione", "all", "N/A", "-"
    
    if fai_imputazione:
        print("\n   [MASCHERA] Su quali pazienti fittare l'imputatore?")
        print("   [1] - baseline_all (Nessun filtro, impara da tutti i pazienti)")
        if fai_mad:
            print("   [2] - mask_tipici (Filtro Univariato: impara solo dai casi clinici standard)")
        else:
            print("   [2] - (Non disponibile, step MAD saltato)")
        
        if usa_dbscan and fai_mad:
            print("   [3] - mask_tipici_strict (Filtro Composito: massima restrizione statistica)")
        else:
            print("   [3] - (Non disponibile: richiede sia MAD che DBSCAN attivi)")
            
        scelta_mask = input("   -> Digita 1, 2 o 3 [3]: ").strip()

        if scelta_mask == '1': strategia_scelta = "baseline_all"
        elif scelta_mask == '2' and fai_mad: strategia_scelta = "mask_tipici"
        elif scelta_mask == '3' and usa_dbscan and fai_mad: strategia_scelta = "mask_tipici_strict"
        else: strategia_scelta = "mask_tipici_strict" if (usa_dbscan and fai_mad) else ("mask_tipici" if fai_mad else "baseline_all")

        if strategia_scelta != "baseline_all":
            print("\n   [POOL] Scegli il pool di apprendimento:")
            print("   [1] - Casi Tipici (Ordinary) -> Impara dai profili clinici standard")
            print("   [2] - Casi Atipici (Rare) -> Impara dai profili clinici estremi/outlier")
            scelta_pool = input("   -> Digita 1 o 2 [1]: ").strip()
            pool_type = "rare" if scelta_pool == '2' else "ordinary"

        print("\n   [METODO] Scegli l'algoritmo matematico:")
        print("   [1] - KNN (K-Nearest Neighbors - calcola la media sui pazienti più simili)")
        print("   [2] - Mediana (Semplice, veloce e resistente ai valori estremi)")
        scelta_algoritmo_nan = input("   -> Digita 1 o 2 [1]: ").strip()
        strategia_imputazione = "median" if scelta_algoritmo_nan == '2' else "knn"

    # --- FEATURE SELECTION ---
    print("\n" + "-" * 60)
    print("\n[FEATURE SELECTION]")
    fai_feature_selection = input("   -> Vuoi eseguire la Feature Selection (Spearman + Kruskal)? (s/n) [s]: ").strip().lower() != 'n'
    soglia_spearman = 0.90
    if fai_feature_selection:
        if input(f"   -> Usare soglia Spearman di default per la ridondanza (corr >= {soglia_spearman})? (s/n) [s]: ").strip().lower() == 'n':
            try: soglia_spearman = float(input("      Inserisci nuova soglia (es. 0.85): "))
            except ValueError: print("      Valore non valido. Uso il default 0.90.")

    # --- GRAFICI ---
    print("\n" + "-" * 60)
    vuoi_grafici = False
    if not usa_kfold:
        vuoi_grafici = input("\n   -> Vuoi generare i grafici diagnostici? Richiede qualche minuto (s/n) [n]: ").strip().lower() == 's'
    else:
        print("\n   [INFO] Generazione grafici saltata (i grafici sono supportati solo in Holdout, non in K-Fold).")


    # =========================================================================
    # STEP 8: MOTORE DI PREPROCESSING (Zero Leakage, con tutti i salvataggi)
    # =========================================================================
    def esegui_preprocessing_fold(X_tr, X_te, y_tr, y_te, cartella_output, nome_fold):
        print(f"\n   === Elaborazione {nome_fold} in corso... ===")
        
        report_dir = os.path.join(cartella_output, "report")
        split_dir = os.path.join(cartella_output, "split")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)
        
        # Salvataggio Dati Base
        X_tr.to_csv(os.path.join(split_dir, "X_train.csv"), index=False)
        X_te.to_csv(os.path.join(split_dir, "X_test.csv"), index=False)
        y_tr.to_csv(os.path.join(split_dir, "y_train.csv"), index=False)
        y_te.to_csv(os.path.join(split_dir, "y_test.csv"), index=False)
        
        # 1. MAD
        if fai_mad:
            univ_res = apply_univariate_outlier_detection(X_tr, EXCLUDE_COLUMNS, mad_thresh, min_obs)
            X_tr_b = univ_res.dataframe
            univariate_report = univ_res.report
            univariate_report.to_csv(os.path.join(report_dir, "report_univariato.csv"), index=False)
        else:
            X_tr_b, univariate_report = X_tr.copy(), None
            X_tr_b["has_univariate_outlier"] = 0
            
        # 2. DBSCAN
        dbscan_result = None
        if usa_dbscan:
            dbscan_result = apply_dbscan_outlier_detection(X_tr_b, set_scelto, eps_val, min_samp_val)
            X_tr_c = dbscan_result.dataframe
            pd.DataFrame(list(dbscan_result.report.items()), columns=["parameter", "value"]).to_csv(
                os.path.join(report_dir, "report_dbscan.csv"), index=False)
        else:
            X_tr_c = X_tr_b.copy()
            X_tr_c["dbscan_low_density"] = 0
            
        # Salvataggio matricione pre-imputazione
        X_tr_c.to_csv(os.path.join(cartella_output, "matricione_con_outlier_flags.csv"), index=False)

        # 3. NaN Imputation
        colonne_da_imputare = [col for col in X_tr_c.select_dtypes(include='number').columns 
                               if col not in EXCLUDE_COLUMNS and not col.startswith(('dbscan', 'has_', 'n_')) 
                               and 'outlier' not in col and 'zscore' not in col and 'invalid' not in col]
        
        if fai_imputazione:
            mask_experiments = {
                "baseline_all": None,
                "mask_tipici": (X_tr_c["has_univariate_outlier"] == 0),
                "mask_tipici_strict": (X_tr_c["has_univariate_outlier"] == 0) & (X_tr_c["dbscan_low_density"] == 0)
            }
            maschera_train = mask_experiments.get(strategia_scelta, None)
            
            fitted_nan = fit_nan_handler(
                X_train=X_tr_c, columns=colonne_da_imputare, impute=True,
                pool=pool_type, group_mask=maschera_train, strategy=strategia_imputazione, 
                knn_neighbors=5, add_missing_flag=True
            )
            
            # Salvataggio Report NaN
            X_tr_ready = transform_nan(X_tr_c, fitted_nan, output_report_path=os.path.join(report_dir, "report_nan_train.csv"))
            X_te_ready = transform_nan(X_te, fitted_nan, output_report_path=os.path.join(report_dir, "report_nan_test.csv"))
        else:
            X_tr_ready, X_te_ready = X_tr_c.copy(), X_te.copy()

        # Salvataggio dataset imputato
        X_tr_ready.to_csv(os.path.join(split_dir, "X_train_ready.csv"), index=False)
        X_te_ready.to_csv(os.path.join(split_dir, "X_test_ready.csv"), index=False)

        # 4. Feature Selection
        feature_cols_per_fs = [col for col in X_tr_ready.columns if col not in EXCLUDE_COLUMNS 
                               and not col.startswith(('dbscan', 'has_', 'n_')) 
                               and 'outlier' not in col and 'zscore' not in col and 'invalid' not in col]
        
        X_tr_to_select, X_te_to_select = X_tr_ready[feature_cols_per_fs], X_te_ready[feature_cols_per_fs]
        report_stat = None
        X_tr_fs_corr, X_tr_fs = X_tr_to_select.copy(), X_tr_to_select.copy() 
        
        if fai_feature_selection:
            cols_dopo_corr, dropped_corr, rep_corr = remove_collinear_features(X_tr_to_select, y_tr, soglia_spearman, "spearman")
            X_tr_fs_corr, X_te_fs_corr = X_tr_to_select[cols_dopo_corr], X_te_to_select[cols_dopo_corr]
            
            cols_dopo_stat, dropped_stat, report_stat = select_by_statistical_test(X_tr_fs_corr, y_tr, 0.05)
            X_tr_fs, X_te_fs = X_tr_fs_corr[cols_dopo_stat], X_te_fs_corr[cols_dopo_stat]
            
            # Salvataggio Report FS
            rep_corr.to_csv(os.path.join(report_dir, "report_ridondanza_correlazione.csv"), index=False)
            report_stat.to_csv(os.path.join(report_dir, "report_rilevanza_kruskal.csv"), index=False)
        else:
            X_tr_fs, X_te_fs = X_tr_to_select.copy(), X_te_to_select.copy()

        # Salvataggio dataset finale
        X_tr_fs.to_csv(os.path.join(split_dir, "X_train_fs.csv"), index=False)
        X_te_fs.to_csv(os.path.join(split_dir, "X_test_fs.csv"), index=False)
        
        print(f"   === {nome_fold} Completato. Feature finali: {X_tr_fs.shape[1]} ===")
        
        return X_tr_c, X_tr_ready, X_te, dbscan_result, univariate_report, report_stat, X_tr_to_select, X_tr_fs_corr, colonne_da_imputare


    # =========================================================================
    # STEP 9: ESECUZIONE CICLO DI SPLIT E PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("=== ESECUZIONE PIPELINE SUI DATI ===")
    print("=" * 70)

    dati_per_plot = None

    if usa_kfold:
        skf = get_stratified_kfold_splits(X_full, y_full, n_splits=n_splits)
        for k, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full)):
            X_tr, y_tr = X_full.iloc[train_idx].copy(), y_full.iloc[train_idx].copy()
            X_te, y_te = X_full.iloc[test_idx].copy(), y_full.iloc[test_idx].copy()
            cartella_fold = os.path.join(cartella_base, f"Fold_{k+1}")
            esegui_preprocessing_fold(X_tr, X_te, y_tr, y_te, cartella_fold, f"Fold_{k+1}")
            
    elif usa_bootstrap:
        split_res = split_bootstrap(X_full, y_full, random_state=42)
        dati_per_plot = esegui_preprocessing_fold(
            split_res.X_train, split_res.X_test, 
            split_res.y_train, split_res.y_test, 
            cartella_base, "Bootstrap_OOB"
        )
        
    elif usa_intero:
        # Modello Definitivo: Nessuno split
        split_res = split_full_dataset(X_full, y_full)
        dati_per_plot = esegui_preprocessing_fold(
            split_res.X_train, split_res.X_test, 
            split_res.y_train, split_res.y_test, 
            cartella_base, "Full_Dataset_Unico"
        )
        
    else:
        # Default: Holdout
        split_res = split_holdout(X_full, y_full, test_size=0.20, random_state=42)
        dati_per_plot = esegui_preprocessing_fold(
            split_res.X_train, split_res.X_test, 
            split_res.y_train, split_res.y_test, 
            cartella_base, "Holdout_Unico"
        )

    # =========================================================================
    # STEP 10: GRAFICI DIAGNOSTICI
    # =========================================================================
    if vuoi_grafici and not usa_kfold and dati_per_plot is not None:
        print("\n" + "-" * 60)
        print("\n[STEP 10] Generazione grafici diagnostici...")
        X_tr_c, X_tr_ready, X_te, dbscan_result, univariate_report, report_stat, X_tr_to_select, X_tr_fs_corr, colonne_da_imputare = dati_per_plot
        
        cartella_plots = os.path.join(cartella_base, "plots")
        dbscan_feature_cols_plot = dbscan_result.features_used if usa_dbscan else None
        
        genera_grafici_esperimento(
            cartella_plots=cartella_plots,
            X_train_pre=X_tr_c,
            X_train_post=X_tr_ready,
            X_test_pre=X_te,
            colonne_features=colonne_da_imputare,
            univariate_report=univariate_report,
            usa_dbscan=usa_dbscan,
            dbscan_feature_cols=dbscan_feature_cols_plot,
            dbscan_flag_col="dbscan_low_density",
            X_train_fs_before_corr=X_tr_to_select,
            X_train_fs_after_corr=X_tr_fs_corr,
            report_stat=report_stat,
        )

    # =========================================================================
    # STEP 11: CREAZIONE LOG E SALVATAGGIO
    # =========================================================================
    step_saltati = []
    if not fai_mad: step_saltati.append("MAD")
    if not usa_dbscan: step_saltati.append("DBSCAN")
    if not fai_imputazione: step_saltati.append("Imputazione")
    if not fai_feature_selection: step_saltati.append("FeatureSelection")

    etichetta = "PreprocessingSemplice" if len(step_saltati) == 4 else (
        "PreprocessingCompleto" if not step_saltati else "PreprocessingAlternativo(No " + ", No ".join(step_saltati) + ")"
    )

    if usa_kfold: 
        etichetta = "KFold_" + etichetta
        str_val = f"K-Fold ({n_splits} splits)"
    elif usa_bootstrap: 
        etichetta = "Bootstrap_" + etichetta
        str_val = "Bootstrap (Train rimpiazzo / Test OOB)"
    elif usa_intero:
        etichetta = "FullDataset_" + etichetta
        str_val = "Intero Dataset (Nessuno split, 100% addestramento per Modello Finale)"
    else: 
        str_val = "Holdout (80/20)"

    percorso_log = os.path.join(cartella_base, "log_esperimento.txt")
    with open(percorso_log, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"LOG ESPERIMENTO #{numero_esperimento} - {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Etichetta setup: {etichetta}\n")
        f.write(f"Strategia Validazione: {str_val}\n\n")

        f.write("--- STEP OPZIONALI ---\n")
        f.write(f"MAD (outlier univariati): {'Eseguito' if fai_mad else 'Saltato'}\n")
        if fai_mad: f.write(f"  Soglia Z-Score MAD: {mad_thresh}\n  Min Osservazioni MAD: {min_obs}\n")

        f.write(f"DBSCAN (outlier multivariati): {'Eseguito' if usa_dbscan else 'Saltato'}\n")
        if usa_dbscan: f.write(f"  Parametri: eps={eps_val}, min_samples={min_samp_val}\n  Feature set usato: {set_scelto}\n")

        f.write(f"Imputazione NaN: {'Eseguita' if fai_imputazione else 'Saltata (NaN lasciati nel dataset)'}\n")
        if fai_imputazione:
            f.write(f"  Strategia maschera: Opzione {scelta_mask} ({strategia_scelta})\n")
            f.write(f"  Pool di addestramento (imputer): {pool_type}\n")
            f.write(f"  Metodo matematico: {strategia_imputazione}\n")

        f.write(f"Feature Selection (Spearman + Kruskal): {'Eseguita' if fai_feature_selection else 'Saltata'}\n")
        if fai_feature_selection: f.write(f"  Soglia Spearman: {soglia_spearman}\n")

        f.write(f"\nGrafici diagnostici generati: {vuoi_grafici}\n")

    cartella_finale = os.path.join("risultati_finali", f"Run_{timestamp}_Esperimento{numero_esperimento}_{etichetta}")
    try:
        os.rename(cartella_base, cartella_finale)
        print(f"\n  -> Log creato. Cartella rinominata in: {cartella_finale}")
    except Exception as e:
        print(f"\n  -> Log creato in: {cartella_base} (Rinominazione fallita: {e})")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETATA CON SUCCESSO")
    print("=" * 70)

if __name__ == "__main__":
    main()