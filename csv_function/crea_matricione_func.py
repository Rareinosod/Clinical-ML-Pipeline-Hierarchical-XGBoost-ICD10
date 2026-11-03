### Codice di analisi dei singoli CSV per l'estrazione di feature ed unione in matricione unico




import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os





### Analisi del file ICD -----------------------------------------------

def process_ICD(db_ICD):
    db_ICD_mod = db_ICD.copy()

    #Inserimento nel dataset di features + OneHotEncoding
    #df_features = pd.get_dummies(db_ICD_mod, columns=['ICD chapter'], prefix='ICD', dtype=int)  #Il comando riscrive tutte le colonne invariate tranne quella encoded


    # Inizializziamo il LabelEncoder per trasformare le etichette di testo in numeri
    le = LabelEncoder()

    # Creiamo la colonna 'target' con i valori numerici (0, 1, 2, 3, 4)
    db_ICD_mod['target'] = le.fit_transform(db_ICD_mod['ICD chapter'])

    # Rimuoviamo la colonna testuale originale, mantenendo solo hadm_id e target
    db_ICD_mod = db_ICD_mod.drop('ICD chapter', axis=1)

    return db_ICD_mod





### Analisi del file Demographic -----------------------------------------------

def process_Demographic(db_Demographic):
    db_Demographic_mod = db_Demographic.copy()

    #Conversione dei tempi di degenza in intervalli temporali "LOS"
    db_Demographic_mod["admittime"] = pd.to_datetime(db_Demographic_mod["admittime"])
    db_Demographic_mod["dischtime"] = pd.to_datetime(db_Demographic_mod["dischtime"])

    db_Demographic_mod["los_hours"] = (db_Demographic_mod["dischtime"] - db_Demographic_mod["admittime"]).dt.total_seconds() / 3600 #Differenza in secondi tradotta in ore

    db_Demographic_mod = db_Demographic_mod.drop(columns=["admittime","dischtime"])

    #Inserimento della colonna di mortalità

    db_Demographic_mod["mortality"] = db_Demographic_mod["deathtime"].notnull().astype(int)
    db_Demographic_mod = db_Demographic_mod.drop(columns=["deathtime"])

    #One-hot Encoding per etichette testuali
    db_Demographic_mod = pd.get_dummies(db_Demographic_mod, columns=["admission_type"], dtype=int, drop_first=True)
    db_Demographic_mod = pd.get_dummies(db_Demographic_mod, columns=["race"], dtype=int, drop_first=True)
    db_Demographic_mod = pd.get_dummies(db_Demographic_mod, columns=["gender"], dtype=int, drop_first=True )

    return db_Demographic_mod





###  Analisi del file Triage.csv -----------------------------------------------

def process_Triage(db_Triage):
    db_Triage_mod = db_Triage.copy()
    
    return db_Triage_mod






### Analisi del file BloodTest.csv (Parte 1: Preparazione base e conversione liste) -----------------------------------------------

def process_Bloodtest(db_BloodTest):

    db_BloodTest_mod = db_BloodTest.copy()


    # Funzione interna per convertire le stringhe in liste numeriche
    def string_to_list(cell):
        # Gestione celle vuote o NaN: restituisce una lista vuota
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        
        # Divisione tramite punto e virgola e conversione in float
        return [float(x) for x in str(cell).split(";") if x.strip()]

    # Applichiamo la conversione a tutte le misurazioni ematiche
    for column in db_BloodTest_mod.columns:
        if column != "hadm_id":
            db_BloodTest_mod[column] = db_BloodTest_mod[column].apply(string_to_list)

    return db_BloodTest_mod



# Analisi proteina C reattiva 

def analyze_CRP(df_features):
    
    # Definizione soglia per la Proteina C-Reattiva
    CRP_THRESHOLD = 10.0
    
    # Controlliamo se la colonna della Proteina C-Reattiva è presente nel matricione
    if "C-Reactive Protein" in df_features.columns:
        
        # 'x' ora è direttamente una lista. Iteriamo subito sui suoi valori!
        df_features["crp_elevated"] = df_features["C-Reactive Protein"].apply(
            lambda x: 1 if isinstance(x, list) and any(v > CRP_THRESHOLD for v in x) else 0)
        
        # Rimuoviamo la colonna originale con i valori testuali multipli
        df_features = df_features.drop("C-Reactive Protein", axis=1)

    return df_features



#Estrazione statistiche per colonne target




def analyze_blood_statistics(df_features, columns_to_process):
    
    # Definisco la logica matematica adattata per ricevere direttamente una lista Python
    def calculate_stats(val_list):
        
        # 1. Gestione lista vuota
        if not isinstance(val_list, list) or len(val_list) == 0: 
            return pd.Series({"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan,
                "std": np.nan, "first": np.nan, "last": np.nan,
                "count": 0, "trend": np.nan})

        # 2. La lista contiene già i float, contiamo solo gli elementi
        n_vals = len(val_list)

        # 3. Calcolo statistiche con la logica originale
        res = {
            "mean": np.mean(val_list),
            "median": np.median(val_list),
            "min": np.min(val_list),
            "max": np.max(val_list),
            "std": np.std(val_list),
            "first": val_list[0],
            "last": val_list[-1],
            "count": n_vals,
            "trend": val_list[-1] - val_list[0] if n_vals > 1 else 0.0
        }
        
        return pd.Series(res)


    # LISTE TEMPORANEE: Raccogliamo i dati per evitare la frammentazione della RAM
    nuovi_dataframe_statistici = []
    colonne_da_rimuovere = []

    # Iteriamo solo sulle colonne che abbiamo passato in input
    for column in columns_to_process:
        
        # Ci assicuriamo che la colonna esista nel dataset
        if column in df_features.columns:
            
            # Calcolo tutte le statistiche per la colonna corrente
            column_stats = df_features[column].apply(calculate_stats)

            # Rinomino con prefisso (es. "Glucose_mean", "Glucose_median", ecc.)
            column_stats.columns = [f"{column}_{stat}" for stat in column_stats.columns]

            # Invece di fare l'insert, salvo le nuove colonne e mi appunto quale vecchia colonna eliminare
            nuovi_dataframe_statistici.append(column_stats)
            colonne_da_rimuovere.append(column)

    # CONCATENAZIONE IN BLOCCO (Fine del ciclo)
    if nuovi_dataframe_statistici:
        # 1. Rimuovo in un colpo solo tutte le vecchie colonne con le liste
        df_features = df_features.drop(columns=colonne_da_rimuovere)
        
        # 2. Unisco il dataframe originale con tutti i nuovi blocchi di statistiche contemporaneamente
        df_features = pd.concat([df_features] + nuovi_dataframe_statistici, axis=1)

    return df_features



















