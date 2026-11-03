### Codice di separazione del dataset in target e feature -----------------------------------------------

def split_features_target(df_features):
    
    # Rimuoviamo 'hadm_id' perché è solo un identificatore e non va dato al modello
    # Rimuoviamo 'target' perché è la colonna che vogliamo indovinare
    X = df_features.drop(columns=['hadm_id', 'target'])

    # y contiene solo le risposte corrette (i capitoli ICD)
    y = df_features['target']

    print("Dimensioni di X (Feature):", X.shape)
    print("Dimensioni di y (Target):", y.shape)


  
    
    return X, y