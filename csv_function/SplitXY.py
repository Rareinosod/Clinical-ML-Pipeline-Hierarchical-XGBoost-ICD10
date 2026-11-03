##  Codice di separazione del dataset in target e feature

# >possibile modifica? lasciamo l'ID del paziente e lo togliamo alla fine oppure boh?



import pandas as pd


#Import del csv dati 

df_features = pd.read_csv("df_features.csv")



# Rimuoviamo 'hadm_id' perché è solo un identificatore e non va dato al modello
# Rimuoviamo 'target' perché è la colonna che vogliamo indovinare
X = df_features.drop(columns=['hadm_id', 'target'])

# y contiene solo le risposte corrette (i capitoli ICD)
y = df_features['target']

print("Dimensioni di X (Feature):", X.shape)
print("Dimensioni di y (Target):", y.shape)


# Salvataggio di X (le feature) e y (il target)
X.to_csv("X_preprocessed.csv", index=False)
y.to_csv("y_target.csv", index=False)
print("File X e y salvati correttamente!")