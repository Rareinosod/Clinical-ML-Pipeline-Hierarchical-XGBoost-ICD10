README: Modulo di Gestione NaN, Imputazione e Scaling (nan_and_imputation_ handling.py)
📋 Panoramica del Modulo
Il modulo nan_and_imputation_handling.py rappresenta il Blocco 3 della pipeline clinica. È progettato per gestire in modo rigoroso e modulare i valori mancanti (NaN) e la preparazione numerica dei dati per i modelli di Machine Learning, evitando qualsiasi forma di data leakage.

🛠️ Responsabilità Principali
Creazione di Shadow Features (_is_missing):
Per ogni colonna numerica trattata, il modulo genera automaticamente una colonna binaria di supporto (<col>_is_missing con valori 1 o 0) che traccia la presenza originaria del dato mancante. Questo permette ai modelli (specialmente quelli basati su alberi) di sfruttare l'informazione portata dall'assenza stessa del dato.

Imputazione Condizionale Flessibile:
Permette di scegliere tra due strategie di imputazione tramite il parametro strategy:

"knn": Utilizza KNNImputer per stimare i buchi basandosi sulla vicinanza geometrica tra pazienti.

"median": Utilizza SimpleImputer con la mediana, calcolata in modo indipendente colonna per colonna.

Scaling Automatico Condizionale (StandardScaler):
Poiché il KNNImputer calcola distanze euclidee, richiede obbligatoriamente che le feature siano sulla stessa scala (evitando che variabili con range ampi dominino su quelle con range ristretti). Il modulo applica automaticamente lo StandardScaler solo se strategy="knn", azzerando lo scaling se viene scelta la mediana.

Filtraggio per Pool di Riferimento (pool):
Consente di decidere su quale sottoinsieme di righe (training set) fittare l'imputer e lo scaler:

"all": Considera l'intero training set.

"ordinary": Filtra e fitta l'imputer solo sui pazienti "sani/ordinari" identificati da una maschera booleana esterna (group_mask == True).

"rare": Filtra e fitta l'imputer solo sui pazienti "rari/patologici" (group_mask == False).

🔒 Prevenzione del Data Leakage
Il modulo separa nettamente la fase di apprendimento (fit_nan_handler) da quella di applicazione (transform_nan):

L'imputer, lo scaler e le mediane vengono calcolati esclusivamente sul training set (e nello specifico solo sul pool di riferimento scelto) e solo se viene richiesto.

La stessa regola viene poi applicata in modo identico e immodificato a validation set, test set o nuovi dati tramite transform_nan, garantendo che nessun'informazione del test set "fuoriesca" durante il preprocessing.