📋 Panoramica dello Script
Il file main.py è il coordinatore centrale dell'intera pipeline. Si occupa di orchestrare sequenzialmente tutti i blocchi del progetto: dal caricamento dei dati grezzi aggregati, alla separazione delle responsabilità, fino alla preparazione dei dataset finali pronti per l'addestramento dei modelli predittivi.

🔄 Flusso Operativo dei 6 Step
Step 1: Caricamento e Separazione Target
Richiama la funzione di creazione del matricione e separa immediatamente le feature indipendenti (X_full) dalla variabile obiettivo (y_full), impedendo a qualsiasi algoritmo successivo di vedere o sfruttare il target in fase di preprocessing.

Step 2: Split Stratificato Anti-Leakage
Esegue una divisione iniziale (Hold-out o configurazione K-Fold) isolando un Test Set bloccato al 20% tramite stratificazione sul target. Questo garantisce che il test set rimanga incontaminato fino alla valutazione finale delle performance.

Step 3: Outlier Detection (Eseguita SOLO sul Train Set)

Parte B (Univariata): Applica il metodo MAD (Median Absolute Deviation) sul training set per identificare e marcare gli outlier univariati.

Parte C (Multivariata): Esegue il clustering DBSCAN sul training set per individuare i casi a bassa densità (rumore/anomalie multivariate).

Nota metodologica: Le soglie e i parametri statistici di outlier detection imparano esclusivamente dalla distribuzione del training set.

Step 4: Configurazione delle Maschere Sperimentali
Costruisce un dizionario di esperimenti (mask_experiments) combinando le flag statistiche e cliniche per definire differenti definizioni di "sano/ordinario" vs "patologico/raro":

baseline_all: Nessun filtro (usa tutti i dati).

multivariata_dbscan: Filtra basandosi sui cluster di DBSCAN.

univariata_mad: Filtra basandosi sugli outlier univariati MAD.

clinica_crp: Filtra basandosi sui livelli di Proteina C-Reattiva.

mask_super_sani: Filtro restrittivo combinato (privo sia di outlier univariati che multivariati).

Step 5: Gestione NaN, Scaling e Imputazione
Chiama il modulo nan_handling passando la strategia di imputazione scelta (es. strategy="knn" con pooling sui pazienti ordinari), applicando il fit sul train e il transform su train e test.

Step 6: Salvataggio dei Dataset Finali
Esporta i dataframe puliti, imputati, scalati e dotati di shadow features (X_train_ready.csv, X_test_ready.csv, y_train.csv, y_test.csv), insieme ai report analitici intermedi per la discussione nel report di progetto.

🎯 Vantaggi Architetturali
Modularità: Ogni blocco logico risiede in un modulo dedicato (outlier_functions, nan_handling, validation_strategies), mantenendo il codice pulito e manutenibile.

Rigore Metodologico: Controllo ferreo contro il data leakage, rispettando appieno le linee guida di validazione per modelli clinici predittivi.