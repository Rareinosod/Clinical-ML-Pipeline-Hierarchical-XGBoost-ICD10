# 🏥 Clinical ML Pipeline & Hierarchical XGBoost Cascade for ICD-10 Emergency Prediction

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Optuna%20%7C%20Scikit--Learn-orange.svg)](https://xgboost.readthedocs.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Hierarchical%20Cascade%20%26%20Zero--Leakage-emerald.svg)](#-architettura-del-modello-cascata-gerarchica-xgboost)
[![Metric](https://img.shields.io/badge/Clinical%20Metric-Macro--F2%20(0.7825)-crimson.svg)](#-metriche-cliniche--risultati-sperimentali)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **Progetto di Machine Learning & Bioingegneria dei Dati Clinici**  
> **Autori (Gruppo 4):** Chiara Parenti, Elisa Galanti, Filippo Baldo, **Ronan Alessandro Reinoso Dominguez**  
> **Dataset:** 3.198 ricoveri ospedalieri de-identificati in Emergenza-Urgenza (Pronto Soccorso / ICU)

---

## 📌 Executive Summary

Questo progetto sviluppa un'infrastruttura algoritmica end-to-end avanzata per la **previsione precoce e la classificazione supervisionata del capitolo diagnostico ICD-10** in pazienti afferenti al dipartimento di emergenza-urgenza.

Integrando 4 sorgenti eterogenee di cartelle cliniche elettroniche (EHR: *Demographics, Triage, Blood Tests, ICD Diagnosis*), la pipeline implementa un protocollo matematico rigoroso anti-leakage, strategie di outlier detection non distruttive (*Shadow Features*), imputazione guidata da maschere cliniche e un'innovativa **architettura a Cascata Gerarchica (Hierarchical XGBoost)** capace di aggirare l'errore irriducibile causato dalla sovrapposizione sintomatologica tra patologie acute.

```
+---------------------------------------------------------------------------------------------------------+
|                                        EHR DATA INGESTION & PIPELINE                                    |
| [Triage Vitals] + [Blood Series] + [Demographics] -> Hard-Clipping -> Split (Holdout/K-Fold/Bootstrap) |
|         |                                                                                               |
|         v                                                                                               |
| [Outlier Detection (Train Only)] -> Univariate MAD (Z>3.5) + Spatial Triage DBSCAN (eps=1.5, min=10)   |
|         |                           => Preservazione del dato tramite Shadow Features                   |
|         v                                                                                               |
| [Masked Imputation] -> KNN Imputer addestrato SOLO su "Mask_Strict" (pazienti ordinari/omogenei)        |
|         |                                                                                               |
|         v                                                                                               |
| [Feature Selection] -> Spearman (|rho| > 0.90) + Kruskal-Wallis (p < 0.05 vs Target)                   |
|         |                                                                                               |
|         v                                                                                               |
| [MODELLO VINCITORE] -> HIERARCHICAL XGBOOST CASCADE (Macro-F2 = 0.7825, zero leakage, OOF Cut-off: 0.51)|
+---------------------------------------------------------------------------------------------------------+
```

---

## 🎯 Il Task di Classificazione Clinica

Il dataset raccoglie **3.198 ricoveri** con diagnosi suddivise in 3 macro-classi cliniche:

| Classe | Denominazione Clinica | Capitoli ICD-10 | Campioni | Quota | Criticità Clinico-Topologica |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **0** | **Disturbi Psichici** | Cap. V | 1.770 | 55.3% | Sparsità critica ematica (>52% NaN per mancata prescrizione di prelievi). Facilmente isolabile dai parametri vitali. |
| **1** | **Patologie Organiche Acute** | Cap. VI, X, XIII *(Neuro, Respiratorio, Muscolo-scheletrico)* | 651 | 20.4% | Sintomatologia infiammatoria sovrapposta (tachicardia, PCR/leucocitosi elevata). |
| **2** | **Sistema Digerente** | Cap. XI *(Stomaco, Fegato, Pancreas)* | 777 | 24.3% | Alta frequenza di biochimica ematica completa; forte sovrapposizione biologica con la Classe 1. |

> ⚠️ **Il Muro della Classe 1 (L'Errore Irriducibile):**  
> Una sepsi/infezione sistemica o una polmonite acuta (Classe 1) scatenano risposte fisiologiche identiche ad un'addome acuto o peritonite (Classe 2): leucocitosi, febbre, tachicardia e impennata della proteina C-reattiva. Questa vicinanza produce una forte indistinguibilità topologica nei dati di triage che nessun classificatore flat standard può risolvere senza overfitting.

---

## 🔬 Metodologia & Pipeline End-to-End

### 1. Hard-Clipping Fisiologico & Costruzione del Matricione
- **Filtro Pre-Statistico Deterministico**: Eliminazione preventiva di refusi di digitazione e valori incompatibili con la vita prima di qualsiasi calcolo aggregato:
  - Parametri Vitali: Temperatura $[80.0 - 115.0]^\circ\text{F}$, FC $[10 - 300]\text{ bpm}$, $\text{SpO}_2$ $[50 - 100]\%$, Pressione $[40-300]/[20-200]\text{ mmHg}$.
  - Esami Ematici: Emoglobina $[2.0 - 25.0]\text{ g/dL}$, Leucociti (WBC) $[0.5 - 60.0]\times 10^9/\text{L}$, Piastrine $[5.0 - 1200.0]\times 10^9/\text{L}$.
  - I valori non fisiologici vengono convertiti in `NaN` per evitare distorsioni nei momenti statistici.
- **Feature Engineering Temporale ed Ematochimico**:
  - Durata degenza (`los_hours`) calcolata da `admittime` e `dischtime`; indicatore di mortalità intraospedaliera (`mortality`).
  - Serie temporali di 10 analiti ematici aggregate in momenti: $\text{media}$, $\text{mediana}$, $\text{deviazione standard}$, $\text{min}$, $\text{max}$, $\text{primo}$, $\text{ultimo}$, $\text{conteggio}$, $\text{trend}$.
  - Biomarcatori euristici: `crp_elevated = 1` ($\text{PCR} > 10\text{ mg/L}$).

### 2. Prevenzione Ferrea del Data Leakage (Split Train/Test)
Tutte le statistiche (mediane, matrici di correlazione, soglie Z-score, cluster DBSCAN e regole di scaling) vengono **stimate esclusivamente sui fold di Training** e proiettate come trasformazione (`transform`) sul Test Set.
- **Hold-Out (80/20)**: Partizione statica stratificata per prototipazione rapida.
- **Stratified K-Fold (5-Fold)**: 5 partizioni indipendenti con preservazione delle percentuali di classe; stima della varianza e stabilità prestazionale.
- **Bootstrap Out-Of-Bag (OOB)**: Campionamento con reinserimento sul train e validazione sui casi non estratti per la massima verifica di generalizzazione.

### 3. Outlier Detection Non Distruttiva (Shadow Features)
In ambito medico un paziente con valori estremi non è un "errore da eliminare", ma un caso ad alta complessità da analizzare con attenzione:
- **MAD Univariato (Robust Z-Score)**:

$$
Z_{\text{robust}} = \frac{0.6745 \cdot (X - \text{Mediana})}{\text{MAD}}
$$

  Soglia train $Z > 3.5 \implies$ attivazione della flag binaria `has_univariate_outlier = 1` (nessun record eliminato).
- **DBSCAN Spaziale Multivariato**:
  Eseguito unicamente sulle 6 misurazioni simultanee di Triage ($\text{eps}=1.5, \text{min\_samples}=10$). Intercetta combinazioni sindromiche atipiche sincrone (es. tachicardia concomitante a ipotermia e ipertensione severa) $\implies$ flag `dbscan_low_density = 1`.

### 4. Imputazione con Maschere Cliniche (`Mask_Strict`)
L'assenza di esami ematici non è casuale (MCAR), bensì clinicamente diagnostica (un paziente psichiatrico non riceve prelievi biochimici intensivi).
- Viene creata una **Shadow Feature** (`_is_missing = 1`) per ogni variabile imputata.
- **KNN Imputer Guidato da Maschere**: Addestrato non su tutti i pazienti indiscriminatamente (`baseline_all`), ma sulla partizione `Mask_Strict` composta unicamente da pazienti ordinari (privi di outlier univariati e situati in cluster densi DBSCAN). L'imputazione evita così di propagare aberrazioni patologiche estreme nei valori mancanti.

### 5. Feature Selection a Due Stadi
1. **Filtro di Ridondanza (Spearman Rank Correlation)**: Rimozione non parametrica di feature multicollineari ($|\rho| > 0.90$), mantenendo quella con maggiore associazione al target.
2. **Filtro di Rilevanza (Kruskal-Wallis)**: Test d'ipotesi tra classi diagnostiche ($p\text{-value} > 0.05 \implies$ feature scartata).

---

## ⚡ Architettura del Modello: Cascata Gerarchica XGBoost

Per risolvere la barriera predittiva tra la Classe 1 e la Classe 2 senza ricorrere a soglie euristiche arbitrarie, è stata ideata un'architettura **Dicotomica a Due Fasi (Hierarchical Cascade)**:

```
                          [ Dati Paziente in Ingresso ]
                                       |
                                       v
                     +-----------------------------------+
                     |           MODELLO A               |
                     |      XGBoost Binario Primario     |
                     |      (Classe 0 vs [1 + 2])        |
                     +-----------------------------------+
                                    /     \
                                   /       \
                        P(0) >= 0.50       P(Resto) > 0.50
                                 /           \
                                v             v
                       +-------------+   +------------------------------------+
                       |  CLASSE 0   |   |            MODELLO B               |
                       |  Disturbi   |   |    XGBoost "Microscopio" Clinico   |
                       |  Psichici   |   |       (Classe 1 vs Classe 2)       |
                       +-------------+   |   (addestrato SOLO su classi 1 e 2)|
                                         +------------------------------------+
                                                        /     \
                                                       /       \
                                             P(2) >= Cut-off   P(2) < Cut-off
                                                     /           \
                                                    v             v
                                             +-------------+  +-------------+
                                             |  CLASSE 2   |  |  CLASSE 1   |
                                             |  Digerente  |  |  Org. Acute |
                                             +-------------+  +-------------+
```

### Dettagli Algoritmici della Cascata:
1. **Modello A (Filtro Primario)**:
   - Isola la macro-classe psichiatrica (`Classe 0`) dalle patologie organiche acute/digerenti (`Resto: 1+2`).
   - Addestramento profondo (`max_depth = 10`), **Macro-F2 interno = 0.9311**.
2. **Modello B (Microscopio Clinico Focalizzato)**:
   - Addestrato **esclusivamente** sul sottoinsieme di pazienti appartenenti a Classe 1 e 2.
   - Bilanciamento classi tramite `scale_pos_weight`.
   - Modello regolarizzato a bassa profondità (`max_depth = 3`) per scongiurare l'overfitting sul rumore biologico.
3. **Ottimizzazione Autonoma della Soglia (Out-Of-Fold Grid Search)**:
   - Scansione autonoma di 81 punti soglia (da 0.10 a 0.90 con passo 0.01) calcolata in Cross-Validation Out-Of-Fold per massimizzare il Macro-F2.
   - **Sorprendente Convergenza Matematica**: L'algoritmo ha individuato in piena autonomia il punto di lavoro ottimale a **Soglia = 0.51** (Score OOF: 0.7669), convalidando oggettivamente e per via puramente numerica l'euristica clinica dei medici del triage!

---

## 📊 Metriche Cliniche & Risultati Sperimentali

In medicina d'urgenza la metrica standard F1 (media armonica bilanciata tra precisione e richiamo) è insufficiente:

$$
\text{Macro-}F_\beta = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}
$$

Nel nostro framework è stato adottato **$\beta = 2$ (Macro-F2)**:
- Un **Falso Positivo** comporta accertamenti diagnostici addizionali (costo lieve).
- Un **Falso Negativo** comporta la dimissione prematura o il mancato ricovero di un paziente acuto critico (**rischio di vita**). Assegnare **peso doppio al Recall** riflette l'imperativo etico e clinico del Pronto Soccorso.

### Classifica Finale dei Modelli Benchmark

| Pos. | Modello / Architettura | Strategia Preprocessing | Macro-F2 Score | Macro-F1 Score | Note di Valutazione |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 🥇 | **Fallback Statico (Softmax Heuristic)** | Completo (MAD + DBSCAN + Mask Strict) | **0.7876** | 0.7701 | Regola esperta clinica hard-coded (P > 0.51). |
| 🥈 | **XGBoost Cascata Gerarchica** | Completo (MAD + DBSCAN + Mask Strict) | **0.7825** | 0.7634 | **Vincitore Metodologico**: 100% data-driven, nessun bias umano, generalizzazione ottimale. |
| 🥉 | **XGBoost Multiclasse Flat Standard** | Completo + Optuna Tuning | 0.7410 | 0.7480 | Inibito dal trade-off sulla Classe 1 (Macro-F su Classe 1 < 0.52). |
| 4 | **Random Forest (Bagging)** | Completo + Class Weight | 0.7290 | 0.7320 | Elevata stabilità inter-fold, ma penalizzato sulla separazione non lineare. |
| 5 | **SVM Kernel RBF / ECOC** | Completo + StandardScaler | 0.7180 | 0.7240 | Buona precisione su Classe 0, ma computazionalmente oneroso e rigido sui confini 1-2. |
| 6 | **AdaBoost** | Completo + Decision Stumps | 0.6940 | 0.7010 | Tendenza all'overfitting sugli outlier di triage. |
| 7 | **KNN (K-Nearest Neighbors)** | Imputazione Mediana + FS | 0.6720 | 0.6810 | Sensibile alla dimensionalità e alle zone a densità spuria. |
| 8 | **Decision Tree (Cart Singolo)** | Semplice | 0.6120 | 0.6200 | Alta varianza, white-box interpretabile ma prestazionalmente inadeguato. |

---

## 🔍 Causalità Clinica & Feature Importance Globale

Analizzando la graduatoria aggregata su oltre **150 run sperimentali**, la top 5 dei biomarcatori rivela una conclusione illuminante:

1. **`pain` (Livello di Dolore in Triage)** — *Rank medio 1.2*: Il parametro cardine per discriminare disturbi somatoformi/psichiatrici rispetto a patologie organiche acute.
2. **`acuity` (Indice di Codice Triage)** — *Rank medio 2.7*: Priorità clinica stimata all'ingresso.
3. **`Calcium_min` (Calcio Sierico Minimo)** — *Rank medio 4.2*: Forte indicatore prognostico in quadri addominali gravi e pancreatiti (ipocalcemia).
4. **`los_hours` (Durata della Degenza)** — *Rank medio 5.9*: Proxy dell'invasività delle cure richieste.
5. **`age` (Età Anagrafica)** — *Rank medio 7.1*: Fattore determinante di vulnerabilità organica.

> 💡 **Evidenza Nosocomiale:** I parametri immediati raccolti al Triage (dolore, gravità percepita, età) superano persino la complessità degli esami biochimici nella fase di triage differenziale rapido.

---

## 📂 Struttura del Repository

```
progetto.finale/
├── Analisi_Comparativa/             # Script e plot aggregati (F1, F2, varianza, feature ranking)
├── Analisi_Dataset/                 # Notebook esplorativi, PCA 3D, t-SNE e matrici di sparsità
├── csv_function/                    # ETL, fusione dei 4 database EHR e creazione del Matricione
├── Data/                            # Dati EHR grezzi de-identificati (Triage, Blood, Demographic, ICD)
├── feature_selection_function/      # Spearman rank correlation & Kruskal-Wallis hypothesis testing
├── gestione_launcher/               # Coordinatore orchestrale: main.py interattivo
│   └── main.py                      # Pipeline principale a 6 step (Holdout / K-Fold / Bootstrap)
├── modelli/                         # Famiglie di classificatori con Optuna Hyperparameter Tuning
│   ├── AdaBoost/                    # Training AdaBoost
│   ├── DecisionTree/                # Training Decision Tree
│   ├── KNN/                         # Training KNN
│   ├── RandomForest/                # Training Random Forest
│   ├── SVM_ECOC_training/           # SVM Error-Correcting Output Codes
│   ├── SVM_standard_training/       # SVM RBF standard
│   ├── XGBoost/                     # Modelli XGBoost
│   ├── esegui_cascata.py            # Motore dell'architettura Hierarchical Cascade
│   └── gestione_modelli.py          # Logica comune di scoring F2 e validazione
├── MODELLO DEFINITIVO/              # Deployment artefatto addestrato sul 100% dei dati (3.198 pazienti)
│   ├── Modello_Clinico_Definitivo_XGBoost_Fallback.pkl
│   └── Parametri_Modello_Clinico_Definitivo_XGBoost_Fallback.txt
├── nan_and_imputation_function/     # KNN Imputer condizionato su maschere cliniche (Mask_Strict)
├── outlier_function/                # MAD univariato e DBSCAN spaziale su parametri di Triage
├── plotting_function/               # Diagnostica grafica automatica
├── risultati_finali/                # Cartelle di Run storiche con report intermedi e split
├── validation_strategies/           # Funzioni di split anti-leakage (Holdout, Stratified K-Fold, Bootstrap)
├── Launcher_colab.ipynb             # Notebook interattivo per replicazione su Google Colab
├── Progetto di Machine Learning_REPORT.docx # Relazione scientifica integrale di progetto
└── README.md                        # Questa documentazione tecnica
```

---

## 🚀 Guida all'Installazione & Utilizzo

### 1. Clonazione del Repository e Setup Ambiente
```bash
git clone https://github.com/Rareinosod/Clinical-ML-Pipeline-Hierarchical-XGBoost-ICD10.git
cd Clinical-ML-Pipeline-Hierarchical-XGBoost-ICD10

# Creazione dell'ambiente virtuale
python -m venv venv
# Attivazione su Windows:
venv\Scripts\activate
# Attivazione su Linux/macOS:
source venv/bin/activate

# Installazione delle dipendenze
pip install numpy pandas scikit-learn xgboost optuna matplotlib seaborn scipy openpyxl
```

### 2. Esecuzione della Pipeline Principale (`main.py`)
Il file `gestione_launcher/main.py` offre un'interfaccia interattiva da terminale per configurare l'intero flusso di preprocessing:
```bash
python gestione_launcher/main.py
```
Il wizard guiderà nella scelta di:
- Strategia di validazione (`Hold-Out 80/20`, `5-Fold Stratificata`, `Bootstrap OOB`, o `100% Dati per Produzione`).
- Parametri di Outlier Detection MAD e raggio $\varepsilon$ di DBSCAN.
- Maschera di Imputazione NaN (`baseline_all`, `mask_tipici`, `mask_tipici_strict`).
- Soglia per la multicollinearità di Spearman.

### 3. Addestramento della Cascata Gerarchica
```bash
python modelli/esegui_cascata.py
```
Questo script esegue l'ottimizzazione Bayesiana Optuna a due fasi, calcola la matrice Out-Of-Fold per individuare il cut-off di equilibrio e salva le metriche diagnostiche dettagliate.

---

## 🔮 Sviluppi Futuri
1. **Elaborazione del Linguaggio Naturale (NLP)**: Integrazione di modelli linguistici biomedicali (*BioBERT / ClinicalBERT*) per analizzare il testo libero del diario infermieristico e del referto di triage.
2. **Computer Vision (Imaging Clinico)**: Integrazione di reti convoluzionali (CNN) su radiografie del torace (RX) ed ecografie FAST d'urgenza.
3. **Late Fusion Multimodale**: Fusione gerarchica tra vettori tabulari EHR, rappresentazioni dense testuali e feature visive per azzerare l'errore irriducibile.

---

## 📄 Licenza e Ringraziamenti
Progetto sviluppato ad uso accademico e di ricerca da **Chiara Parenti, Elisa Galanti, Filippo Baldo e Ronan Alessandro Reinoso Dominguez**.  
Tutti i dati clinici originali sono stati trattati secondo gli standard di de-identificazione e privacy sanitaria HIPAA/GDPR.
