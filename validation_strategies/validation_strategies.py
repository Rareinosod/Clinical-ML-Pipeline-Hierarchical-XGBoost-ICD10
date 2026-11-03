"""
validation_strategies.py
========================
Modulo per la gestione delle strategie di split e validazione (Hold-out, K-Fold, Nested CV).
Garantisce la stratificazione per preservare la proporzione delle classi.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample

@dataclass
class HoldOutResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_holdout(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> HoldOutResult:
    """
    Esegue una divisione classica Hold-out Stratificata (Outer Loop bloccato).
    Garantisce il bilanciamento delle classi tramite stratify=y.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    return HoldOutResult(
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test
    )


def get_stratified_kfold_splits(
    X: pd.DataFrame, 
    y: pd.Series, 
    n_splits: int = 5, 
    random_state: int = 42
) -> StratifiedKFold:
    """
    Restituisce un oggetto StratifiedKFold dinamicamente configurabile.
    Viene utilizzato sia per l'Outer Loop (nel main.py per isolare le cartelle dei fold)
    sia per l'Inner Loop (nei modelli per l'ottimizzazione degli iperparametri senza leakage).
    """
    return StratifiedKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=random_state
    )




def split_bootstrap(
    X: pd.DataFrame, 
    y: pd.Series, 
    random_state: int = 42
) -> HoldOutResult:
    """
    Esegue uno split basato su Bootstrap Stratificato.
    Crea un set di addestramento campionando con rimpiazzo (stessa dimensione dell'originale).
    I campioni non estratti (Out-of-Bag, OOB) costituiscono il set di test.
    Garantisce il bilanciamento delle classi nel set di addestramento tramite stratify=y.
    """
    
    # Campionamento con rimpiazzo per creare il training set
    # Di default, resample estrae un numero di campioni pari a len(X)
    X_train, y_train = resample(
        X, y, 
        replace=True, 
        stratify=y, #Mantenimento proporzione su etichette
        random_state=random_state
    )
    
    # Identificazione dei campioni Out-of-Bag (OOB)
    # Poiché resample preserva gli indici di Pandas, usiamo .difference() 
    # per trovare gli indici originali che NON sono stati estratti nel training set
    oob_indices = X.index.difference(X_train.index)
    
    # Creiamo il test set usando gli indici OOB
    X_test = X.loc[oob_indices]
    y_test = y.loc[oob_indices]
    
    return HoldOutResult(
        X_train=X_train, 
        X_test=X_test, 
        y_train=y_train, 
        y_test=y_test
    )


def split_full_dataset(
    X: pd.DataFrame, 
    y: pd.Series
) -> HoldOutResult:
    """
    Restituisce l'intero dataset. 
    Il test set è una copia del training set per mantenere l'architettura 
    della pipeline intatta ed evitare crash nei report delle metriche.
    Utile ESCLUSIVAMENTE per addestrare il modello finale in produzione.
    """
    return HoldOutResult(
        X_train=X.copy(), 
        X_test=X.copy(), 
        y_train=y.copy(), 
        y_test=y.copy()
    )