"""
nan_handling.py
================

Modulo per la gestione dei valori mancanti (NaN) — Blocco 3 della pipeline.

Responsabilità di QUESTO modulo:
- creare shadow feature "<col>_is_missing" per le colonne trattate in O(1) allocazioni.
- imputare (in modo OPZIONALE, disattivabile) i NaN, con SimpleImputer
  (mediana) o KNNImputer.
- permettere di scegliere il pool di righe su cui FITTARE l'imputer:
  "all", "ordinary" o "rare", sulla base di una maschera booleana
  fornita dall'esterno.

- decide cosa sia "tipico" o "atipico": il criterio (acuity, flag
  outlier, o altro) è deciso fuori da qui e passato come group_mask.
  Questo per non introdurre nel modulo NaN una definizione clinica
  arbitraria, e per evitare ogni rischio di usare informazioni non
  disponibili a predict-time (es. il target).

- fa scaling (solo se si usa il KNN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler


PoolType = Literal["all", "ordinary", "rare"]
StrategyType = Literal["knn", "median"]


# =============================================================================
# CONTENITORE DI RISULTATO (memoria del fit, da riusare in transform)
# =============================================================================

@dataclass
class FittedNanHandler:
    """
    Tutto ciò che fit_nan_handler ha imparato dal training set.

    Attributes
    ----------
    columns: Colonne su cui è stata applicata la gestione NaN.
    impute: Se False, non è stato fittato alcun imputer: solo flag di missing.
    pool: Pool di righe usato per fittare l'imputer ("all"/"ordinary"/"rare").
    strategy: Strategia di imputazione ("knn" o "median").
    imputer: Oggetto sklearn fittato (None se impute=False).
    scaler: Oggetto StandardScaler fittato (usato solo se strategy="knn").
    add_missing_flag: Se True, sono state create le colonne "<col>_is_missing".
    n_rows_used_for_fit: Righe effettivamente usate per il fit dell'imputer.
    """

    columns: list[str]
    impute: bool
    pool: PoolType
    strategy: StrategyType
    imputer: Optional[object]
    scaler: Optional[object]
    add_missing_flag: bool
    n_rows_used_for_fit: int


# =============================================================================
# FUNZIONI INTERNE
# =============================================================================

def _select_columns(df: pd.DataFrame, columns: Optional[list[str]]) -> list[str]:
    """Se columns è None, usa tutte le colonne numeriche del dataframe."""
    if columns is not None:
        return [c for c in columns if c in df.columns]
    return list(df.select_dtypes(include=[np.number]).columns)


def _add_missing_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Crea <col>_is_missing = 1/0 con una singola allocazione di memoria (Zero Fragmentation).
    Risolve il PerformanceWarning derivante dall'inserimento iterativo in ciclo.
    """
    # 1. Calcoliamo tutte le nuove feature vettorialmente in memoria
    new_cols = {f"{col}_is_missing": df[col].isna().astype("int8") for col in columns}
    
    # 2. Le trasformiamo in un blocco unico
    new_cols_df = pd.DataFrame(new_cols, index=df.index)
    
    # 3. Riapplichiamo la concatenazione in O(1) allocazioni
    return pd.concat([df, new_cols_df], axis=1)


def _select_pool(
    df: pd.DataFrame,
    pool: PoolType,
    group_mask: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Seleziona il sottoinsieme di righe su cui fittare l'imputer[cite: 7].

    group_mask: Series booleana allineata a df.index[cite: 7]. True = "tipico" (ordinary).
    Obbligatoria se pool != "all"[cite: 7].
    """
    if pool == "all":
        return df

    if group_mask is None:
        raise ValueError(
            f"pool='{pool}' richiede group_mask (Series booleana), ma è None[cite: 7]."
        )

    mask_aligned = group_mask.reindex(df.index)
    if mask_aligned.isna().any():
        raise ValueError(
            "group_mask non copre tutte le righe di df (indici mancanti o disallineati)[cite: 7]."
        )

    if pool == "ordinary":
        subset = df.loc[mask_aligned == True]  
    elif pool == "rare":
        subset = df.loc[mask_aligned == False] 
    else:
        raise ValueError(f"pool non riconosciuto: {pool}[cite: 7]")

    if subset.empty:
        raise ValueError(
            f"Il pool '{pool}' selezionato con group_mask è vuoto: "
            "impossibile fittare l'imputer su zero righe[cite: 7]."
        )

    return subset


# =============================================================================
# FUNZIONI PUBBLICHE
# =============================================================================

def fit_nan_handler(
    X_train: pd.DataFrame,
    columns: Optional[list[str]] = None,
    impute: bool = True,
    pool: PoolType = "all",
    group_mask: Optional[pd.Series] = None,
    strategy: StrategyType = "knn",
    knn_neighbors: int = 5,
    add_missing_flag: bool = True,
) -> FittedNanHandler:
    """
    Impara la gestione dei NaN SOLO dal training set[cite: 7].
    """
    cols = _select_columns(X_train, columns)

    if not impute:
        return FittedNanHandler(
            columns=cols,
            impute=False,
            pool=pool,
            strategy=strategy,
            imputer=None,
            scaler=None,
            add_missing_flag=add_missing_flag,
            n_rows_used_for_fit=0,
        )

    # 1. Selezioniamo il pool di righe per il fit ("all", "ordinary" o "rare")
    fit_pool_df = _select_pool(X_train[cols], pool=pool, group_mask=group_mask)

    scaler = None
    data_for_imputer = fit_pool_df

    # 2. Fit Imputer (con scaling preventivo in caso di KNN)
    if strategy == "knn":
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(fit_pool_df)
        data_for_imputer = pd.DataFrame(scaled_array, columns=cols, index=fit_pool_df.index)
        imputer = KNNImputer(n_neighbors=knn_neighbors)
    elif strategy == "median":
        imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"strategy non riconosciuta: {strategy}")

    imputer.fit(data_for_imputer)

    return FittedNanHandler(
        columns=cols,
        impute=True,
        pool=pool,
        strategy=strategy,
        imputer=imputer,
        scaler=scaler,
        add_missing_flag=add_missing_flag,
        n_rows_used_for_fit=len(fit_pool_df),
    )


def transform_nan(X: pd.DataFrame, fitted: FittedNanHandler, output_report_path: str = "report_imputazione_nan.csv") -> pd.DataFrame:
    """
    Applica a un nuovo dataframe (train, val, test) la stessa
    logica appresa da fit_nan_handler[cite: 7].
    """
    X = X.copy()
    cols = fitted.columns

    # Analisi dei NaN prima dell'imputazione
    nan_counts_before = X[cols].isna().sum()
    total_nan_before = nan_counts_before.sum()
    
    print(f"\n[REPORT NaN] Totale valori mancanti (NaN) trovati: {total_nan_before}")
    print("Dettaglio NaN per colonna (prime colonne con NaN):")
    print(nan_counts_before[nan_counts_before > 0].head(10))

    if fitted.add_missing_flag:
        X = _add_missing_flags(X, cols)

    if not fitted.impute:
        return X

    data_to_impute = X[cols]

    # Applicazione scaling se necessario
    if fitted.scaler is not None:
        scaled_array = fitted.scaler.transform(data_to_impute)
        data_to_impute = pd.DataFrame(scaled_array, columns=cols, index=X.index)

    # Imputazione
    imputed_array = fitted.imputer.transform(data_to_impute)

    # Compilazione report
    report_rows = []
    for col in cols:
        n_nan = int(nan_counts_before[col])
        if n_nan > 0:
            report_rows.append({
                "feature": col,
                "n_nan_iniziali": n_nan,
                "strategia_usata": fitted.strategy,
                "pool_fit": fitted.pool,
                "righe_coinvolte_imputazione": n_nan
            })

    report_df = pd.DataFrame(report_rows)
    if not report_df.empty:
        report_df.to_csv(output_report_path, index=False)
        print(f"  Report dettagliato dei NaN salvato in: {output_report_path}")

    # (Per KNN) Ripristino scalatura per mantenere i dati comparabili a quelli originali
    if fitted.scaler is not None:
        imputed_array = fitted.scaler.inverse_transform(imputed_array)

    X[cols] = imputed_array
    
    return X