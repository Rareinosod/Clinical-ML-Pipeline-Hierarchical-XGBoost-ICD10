"""
outlier_pipeline.py
===================

Pipeline di controllo qualità e outlier detection per dati clinici.

AGGIORNAMENTO IMPORTANTE
-------------------------
Il codice reale di Filippo (process_Bloodtest) NON lascia le celle del
BloodTest come stringhe "12;13". Le trasforma già in LISTE PYTHON di float,
es. [12.0, 13.0], oppure lista vuota [] se la cella era mancante.

Di conseguenza la pulizia della Parte A per il BloodTest non deve più
splittare stringhe: deve filtrare liste già pronte, rimuovendo i valori
fuori dai limiti fisiologici e lasciando una lista più corta (o vuota).
Questo è compatibile con calculate_stats() di Filippo, che tratta una
lista vuota come "nessun dato disponibile".

STRUTTURA DI UTILIZZO
---------------------
PARTE A (prima delle statistiche aggregate di Filippo)
    clean_raw_measurements()              -> Triage (celle a valore singolo)
    clean_raw_bloodtest_lists()            -> BloodTest (celle come liste)

PARTE B (dopo il matricione)
    apply_univariate_outlier_detection()

PARTE C (dopo il matricione)
    apply_dbscan_outlier_detection()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    
    from sklearn.preprocessing import RobustScaler
except ImportError as error:
    raise ImportError(
        "La Parte C richiede scikit-learn. Installa con: pip install scikit-learn"
    ) from error

from .physiological_ranges import (
    DEFAULT_PHYSIOLOGICAL_RANGES,
    FEATURE_SETS,
)


# =============================================================================
# PARAMETRI GENERALI
# =============================================================================

MAD_THRESHOLD: float = 3.5
MIN_OBSERVATIONS: int = 10
EPS_DEFAULT: float = 1.5
MIN_SAMPLES_DEFAULT: int = 10
DBSCAN_METRIC: str = "euclidean"


# =============================================================================
# CLASSI DI RISULTATO
# =============================================================================

@dataclass
class CleaningResult:
    """Risultato della Parte A per celle a valore singolo (es. Triage)."""

    dataframe: pd.DataFrame
    report: pd.DataFrame
    available_columns: list[str]
    missing_expected_columns: list[str]


@dataclass
class ListCleaningResult:
    """Risultato della Parte A per celle contenenti liste (es. BloodTest)."""

    dataframe: pd.DataFrame
    report: pd.DataFrame
    available_columns: list[str]
    missing_expected_columns: list[str]


@dataclass
class UnivariateOutlierResult:
    """Risultato della Parte B."""

    dataframe: pd.DataFrame
    report: pd.DataFrame
    analyzed_columns: list[str]
    skipped_columns: list[str]


@dataclass
class DBSCANResult:
    """Risultato della Parte C."""

    dataframe: pd.DataFrame
    report: dict[str, Any]
    features_used: list[str]
    n_evaluable: int
    n_not_evaluable: int


# =============================================================================
# PARTE A  - CELLE A VALORE SINGOLO (ES. TRIAGE)
# =============================================================================

def validate_input_dataframe(
    df: pd.DataFrame,
    expected_columns: Iterable[str] | None = None,
    id_column: str = "hadm_id",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Controlla che il dataframe sia valido per il preprocessing."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df deve essere un pandas.DataFrame.")

    if df.empty:
        raise ValueError("Il dataframe è vuoto.")

    validated_df = df.copy(deep=True)
    expected_columns = list(expected_columns or [])

    available_columns = [
        column for column in expected_columns
        if column in validated_df.columns
    ]
    missing_expected_columns = [
        column for column in expected_columns
        if column not in validated_df.columns
    ]

    if id_column in validated_df.columns and validated_df[id_column].isna().all():
        raise ValueError(
            f"La colonna identificativa '{id_column}' contiene solo NaN."
        )

    return validated_df, available_columns, missing_expected_columns


def clean_impossible_values(
    df: pd.DataFrame,
    physiological_ranges: dict[str, tuple[float, float]],
    add_column_flags: bool = True,
    invalid_suffix: str = "_invalid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converte in NaN i valori fuori dai limiti, per colonne a valore singolo.

    Usare SOLO su colonne dove ogni cella è un singolo numero (es. Triage).
    Per colonne con celle-lista (es. BloodTest dopo process_Bloodtest) usare
    invece clean_bloodtest_lists().
    """
    cleaned_df = df.copy(deep=True)
    report_rows: list[dict[str, Any]] = []

    for column, limits in physiological_ranges.items():
        lower_bound, upper_bound = limits

        if column not in cleaned_df.columns:
            report_rows.append({
                "feature": column,
                "feature_present": False,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_values_before": 0,
                "n_missing_before": np.nan,
                "n_non_numeric_converted_to_nan": np.nan,
                "n_invalid_values": 0,
                "n_missing_after": np.nan,
                "invalid_rate_pct": np.nan,
            })
            continue

        original_values = cleaned_df[column]
        numeric_values = pd.to_numeric(original_values, errors="coerce")

        n_values_before = int(numeric_values.notna().sum())
        n_missing_before = int(numeric_values.isna().sum())
        n_non_numeric = int(
            original_values.notna().sum() - numeric_values.notna().sum()
        )

        invalid_mask = (
            numeric_values.notna()
            & ((numeric_values < lower_bound) | (numeric_values > upper_bound))
        )
        n_invalid_values = int(invalid_mask.sum())

        if add_column_flags:
            cleaned_df[f"{column}{invalid_suffix}"] = invalid_mask.astype("int8")

        numeric_values.loc[invalid_mask] = np.nan
        cleaned_df[column] = numeric_values

        n_missing_after = int(cleaned_df[column].isna().sum())
        invalid_rate_pct = (
            n_invalid_values / n_values_before * 100
            if n_values_before > 0
            else np.nan
        )

        report_rows.append({
            "feature": column,
            "feature_present": True,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_values_before": n_values_before,
            "n_missing_before": n_missing_before,
            "n_non_numeric_converted_to_nan": n_non_numeric,
            "n_invalid_values": n_invalid_values,
            "n_missing_after": n_missing_after,
            "invalid_rate_pct": invalid_rate_pct,
        })

    return cleaned_df, pd.DataFrame(report_rows)


def create_quality_flags(
    df: pd.DataFrame,
    physiological_ranges: dict[str, tuple[float, float]],
    invalid_suffix: str = "_invalid",
) -> pd.DataFrame:
    """Crea flag riassuntive di qualità a livello di riga."""
    result_df = df.copy(deep=True)

    invalid_columns = [
        f"{column}{invalid_suffix}"
        for column in physiological_ranges
        if f"{column}{invalid_suffix}" in result_df.columns
    ]

    if not invalid_columns:
        result_df["n_invalid_values"] = 0
        result_df["has_invalid_value"] = 0
        return result_df

    result_df["n_invalid_values"] = (
        result_df[invalid_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        .astype("int16")
    )
    result_df["has_invalid_value"] = (
        result_df["n_invalid_values"] > 0
    ).astype("int8")

    return result_df


def clean_raw_measurements(
    df: pd.DataFrame,
    physiological_ranges: dict[str, tuple[float, float]] | None = None,
    id_column: str = "hadm_id",
) -> CleaningResult:
    """
    Pulisce dati con UN valore singolo per cella (tipicamente il Triage).

    Da richiamare prima di qualsiasi aggregazione sui dati di triage.
    """
    if physiological_ranges is None:
        physiological_ranges = DEFAULT_PHYSIOLOGICAL_RANGES

    validated_df, available_columns, missing_expected_columns = (
        validate_input_dataframe(
            df=df,
            expected_columns=physiological_ranges.keys(),
            id_column=id_column,
        )
    )

    cleaned_df, cleaning_report = clean_impossible_values(
        df=validated_df,
        physiological_ranges=physiological_ranges,
        add_column_flags=True,
    )

    cleaned_df = create_quality_flags(
        df=cleaned_df,
        physiological_ranges=physiological_ranges,
    )

    return CleaningResult(
        dataframe=cleaned_df,
        report=cleaning_report,
        available_columns=available_columns,
        missing_expected_columns=missing_expected_columns,
    )


# =============================================================================
# PARTE A (VARIANTE 2) - CELLE CONTENENTI LISTE (ES. BLOODTEST DI FILIPPO)
# =============================================================================
#
# Dopo process_Bloodtest() di Filippo, ogni cella del BloodTest è una lista
# Python di float, es. [12.1, 9999.0, 11.9], oppure [] se non c'erano dati.
#
# Questa funzione filtra ogni lista, tenendo solo i valori dentro i limiti
# fisiologici configurati. Una lista con valori scartati diventa più corta.
# Se tutti i valori vengono scartati, la lista diventa vuota [], che è
# esattamente il formato che calculate_stats() di Filippo tratta già come
# "nessun dato disponibile" (restituisce tutte le statistiche NaN).
# =============================================================================

def _clean_value_list(
    value_list: Any,
    lower_bound: float,
    upper_bound: float,
) -> tuple[list[float], int, int]:
    """
    Filtra una singola cella-lista contro i limiti fisiologici.

    Parameters
    ----------
    value_list:
        Il contenuto della cella. Deve essere una lista di numeri, ma la
        funzione gestisce anche NaN o tipi inattesi restituendoli invariati.
    lower_bound, upper_bound:
        Limiti di plausibilità.

    Returns
    -------
    cleaned_list:
        Lista con solo i valori validi. Lista vuota se nessun valore resta.
    n_total:
        Numero di valori originariamente presenti nella lista.
    n_invalid:
        Numero di valori scartati perché fuori range o non numerici.
    """
    if not isinstance(value_list, list):
        # Cella non conforme (es. NaN residuo): la lasciamo invariata,
        # non è compito di questa funzione gestire formati diversi da lista.
        return value_list, 0, 0

    n_total = len(value_list)
    valid_values: list[float] = []
    n_invalid = 0

    for value in value_list:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            n_invalid += 1
            continue

        if np.isnan(numeric_value):
            n_invalid += 1
            continue

        if numeric_value < lower_bound or numeric_value > upper_bound:
            n_invalid += 1
            continue

        valid_values.append(numeric_value)

    return valid_values, n_total, n_invalid


def clean_bloodtest_lists(
    df: pd.DataFrame,
    physiological_ranges: dict[str, tuple[float, float]],
    add_column_flags: bool = True,
    invalid_suffix: str = "_invalid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pulisce colonne le cui celle sono LISTE di valori (BloodTest di Filippo,
    dopo process_Bloodtest).

    Per ogni cella:
    1. scorre i valori della lista;
    2. scarta i valori non numerici o fuori dai limiti configurati;
    3. restituisce la lista filtrata (eventualmente vuota).

    Questa funzione va chiamata DOPO process_Bloodtest() (che crea le liste)
    e PRIMA di analyze_blood_statistics() e analyze_CRP() di Filippo (che
    calcolano le statistiche a partire dalle liste).

    Parameters
    ----------
    df:
        Dataframe con colonne a celle-lista (es. df_features dopo il merge
        del BloodTest già convertito in liste).
    physiological_ranges:
        Dizionario {colonna: (minimo, massimo)}.
    add_column_flags:
        Se True, crea <colonna>_invalid = 1 se almeno un valore della
        lista è stato scartato in quella riga.
    invalid_suffix:
        Suffisso delle colonne flag.

    Returns
    -------
    cleaned_df:
        Dataframe con le liste ripulite dai valori non plausibili.
    cleaning_report:
        Report con conteggi a livello di singolo valore (non di cella).
    """
    cleaned_df = df.copy(deep=True)
    report_rows: list[dict[str, Any]] = []

    for column, limits in physiological_ranges.items():
        lower_bound, upper_bound = limits

        if column not in cleaned_df.columns:
            report_rows.append({
                "feature": column,
                "feature_present": False,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "n_cells_before": 0,
                "n_values_total": 0,
                "n_values_invalid": 0,
                "n_cells_emptied": 0,
                "invalid_rate_pct": np.nan,
            })
            continue

        original_column = cleaned_df[column]

        n_cells_before = int(
            original_column.apply(
                lambda cell: isinstance(cell, list) and len(cell) > 0
            ).sum()
        )

        results = original_column.apply(
            lambda cell: _clean_value_list(cell, lower_bound, upper_bound)
        )

        cleaned_lists = results.apply(lambda triple: triple[0])
        n_values_per_cell = results.apply(lambda triple: triple[1])
        n_invalid_per_cell = results.apply(lambda triple: triple[2])

        n_values_total = int(n_values_per_cell.sum())
        n_values_invalid = int(n_invalid_per_cell.sum())

        n_cells_emptied = int(
            (
                (n_values_per_cell > 0)
                & cleaned_lists.apply(lambda cell: isinstance(cell, list) and len(cell) == 0)
            ).sum()
        )

        if add_column_flags:
            cleaned_df[f"{column}{invalid_suffix}"] = (
                (n_invalid_per_cell > 0).astype("int8")
            )

        cleaned_df[column] = cleaned_lists

        invalid_rate_pct = (
            n_values_invalid / n_values_total * 100
            if n_values_total > 0
            else np.nan
        )

        report_rows.append({
            "feature": column,
            "feature_present": True,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_cells_before": n_cells_before,
            "n_values_total": n_values_total,
            "n_values_invalid": n_values_invalid,
            "n_cells_emptied": n_cells_emptied,
            "invalid_rate_pct": invalid_rate_pct,
        })

    return cleaned_df, pd.DataFrame(report_rows)


def clean_raw_bloodtest_lists(
    df: pd.DataFrame,
    physiological_ranges: dict[str, tuple[float, float]] | None = None,
    id_column: str = "hadm_id",
) -> ListCleaningResult:
    """
    Pulisce il BloodTest già convertito in liste da process_Bloodtest().

    Da richiamare DOPO process_Bloodtest() e il merge in df_features, ma
    PRIMA di analyze_CRP() e analyze_blood_statistics().

    Parameters
    ----------
    df:
        Dataframe (tipicamente df_features già mergiato) con colonne
        ematiche a celle-lista.
    physiological_ranges:
        Limiti da usare. Se None, usa DEFAULT_PHYSIOLOGICAL_RANGES.
    id_column:
        Nome della colonna identificativa.

    Returns
    -------
    ListCleaningResult:
        Dataframe pulito, report, colonne presenti/mancanti.
    """
    if physiological_ranges is None:
        physiological_ranges = DEFAULT_PHYSIOLOGICAL_RANGES

    validated_df, available_columns, missing_expected_columns = (
        validate_input_dataframe(
            df=df,
            expected_columns=physiological_ranges.keys(),
            id_column=id_column,
        )
    )

    cleaned_df, cleaning_report = clean_bloodtest_lists(
        df=validated_df,
        physiological_ranges=physiological_ranges,
        add_column_flags=True,
    )

    return ListCleaningResult(
        dataframe=cleaned_df,
        report=cleaning_report,
        available_columns=available_columns,
        missing_expected_columns=missing_expected_columns,
    )


# =============================================================================
# PARTE B - OUTLIER UNIVARIATI CON MAD ROBUSTO
# =============================================================================

def compute_median(series: pd.Series) -> float:
    """Calcola la mediana ignorando i NaN."""
    if series.empty or series.isna().all():
        return np.nan
    return float(series.median())


def compute_mad(
    series: pd.Series,
    median_value: float | None = None,
) -> float:
    """Calcola il MAD: Median Absolute Deviation."""
    if series.empty or series.isna().all():
        return np.nan

    if median_value is None:
        median_value = compute_median(series)

    return float((series - median_value).abs().median())


def compute_robust_zscore(
    series: pd.Series,
    median_value: float,
    mad_value: float,
) -> pd.Series:
    """Calcola il robust z-score: 0.6745 * (x - mediana) / MAD."""
    if mad_value == 0 or np.isnan(mad_value):
        return pd.Series(np.nan, index=series.index)

    return 0.6745 * (series - median_value) / mad_value





def detect_univariate_outliers(
    df: pd.DataFrame,
    columns_to_analyze: list[str] | None = None,
    mad_threshold: float = MAD_THRESHOLD,
    min_observations: int = MIN_OBSERVATIONS,
    outlier_suffix: str = "_univariate_outlier",
) -> UnivariateOutlierResult:
    """Identifica outlier per singola feature con robust z-score senza frammentare la memoria e con tracciatori."""
    
    # TRACCIATORE: Stampa iniziale dei parametri e delle dimensioni
    print(f"\n[TRACCIATORE MAD] Avvio Rilevamento Outlier Univariati")
    print(f"  -> Parametri impostati: Soglia Z-Score = {mad_threshold}, Osservazioni Minime = {min_observations}")
    print(f"  -> Dimensioni DataFrame in ingresso: {df.shape}")

    result_df = df.copy(deep=True)
    analyzed_columns: list[str] = []
    skipped_columns: list[str] = []
    report_rows: list[dict[str, Any]] = []
    
    # DIZIONARIO TEMPORANEO: Raccoglie i dati per evitare la frammentazione
    new_columns_data = {}

    if columns_to_analyze is None:
        columns_to_analyze = list(
            result_df.select_dtypes(include=[np.number]).columns
        )

    for column in columns_to_analyze:
        if column not in result_df.columns:
            skipped_columns.append(column)
            report_rows.append({
                "feature": column,
                "feature_present": False,
                "reason_skipped": "Colonna non presente",
                "n_observations": 0,
                "n_missing": np.nan,
                "median_value": np.nan,
                "mad_value": np.nan,
                "n_outliers": 0,
                "outlier_rate_pct": np.nan,
            })
            continue

        series = pd.to_numeric(result_df[column], errors="coerce")
        n_observations = int(series.notna().sum())
        n_missing = int(series.isna().sum())

        if n_observations < min_observations:
            skipped_columns.append(column)
            report_rows.append({
                "feature": column,
                "feature_present": True,
                "reason_skipped": f"Meno di {min_observations} osservazioni",
                "n_observations": n_observations,
                "n_missing": n_missing,
                "median_value": np.nan,
                "mad_value": np.nan,
                "n_outliers": 0,
                "outlier_rate_pct": np.nan,
            })
            continue

        median_value = compute_median(series)
        mad_value = compute_mad(series, median_value)

        if np.isnan(mad_value) or mad_value == 0:
            skipped_columns.append(column)
            report_rows.append({
                "feature": column,
                "feature_present": True,
                "reason_skipped": "MAD non calcolabile o uguale a zero",
                "n_observations": n_observations,
                "n_missing": n_missing,
                "median_value": median_value,
                "mad_value": mad_value,
                "n_outliers": 0,
                "outlier_rate_pct": np.nan,
            })
            continue

        z_scores = compute_robust_zscore(series, median_value, mad_value)
        outlier_mask = z_scores.abs() > mad_threshold
        n_outliers = int(outlier_mask.sum())
        outlier_rate_pct = n_outliers / n_observations * 100

        # INSERIMENTO NEL DIZIONARIO (anziché nel dataframe)
        new_columns_data[f"{column}{outlier_suffix}"] = outlier_mask.astype("int8")
        new_columns_data[f"{column}_robust_zscore"] = z_scores

        analyzed_columns.append(column)
        
        # TRACCIATORE: Stampa i calcoli eseguiti per ogni singola feature valutata
        print(f"    * {column}: Mediana={median_value:.2f}, MAD={mad_value:.2f} -> Outlier trovati: {n_outliers} ({outlier_rate_pct:.1f}%)")

        report_rows.append({
            "feature": column,
            "feature_present": True,
            "reason_skipped": None,
            "n_observations": n_observations,
            "n_missing": n_missing,
            "median_value": median_value,
            "mad_value": mad_value,
            "n_outliers": n_outliers,
            "outlier_rate_pct": outlier_rate_pct,
        })

    # CONCATENAZIONE IN BLOCCO: Unisce il dizionario al dataframe originale in un colpo solo
    if new_columns_data:
        new_cols_df = pd.DataFrame(new_columns_data, index=result_df.index)
        result_df = pd.concat([result_df, new_cols_df], axis=1)

    outlier_flag_columns = [
        f"{column}{outlier_suffix}"
        for column in analyzed_columns
    ]

    if outlier_flag_columns:
        result_df["n_univariate_outliers"] = (
            result_df[outlier_flag_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            .astype("int16")
        )
        result_df["has_univariate_outlier"] = (
            result_df["n_univariate_outliers"] > 0
        ).astype("int8")
    else:
        result_df["n_univariate_outliers"] = 0
        result_df["has_univariate_outlier"] = 0

    # TRACCIATORE: Riepilogo finale
    print(f"[TRACCIATORE MAD] Analisi Completata. Feature analizzate: {len(analyzed_columns)} | Feature saltate: {len(skipped_columns)}")
    print(f"  -> Dimensioni DataFrame in uscita: {result_df.shape}\n")

    return UnivariateOutlierResult(
        dataframe=result_df,
        report=pd.DataFrame(report_rows),
        analyzed_columns=analyzed_columns,
        skipped_columns=skipped_columns,
    )








def apply_univariate_outlier_detection(
    df: pd.DataFrame,
    exclude_columns: list[str] | None = None,
    mad_threshold: float = MAD_THRESHOLD,
    min_observations: int = MIN_OBSERVATIONS,
) -> UnivariateOutlierResult:
    """Wrapper per applicare Parte B a tutte le colonne numeriche non escluse."""
    exclude_columns = exclude_columns or []

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    columns_to_analyze = [
        column for column in numeric_columns
        if column not in exclude_columns
    ]

    return detect_univariate_outliers(
        df=df,
        columns_to_analyze=columns_to_analyze,
        mad_threshold=mad_threshold,
        min_observations=min_observations,
    )


# =============================================================================
# PARTE C - OUTLIER MULTIVARIATI CON DBSCAN
# =============================================================================

def select_dbscan_features(
    df: pd.DataFrame,
    feature_set: str | list[str] = "triage",
    custom_features: list[str] | None = None,
    max_missing_rate: float = 0.30,
) -> tuple[list[str], dict[str, str]]:
    """Seleziona le feature per DBSCAN in base a disponibilità e missingness."""
    if not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate deve essere compreso tra 0 e 1.")

    if custom_features is not None:
        requested_features = list(custom_features)
    elif isinstance(feature_set, str):
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"feature_set '{feature_set}' non riconosciuto. "
                f"Disponibili: {list(FEATURE_SETS)}"
            )
        requested_features = FEATURE_SETS[feature_set]
    else:
        requested_features = list(feature_set)

    selected_features: list[str] = []
    excluded_features: dict[str, str] = {}

    for feature in requested_features:
        if feature not in df.columns:
            excluded_features[feature] = "Colonna non presente"
            continue

        numeric_series = pd.to_numeric(df[feature], errors="coerce")
        missing_rate = float(numeric_series.isna().mean())

        if missing_rate > max_missing_rate:
            excluded_features[feature] = (
                f"Missingness {missing_rate:.2%} > {max_missing_rate:.2%}"
            )
            continue

        selected_features.append(feature)

    return selected_features, excluded_features






def prepare_dbscan_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, RobustScaler | None]:
    """Prepara e scala la matrice da passare a DBSCAN (senza imputazione fittizia)."""
    if not features:
        return pd.DataFrame(), pd.Series(0, index=df.index, dtype="int8"), None

    matrix = df[features].copy()
    for feature in features:
        matrix[feature] = pd.to_numeric(matrix[feature], errors="coerce")

    # NOVITÀ: Troviamo solo i pazienti che NON hanno nessun NaN e usiamo solo quelli
    evaluable_mask = matrix.notna().all(axis=1)
    matrix_complete = matrix.loc[evaluable_mask]

    if matrix_complete.empty:
        return pd.DataFrame(), evaluable_mask.astype("int8"), None

    scaler = RobustScaler()
    matrix_scaled = pd.DataFrame(
        scaler.fit_transform(matrix_complete),
        index=matrix_complete.index,
        columns=features,
    )

    return matrix_scaled, evaluable_mask.astype("int8"), scaler









def detect_multivariate_outliers_dbscan(
    df: pd.DataFrame,
    feature_set: str | list[str] = "triage",
    custom_features: list[str] | None = None,
    eps: float | None = None,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    max_missing_rate: float = 0.30,
    metric: str = DBSCAN_METRIC,
) -> DBSCANResult:
    """Applica DBSCAN per identificare casi rari rispetto a più feature insieme."""
    print(f"\n[TRACCIATORE DBSCAN] Avvio Rilevamento Spaziale")
    
    if eps is None:
        eps = EPS_DEFAULT

    if eps <= 0:
        raise ValueError("eps deve essere maggiore di zero.")
    if min_samples < 2:
        raise ValueError("min_samples deve essere almeno 2.")

    print(f"  -> Parametri: epsilon (eps) = {eps}, min_samples = {min_samples}, metrica = {metric}")

    result_df = df.copy(deep=True)

    features, excluded_features = select_dbscan_features(
        df=df,
        feature_set=feature_set,
        custom_features=custom_features,
        max_missing_rate=max_missing_rate,
    )
    
    print(f"  -> Feature selezionate ({len(features)}): {features}")
    if excluded_features:
        print(f"  -> Feature escluse per alta sparsità: {list(excluded_features.keys())}")

    # Richiamo della funzione prepare_dbscan_matrix (con eventuale imputazione temporanea)
    matrix_scaled, evaluable, _ = prepare_dbscan_matrix(df, features)

    result_df["dbscan_evaluable"] = evaluable
    result_df["dbscan_cluster"] = np.nan
    result_df["dbscan_low_density"] = np.nan

    n_evaluable = int(evaluable.sum())
    n_not_evaluable = len(df) - n_evaluable
    
    print(f"  -> Valutazione Topologica: {n_evaluable} record completi, {n_not_evaluable} record esclusi (NaN residui)")

    if not features or matrix_scaled.empty:
        print(f"[TRACCIATORE DBSCAN] Errore: Matrice vuota. Interruzione processo.\n")
        
        # CODICE RIPRISTINATO: Compilazione report di errore
        report = {
            "features_used": features,
            "excluded_features": excluded_features,
            "n_features": len(features),
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "n_evaluable": n_evaluable,
            "n_not_evaluable": n_not_evaluable,
            "n_clusters": 0,
            "n_noise": 0,
            "noise_rate_pct": np.nan,
            "reason_no_dbscan": "Nessuna matrice completa disponibile",
        }
        return DBSCANResult(
            dataframe=result_df,
            report=report,
            features_used=features,
            n_evaluable=n_evaluable,
            n_not_evaluable=n_not_evaluable,
        )
        
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(matrix_scaled)

    result_df.loc[evaluable == 1, "dbscan_cluster"] = labels
    result_df.loc[evaluable == 1, "dbscan_low_density"] = (
        labels == -1
    ).astype("int8")

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_rate_pct = n_noise / n_evaluable * 100

    print(f"  -> Esito: Trovati {n_clusters} cluster principali. Isolate {n_noise} osservazioni anomale (noise: {noise_rate_pct:.2f}%).")
    print(f"[TRACCIATORE DBSCAN] Analisi Spaziale Completata.\n")
    
    # CODICE RIPRISTINATO: Compilazione report di successo
    report = {
        "features_used": features,
        "excluded_features": excluded_features,
        "n_features": len(features),
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "n_evaluable": n_evaluable,
        "n_not_evaluable": n_not_evaluable,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_rate_pct": noise_rate_pct,
        "reason_no_dbscan": None,
    }

    return DBSCANResult(
        dataframe=result_df,
        report=report,
        features_used=features,
        n_evaluable=n_evaluable,
        n_not_evaluable=n_not_evaluable,
    )







def apply_dbscan_outlier_detection(
    df: pd.DataFrame,
    feature_set: str = "triage",
    eps: float | None = None,
    min_samples: int = MIN_SAMPLES_DEFAULT,
    max_missing_rate: float = 0.30,
) -> DBSCANResult:
    """Wrapper semplice per eseguire la Parte C con un feature set predefinito."""
    return detect_multivariate_outliers_dbscan(
        df=df,
        feature_set=feature_set,
        eps=eps,
        min_samples=min_samples,
        max_missing_rate=max_missing_rate,
    )
