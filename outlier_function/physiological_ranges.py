"""
Limiti fisiologici per il data cleaning di dati clinici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# =============================================================================
# LIMITI DI PLAUSIBILITÀ PER TRIAGE
# =============================================================================

PHYSIOLOGICAL_RANGES_TRIAGE: Dict[str, Tuple[float, float]] = {
    "temperature": (80.0, 115.0),      # °F
    "heartrate": (10.0, 300.0),        # bpm
    "resprate": (4.0, 80.0),           # atti/min
    "o2sat": (50.0, 100.0),            # %
    "sbp": (40.0, 300.0),              # mmHg
    "dbp": (20.0, 200.0),              # mmHg
    "pain": (0.0, 15.0),               # scala NRS
}


# =============================================================================
# LIMITI DI PLAUSIBILITÀ PER ESAMI DEL SANGUE (BLOOD TEST)
# =============================================================================
#
# ATTENZIONE: nel dataset grezzo, ogni cella di BloodTest.csv contiene più
# valori separati da ";" (es. "12.1;13.4;11.9"), non un singolo numero.
# Questi limiti vengono applicati a OGNI singolo valore dentro la cella,
# non alla cella intera. Vedi clean_multivalue_impossible_values() nel
# file outlier_pipeline.py.
# =============================================================================

PHYSIOLOGICAL_RANGES_BLOOD: Dict[str, Tuple[float, float]] = {
    # Emoglobina: 2-25 g/dL
    "Hemoglobin": (2.0, 25.0),

    # Ematocrito: 10-70 %
    "Hematocrit": (10.0, 70.0),

    # Globuli bianchi: 0.5-60 ×10^9/L
    "White Blood Cells": (0.5, 60.0),

    # Piastrine: 5-1200 ×10^9/L
    "Platelet Count": (5.0, 1200.0),

    # Globuli rossi: 1.0-8.0 ×10^6/µL
    "Red Blood Cells": (1.0, 8.0),

    # Volume corpuscolare medio: 50-130 fL
    "Mean Corpuscular Volume": (50.0, 130.0),

    # Bicarbonato: 5-45 mEq/L
    "Bicarbonate": (5.0, 45.0),

    # Calcio: 4-15 mg/dL
    "Calcium": (4.0, 15.0),

    # Magnesio: 0.5-5.0 mg/dL
    "Magnesium": (0.5, 5.0),

    # Fosfato: 0.5-10.0 mg/dL
    "Phosphate": (0.5, 10.0),

    # Proteina C-reattiva: 0-600 mg/L
    "C-Reactive Protein": (0.0, 600.0),

    # Creatinina: 0.1-20 mg/dL (mantenuta se presente nel dataset)
    "Creatinine": (0.1, 20.0),

    # Sodio: 100-180 mEq/L (mantenuta se presente nel dataset)
    "Sodium": (100.0, 180.0),

    # Potassio: 1.5-9.0 mEq/L (mantenuta se presente nel dataset)
    "Potassium": (1.5, 9.0),

    # Glucosio: 20-800 mg/dL (mantenuta se presente nel dataset)
    "Glucose": (20.0, 800.0),
}


DEFAULT_PHYSIOLOGICAL_RANGES: Dict[str, Tuple[float, float]] = {
    **PHYSIOLOGICAL_RANGES_TRIAGE,
    **PHYSIOLOGICAL_RANGES_BLOOD,
}


# =============================================================================
# FEATURE SET PER DBSCAN
# =============================================================================

FEATURE_SETS: Dict[str, list[str]] = {
    "triage": [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
    ],
    "triage_plus_blood": [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "Hemoglobin",
        "White Blood Cells",
        "Platelet Count",
        "C-Reactive Protein",
    ],
    "blood_only": [
        "Hemoglobin",
        "Hematocrit",
        "White Blood Cells",
        "Platelet Count",
        "Red Blood Cells",
        "Mean Corpuscular Volume",
        "Bicarbonate",
        "Calcium",
        "Magnesium",
        "Phosphate",
    ],
}


# =============================================================================
# OGGETTO DI CONFIGURAZIONE
# =============================================================================

@dataclass
class PhysiologicalRangesConfig:
    """Configurazione completa dei limiti fisiologici."""

    triage: Dict[str, Tuple[float, float]]
    blood: Dict[str, Tuple[float, float]]
    combined: Dict[str, Tuple[float, float]]


def get_physiological_ranges() -> PhysiologicalRangesConfig:
    """Restituisce la configurazione completa dei limiti fisiologici."""
    combined = {**PHYSIOLOGICAL_RANGES_TRIAGE, **PHYSIOLOGICAL_RANGES_BLOOD}
    return PhysiologicalRangesConfig(
        triage=PHYSIOLOGICAL_RANGES_TRIAGE,
        blood=PHYSIOLOGICAL_RANGES_BLOOD,
        combined=combined,
    )
