"""
concept_map.py

Canonical concept -> source-specific code mapping.

Every entry here was VERIFIED against the actual downloaded MIMIC-FHIR
and eICU-CRD data (via explore_data.py's find_concept_in_fhir /
find_concept_in_eicu) rather than guessed from memory or documentation —
a wrong code here would silently corrupt the whole pipeline, so nothing
goes in this file until it's been grepped out of real records.

Extend this file as more concepts get verified.
"""

CONCEPT_MAP = {
    "creatinine_serum": {
        "description": (
            "Serum creatinine (SCr) — the primary AKI biomarker per "
            "KDIGO criteria. This is the core trend-detection signal."
        ),
        "mimic_fhir": {
            "resource": "MimicObservationLabevents",
            "include_codes": ["50912"],  # display: "Creatinine"
            "exclude_codes": [
                "51082",  # Creatinine, Urine
                "51070",  # Albumin/Creatinine, Urine
                "51099",  # Protein/Creatinine Ratio
                "50841",  # Creatinine, Ascites
                "51067",  # 24 hr Creatinine
                "52024",  # Creatinine, Whole Blood
            ],
            "expected_unit": "mg/dL",
        },
        "eicu": {
            "table": "lab.csv",
            "name_column": "labname",
            "include_values": ["creatinine"],  # exact match, case-insensitive
            "exclude_values": ["urinary creatinine"],
            "expected_unit": "mg/dL",
        },
    },
    "urine_output": {
        "description": (
            "Urine output — used for the KDIGO oliguria criterion "
            "(<0.5 mL/kg/h over 6h). Summed across all urine-collection "
            "routes; non-urine drains (e.g. chest tubes) are excluded."
        ),
        "mimic_fhir": {
            "resource": "MimicObservationOutputevents",
            "include_codes": [
                "226559",  # Foley
                "226560",  # Void
                "226627",  # OR Urine
                "226631",  # PACU Urine
                "227489",  # GU Irrigant/Urine Volume Out
            ],
            "exclude_codes": [],  # anything else here (e.g. Chest Tube) is NOT urine
            "expected_unit": "ml",
        },
        "eicu": {
            "table": "intakeOutput.csv",
            # NOTE: eICU has no dedicated "urine" column — entries are
            # free-text rows in `celllabel`, charted inconsistently across
            # 200+ hospitals (case variants, whitespace variants, and
            # same-looking labels meaning DIFFERENT things at different
            # sites). Normalize with `.strip().lower()` before matching —
            # plain `.lower()` alone misses whitespace-only duplicates
            # (e.g. "Urine Output (mL)-Urethral Catheter" appeared with
            # 0, 1, and 2 trailing spaces as three "different" strings).
            "name_column": "celllabel",
            "include_if_contains": ["urine", "foley", "void"],
            # VERIFIED TRAPS: these are COUNTS or yes/no FLAGS, not volumes
            # in mL. Confirmed by implausibly small sample values (0-2).
            "exclude_values_normalized": [
                "number of incontinent voids",
                "incontinent urine",
                "urine incontinence",
                "urine occurrence",
                "urine count",  # lowercase variant only — see AMBIGUOUS below
            ],
            # AMBIGUOUS — needs a follow-up check before deciding:
            # 'Urine Count' (capitalized) had sample_value=650, which is
            # far too large to be a real count. Likely a DIFFERENT site's
            # flowsheet reusing a similar-looking label for an actual
            # volume. Pull a larger sample of this exact label before
            # deciding whether to include or exclude it.
            "pending_review": ["Urine Count"],
            "expected_unit": "ml",
            "unit_verification_status": (
                "CONFIRMED for most labels — sample values (tens to "
                "low-thousands) look like plausible urine volumes in mL. "
                "Still pending: the 'Urine Count' ambiguity above."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Time axis normalization
# ---------------------------------------------------------------------------
#
# MIMIC-FHIR: absolute, date-shifted ISO datetimes.
#   minutes_since_admission = (event_time - encounter.period.start) / 60
#
# eICU-CRD: already relative, in MINUTES from ICU admission
#   (e.g. `labresultoffset`, `observationoffset`, `intakeoutputoffset`).
#   -> use directly as minutes_since_admission.
#
# CRITICAL — leakage risk on lab timestamps:
#   For MIMIC labs, use `issued` (result-available time), NOT
#   `effectiveDateTime` (specimen-draw time). Verified in real records
#   that these two timestamps DIFFER — using draw time would let the
#   model "see" a lab value before a clinician could have known it.
#   For eICU, `labresultoffset` is assumed to represent result-availability
#   time (not draw time) — confirm against the eICU data dictionary
#   before relying on this for real training.
#
# ---------------------------------------------------------------------------
# Structural notes (see full write-up in chat / project doc)
# ---------------------------------------------------------------------------
#
# 1. MIMIC chartevents = long format (1 row = 1 variable at 1 time).
#    eICU vitalPeriodic  = wide format (1 row = all vitals at 1 time,
#    mostly NaN). eICU vitals must be melted to long format before
#    tokenizing so both sources yield the same (concept, value, time) shape.
#
# 2. Medications: MIMIC MedicationAdministrationICU = discrete point-in-time
#    dose events. eICU medication.csv = order-level records with
#    drugorderoffset / drugstartoffset / drugstopoffset (an exposure
#    INTERVAL, not repeated discrete doses). These need different
#    tokenization treatment.
#
# 3. Relational hierarchy matches cleanly across both sources:
#      MIMIC Patient.id        <-> eICU uniquepid
#      MIMIC parent Encounter  <-> eICU patienthealthsystemstayid
#      MIMIC EncounterICU      <-> eICU patientunitstayid
#
# 4. eICU age is a STRING, not an int — ages over 89 are masked as "89+"
#    for de-identification. Needs custom parsing, not a plain int() cast.