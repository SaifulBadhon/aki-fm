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
            # in mL. Confirmed by implausibly small sample values (0-2),
            # or (for 'Urine Count') by a 20-sample check where 19/20
            # values were exactly 1.0 with a single 650.0 outlier — a
            # textbook count distribution, not a volume distribution.
            "exclude_values_normalized": [
                "number of incontinent voids",
                "incontinent urine",
                "urine incontinence",
                "urine occurrence",
                "urine count",  # RESOLVED: both cases confirmed as counts
            ],
            "expected_unit": "ml",
            "unit_verification_status": (
                "CONFIRMED — sample values for all included labels look "
                "like plausible urine volumes in mL (tens to low-thousands). "
                "Creatinine + urine output mapping is now COMPLETE for "
                "both MIMIC-FHIR and eICU-CRD."
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
# Chartevents category scoping
# ---------------------------------------------------------------------------
#
# Derived from a real category-level inventory (inventory_concepts.py) —
# chartevents mixes genuine physiological signal with a LARGE volume of
# nursing documentation/workflow items (e.g. "Restraint/Support Systems"
# was more frequent than Heart Rate). This is CATEGORY-LEVEL scoping,
# not per-variable feature selection — excluding a whole class of
# non-clinical documentation, not cherry-picking predictive columns.

CHARTEVENTS_CATEGORY_SCOPE = {
    # COMPLETE — all 35 categories from the full inventory are classified below.
    "include": [
        "Routine Vital Signs",
        "Respiratory",
        "Pulmonary",
        "Cardiovascular",
        "Cardiovascular (Pulses)",
        "Cardiovascular (Pacer Data)",
        "GI/GU",
        "Pain/Sedation",
        "Neurological",
        "Hemodynamics",
        "Toxicology",  # nephrotoxic drug level monitoring — directly relevant
        # Mechanical circulatory support devices — hemodynamically critical
        # patients, high AKI incidence via perfusion-dependent injury:
        "Impella",
        "Heartware",
        "IABP",
        "ECMO",
        "NICOM",
        "Tandem Heart",  # n=1 in this sample, negligible but harmless to include
    ],
    "exclude": [
        "Restraint/Support Systems",
        "Care Plans",
        "Alarms",
        "Skin - Assessment",
        "Skin - Impairment",
        "Skin - Incisions",
        "Access Lines - Invasive",
        "Access Lines - Peripheral",
        "Adm History/FHPA",
        # Free-text notes / niche non-ICU-general categories — out of scope,
        # consistent with the project-wide "no free text" decision:
        "MD Progress Note",
        "OB-GYN",
        "OT Notes",
        "Swallow Evaluation",
        "RNTriggerNote",
    ],
    "needs_review": [
        "Treatments",  # too generic to classify by name alone — inspect sample records
        "General",     # same — inspect sample records before deciding
    ],
    "special_handling": {
        "Dialysis": (
            "NOT a plain include/exclude. (1) Used as a COHORT EXCLUSION "
            "signal — patients on dialysis before/at admission should be "
            "excluded entirely, matching the source AKI paper's exclusion "
            "criteria (maintenance HD patients don't fit a de novo AKI "
            "prediction task). (2) Used as a FEATURE for patients who "
            "start dialysis DURING the stay as a treatment response."
        ),
        "Labs": (
            "PENDING — this chartevents category appears to duplicate real "
            "labevents values (e.g. potassium, HCO3 seen under both). "
            "Need to check whether these are genuinely separate point-of-"
            "care/bedside draws or re-entered duplicates of the same lab "
            "before deciding whether to include, exclude, or deduplicate."
        ),
    },
}


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