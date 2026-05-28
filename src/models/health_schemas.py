"""
CHW Health Schemas — LMH Interview Extension
=============================================
Demonstrates how the Digital Postcard HITL pipeline maps to
Last Mile Health's HEP Assist architecture.

Same pattern:
  AI recommendation → Human review gate → Action taken
  ↓ adapted to →
  Claude evaluates patient report → CHW supervisor reviews → CHW acts

FHIR mappings included to signal health informatics awareness.
"""

from pydantic import BaseModel, field_validator
from enum import Enum
from typing import Optional


class TriageDecision(str, Enum):
    """
    iCCM triage levels aligned with WHO community case management protocols.
    Maps to the QAStatus enum in the postcard pipeline — same HITL pattern,
    clinical domain.
    """
    REFER_IMMEDIATELY = "REFER_IMMEDIATELY"   # emergency — go to facility now
    REFER_WITHIN_24H  = "REFER_WITHIN_24H"   # non-emergency referral
    TREAT_AT_HOME     = "TREAT_AT_HOME"       # CHW can manage with guidance
    NEEDS_SUPERVISOR  = "NEEDS_SUPERVISOR"    # uncertain — escalate to supervisor


class CHWTriageInput(BaseModel):
    """
    Patient report submitted by a CHW at point of care, possibly offline.

    FHIR mapping:
    - This model → FHIR Encounter + Observation resources
    - muac_mm     → FHIR Observation (LOINC: 56072-2, mid-arm circumference)
    - vital_signs → FHIR Observation bundle
    """
    patient_age_months: int
    symptoms:           str            # free text from CHW
    vital_signs:        Optional[dict] = None   # {"temp_c": 38.5, "muac_mm": 115}
    chw_id:             str
    community_id:       str

    @field_validator("vital_signs")
    @classmethod
    def validate_vital_signs(cls, v):
        if v is None:
            return v
        if "muac_mm" in v:
            muac = v["muac_mm"]
            if not (60 <= muac <= 250):
                raise ValueError(
                    f"MUAC {muac}mm outside plausible range (60–250mm). "
                    f"Note: MUAC <115mm = severe acute malnutrition (emergency)."
                )
        if "temp_c" in v:
            temp = v["temp_c"]
            if not (30.0 <= temp <= 43.0):
                raise ValueError(f"Temperature {temp}°C outside plausible range (30–43°C).")
        return v


class CHWTriageOutput(BaseModel):
    """
    Structured triage output from the AI step.
    Pydantic enforcement prevents hallucination from leaking into action steps.

    FHIR mapping: → FHIR ClinicalImpression resource

    Maps to PostcardEvaluation in the postcard pipeline — same structure,
    clinical domain with richer disaggregation for MERL reporting.
    """
    decision:         TriageDecision
    reasoning:        str          # plain language for the CHW
    confidence:       str          # HIGH / MEDIUM / LOW
    red_flags:        list[str]    # specific danger signs identified
    chw_action_steps: list[str]    # plain-language instructions the CHW follows now
    referral_facility: Optional[str] = None
    data_quality_flags: list[str] = []   # e.g. "MUAC below plausible range"
