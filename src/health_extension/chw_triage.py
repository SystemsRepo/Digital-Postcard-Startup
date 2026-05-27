"""
CHW Triage Extension for Digital Postcard Pipeline
====================================================
Demonstrates how the LangGraph HITL (Human-in-the-Loop) pipeline
from this project can be adapted for community health worker (CHW)
clinical decision support.

The same pattern applies:
  AI recommendation → Human review gate → Action taken

In a CHW context:
  Claude evaluates patient report → CHW supervisor reviews → CHW acts

This maps to LMH's HEP Assist architecture.
"""

from pydantic import BaseModel, field_validator
from enum import Enum
from typing import Optional
from datetime import datetime


class TriageDecision(str, Enum):
    """
    iCCM (Integrated Community Case Management) triage levels.
    CHW-appropriate decision categories aligned with WHO protocols.
    """
    REFER_EMERGENCY = "REFER_EMERGENCY"   # immediate facility referral
    REFER_ROUTINE   = "REFER_ROUTINE"     # non-emergency referral within 24h
    TREAT_IN_PLACE  = "TREAT_IN_PLACE"    # CHW manages with protocol guidance
    ESCALATE        = "ESCALATE"          # needs supervisor, insufficient info


class CHWPatientReport(BaseModel):
    """
    Patient report submitted by a CHW at point of care.

    FHIR mapping:
    - This model → FHIR Encounter + Observation resources
    - Field: patient_age_months → FHIR Observation (age in months)
    - Field: muac_mm → FHIR Observation (LOINC: 56072-2, mid-arm circumference)
    - Field: temperature_c → FHIR Observation (LOINC: 8310-5, body temperature)
    """
    report_id:          str
    chw_id:             str
    community_id:       str
    patient_age_months: int
    chief_complaint:    str
    temperature_c:      Optional[float] = None
    muac_mm:            Optional[float] = None   # mid-upper arm circumference
    respiratory_rate:   Optional[int]   = None
    additional_notes:   str = ""
    submitted_at:       datetime = datetime.utcnow()

    @field_validator("muac_mm")
    @classmethod
    def validate_muac(cls, v):
        """
        Plausible MUAC range for children: 60–250mm.
        <115mm = severe acute malnutrition (emergency referral trigger).
        115–125mm = moderate acute malnutrition (routine referral).
        Data quality flag if outside this range — don't block submission,
        but flag for MERL review.
        """
        if v is not None and not (60 <= v <= 250):
            raise ValueError(
                f"MUAC {v}mm is outside the plausible range (60–250mm). "
                f"Please recheck the measurement. "
                f"Note: MUAC <115mm triggers emergency referral."
            )
        return v

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v):
        if v is not None and not (30.0 <= v <= 43.0):
            raise ValueError(
                f"Temperature {v}°C is outside plausible range (30–43°C). "
                f"Please recheck the reading."
            )
        return v


class CHWTriageOutput(BaseModel):
    """
    Structured output from the AI triage step.

    FHIR mapping:
    - This model → FHIR ClinicalImpression resource
    - Pydantic enforcement prevents hallucination from leaking into action steps.
    """
    report_id:     str
    decision:      TriageDecision
    reasoning:     str          # plain language explanation for CHW
    red_flags:     list[str]    # specific danger signs identified
    action_steps:  list[str]    # numbered instructions the CHW should follow now
    confidence:    str          # HIGH / MEDIUM / LOW
    dq_flags:      list[str]    # data quality issues for MERL review
    ai_model_used: str
    fallback_used: bool = False


# LangGraph State for HITL CHW Pipeline
# =======================================
# This extends the existing Digital Postcard pipeline state
# to include clinical review checkpoints.
#
# Pipeline flow:
#
#   [CHW submits report]
#          │
#          ▼
#   [validate_data]   ← Pydantic checks (rejects impossible MUAC, temp)
#          │
#          ▼
#   [ai_triage]       ← Claude evaluates clinical urgency
#          │
#          ▼
#   [human_review]    ← Supervisor sees recommendation, can override
#          │                (this is the HITL node — same as postcard approval)
#          ▼
#   [dispatch_action] ← SMS to CHW, log to DHIS2, alert facility if REFER_EMERGENCY
#
# Key principle: AI recommends, human decides, system acts.
# This is the correct governance model for clinical AI in low-resource settings.

TRIAGE_SYSTEM_PROMPT = """
You are a clinical decision support assistant for community health workers (CHWs)
trained in Integrated Community Case Management (iCCM) in sub-Saharan Africa.

Rules:
1. Base decisions ONLY on the information provided. Never invent symptoms.
2. When in doubt, choose ESCALATE — a false escalation is safer than a missed referral.
3. REFER_EMERGENCY for: convulsions, unconsciousness, inability to drink/breastfeed,
   stridor, severe acute malnutrition (MUAC <115mm), fast breathing >50/min for infants,
   chest in-drawing, high fever >39.5°C, or any sign of severe illness.
4. Use plain language a CHW with primary school education can understand.
5. Output MUST be valid JSON matching the TriageOutput schema exactly.
"""
