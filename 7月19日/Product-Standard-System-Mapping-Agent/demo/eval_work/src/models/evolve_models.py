from __future__ import annotations
from dataclasses import dataclass, field

from src.models.enums import EvolveActionType, ExpansionStatus


@dataclass
class SynonymVerifyResult:
    is_synonym: bool = False
    confidence: float = 0.0
    reason: str = ""


@dataclass
class CategoryAnalysisResult:
    category_name: str = ""
    parent_category: str = ""
    level_position: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class SynonymUpdateAction:
    category_id: str = ""
    product_name: str = ""
    llm_verified: bool = False
    timestamp: str = ""


@dataclass
class ExpansionAction:
    product_name: str = ""
    category_analysis: str = ""
    suggested_parent_id: str | None = None
    suggested_name: str | None = None
    suggested_level: str | None = None
    status: ExpansionStatus = ExpansionStatus.PENDING_REVIEW
    timestamp: str = ""


@dataclass
class ExpansionSuggestion:
    product_name: str = ""
    suggested_parent_id: str = ""
    suggested_category_name: str = ""
    suggested_level_position: str = ""
    llm_analysis: str = ""
    status: ExpansionStatus = ExpansionStatus.PENDING_REVIEW
    timestamp: str = ""


@dataclass
class EvolveAction:
    action_type: EvolveActionType = EvolveActionType.NONE
    trigger_reason: str = ""
    timestamp: str = ""
