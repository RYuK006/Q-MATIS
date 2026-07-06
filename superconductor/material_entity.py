import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ==========================================
# ID GENERATION UTILS
# ==========================================
def gen_id(prefix: str) -> str:
    return f"QMATIS-{prefix}-{str(uuid.uuid4())[:8].upper()}"

# ==========================================
# EXPERIMENTS & MODELS
# ==========================================
@dataclass
class ExperimentRecord:
    id: str = field(default_factory=lambda: gen_id("EXP"))
    pipeline_version: str = "v1"
    git_commit: str = "unknown"
    random_seed: int = 42
    gpu_info: str = "unknown"
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: Optional[str] = None
    status: str = "RUNNING"

@dataclass
class ModelRecord:
    id: str = field(default_factory=lambda: gen_id("MOD"))
    name: str = "unknown"
    architecture: str = "unknown"
    version: str = "v1"
    training_dataset: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class CheckpointRecord:
    id: str = field(default_factory=lambda: gen_id("CHK"))
    model_id: str = ""
    epoch: int = 0
    val_loss: float = 0.0
    weights_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

# ==========================================
# MATERIAL EVENTS (EVENT SOURCING)
# ==========================================
@dataclass
class PhysicsAuditRecord:
    id: str = field(default_factory=lambda: gen_id("FLT"))
    experiment_id: str = ""
    filter_name: str = ""
    status: str = "Pass" 
    score: float = 1.0
    reason: str = ""
    threshold: Optional[float] = None
    intermediate_values: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class DecisionRecord:
    id: str = field(default_factory=lambda: gen_id("DEC"))
    experiment_id: str = ""
    action: str = "" # "Imported", "Generated", "Rejected", "Accepted"
    reason: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    responsible_module: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class PredictionRecord:
    id: str = field(default_factory=lambda: gen_id("PRD"))
    experiment_id: str = ""
    checkpoint_id: str = ""
    predicted_tc: float = 0.0
    uncertainty: float = 0.0
    physics_score: float = 0.0
    stability_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class EmbeddingRecord:
    id: str = field(default_factory=lambda: gen_id("EMB"))
    experiment_id: str = ""
    prediction_id: str = ""
    dimension: int = 0
    embedding_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

# ==========================================
# MATERIAL ENTITY
# ==========================================
@dataclass
class MaterialEntity:
    id: str = field(default_factory=lambda: gen_id("MAT"))
    formula: str = ""
    reduced_formula: str = ""
    source: str = ""
    
    # Lineage & Provenance
    parent_id: Optional[str] = None
    generation_strategy: Optional[str] = None
    
    # Core structure data
    structure_json: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed State (Derived from decisions, not directly updated)
    is_rejected: bool = False
    
    # History Collections (Append-only)
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    physics_audits: List[PhysicsAuditRecord] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    predictions: List[PredictionRecord] = field(default_factory=list)
    embeddings: List[EmbeddingRecord] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

