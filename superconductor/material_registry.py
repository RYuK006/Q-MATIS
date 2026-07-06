import json
import sqlite3
import logging
import os
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

from .materials_lake import MaterialsLake
from .material_entity import MaterialEntity, DecisionRecord, PhysicsAuditRecord, PredictionRecord, EmbeddingRecord

logger = logging.getLogger(__name__)

class MaterialRegistry:
    """
    Central ORM for managing Materials Entities via Event Sourcing.
    Never overwrites scientific data.
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    def register_material(self, entity: MaterialEntity) -> str:
        """
        Registers a material if it doesn't exist, and strictly appends all new events.
        """
        # Try to insert base material if it's new
        try:
            self.lake.execute_write("""
                INSERT INTO materials (id, formula, reduced_formula, source, parent_id, generation_strategy, structure_json, metadata_json, is_rejected, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id, entity.formula, entity.reduced_formula, entity.source, 
                entity.parent_id, entity.generation_strategy, json.dumps(entity.structure_json), 
                json.dumps(entity.metadata), entity.is_rejected, entity.created_at
            ))
            logger.info(f"Registered new material: {entity.id} ({entity.reduced_formula})")
        except sqlite3.IntegrityError:
            # Material already exists, we will just append new events
            logger.debug(f"Material {entity.id} already exists. Appending events.")

        self._append_decisions(entity.id, entity.decisions)
        self._append_physics_audits(entity.id, entity.physics_audits)
        self._append_predictions(entity.id, entity.predictions)
        self._append_embeddings(entity.id, entity.embeddings)
        
        return entity.id

    def _append_decisions(self, material_id: str, decisions: List[DecisionRecord]):
        for d in decisions:
            try:
                self.lake.execute_write("""
                    INSERT INTO decision_history (id, material_id, experiment_id, action, reason, parameters, responsible_module, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (d.id, material_id, d.experiment_id, d.action, d.reason, json.dumps(d.parameters), d.responsible_module, d.timestamp))
            except sqlite3.IntegrityError:
                pass # Already recorded

    def _append_physics_audits(self, material_id: str, audits: List[PhysicsAuditRecord]):
        for a in audits:
            try:
                self.lake.execute_write("""
                    INSERT INTO physics_audits (id, material_id, experiment_id, filter_name, status, score, reason, threshold, intermediate_values, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (a.id, material_id, a.experiment_id, a.filter_name, a.status, a.score, a.reason, a.threshold, json.dumps(a.intermediate_values), a.timestamp))
            except sqlite3.IntegrityError:
                pass

    def _append_predictions(self, material_id: str, preds: List[PredictionRecord]):
        for p in preds:
            try:
                self.lake.execute_write("""
                    INSERT INTO predictions (id, material_id, experiment_id, checkpoint_id, predicted_tc, uncertainty, physics_score, stability_score, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (p.id, material_id, p.experiment_id, p.checkpoint_id, p.predicted_tc, p.uncertainty, p.physics_score, p.stability_score, p.timestamp))
            except sqlite3.IntegrityError:
                pass

    def _append_embeddings(self, material_id: str, embs: List[EmbeddingRecord]):
        for e in embs:
            try:
                self.lake.execute_write("""
                    INSERT INTO embeddings (id, material_id, experiment_id, prediction_id, dimension, embedding_path, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (e.id, material_id, e.experiment_id, e.prediction_id, e.dimension, e.embedding_path, e.timestamp))
            except sqlite3.IntegrityError:
                pass

    def save_latent_vector(self, material_id: str, prediction_id: str, experiment_id: str, vector: np.ndarray) -> EmbeddingRecord:
        """
        Saves a latent graph embedding vector to disk and records it in the lake.
        """
        dim = vector.shape[0] if len(vector.shape) == 1 else vector.shape[-1]
        
        emb_record = EmbeddingRecord(
            experiment_id=experiment_id,
            prediction_id=prediction_id,
            dimension=dim
        )
        
        emb_filename = f"{emb_record.id}.npy"
        emb_path = os.path.join(self.lake.embeddings_dir, emb_filename)
        np.save(emb_path, vector)
        
        emb_record.embedding_path = emb_path
        
        self._append_embeddings(material_id, [emb_record])
        return emb_record
