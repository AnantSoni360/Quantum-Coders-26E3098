"""
Model Registry
Manages model metadata and status
"""

import yaml
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

from config import settings
from database.models import ModelRegistry as DBModelRegistry
from database.session import AsyncSessionLocal

class ModelRegistry:
    """Model metadata registry"""
    
    def __init__(self):
        self.models_config = {}
        self.models = {}
    
    async def initialize(self):
        """Initialize model registry from config"""
        config = settings.load_models_config()
        self.models_config = config.get("models", {})
        
        # Update database with model definitions
        async with AsyncSessionLocal() as session:
            for model_id, model_config in self.models_config.items():
                # Check if model exists
                from sqlalchemy import select
                query = select(DBModelRegistry).where(DBModelRegistry.model_id == model_id)
                result = await session.execute(query)
                db_model = result.scalar_one_or_none()
                
                if not db_model:
                    # Create new model entry
                    db_model = DBModelRegistry(
                        model_id=model_id,
                        model_name=model_config.get("name", model_id),
                        model_type=model_config.get("type", "unknown"),
                        provider=model_config.get("provider", ""),
                        format=model_config.get("format"),
                        quantization=model_config.get("quantization"),
                        capabilities=model_config.get("capabilities", []),
                        context_length=model_config.get("context_length", 4096),
                        memory_required_gb=model_config.get("resources", {}).get("memory_gb"),
                        config=model_config
                    )
                    session.add(db_model)
            
            await session.commit()
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by ID"""
        return self.models_config.get(model_id)
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get all model configurations"""
        return list(self.models_config.values())
    
    def get_models_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get models with specific capability"""
        models = []
        for model_id, config in self.models_config.items():
            if capability in config.get("capabilities", []):
                models.append(config)
        return models
    
    def get_model_status(self, model_id: str) -> str:
        """Get model loading status"""
        from backends.backend_factory import BackendFactory
        backend = BackendFactory.get_backend(model_id)
        if backend and backend.loaded_at:
            return "loaded"
        return "ready"  # Available but not loaded