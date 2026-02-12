"""
Model Loader
Dynamic model loading with memory management
"""

import torch
import gc
from typing import Dict, Any, Optional
import psutil
import GPUtil

from models.model_registry import ModelRegistry
from backends.backend_factory import BackendFactory
from core.exceptions import ModelLoadError

class ModelLoader:
    """Dynamic model loader with memory management"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.memory_pool = MemoryPool()
    
    async def load_model(self, model_id: str, quantization: Optional[str] = None) -> bool:
        """Load a model into memory"""
        # Get model configuration
        config = self.registry.get_model_config(model_id)
        if not config:
            raise ModelLoadError(model_id, "Model not found in registry")
        
        # Apply quantization if specified
        if quantization:
            config["quantization"] = quantization
        
        # Check if already loaded
        existing = BackendFactory.get_backend(model_id)
        if existing:
            return True
        
        # Check memory availability
        memory_required = config.get("resources", {}).get("memory_gb", 4.0)
        if not self.memory_pool.allocate(model_id, memory_required):
            # Try to free memory
            self.memory_pool.cleanup()
            if not self.memory_pool.allocate(model_id, memory_required):
                raise ModelLoadError(model_id, "Insufficient memory")
        
        # Create and load backend
        try:
            backend = BackendFactory.create_backend(config)
            success = await backend.load()
            
            if success:
                BackendFactory.register_backend(model_id, backend)
                return True
            else:
                self.memory_pool.deallocate(model_id)
                return False
                
        except Exception as e:
            self.memory_pool.deallocate(model_id)
            raise ModelLoadError(model_id, str(e))
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        backend = BackendFactory.get_backend(model_id)
        if backend:
            success = await backend.unload()
            if success:
                BackendFactory.unregister_backend(model_id)
                self.memory_pool.deallocate(model_id)
            return success
        return True
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        backends = BackendFactory.get_all_backends()
        return {
            model_id: {
                "loaded_at": backend.loaded_at,
                "total_requests": backend.total_requests,
                "average_latency": backend.latency_sum / backend.total_requests if backend.total_requests > 0 else 0
            }
            for model_id, backend in backends.items()
        }

class MemoryPool:
    """GPU/CPU memory pool management"""
    
    def __init__(self):
        self.allocated = {}  # model_id -> memory_gb
        self.total_gpu_memory = self._get_total_gpu_memory()
        self.total_cpu_memory = psutil.virtual_memory().total / (1024 ** 3)
    
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except:
            pass
        return 0
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        try:
            if torch.cuda.is_available():
                free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                return free_memory / (1024 ** 3)
        except:
            pass
        return 0
    
    def allocate(self, model_id: str, memory_gb: float, device: str = "auto") -> bool:
        """Allocate memory for model"""
        if device == "auto":
            device = "gpu" if self.total_gpu_memory > 0 else "cpu"
        
        if device == "gpu":
            available = self._get_available_gpu_memory()
            if available >= memory_gb:
                self.allocated[model_id] = {"device": "gpu", "memory": memory_gb}
                return True
        else:
            available = psutil.virtual_memory().available / (1024 ** 3)
            if available >= memory_gb:
                self.allocated[model_id] = {"device": "cpu", "memory": memory_gb}
                return True
        
        return False
    
    def deallocate(self, model_id: str):
        """Deallocate memory for model"""
        if model_id in self.allocated:
            del self.allocated[model_id]
        
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run garbage collector
        gc.collect()
    
    def cleanup(self):
        """Clean up memory by unloading least recently used models"""
        # Sort by last used time
        loaded_models = BackendFactory.get_all_backends()
        
        if len(loaded_models) > 1:
            # Unload model with least requests (simple LRU approximation)
            model_to_unload = min(
                loaded_models.items(),
                key=lambda x: x[1].total_requests
            )
            
            import asyncio
            asyncio.create_task(self._unload_model(model_to_unload[0]))
    
    async def _unload_model(self, model_id: str):
        """Async model unloading"""
        from models.model_loader import ModelLoader
        loader = ModelLoader()
        await loader.unload_model(model_id)