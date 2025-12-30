"""
Embedded Local LLM - No external services required!

Uses llama-cpp-python to run small quantized models directly.
Models are downloaded on first use and cached locally.

Copyright (c) 2024-2025 Ihsan Mokhlis
Licensed under CC-BY-NC-SA-4.0
"""

import logging
import os
from pathlib import Path
from typing import Generator

logger = logging.getLogger("docling_kb.llm")

# Model configurations - small models that run well on CPU
AVAILABLE_MODELS = {
    "qwen2-0.5b": {
        "repo_id": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "filename": "qwen2-0_5b-instruct-q4_k_m.gguf",
        "size_mb": 350,
        "context_length": 4096,
        "description": "Fast & lightweight, good for simple Q&A",
    },
    "tinyllama-1.1b": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 670,
        "context_length": 2048,
        "description": "Better quality, still fast",
    },
    "phi-3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_mb": 2300,
        "context_length": 4096,
        "description": "Best quality, needs more RAM",
    },
    "smollm-360m": {
        "repo_id": "HuggingFaceTB/SmolLM-360M-Instruct-GGUF",
        "filename": "smollm-360m-instruct-q8_0.gguf",
        "size_mb": 380,
        "context_length": 2048,
        "description": "Tiny but capable",
    },
}

DEFAULT_MODEL = "qwen2-0.5b"


class LocalLLM:
    """
    Embedded LLM that runs locally without external services.
    
    Features:
    - Automatic model download on first use
    - Runs on CPU (no GPU required)
    - Small memory footprint
    - Fast inference for small models
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir or os.environ.get("LLM_CACHE_DIR", "/app/cache/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config = AVAILABLE_MODELS.get(model_name, AVAILABLE_MODELS[DEFAULT_MODEL])
        self.n_ctx = n_ctx or self.model_config["context_length"]
        self.n_threads = n_threads or os.cpu_count() or 4
        
        self._llm = None
        self._initialized = False
    
    @property
    def model_path(self) -> Path:
        """Path to the downloaded model file."""
        return self.cache_dir / self.model_config["filename"]
    
    @property
    def is_downloaded(self) -> bool:
        """Check if model is already downloaded."""
        return self.model_path.exists()
    
    async def initialize(self) -> bool:
        """
        Initialize the LLM. Downloads model if not present.
        Returns True if successful, False otherwise.
        """
        if self._initialized:
            return True
        
        try:
            # Check if llama-cpp-python is available
            from llama_cpp import Llama
        except ImportError:
            logger.warning("llama-cpp-python not installed. LLM features disabled.")
            return False
        
        # Download model if needed
        if not self.is_downloaded:
            logger.info(f"Downloading model {self.model_name} (~{self.model_config['size_mb']}MB)...")
            success = await self._download_model()
            if not success:
                return False
        
        # Load the model
        try:
            logger.info(f"Loading LLM: {self.model_name}...")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only for compatibility
                verbose=False,
            )
            self._initialized = True
            logger.info(f"✅ LLM ready: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            return False
    
    async def _download_model(self) -> bool:
        """Download the model from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            
            logger.info(f"Downloading from {self.model_config['repo_id']}...")
            
            hf_hub_download(
                repo_id=self.model_config["repo_id"],
                filename=self.model_config["filename"],
                local_dir=str(self.cache_dir),
                local_dir_use_symlinks=False,
            )
            
            logger.info(f"✅ Model downloaded: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            stop: List of stop sequences
            
        Returns:
            Generated text response
        """
        if not self._initialized or self._llm is None:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "<|end|>", "<|eot_id|>"],
            echo=False,
        )
        
        return response["choices"][0]["text"].strip()
    
    def chat(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Chat-style generation with context (RAG).
        
        Args:
            question: User's question
            context: Retrieved context from documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
Rules:
- Only answer based on the provided context
- If the context doesn't contain relevant information, say "I don't have enough information to answer that"
- Be concise but thorough
- Cite the source documents when relevant"""
        
        # Format prompt based on model
        if "qwen" in self.model_name.lower():
            prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Context from documents:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
        elif "phi" in self.model_name.lower():
            prompt = f"""<|system|>
{system_prompt}<|end|>
<|user|>
Context from documents:
{context}

Question: {question}<|end|>
<|assistant|>
"""
        else:
            # Generic chat format
            prompt = f"""### System:
{system_prompt}

### Context:
{context}

### Question:
{question}

### Answer:
"""
        
        return self.generate(prompt, max_tokens=512, temperature=0.3)
    
    def health_check(self) -> dict:
        """Return LLM health status."""
        return {
            "available": self._initialized,
            "model": self.model_name,
            "model_path": str(self.model_path) if self.is_downloaded else None,
            "downloaded": self.is_downloaded,
            "config": self.model_config,
        }


# Singleton instance
_llm_instance: LocalLLM | None = None


async def get_llm(model_name: str | None = None) -> LocalLLM | None:
    """
    Get or create the LLM singleton.
    
    Returns None if LLM is disabled or unavailable.
    """
    global _llm_instance
    
    # Check if LLM is disabled
    if os.environ.get("DISABLE_LLM", "").lower() in ("true", "1", "yes"):
        return None
    
    # Get model from env or use default
    model = model_name or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
    
    # Create instance if needed
    if _llm_instance is None or _llm_instance.model_name != model:
        _llm_instance = LocalLLM(model_name=model)
        success = await _llm_instance.initialize()
        if not success:
            _llm_instance = None
    
    return _llm_instance


def list_available_models() -> dict:
    """List all available models and their configurations."""
    return AVAILABLE_MODELS

