"""Batch inference module for efficient LLM processing."""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt: str
    adapter_id: Optional[str] = None
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass
class InferenceResponse:
    """Response from inference."""
    request_id: str
    text: str
    tokens_generated: int = 0
    success: bool = True
    error: Optional[str] = None


class BatchInference:
    """Batch inference handler with communication budget enforcement."""
    
    def __init__(
        self,
        llm_backend,
        max_batch_size: int = 32,
        communication_budget: int = 5,
        max_tokens: int = 128
    ):
        self.llm_backend = llm_backend
        self.max_batch_size = max_batch_size
        self.communication_budget = communication_budget
        self.max_tokens = max_tokens
        
        # Track message counts per agent per timestep
        self._message_counts: Dict[str, int] = {}
        self._current_timestep: int = 0
        self._lock = threading.Lock()
    
    def reset_timestep(self, timestep: int):
        """Reset message counts for a new timestep."""
        with self._lock:
            self._current_timestep = timestep
            self._message_counts.clear()
    
    def can_send_message(self, agent_id: str) -> bool:
        """Check if agent can send more messages this timestep."""
        with self._lock:
            count = self._message_counts.get(agent_id, 0)
            return count < self.communication_budget
    
    def get_remaining_budget(self, agent_id: str) -> int:
        """Get remaining message budget for an agent."""
        with self._lock:
            count = self._message_counts.get(agent_id, 0)
            return max(0, self.communication_budget - count)
    
    def _increment_message_count(self, agent_id: str):
        """Increment message count for an agent."""
        with self._lock:
            self._message_counts[agent_id] = self._message_counts.get(agent_id, 0) + 1
    
    def _truncate_to_max_tokens(self, text: str) -> str:
        """Truncate text to max tokens (approximate by words)."""
        words = text.split()
        if len(words) > self.max_tokens:
            return " ".join(words[:self.max_tokens])
        return text
    
    def infer_single(
        self,
        prompt: str,
        agent_id: str,
        adapter_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        enforce_budget: bool = True
    ) -> InferenceResponse:
        """Run inference for a single request with budget enforcement."""
        if enforce_budget and not self.can_send_message(agent_id):
            return InferenceResponse(
                request_id=agent_id,
                text="",
                success=False,
                error="Communication budget exceeded"
            )
        
        try:
            max_tokens = max_tokens or self.max_tokens
            response_text = self.llm_backend.generate(
                prompt=prompt,
                adapter_id=adapter_id,
                max_new_tokens=max_tokens
            )
            
            # Truncate if needed
            response_text = self._truncate_to_max_tokens(response_text)
            
            if enforce_budget:
                self._increment_message_count(agent_id)
            
            return InferenceResponse(
                request_id=agent_id,
                text=response_text,
                tokens_generated=len(response_text.split()),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Inference error for {agent_id}: {e}")
            return InferenceResponse(
                request_id=agent_id,
                text="",
                success=False,
                error=str(e)
            )
    
    def infer_batch(
        self,
        requests: List[InferenceRequest],
        enforce_budget: bool = True
    ) -> List[InferenceResponse]:
        """Run batch inference for multiple requests."""
        responses = []
        
        # Filter requests based on budget
        valid_requests = []
        for req in requests:
            agent_id = req.adapter_id or req.request_id
            if not enforce_budget or self.can_send_message(agent_id):
                valid_requests.append(req)
            else:
                responses.append(InferenceResponse(
                    request_id=req.request_id,
                    text="",
                    success=False,
                    error="Communication budget exceeded"
                ))
        
        # Process in batches
        for i in range(0, len(valid_requests), self.max_batch_size):
            batch = valid_requests[i:i + self.max_batch_size]
            batch_responses = self._process_batch(batch, enforce_budget)
            responses.extend(batch_responses)
        
        return responses
    
    def _process_batch(
        self,
        batch: List[InferenceRequest],
        enforce_budget: bool
    ) -> List[InferenceResponse]:
        """Process a single batch of requests."""
        responses = []
        
        for req in batch:
            agent_id = req.adapter_id or req.request_id
            response = self.infer_single(
                prompt=req.prompt,
                agent_id=agent_id,
                adapter_id=req.adapter_id,
                max_tokens=req.max_tokens,
                enforce_budget=enforce_budget
            )
            responses.append(response)
        
        return responses
    
    def infer_parallel(
        self,
        requests: List[InferenceRequest],
        num_workers: int = 4,
        enforce_budget: bool = True
    ) -> List[InferenceResponse]:
        """Run inference in parallel using thread pool."""
        responses = [None] * len(requests)
        
        def process_request(idx: int, req: InferenceRequest):
            agent_id = req.adapter_id or req.request_id
            response = self.infer_single(
                prompt=req.prompt,
                agent_id=agent_id,
                adapter_id=req.adapter_id,
                max_tokens=req.max_tokens,
                enforce_budget=enforce_budget
            )
            responses[idx] = response
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(process_request, idx, req)
                for idx, req in enumerate(requests)
            ]
            for future in futures:
                future.result()
        
        return responses
