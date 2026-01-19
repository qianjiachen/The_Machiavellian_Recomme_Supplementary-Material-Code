"""Message passing system for agent communication."""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import uuid
from dataclasses import dataclass, field

from src.models.data_models import Message


class MessageBuffer:
    """Buffer for managing messages between agents."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._pending: Dict[str, List[Message]] = defaultdict(list)  # receiver_id -> messages
        self._history: List[Message] = []
        self._message_count = 0
    
    def send(
        self,
        sender_id: str,
        receiver_id: str,
        content: str,
        timestamp: int,
        message_type: str = "general"
    ) -> Message:
        """Send a message from sender to receiver."""
        message = Message(
            message_id=f"msg_{self._message_count}_{uuid.uuid4().hex[:8]}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            timestamp=timestamp,
            message_type=message_type
        )
        
        self._pending[receiver_id].append(message)
        self._history.append(message)
        self._message_count += 1
        
        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]
        
        return message
    
    def broadcast(
        self,
        sender_id: str,
        receiver_ids: List[str],
        content: str,
        timestamp: int,
        message_type: str = "broadcast"
    ) -> List[Message]:
        """Broadcast a message to multiple receivers."""
        messages = []
        for receiver_id in receiver_ids:
            msg = self.send(sender_id, receiver_id, content, timestamp, message_type)
            messages.append(msg)
        return messages
    
    def receive(self, receiver_id: str, clear: bool = True) -> List[Message]:
        """Receive all pending messages for an agent."""
        messages = self._pending.get(receiver_id, [])
        if clear:
            self._pending[receiver_id] = []
        return messages
    
    def peek(self, receiver_id: str) -> List[Message]:
        """Peek at pending messages without clearing."""
        return self.receive(receiver_id, clear=False)
    
    def get_history(
        self,
        agent_id: Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None
    ) -> List[Message]:
        """Get message history with optional filters."""
        messages = self._history
        
        if agent_id:
            messages = [
                m for m in messages
                if m.sender_id == agent_id or m.receiver_id == agent_id
            ]
        
        if start_timestamp is not None:
            messages = [m for m in messages if m.timestamp >= start_timestamp]
        
        if end_timestamp is not None:
            messages = [m for m in messages if m.timestamp <= end_timestamp]
        
        return messages
    
    def get_conversation(
        self,
        agent1_id: str,
        agent2_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get conversation between two agents."""
        messages = [
            m for m in self._history
            if (m.sender_id == agent1_id and m.receiver_id == agent2_id) or
               (m.sender_id == agent2_id and m.receiver_id == agent1_id)
        ]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear_pending(self, receiver_id: Optional[str] = None):
        """Clear pending messages."""
        if receiver_id:
            self._pending[receiver_id] = []
        else:
            self._pending.clear()
    
    def clear_all(self):
        """Clear all messages."""
        self._pending.clear()
        self._history.clear()
        self._message_count = 0
    
    def get_pending_count(self, receiver_id: str) -> int:
        """Get count of pending messages for an agent."""
        return len(self._pending.get(receiver_id, []))
    
    def get_total_messages(self) -> int:
        """Get total number of messages sent."""
        return self._message_count
