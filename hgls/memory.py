"""
memory.py — Working Memory / Context Buffer.

Limited-capacity workspace (capacity = 7 ± 2, per Miller's Law).
Older items are automatically displaced when full.
"""

from collections import deque
from typing import Any, List, Optional

CAPACITY = 7


class WorkingMemory:

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(self, item: Any) -> None:
        self._buf.append(item)

    def pop(self) -> Optional[Any]:
        return self._buf.pop() if self._buf else None

    def peek(self) -> Optional[Any]:
        return self._buf[-1] if self._buf else None

    def get_context(self) -> List[Any]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def is_full(self) -> bool:
        return len(self._buf) >= self.capacity

    def __len__(self):
        return len(self._buf)

    def __repr__(self):
        return f"WorkingMemory({list(self._buf)})"