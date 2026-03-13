from __future__ import annotations
from collections.abc import Iterable
from typing import Callable, Final, Optional, TypeAlias, TypeVar, Generic, Protocol
import time, random, statistics
import gc, sys

T = TypeVar('T')
class Heap(Generic[T]):
    """
    A binary heap implementation supporting both min-heap and max-heap behavior,
    with support for a custom key function for flexible ordering.

    Attributes:
        _arr (list[T]): Internal array representing the heap.
        _is_min_heap (bool): If True, heap behaves as a min-heap; otherwise, as a max-heap.
        _key (Callable): Function to extract comparison key from elements.
    """
    __slots__ = ("_arr", "_is_min_heap", "_key")

    def __init__(self, arr: list[T], isMinHeap: bool = False, key: Callable = lambda x: x) -> None:
        """
        Initialize a Heap from an existing list of integers.

        Args:
            arr (list[T]): Initial elements to store in the heap.
            isMinHeap (bool): Whether the heap is a min-heap (default False = max-heap).
            key (Callable): Optional key function for custom ordering (default identity).
        """
        self._arr = list(arr)
        self._is_min_heap = isMinHeap
        self._key = key
        self.build_heap()

    # Helpers
    def _left(self, index: int) -> int:
        """Return the index of the left child of a node."""
        return (index * 2) + 1

    def _right(self, index: int) -> int:
        """Return the index of the right child of a node."""
        return (index * 2) + 2

    def _parent(self, index: int) -> int:
        """Return the index of the parent of a node."""
        return (index - 1) // 2

    def _swap(self, a: int, b: int) -> None:
        """Swap two elements in the heap array."""
        val_a, val_b = self._arr[a], self._arr[b]
        self._arr[a], self._arr[b] = val_b, val_a

    @property
    def size(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._arr)

    def is_empty(self) -> bool:
        """Check whether the heap is empty."""
        return len(self._arr) == 0

    def heapify_down(self, index: int, size: int) -> None:
        """Restore the heap property by moving an element downwards."""
        cumulative: int = index
        left: int = self._left(index)
        right: int = self._right(index)

        if left < size and self._compare(left, cumulative):
            cumulative = left

        if right < size and self._compare(right, cumulative):
            cumulative = right

        if cumulative != index:
            self._swap(index, cumulative)
            self.heapify_down(cumulative, size)

    def heapify_up(self, index: int) -> None:
        """Restore the heap property by moving an element upwards."""
        while index > 0:
            parent = self._parent(index)
            if self._compare(index, parent):
                self._swap(index, parent)
                index = parent
            else:
                break

    @classmethod
    def heapify_list(cls, arr: list[T], isMinHeap=False, key=None) -> list[T]:
        """
        Convert a list into a heap and return the internal array.

        Args:
            arr (list[T]): List to convert.
            isMinHeap (bool): Whether the heap should be min-heap.
            key (Callable): Optional key function.

        Returns:
            list[T]: Heapified array.
        """
        h = cls(arr, isMinHeap=isMinHeap, key=key)
        return h._arr

    def build_heap(self) -> None:
        """Convert the internal array into a valid heap."""

        start_index: int = (self.size // 2) - 1
        for i in range(start_index, -1, -1):
            self.heapify_down(i, self.size)

    def peek(self) -> Optional[T]:
        """Return the top element of the heap without removing it."""
        return None if self.is_empty() else self._arr[0]

    def peek_n(self, n: int) -> list[T]:
        """
        Return the top n elements from the heap without modifying it.

        Args:
            n (int): Number of elements to return.

        Returns:
            list[T]: The n highest-priority elements.
        """
        temp = Heap(self._arr.copy(), isMinHeap=self._is_min_heap, key=self._key)
        return [temp.pop() for _ in range(n)]

    def push(self, value: T) -> None:
        """Insert a new value into the heap."""
        self._arr.append(value)
        self.heapify_up(len(self._arr) - 1)

    def pop(self) -> T:
        """Remove and return the root element of the heap."""
        if self.is_empty():
            raise IndexError("Pop from empty heap")
        
        root = self._arr[0]
        last_index = self.size - 1
        if last_index > 0:
            self._swap(0, last_index)

        self._arr.pop()

        if not self.is_empty():
            self.heapify_down(0, self.size)
            
        return root

    def push_pop(self, value: T) -> T:
        """
        Push a value onto the heap and pop the root in a single efficient operation.

        Args:
            value (int): Value to push.

        Returns:
            int: The popped root value.
        """
        if self.is_empty():
            self.push(value)
            return value

        root = self._arr[0]
        if (self._is_min_heap and value > root) or not (self._is_min_heap and value < root):
            self._arr[0] = value
            self.heapify_down(0, self.size)
            return root
        else:
            return value

    def replace(self, oldValue: T, newValue: T) -> None:
        """
        Replace an existing value in the heap with a new value.

        Args:
            oldValue (T): Value to replace.
            newValue (T): New value to insert.

        Raises:
            ValueError: If heap is empty or value is not found.
        """
        if self.is_empty():
            raise ValueError("Heap is empty")
        try:
            index = self._arr.index(oldValue)
        except ValueError:
            raise KeyError(f"{oldValue} not found in heap")

        self._arr[index] = newValue

        parent = self._parent(index)
        if index > 0 and self._compare(index, parent):
            self.heapify_up(index)
        else:
            self.heapify_down(index, self.size)

    def clear(self) -> None:
        """Remove all elements from the heap."""
        self._arr.clear()

    def merge(self, heap: "Heap | list[T]") -> "Heap":
        """
        Merge another heap or list into this heap.

        Args:
            heap (Heap | list[T]): Heap or list to merge.

        Returns:
            Heap: Self after merging.
        """
        values = heap._arr if isinstance(heap, Heap) else heap
        self._arr.extend(values)
        self.build_heap()
        return self

    def nlargest(self, n: int) -> list[T]:
        """
        Return the n largest elements.

        Args:
            n (int): Number of elements to return.

        Returns:
            list[T]: n largest elements in descending order.
        """
        if n <= 0:
            return []
        if self.size <= n:
            return sorted(self._arr, reverse=True)

        temp_heap = Heap(self._arr[:n], isMinHeap=True, key=self._key)
        for val in self._arr[n:]:
            if self._key(val) > self._key(temp_heap.peek()):
                temp_heap.pop()
                temp_heap.push(val)

        return sorted(temp_heap._copy(), reverse=True)

    def nsmallest(self, n: int) -> list[T]:
        """
        Return the n smallest elements.

        Args:
            n (int): Number of elements to return.

        Returns:
            list[T]: n smallest elements in ascending order.
        """
        if n <= 0:
            return []
        if self.size <= n:
            return sorted(self._arr)

        temp_heap = Heap(self._arr[:n], isMinHeap=False, key=self._key)
        for val in self._arr[n:]:
            if self._key(val) < self._key(temp_heap.peek()):
                temp_heap.pop()
                temp_heap.push(val)

        return sorted(temp_heap._copy())

    def update_key(self, key: Callable = lambda x: x) -> None:
        """
        Update the key function used for ordering and rebuild the heap.

        Args:
            key (Callable): New key function.
        """
        self._key = key
        self.build_heap()

    def display_contents(self) -> None:
        """Display the heap in a tree-like, level-by-level format."""
        if self.is_empty():
            print("Empty Heap")
            return
        print("[MinHeap]:" if self._is_min_heap else "[MaxHeap]:")
        n, level, index = self.size, 0, 0
        while index < n:
            level_size = 2 ** level
            level_nodes = self._arr[index: index + level_size]
            indent = " " * (2 ** (max(0, self.size.bit_length() - level - 1)))
            spacing = " " * (2 ** (max(0, self.size.bit_length() - level)))
            print(indent + spacing.join(str(v) for v in level_nodes))
            index += level_size
            level += 1

    def print_contents(self) -> None:
        """Print the raw internal array representing the heap."""
        if self.is_empty():
            print("Empty tree")
            return
        print("[MinHeap]:" if self._is_min_heap else "[MaxHeap]:")
        print(self._arr)

    def __repr__(self) -> str:
        self.display_contents()

    def __str__(self) -> str:
        """Return a string representation of the heap."""
        heap_type = "MinHeap" if self._is_min_heap else "MaxHeap"
        return f"[{heap_type}] {self._arr}"

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return self.size

    def __bool__(self) -> bool:
        """Return True if the heap is not empty."""
        return not self.is_empty()

    def __iter__(self):
        """Return an iterator over a copy of the heap array."""
        return iter(self._arr.copy())

    def __contains__(self, item):
        """Check if an item exists in the heap."""
        return item in self._arr

    def _compare(self, a, b) -> bool:
        """
        Compare two elements based on the heap type and key function.

        Args:
            a (int): Index of first element.
            b (int): Index of second element.

        Returns:
            bool: True if a has higher priority than b.
        """
        ka, kb = self._key(self._arr[a]), self._key(self._arr[b])
        return ka < kb if self._is_min_heap else ka > kb

    def _copy(self) -> list[T]:
        """
        Return a shallow copy of the heap elements.

        Returns:
            list[T]: Heap contents.
        """
        return self._arr.copy()