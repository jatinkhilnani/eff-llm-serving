import queue, random, statistics, threading, time, torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


@dataclass
class KVCache:
    """Representation of a KV cache entry"""

    key_states: torch.Tensor
    value_states: torch.Tensor
    request_id: str
    last_accessed: float
    size_bytes: int

    def update_access_time(self):
        self.last_accessed = time.time()


class MemoryPool:
    """Elastic memory pool for managing GPU memory across multiple devices"""

    def __init__(self, devices: List[int], memory_per_device: List[int]):
        """
        Initialize memory pool

        Args:
            devices: List of GPU device IDs
            memory_per_device: Memory allocation per device in MB
        """
        self.devices = devices
        self.memory_per_device = memory_per_device
        self.total_memory = sum(memory_per_device)
        self.available_memory = self.total_memory
        self.device_usage = {device: 0 for device in devices}
        self.lock = threading.Lock()

    def allocate(
        self, size_mb: int, preferred_device: Optional[int] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Allocate memory from the pool

        Args:
            size_mb: Size to allocate in MB
            preferred_device: Preferred GPU device, if any

        Returns:
            Tuple of (success, device_id)
        """
        with self.lock:
            if size_mb > self.available_memory:
                return False, None

            if (
                preferred_device is not None
                and self.device_usage[preferred_device] + size_mb
                <= self.memory_per_device[self.devices.index(preferred_device)]
            ):
                self.device_usage[preferred_device] += size_mb
                self.available_memory -= size_mb
                return True, preferred_device

            # Find device with most free memory
            best_device = None
            most_free = -1

            for i, device in enumerate(self.devices):
                free_memory = self.memory_per_device[i] - self.device_usage[device]
                if free_memory >= size_mb and free_memory > most_free:
                    most_free = free_memory
                    best_device = device

            if best_device is not None:
                self.device_usage[best_device] += size_mb
                self.available_memory -= size_mb
                return True, best_device

            return False, None

    def free(self, size_mb: int, device: int):
        """Free memory from a specific device"""
        with self.lock:
            self.device_usage[device] = max(0, self.device_usage[device] - size_mb)
            self.available_memory += (
                size_mb  ## what if size_mb > self.device_usage[device]?
            )


class CacheManager:
    """Manages KV cache across devices with eviction policies"""

    def __init__(
        self,
        memory_pool: MemoryPool,
        cache_capacity_percent: float = 80.0,
        eviction_policy: str = "LRU",
    ):
        """
        Initialize cache manager

        Args:
            memory_pool: The shared memory pool
            cache_capacity_percent: Percentage of memory pool to use for caching
            eviction_policy: Cache eviction policy (LRU, FIFO, etc.)
        """
        self.memory_pool = memory_pool
        self.cache_capacity = int(
            memory_pool.total_memory * (cache_capacity_percent / 100.0)
        )
        self.current_cache_size = 0
        self.eviction_policy = eviction_policy
        self.cache_entries: Dict[str, Dict[str, KVCache]] = (
            {}
        )  # request_id -> {layer_id -> KVCache}
        self.device_mapping: Dict[str, int] = {}  # request_id -> device_id
        self.lock = threading.Lock()

        # Add cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.eviction_count = 0

    def store(
        self,
        request_id: str,
        layer_id: str,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        preferred_device: Optional[int] = None,
    ) -> bool:
        """
        Store KV cache for a request and layer

        Args:
            request_id: Unique request identifier
            layer_id: Layer identifier
            key_states: Key states tensor
            value_states: Value states tensor
            preferred_device: Preferred device for storage (if any)

        Returns:
            Success flag
        """
        # Calculate memory size in MB
        key_size = key_states.element_size() * key_states.nelement() / (1024 * 1024)
        value_size = (
            value_states.element_size() * value_states.nelement() / (1024 * 1024)
        )
        total_size = key_size + value_size

        with self.lock:
            # Check if we already have an entry for this request
            if request_id in self.cache_entries:
                device = self.device_mapping[request_id]

                # If this layer is already cached, update it
                if layer_id in self.cache_entries[request_id]:
                    old_cache = self.cache_entries[request_id][layer_id]
                    old_size = old_cache.size_bytes / (1024 * 1024)

                    # Free the old allocation
                    self.memory_pool.free(old_size, device)

                    self.current_cache_size -= old_size

                    # Try to allocate new size
                    success, device = self.memory_pool.allocate(total_size, device)
                    if not success:
                        # Try to evict if allocation failed
                        self._evict(total_size - old_size)
                        success, device = self.memory_pool.allocate(total_size, device)
                        if not success:
                            return False

                    # Update cache entry
                    cache_entry = KVCache(
                        key_states=key_states.to(f"cuda:{device}"),
                        value_states=value_states.to(f"cuda:{device}"),
                        request_id=request_id,
                        last_accessed=time.time(),
                        size_bytes=int(total_size * 1024 * 1024),
                    )
                    self.cache_entries[request_id][layer_id] = cache_entry
                    self.current_cache_size += total_size
                    return True
            else:
                # New request - try to allocate
                if preferred_device is None and len(self.cache_entries) > 0:
                    # Try to use the same device as other layers from this request
                    similar_requests = [
                        req
                        for req in self.device_mapping.keys()
                        if req.split("_")[0] == request_id.split("_")[0]
                    ]
                    if similar_requests:
                        preferred_device = self.device_mapping[similar_requests[0]]

                success, device = self.memory_pool.allocate(
                    total_size, preferred_device
                )
                if not success:
                    # Try to evict
                    self._evict(total_size)
                    success, device = self.memory_pool.allocate(
                        total_size, preferred_device
                    )
                    if not success:
                        return False

                # Create new cache entry
                cache_entry = KVCache(
                    key_states=key_states.to(f"cuda:{device}"),
                    value_states=value_states.to(f"cuda:{device}"),
                    request_id=request_id,
                    last_accessed=time.time(),
                    size_bytes=int(total_size * 1024 * 1024),
                )

                if request_id not in self.cache_entries:
                    self.cache_entries[request_id] = {}
                    self.device_mapping[request_id] = device

                self.cache_entries[request_id][layer_id] = cache_entry
                self.current_cache_size += total_size
                return True

    def retrieve(
        self, request_id: str, layer_id: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve KV cache for a request and layer

        Args:
            request_id: Unique request identifier
            layer_id: Layer identifier

        Returns:
            Tuple of (key_states, value_states) or (None, None) if not found
        """
        with self.lock:
            if (
                request_id in self.cache_entries
                and layer_id in self.cache_entries[request_id]
            ):
                cache_entry = self.cache_entries[request_id][layer_id]
                cache_entry.update_access_time()
                self.cache_hits += 1
                return cache_entry.key_states, cache_entry.value_states

            self.cache_misses += 1
            return None, None

    def _evict(self, required_mb: float) -> bool:
        """
        Evict cache entries to free up required memory

        Args:
            required_mb: Required memory in MB

        Returns:
            Success flag
        """
        if self.current_cache_size == 0:
            return False

        if self.eviction_policy == "LRU":
            # Get all cache entries flattened
            all_entries = []
            for request_id, layers in self.cache_entries.items():
                for layer_id, cache in layers.items():
                    all_entries.append((request_id, layer_id, cache))

            # Sort by last accessed time
            all_entries.sort(key=lambda x: x[2].last_accessed)

            freed_mb = 0
            evicted = []

            # Evict oldest entries first
            for request_id, layer_id, cache in all_entries:
                entry_size_mb = cache.size_bytes / (1024 * 1024)
                device = self.device_mapping[request_id]
                self.memory_pool.free(entry_size_mb, device)
                freed_mb += entry_size_mb
                evicted.append((request_id, layer_id))

                if freed_mb >= required_mb:
                    break

            # Remove evicted entries from cache
            for request_id, layer_id in evicted:
                del self.cache_entries[request_id][layer_id]
                if not self.cache_entries[request_id]:
                    del self.cache_entries[request_id]
                    del self.device_mapping[request_id]

            self.current_cache_size -= freed_mb
            self.eviction_count += len(evicted)
            return freed_mb >= required_mb

        elif self.eviction_policy == "FIFO":
            # Implement FIFO eviction if needed
            pass

        return False

    def clear(self, request_id: Optional[str] = None):
        """
        Clear cache entries

        Args:
            request_id: Specific request to clear, or all if None
        """
        with self.lock:
            if request_id is not None:
                if request_id in self.cache_entries:
                    device = self.device_mapping[request_id]
                    for layer_id, cache in self.cache_entries[request_id].items():
                        entry_size_mb = cache.size_bytes / (1024 * 1024)
                        self.memory_pool.free(entry_size_mb, device)
                        self.current_cache_size -= entry_size_mb

                    del self.cache_entries[request_id]
                    del self.device_mapping[request_id]
            else:
                # Clear all cache entries
                for request_id, layers in self.cache_entries.items():
                    device = self.device_mapping[request_id]
                    for layer_id, cache in layers.items():
                        entry_size_mb = cache.size_bytes / (1024 * 1024)
                        self.memory_pool.free(entry_size_mb, device)

                self.current_cache_size = 0
                self.cache_entries.clear()
                self.device_mapping.clear()

    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "eviction_count": self.eviction_count,
                "current_cache_size_mb": self.current_cache_size,
                "cache_capacity_mb": self.cache_capacity,
                "utilization": (
                    self.current_cache_size / self.cache_capacity
                    if self.cache_capacity > 0
                    else 0
                ),
            }


class RequestScheduler:
    """Schedules and routes inference requests"""

    def __init__(self, memory_pool: MemoryPool, cache_manager: CacheManager):
        """
        Initialize request scheduler

        Args:
            memory_pool: The shared memory pool
            cache_manager: The KV cache manager
        """
        self.memory_pool = memory_pool
        self.cache_manager = cache_manager
        self.request_queue = queue.PriorityQueue()
        self.active_requests = set()
        self.lock = threading.Lock()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.running = False

        # Add timing metrics
        self.request_times = {}

    def start(self):
        """Start the scheduler thread"""
        self.running = True
        self.scheduler_thread.start()

    def stop(self):
        """Stop the scheduler thread"""
        self.running = False
        if self.scheduler_thread.is_alive():
            self.scheduler_thread.join()

    def submit_request(
        self, request_id: str, priority: int, prefill_tokens: int, decode_tokens: int
    ):
        """
        Submit a request for scheduling

        Args:
            request_id: Unique request identifier
            priority: Request priority (lower value = higher priority)
            prefill_tokens: Number of tokens for prefill phase
            decode_tokens: Number of tokens for decode phase
        """
        with self.lock:
            self.request_times[request_id] = {
                "submit_time": time.time(),
                "queue_start": None,
                "processing_start": None,
                "processing_end": None,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "cache_hit": False,
                "user_id": request_id.split("_")[0] if "_" in request_id else "unknown",
            }
            self.request_queue.put(
                (priority, time.time(), request_id, prefill_tokens, decode_tokens)
            )

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                if self.request_queue.empty():
                    time.sleep(0.01)
                    continue

                priority, submit_time, request_id, prefill_tokens, decode_tokens = (
                    self.request_queue.get(block=False)
                )

                # Record queue processing start time
                with self.lock:
                    if request_id in self.request_times:
                        self.request_times[request_id]["queue_start"] = time.time()

                # Check if we can process this request
                with self.lock:
                    # Estimate memory requirements
                    # In a real implementation, this would be based on model specs
                    prefill_memory_mb = prefill_tokens * 0.1  # Example estimate
                    decode_memory_mb = decode_tokens * 0.05  # Example estimate

                    # Check cache hit
                    cache_hit = False
                    for layer_id in [
                        f"layer_{i}" for i in range(12)
                    ]:  # Example with 12 layers
                        k, v = self.cache_manager.retrieve(request_id, layer_id)
                        if k is not None and v is not None:
                            cache_hit = True
                            break

                    # Record cache hit status
                    if request_id in self.request_times:
                        self.request_times[request_id]["cache_hit"] = cache_hit

                    # Adjust memory requirements based on cache hits
                    if cache_hit:
                        required_memory = decode_memory_mb
                    else:
                        required_memory = prefill_memory_mb + decode_memory_mb

                    # Try to allocate
                    success, device = self.memory_pool.allocate(required_memory)

                    if success:
                        # Record processing start time
                        if request_id in self.request_times:
                            self.request_times[request_id][
                                "processing_start"
                            ] = time.time()

                        # Process the request (in a real implementation, this would launch processing)
                        self.active_requests.add(request_id)
                        print(
                            f"Processing request {request_id} on device {device}, cache hit: {cache_hit}"
                        )

                        # Simulate processing
                        threading.Thread(
                            target=self._process_request,
                            args=(request_id, device, required_memory),
                        ).start()
                    else:
                        # Re-queue the request with the same priority
                        self.request_queue.put(
                            (
                                priority,
                                submit_time,
                                request_id,
                                prefill_tokens,
                                decode_tokens,
                            )
                        )
                        time.sleep(0.1)  # Avoid tight loop

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error in scheduler loop: {e}")
                time.sleep(0.1)

    def _process_request(self, request_id: str, device: int, memory_mb: float):
        """
        Simulate processing a request

        Args:
            request_id: Request identifier
            device: Assigned device
            memory_mb: Allocated memory in MB
        """
        try:
            # Simulate processing time
            time.sleep(np.random.uniform(0.5, 2.0))

            # In a real implementation, this would run model inference
            # and store KV cache in the cache manager

            # Example of storing cache for the request (just a simulation)
            for layer_id in [f"layer_{i}" for i in range(12)]:
                # Create mock tensors
                key_size = [1, 8, 1, 32]  # batch, heads, seq_len, head_dim
                value_size = [1, 8, 1, 32]

                key_states = torch.rand(key_size, device=f"cuda:{device}")
                value_states = torch.rand(value_size, device=f"cuda:{device}")

                self.cache_manager.store(
                    request_id=request_id,
                    layer_id=layer_id,
                    key_states=key_states,
                    value_states=value_states,
                    preferred_device=device,
                )

            # Free the allocated memory
            with self.lock:
                if request_id in self.active_requests:
                    self.active_requests.remove(request_id)

                # Record processing end time
                if request_id in self.request_times:
                    self.request_times[request_id]["processing_end"] = time.time()

                self.memory_pool.free(memory_mb, device)

        except Exception as e:
            print(f"Error processing request {request_id}: {e}")
            # Ensure memory is freed even on error
            with self.lock:
                if request_id in self.active_requests:
                    self.active_requests.remove(request_id)
                self.memory_pool.free(memory_mb, device)

    def get_timing_stats(self):
        """Get timing statistics for all processed requests"""
        with self.lock:
            stats = []
            for request_id, times in self.request_times.items():
                if times.get("processing_end") is not None:
                    queue_time = (
                        (times.get("processing_start", 0) - times.get("queue_start", 0))
                        if times.get("queue_start") is not None
                        else 0
                    )
                    processing_time = (
                        times.get("processing_end", 0)
                        - times.get("processing_start", 0)
                        if times.get("processing_start") is not None
                        else 0
                    )
                    total_time = times.get("processing_end", 0) - times.get(
                        "submit_time", 0
                    )

                    stats.append(
                        {
                            "request_id": request_id,
                            "user_id": times.get("user_id", "unknown"),
                            "queue_time": queue_time,
                            "processing_time": processing_time,
                            "total_time": total_time,
                            "prefill_tokens": times.get("prefill_tokens", 0),
                            "decode_tokens": times.get("decode_tokens", 0),
                            "cache_hit": times.get("cache_hit", False),
                        }
                    )
            return stats


class ModelRunner:
    """Runs model inference with memory-aware execution"""

    def __init__(self, model_path: str, cache_manager: CacheManager):
        """
        Initialize model runner

        Args:
            model_path: Path to model weights
            cache_manager: KV cache manager
        """
        self.model_path = model_path
        self.cache_manager = cache_manager
        # In a real implementation, this would load the model
        self.inference_times = []
        self.lock = threading.Lock()

    def run_inference(
        self, request_id: str, input_tokens: List[int], max_new_tokens: int, device: int
    ) -> List[int]:
        """
        Run model inference

        Args:
            request_id: Request identifier
            input_tokens: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            device: Device to run on

        Returns:
            Generated token IDs
        """
        start_time = time.time()

        # This is a simplified example. In a real implementation,
        # this would run the actual model inference.

        # Simulate token generation
        generated_tokens = []

        # Check if we have cached KV states
        cache_hit = True
        for layer_id in [f"layer_{i}" for i in range(12)]:
            k, v = self.cache_manager.retrieve(request_id, layer_id)
            if k is None or v is None:
                cache_hit = False
                break

        # Simulate prefill if no cache hit
        if not cache_hit:
            print(f"Cache miss for request {request_id}. Running prefill.")
            # Simulate prefill computation
            prefill_start = time.time()
            time.sleep(len(input_tokens) * 0.01)
            prefill_time = time.time() - prefill_start

            # Create and store KV cache
            for layer_id in [f"layer_{i}" for i in range(12)]:
                seq_len = len(input_tokens)

                # Create mock tensors
                key_size = [1, 8, seq_len, 32]  # batch, heads, seq_len, head_dim
                value_size = [1, 8, seq_len, 32]

                key_states = torch.rand(key_size, device=f"cuda:{device}")
                value_states = torch.rand(value_size, device=f"cuda:{device}")

                self.cache_manager.store(
                    request_id=request_id,
                    layer_id=layer_id,
                    key_states=key_states,
                    value_states=value_states,
                    preferred_device=device,
                )
        else:
            print(f"Cache hit for request {request_id}.")
            prefill_time = 0

        # Simulate decoding
        decode_start = time.time()
        for i in range(max_new_tokens):
            # In a real implementation, this would run the model's forward pass
            # using the cached KV states

            # Simulate decoding computation
            time.sleep(0.02)

            # Generate a random token (in a real implementation, this would be model output)
            next_token = np.random.randint(0, 50000)
            generated_tokens.append(next_token)

            # Update KV cache with the new token
            for layer_id in [f"layer_{i}" for i in range(12)]:
                # Get existing cache
                k, v = self.cache_manager.retrieve(request_id, layer_id)

                if k is not None and v is not None:
                    # Append new token's KV states
                    new_k = torch.cat(
                        [k, torch.rand(1, 8, 1, 32, device=k.device)], dim=2
                    )
                    new_v = torch.cat(
                        [v, torch.rand(1, 8, 1, 32, device=v.device)], dim=2
                    )

                    # Store updated cache
                    self.cache_manager.store(
                        request_id=request_id,
                        layer_id=layer_id,
                        key_states=new_k,
                        value_states=new_v,
                    )
            # Simulate end of text
            if next_token == 50000 - 1:  # EOS token in this simulation
                break

        decode_time = time.time() - decode_start
        total_time = time.time() - start_time

        # Record timing info
        with self.lock:
            self.inference_times.append(
                {
                    "request_id": request_id,
                    "total_time": total_time,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "input_tokens": len(input_tokens),
                    "output_tokens": len(generated_tokens),
                    "cache_hit": cache_hit,
                }
            )

        return generated_tokens

    def get_inference_stats(self):
        """Get inference timing statistics"""
        with self.lock:
            if not self.inference_times:
                return {}

            cache_hit_times = [
                t["total_time"] for t in self.inference_times if t["cache_hit"]
            ]
            cache_miss_times = [
                t["total_time"] for t in self.inference_times if not t["cache_hit"]
            ]

            stats = {
                "avg_total_time": statistics.mean(
                    [t["total_time"] for t in self.inference_times]
                ),
                "avg_prefill_time": statistics.mean(
                    [t["prefill_time"] for t in self.inference_times]
                ),
                "avg_decode_time": statistics.mean(
                    [t["decode_time"] for t in self.inference_times]
                ),
                "avg_time_cache_hit": (
                    statistics.mean(cache_hit_times) if cache_hit_times else 0
                ),
                "avg_time_cache_miss": (
                    statistics.mean(cache_miss_times) if cache_miss_times else 0
                ),
                "cache_hit_speedup": (
                    statistics.mean(cache_miss_times) / statistics.mean(cache_hit_times)
                    if cache_hit_times and cache_miss_times
                    else 0
                ),
            }
            return stats


class MemServer:
    """Main server implementation integrating all components"""

    def __init__(
        self,
        model_path: str,
        gpu_devices: List[int],
        memory_per_device: List[int],
        cache_capacity_percent: float = 80.0,
    ):
        """
        Initialize MemServer

        Args:
            model_path: Path to model weights
            gpu_devices: List of GPU device IDs to use
            memory_per_device: Memory allocation per device in MB
            cache_capacity_percent: Percentage of memory for caching
        """
        # Initialize memory pool
        self.memory_pool = MemoryPool(gpu_devices, memory_per_device)

        # Initialize cache manager
        self.cache_manager = CacheManager(
            memory_pool=self.memory_pool,
            cache_capacity_percent=cache_capacity_percent,
            eviction_policy="LRU",
        )

        # Initialize request scheduler
        self.scheduler = RequestScheduler(
            memory_pool=self.memory_pool, cache_manager=self.cache_manager
        )

        # Initialize model runner
        self.model_runner = ModelRunner(
            model_path=model_path, cache_manager=self.cache_manager
        )

        self.request_counter = 0
        self.lock = threading.Lock()

        # Add metrics container
        self.request_metrics = []

    def start(self):
        """Start the server"""
        self.scheduler.start()
        print(f"MemServer started with {len(self.memory_pool.devices)} devices")

    def stop(self):
        """Stop the server"""
        self.scheduler.stop()
        print("MemServer stopped")

    def process_request(
        self,
        user_id: str,
        input_text: str,
        priority: int = 0,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Process a new request

        Args:
            user_id: User identifier
            input_text: Input text
            priority: Request priority (lower value = higher priority)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        start_time = time.time()

        with self.lock:
            self.request_counter += 1
            request_id = f"{user_id}_req_{self.request_counter}"

        # In a real implementation, this would tokenize the input
        input_tokens = list(range(len(input_text.split())))

        # Submit for scheduling
        self.scheduler.submit_request(
            request_id=request_id,
            priority=priority,
            prefill_tokens=len(input_tokens),
            decode_tokens=max_new_tokens,
        )

        # Simulate the model runner (which would typically be asynchronous)
        # In a real implementation, we'd wait for completion or use callbacks
        device = 0  # Mock assigned device
        self.model_runner.run_inference(
            request_id, input_tokens, max_new_tokens, device
        )

        end_time = time.time()

        # Record metrics
        with self.lock:
            self.request_metrics.append(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "input_length": len(input_tokens),
                    "output_length": max_new_tokens,
                    "processing_time": end_time - start_time,
                }
            )

        return (
            f"Response for '{input_text[:20]}...' (Generated {max_new_tokens} tokens)"
        )

    def get_metrics(self):
        """Get all metrics including cache and timing statistics"""
        cache_stats = self.cache_manager.get_stats()
        timing_stats = self.scheduler.get_timing_stats()
        inference_stats = self.model_runner.get_inference_stats()

        return {
            "cache_stats": cache_stats,
            "timing_stats": timing_stats,
            "inference_stats": inference_stats,
            "request_metrics": self.request_metrics,
        }


class UserSimulator:
    """Simulates multiple users making requests to the MemServer"""

    def __init__(self, server: MemServer, num_users: int, prompt_templates: List[str]):
        """
        Initialize user simulator

        Args:
            server: The MemServer instance
            num_users: Number of simulated users
            prompt_templates: List of prompt templates to use
        """
        self.server = server
        self.user_ids = [f"user_{i}" for i in range(num_users)]
        self.prompt_templates = prompt_templates
        self.simulation_stats = []

    def _user_session(
        self, user_id: str, num_requests: int, repeat_probability: float = 0.3
    ):
        """
        Simulate a user session with multiple requests

        Args:
            user_id: User identifier
            num_requests: Number of requests to make
            repeat_probability: Probability of repeating a previous prompt
        """
        previous_prompts = []

        for i in range(num_requests):
            # Decide whether to repeat a previous prompt
            if previous_prompts and random.random() < repeat_probability:
                prompt = random.choice(previous_prompts)
                is_repeat = True
            else:
                prompt = random.choice(self.prompt_templates)
                previous_prompts.append(prompt)
                is_repeat = False

            # Add some randomness to the prompt to simulate variations
            if not is_repeat:
                prompt = f"{prompt} {random.choice(['Please', 'Could you', 'I would like to'])} elaborate."

            # Determine priority (some users might have higher priority)
            priority = 0 if "vip" in user_id else random.randint(1, 3)

            # Determine max tokens based on prompt complexity
            max_tokens = random.randint(50, 200)

            # Process the request
            start_time = time.time()
            response = self.server.process_request(
                user_id=user_id,
                input_text=prompt,
                priority=priority,
                max_new_tokens=max_tokens,
            )
            end_time = time.time()

            # Record statistics
            self.simulation_stats.append(
                {
                    "user_id": user_id,
                    "request_num": i + 1,
                    "prompt": prompt[:30] + "...",
                    "is_repeat": is_repeat,
                    "priority": priority,
                    "max_tokens": max_tokens,
                    "response_time": end_time - start_time,
                }
            )

            # Simulate thinking time between requests
            think_time = random.uniform(0.5, 3.0)
            time.sleep(think_time)

    def run_simulation(self, requests_per_user: int, concurrent_users: int = None):
        """
        Run the simulation with multiple users

        Args:
            requests_per_user: Number of requests per user
            concurrent_users: Max number of concurrent users (default: all)
        """
        if concurrent_users is None:
            concurrent_users = len(self.user_ids)

        # Create a thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Submit user sessions to the executor
            futures = [
                executor.submit(self._user_session, user_id, requests_per_user)
                for user_id in self.user_ids
            ]

            # Wait for all futures to complete
            for future in futures:
                future.result()

    def get_simulation_stats(self):
        """Get simulation statistics"""
        return self.simulation_stats

    def generate_analysis_report(self):
        """Generate analysis of simulation results"""
        if not self.simulation_stats:
            return "No simulation data available."

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.simulation_stats)

        # Group by user
        user_stats = df.groupby("user_id").agg(
            {
                "response_time": ["mean", "min", "max"],
                "is_repeat": ["mean", "sum"],
                "request_num": "max",
            }
        )

        # Analyze repeat vs. non-repeat times
        repeat_times = df[df["is_repeat"]]["response_time"].mean()
        new_times = df[~df["is_repeat"]]["response_time"].mean()
        speedup = new_times / repeat_times if repeat_times > 0 else 0

        report = (
            f"Simulation Results:\n"
            f"- Total users: {len(self.user_ids)}\n"
            f"- Total requests: {len(self.simulation_stats)}\n"
            f"- Average response time: {df['response_time'].mean():.3f}s\n"
            f"- Repeat requests: {df['is_repeat'].sum()} ({df['is_repeat'].mean()*100:.1f}%)\n"
            f"- New request avg time: {new_times:.3f}s\n"
            f"- Repeat request avg time: {repeat_times:.3f}s\n"
            f"- Speedup factor for repeat requests: {speedup:.2f}x\n"
        )

        return report


# Analytics class for visualizing and analyzing results
class PerformanceAnalyzer:
    """Analyzes and visualizes performance metrics"""

    def __init__(self, metrics: dict):
        """
        Initialize performance analyzer

        Args:
            metrics: Dictionary of metrics from MemServer
        """
        self.metrics = metrics

    def create_timing_df(self):
        """Create a DataFrame of timing statistics"""
        if "timing_stats" not in self.metrics or not self.metrics["timing_stats"]:
            return pd.DataFrame()

        return pd.DataFrame(self.metrics["timing_stats"])

    def create_request_df(self):
        """Create a DataFrame of request metrics"""
        if "request_metrics" not in self.metrics or not self.metrics["request_metrics"]:
            return pd.DataFrame()

        return pd.DataFrame(self.metrics["request_metrics"])

    def generate_summary_report(self):
        """Generate a summary report of performance metrics"""
        cache_stats = self.metrics.get("cache_stats", {})
        inference_stats = self.metrics.get("inference_stats", {})

        report = "Performance Summary:\n\n"

        # Cache statistics
        report += "Cache Statistics:\n"
        if cache_stats:
            report += f"- Cache hit rate: {cache_stats.get('hit_rate', 0)*100:.2f}%\n"
            report += f"- Cache hits: {cache_stats.get('cache_hits', 0)}\n"
            report += f"- Cache misses: {cache_stats.get('cache_misses', 0)}\n"
            report += f"- Eviction count: {cache_stats.get('eviction_count', 0)}\n"
            report += (
                f"- Cache utilization: {cache_stats.get('utilization', 0)*100:.2f}%\n"
            )
        else:
            report += "- No cache statistics available\n"

        report += "\nInference Statistics:\n"
        if inference_stats:
            report += f"- Avg. total inference time: {inference_stats.get('avg_total_time', 0):.3f}s\n"
            report += f"- Avg. prefill time: {inference_stats.get('avg_prefill_time', 0):.3f}s\n"
            report += f"- Avg. decode time: {inference_stats.get('avg_decode_time', 0):.3f}s\n"
            report += f"- Avg. time with cache hit: {inference_stats.get('avg_time_cache_hit', 0):.3f}s\n"
            report += f"- Avg. time with cache miss: {inference_stats.get('avg_time_cache_miss', 0):.3f}s\n"
            report += f"- Cache hit speedup: {inference_stats.get('cache_hit_speedup', 0):.2f}x\n"
        else:
            report += "- No inference statistics available\n"

        return report

    def plot_timing_comparison(self):
        """Plot timing comparison between cache hits and misses"""
        df = self.create_timing_df()
        if df.empty:
            return None

        # Create comparison data
        hit_times = df[df["cache_hit"]]["processing_time"].tolist()
        miss_times = df[~df["cache_hit"]]["processing_time"].tolist()

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot([hit_times, miss_times], labels=["Cache Hit", "Cache Miss"])
        ax.set_title("Processing Time: Cache Hit vs Miss")
        ax.set_ylabel("Time (seconds)")

        return fig

    def plot_user_experience(self):
        """Plot user experience metrics"""
        df = self.create_timing_df()
        if df.empty:
            return None

        # Group by user and get average response times
        user_df = (
            df.groupby("user_id")
            .agg({"total_time": "mean", "cache_hit": "mean"})
            .reset_index()
        )

        # Sort by cache hit rate
        user_df = user_df.sort_values("cache_hit", ascending=False)

        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 7))

        x = range(len(user_df))
        ax1.bar(x, user_df["total_time"], color="skyblue")
        ax1.set_xlabel("User")
        ax1.set_ylabel("Avg Response Time (s)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xticks(x)
        ax1.set_xticklabels(user_df["user_id"], rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(x, user_df["cache_hit"] * 100, "r-o", linewidth=2)
        ax2.set_ylabel("Cache Hit Rate (%)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        fig.tight_layout()
        fig.suptitle("User Experience vs Cache Hit Rate")
        plt.subplots_adjust(top=0.9)

        return fig


# Enhanced run example function with multi-user simulation
def run_example():
    # Initialize server
    server = MemServer(
        model_path="meta-llama/Meta-Llama-2-13B",
        gpu_devices=[0, 1],  # Use 2 GPUs
        memory_per_device=[16000, 16000],  # 16GB per GPU
        cache_capacity_percent=95.0,
    )

    # Start the server
    server.start()

    try:
        # Define prompt templates for simulation
        prompt_templates = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "What are the main features of Python?",
            "How does photosynthesis work?",
            "What is the meaning of life?",
            "Can you explain machine learning to a beginner?",
            "What are the benefits of regular exercise?",
            "How do I make a chocolate cake?",
            "What is the difference between AI and ML?",
            "Tell me about the history of the internet.",
            "How do I learn programming?",
            "What are the best practices for data science?",
            "Can you explain how LLMs work?",
            "How does KV caching benefit LLM inference?",
        ]

        # Create user simulator
        num_users = 50
        rpu = 100
        con_users = 25
        user_simulator = UserSimulator(server, num_users, prompt_templates)

        # Run simulation
        print(f"Starting simulation with {num_users} users...")
        user_simulator.run_simulation(requests_per_user=rpu, concurrent_users=con_users)

        # Get simulation results
        simulation_report = user_simulator.generate_analysis_report()
        print("\nSimulation Report:")
        print(simulation_report)

        # Gather metrics
        metrics = server.get_metrics()

        # Analyze performance
        analyzer = PerformanceAnalyzer(metrics)
        summary_report = analyzer.generate_summary_report()
        print("\nPerformance Analysis:")
        print(summary_report)

        # Plot results (in a real environment, you would save these plots)
        try:
            timing_plot = analyzer.plot_timing_comparison()
            user_plot = analyzer.plot_user_experience()

            # Save plots if possible (comment out if running in environment without plot support)
            if timing_plot:
                timing_plot.savefig("cache_timing_comparison.png")
                print("Saved timing comparison plot to cache_timing_comparison.png")

            if user_plot:
                user_plot.savefig("user_experience.png")
                print("Saved user experience plot to user_experience.png")
        except Exception as e:
            print(f"Could not generate plots: {e}")

        # Convert timing stats to DataFrame for more detailed analysis
        timing_df = pd.DataFrame(metrics["timing_stats"])
        if not timing_df.empty:
            # Calculate speedup ratio per user
            user_stats = timing_df.groupby(["user_id", "cache_hit"]).agg(
                {"total_time": ["mean", "count"]}
            )

            print("\nDetailed User Performance:")
            for user_id in timing_df["user_id"].unique():
                user_data = user_stats.loc[user_id]
                if (True in user_data.index) and (False in user_data.index):
                    hit_time = user_data.loc[True][("total_time", "mean")]
                    miss_time = user_data.loc[False][("total_time", "mean")]
                    speedup = miss_time / hit_time if hit_time > 0 else 0
                    hit_count = user_data.loc[True][("total_time", "count")]
                    miss_count = user_data.loc[False][("total_time", "count")]
                    total = hit_count + miss_count
                    hit_rate = hit_count / total if total > 0 else 0

                    print(f"User {user_id}:")
                    print(
                        f"  - Cache hit rate: {hit_rate*100:.1f}% ({hit_count}/{total})"
                    )
                    print(f"  - Avg time with cache hit: {hit_time:.3f}s")
                    print(f"  - Avg time with cache miss: {miss_time:.3f}s")
                    print(f"  - Speedup: {speedup:.2f}x")

    finally:
        # Stop the server
        server.stop()


if __name__ == "__main__":
    run_example()
