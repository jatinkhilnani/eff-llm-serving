def calculate_kv_cache_size(
    batch_size, seq_len, num_layers, num_heads, head_dim, dtype_size
):
    # Key states: [batch_size, num_heads, seq_len, head_dim]
    key_size = batch_size * num_heads * seq_len * head_dim * dtype_size

    # Value states: [batch_size, num_heads, seq_len, head_dim]
    value_size = batch_size * num_heads * seq_len * head_dim * dtype_size

    # Total across all layers (bytes)
    total_bytes = (key_size + value_size) * num_layers

    # Convert to MB
    total_mb = total_bytes / (1024 * 1024)

    return total_mb


def estimate_memory_requirements(model_config, batch_size, input_length, output_length):
    """
    Estimate memory requirements for prefill and decode phases

    Args:
        model_config: Dictionary containing model parameters
        batch_size: Batch size for inference
        input_length: Number of input tokens (prefill phase)
        output_length: Number of output tokens (decode phase)

    Returns:
        Tuple of (prefill_memory_mb, decode_memory_mb)
    """
    # Model configuration
    num_layers = model_config["num_layers"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    hidden_size = model_config["hidden_size"]
    dtype_size = model_config["dtype_size"]  # bytes per parameter

    # KV cache for prefill (all input tokens)
    prefill_kv_cache_mb = calculate_kv_cache_size(
        batch_size, input_length, num_layers, num_heads, head_dim, dtype_size
    )

    # Activation memory for prefill (rough estimate)
    # This includes attention scores, intermediate activations, etc.
    activation_factor = 2.5  # typical multiplier for transformer models
    prefill_activation_mb = (
        batch_size * input_length * hidden_size * dtype_size * activation_factor
    ) / (1024 * 1024)

    # Total prefill memory
    prefill_memory_mb = prefill_kv_cache_mb + prefill_activation_mb

    # For decode phase, we only need KV cache for all tokens (input + newly generated)
    # and activations for the new token being processed
    total_length = input_length + output_length
    decode_kv_cache_mb = calculate_kv_cache_size(
        batch_size, total_length, num_layers, num_heads, head_dim, dtype_size
    )

    # Activation memory for just one new token during decoding
    decode_activation_mb = (
        batch_size * 1 * hidden_size * dtype_size * activation_factor
    ) / (1024 * 1024)

    # Total decode memory
    decode_memory_mb = decode_kv_cache_mb + decode_activation_mb

    return prefill_memory_mb, decode_memory_mb


def profile_memory_usage(model, input_ids, max_new_tokens):
    """Profile actual memory usage during model execution"""
    import torch

    # Record initial memory
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    # Run prefill phase
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

    # Record memory after prefill
    torch.cuda.synchronize()
    after_prefill_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    prefill_memory = after_prefill_memory - start_memory

    # Run decode phase (one step)
    with torch.no_grad():
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model(next_token, use_cache=True, past_key_values=past_key_values)

    # Record memory after one decode step
    torch.cuda.synchronize()
    after_decode_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    one_step_decode_memory = after_decode_memory - after_prefill_memory

    # Extrapolate for all decode steps
    decode_memory = one_step_decode_memory * max_new_tokens

    return prefill_memory, decode_memory


def get_model_config(model_name_or_path):
    """
    Get model configuration parameters needed for memory estimation

    Args:
        model_name_or_path: Model name (e.g., 'gpt2-large') or path to local model

    Returns:
        Dictionary with model configuration parameters
    """
    from transformers import AutoConfig

    # Load the model configuration
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Extract relevant parameters
    model_config = {
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "hidden_size": config.hidden_size,
        "head_dim": config.hidden_size // config.num_attention_heads,
    }

    # Determine dtype_size based on model precision
    import torch

    if hasattr(config, "torch_dtype"):
        if config.torch_dtype == torch.float16 or config.torch_dtype == "float16":
            model_config["dtype_size"] = 2  # fp16
        elif config.torch_dtype == torch.bfloat16 or config.torch_dtype == "bfloat16":
            model_config["dtype_size"] = 2  # bf16
        elif config.torch_dtype == torch.float32 or config.torch_dtype == "float32":
            model_config["dtype_size"] = 4  # fp32
        elif config.torch_dtype == torch.int8 or config.torch_dtype == "int8":
            model_config["dtype_size"] = 1  # int8
        else:
            model_config["dtype_size"] = 4  # default to fp32
    else:
        # Default to fp16 for most modern LLMs
        model_config["dtype_size"] = 2

    return model_config


def profile_model_memory_usage(
    self, model, sample_input_length=128, sample_output_length=32
):
    """
    Profile actual memory usage of the model with sample inputs

    Args:
        model: The loaded model
        sample_input_length: Sample input sequence length
        sample_output_length: Sample output sequence length to estimate

    Returns:
        Tuple of (prefill_memory_per_token, decode_memory_per_token)
    """
    import torch

    # Create dummy input (batch size 1)
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 50000, (1, sample_input_length), device=device)

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()

    # Run prefill
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

    torch.cuda.synchronize()
    prefill_memory = torch.cuda.memory_allocated() - start_memory
    prefill_memory_per_token = prefill_memory / sample_input_length

    # Run one decode step
    with torch.no_grad():
        next_token = torch.randint(0, 50000, (1, 1), device=device)
        _ = model(next_token, use_cache=True, past_key_values=past_key_values)

    torch.cuda.synchronize()
    decode_memory = torch.cuda.memory_allocated() - start_memory
    decode_memory_growth = decode_memory - prefill_memory

    # Memory per token for decode phase
    decode_memory_per_token = decode_memory_growth

    # Convert to MB
    prefill_memory_per_token_mb = prefill_memory_per_token / (1024 * 1024)
    decode_memory_per_token_mb = decode_memory_per_token / (1024 * 1024)

    return prefill_memory_per_token_mb, decode_memory_per_token_mb
