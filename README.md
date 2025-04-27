# Efficient LLM Serving
This repository contains code for efficient LLM serving, replication of MemServe by Hu et al., for handling of large language models (LLMs) in production. The code is based on the paper "MemServe: Efficient LLM Serving with Memory-Centric Design" by Hu et al. (2023).

The code is implemented in Python and uses the PyTorch framework. It is designed to be run on a single machine with multiple GPUs. The code has been evaluated on simulated workloads using 15 prompts with additional randomization, 50 users (25 concurrent) sending 100 requests each.