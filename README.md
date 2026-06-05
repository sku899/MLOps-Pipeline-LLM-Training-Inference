# MLOps-Pipeline-LLM-Training-Inference
### This repository contains a production-grade pipeline for fine-tuning and ### serving Large Language Models (LLMs) using Llama-3.1-8B. The architecture ### is designed for scalability, observability, and reliability, mirroring ### modern enterprise standards.
## Quick Start
To deploy the entire stack locally using kind and docker, run:
Bash
chmod +x ./run-all.sh
./run-all.sh
## Cost Optimisation Strategy
A critical requirement for production LLMs is managing infrastructure costs while maintaining performance. In this pipeline, we implement several strategies to reduce VRAM footprint and compute overhead:
### 1. PEFT/LoRA (Parameter-Efficient Fine-Tuning)
Instead of fine-tuning the full weight matrix of the Llama-3.1-8B model, which would require massive VRAM, we utilise LoRA (Low-Rank Adaptation).
•   The Problem: Full-parameter fine-tuning of an 8B model requires storing gradients and optimizer states for every parameter. For an 8B model, this typically exceeds 80GB of VRAM, necessitating expensive A100/H100 clusters.
•   The Solution: By injecting small, trainable rank-decomposition matrices (adapters) into the attention layers, we freeze the base model and only train <1% of the total parameters.
•   The Impact: This reduces the VRAM requirement to ~24GB, allowing training on a single high-end consumer GPU (like an RTX 3090/4090) or mid-range enterprise GPUs (L4/A10G), resulting in a >70% reduction in infrastructure costs.
### 2. PagedAttention (Inference Serving)
For inference, we employ vLLM, which utilises PagedAttention.
•   Optimisation: Standard serving engines pre-allocate fixed-size memory blocks for KV caches, leading to significant fragmentation. PagedAttention partitions the KV cache into non-contiguous blocks, similar to virtual memory in operating systems.
•   The Result: Near-zero memory waste, allowing for higher batch sizes and lower latency on the same hardware.
### 3. Mixed Precision (BF16)
We utilise bfloat16 instead of float32. This halves the memory usage for model weights and activation buffers without sacrificing numerical stability, essential for training Llama-3.1 series models.
## Pipeline Components
#### Training (/train)
•   FSDP (Fully Sharded Data Parallel): Used to shard model states, gradients, and optimizer across GPUs.
•   Sanity Test: The script is pre-configured to run 100 steps to validate convergence before scaling to larger datasets.
#### Serving (/inference)
•   vLLM: Configured to serve the fine-tuned LoRA adapters dynamically over the base model.
•   Canary Deployment: The Helm chart manages traffic split, routing 10% of production traffic to the new model iteration to monitor performance and error rates before full rollout.
#### Observability
•   Grafana: The dashboard provides real-time monitoring of:
o   GPU Utilisation & VRAM: Tracking hardware efficiency.
o   Throughput (tokens/s): Measuring service capability.
o   P99 Latency: Ensuring SLA compliance.
#### Reliability & Rollback
Production safety is managed through the .github/workflows/ CI/CD pipeline and the rollback.sh utility.
1.  CI/CD: Automated linting and unit tests ensure code quality on every push.
2.  Health Probes: Liveness/readiness probes detect OOM (Out of Memory) conditions.
3.  Automated Rollback: If the error_rate exceeds 5% for over 5 minutes, the rollback.sh script automatically reverts the Kubernetes deployment to the previous stable image tag, minimizing downtime.
## Architecture Overview
For a detailed look at the integration between the Slurm-based training cluster and the Kubernetes-based inference service, refer to architecture.pdf.
