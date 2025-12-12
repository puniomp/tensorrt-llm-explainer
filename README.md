# TensorRT-LLM Explainer Notebook

This repository contains an educational, end-to-end walkthrough of **why LLM
decode is slow in vanilla frameworks** and **how TensorRT-LLM accelerates
inference on NVIDIA GPUs**.

The notebook breaks down the full performance story:

- Why decode is memory-bound  
- Why PyTorch eager-mode launches so many tiny kernels  
- How fused kernels reduce memory traffic and launch overhead  
- What KV Cache is, and how Paged KV Cache enables high concurrency  
- How Tensor Cores + FP8/INT8 yield major speedups  
- Conceptual differences between **PyTorch**, **vLLM**, and **TensorRT-LLM**  
- Why TRT-LLM often achieves **2–4× higher decode throughput**  
- Cost per token modeling and real-world implications  

The goal of this explainer is to make GPU inference **intuitive** for engineers,
interviewers, and customers—without requiring CUDA or TensorRT internals.

---

## 📘 Notebook

👉 **[Open the TensorRT-LLM Explainer Notebook](notebooks/tensorrt_llm_explainer_notebook.ipynb)**

The notebook includes:

- CPU-based toy simulations for attention, paging, and fusion  
- Conceptual illustrations of memory and compute bottlenecks  
- Clear summaries and talking points suitable for interviews or customer conversations  

This is *not* a benchmarking repo—it's a teaching tool.

---

## 🎯 Who This Is For

- Engineers learning transformer inference performance  
- Candidates preparing for NVIDIA AI/SA interviews  
- Teams optimizing LLM serving workloads  
- Anyone wanting to understand why frameworks struggle and how TRT-LLM fixes it  

---

## 💡 Key Topics Covered

- Memory-bound vs compute-bound decode  
- Kernel fusion and Tensor Core utilization  
- Paged KV Cache fundamentals  
- FP8/INT8 acceleration  
- vLLM vs TensorRT-LLM architecture  
- Back-of-the-envelope performance reasoning  
- Cost-per-million-token estimation  

---

## 📄 License

This project is licensed under the MIT License.
