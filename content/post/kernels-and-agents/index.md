---
title: GPU
description: Running reservoir simulations on CUDA, AMD and CPU with a single code
slug: kernels-and-agents
date: 2025-09-08
# image: kernels.png
categories:
    - Julia
    - JutulDarcy
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

## A bit of background

I have just hit the merge button on the 0.4.0 release of JutulDarcy. The biggest change since the 0.3 series is full support for executing models on GPUs. We already had GPU support for linear solves on CUDA through NVIDIAs CuSPARSE and AMGX libraries, but the new release moves all compute-intensive parts to the GPUs in a single unified implementation that can execute on AMD GPUs, CUDA GPUs and CPUs, all written in Julia.

Running models on GPUs can be a major performance benefit, as modern GPUs offer much higher throughput than CPUs when performing repetitive numerical calculations. Other reservoir simulators that were originally written for CPUs before GPUs for computations was established have moved some or all calculations to GPUs (e.g. Intersect or tNavigator) and several simulators have been developed with the express intent of targeting GPUs. This includes both commerical offerings like Echelon from StoneRidge and research simulators like GEOS that was written from the ground-up for GPUs and DARTS where the operator-based linearization allows fast GPU execution by caching operators that are sparsely evaluated on the CPU.

There are a few pain points when considering GPU solves for reservoir simulation. As a single code often supports many different types of governing equations that have their own highly performance sensitive kernels for residual and Jacobians, porting to GPU can be a highly invasive process that touches large parts of the code. There is a risk of having separate GPU implementations that live side-by-side with the CPU version and has to be maintained in sync, or to end up with highly GPU-specialized code that is hard to manage and may have worse performance on CPU. NVIDIA is the most popular vendor for GPUs and is programmed by using the the proprietary CUDA library, so you may then naturally run into issues when you want to run on e.g. an AMD card - or some future accelerator that could appear.

This blog post is then not about the great performance offered by GPUs, which are for the most part a given, but rather how reservoir simulation can be executed in a fast manner on GPUs without making the code a "GPU-ified" code that has a lot of complexity. The ingredients we are going to use are:

1. The Jutul+JutulDarcy framework for automatic differentiation
1. KernelAbstractions for vendor-neutral parallelization
1. Julia package extensions for load-on-demand functionality


## The parts of a reservoir simulator

### Properties

The largest cost in a forward simulation is typically the linear solver and this is a fairly self-contained

### Equations

### Linear solvers

Reservoir simulators running on GPUs is hardly a new development, so the reason for this blog post is instead to highlight that this port was done without altering the implementations of equations themselves. It is also 


### Convergence criteria, updates and miscellanious


I started what eventually became Jutul.jl back in 2020 with three goals:

1. Learn Julia
2. Fast automatic differentiation
3. Write once, execute anywhere

The KA code was eventually decided to be too brittle to keep maintaing.


 and potentially in the future in Apple Metal
##

## Motivation




- We had code already
- Primary variables, equations, etc, discuss
- Adapt

## Hello Sol

[^1]: DARTS uses a model where linearization operators are evaluated on the CPU, but cached and retrieved with numeric differentiation on GPU
