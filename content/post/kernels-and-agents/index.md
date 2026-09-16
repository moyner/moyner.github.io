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

Running models on GPUs can be a major performance benefit, as modern GPUs offer much higher throughput than CPUs when performing repetitive numerical calculations. Other reservoir simulators that were originally written for CPUs before GPUs for computations was established have moved some or all calculations to GPUs (e.g. Intersect or tNavigator) and several simulators have been developed with the express intent of targeting GPUs. This includes Echelon, GEOS and DARTS


Rob Pike[^1],

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

[^1]: The above quote is excerpted from Rob Pike's [talk](https://www.youtube.com/watch?v=PAAkCSZUG1c) during Gopherfest, November 18, 2015.