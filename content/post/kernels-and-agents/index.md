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

This post is written for those who are interested in at least one of the following:

- GPUs for reservoir simulation
- The Julia programming language
- Vendor neutral GPU programming
- AI assistance

This post is intended to be read by readers who may not be familiar with all of the above, so please bear with me if you are already a GPU-reservoir simulation expert who writes Julia kernels in your sleep (if so, drop me a line, we probably have some overlapping interests!).

## GPU reservoir simulation

Running models on GPUs can be a major performance benefit, as modern GPUs offer much higher throughput than CPUs when performing repetitive numerical calculations. Other reservoir simulators that were originally written for CPUs before GPUs for computations was established have moved some or all calculations to GPUs (e.g. Intersect or tNavigator) and several simulators have been developed with the express intent of targeting GPUs. This includes both commerical offerings like Echelon from StoneRidge and research simulators like GEOS that was written from the ground-up for GPUs and DARTS where the operator-based linearization allows fast GPU execution by caching operators that are sparsely evaluated on the CPU.

There are a few pain points when considering GPU solves for reservoir simulation. As a single code often supports many different types of governing equations that have their own highly performance sensitive kernels for residual and Jacobians, porting to GPU can be a highly invasive process that touches large parts of the code. There is a risk of having separate GPU implementations that live side-by-side with the CPU version and has to be maintained in sync, or to end up with highly GPU-specialized code that is hard to manage and may have worse performance on CPU. NVIDIA is the most popular vendor for GPUs and is programmed by using the the proprietary CUDA library, so you may then naturally run into issues when you want to run on e.g. an AMD card - or some future accelerator that could appear.

This blog post is then not about the great performance offered by GPUs, which are for the most part a given, but rather how reservoir simulation can be executed in a fast manner on GPUs without making the code a "GPU-ified" code that has a lot of complexity and is tied to one particular vendor or execution mode. The ingredients we are going to use are:

1. The Jutul+JutulDarcy framework for automatic differentiation
1. KernelAbstractions for vendor-neutral parallelization
1. Julia package extensions for load-on-demand functionality
1. A bit of coding agents to fill in some gaps in the Julia linear solver ecosystem for our particular usecase

## How does a reservoir simulator work?

Modern reservoir simulators predominantly use a fully or partially implicit scheme for solving the governing equations. For brevity, we will limit our selves to the fully implicit case in this blog post. A rough sketch of a time stepping loop [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) is as follows:

1. __Properties__: Evaluate constitutive laws based on current primary variables $\mathbf{x}$ and the primary variables at the previous time-step $\mathbf{x}_0$.
2. __Equation assembly__: Compute the residual equations $\mathbf{r}$ and the corresponding Jacobian matrix from the evaluated properties and current primary variables
3. __Convergence__: Check convergence by checking the magnitude of $\mathbf{r}$ since equations on residual form are solved when $\mathbf{r} = \mathbf{0}$.If the equations are converged, set $\mathbf{x}_0 \gets \mathbf{x}$ and go to the next time-step, starting from point 1.
4. __Linear solve__: If we have not yet converged, solve the linearized system to obtain an update $\Delta \mathbf{x} = -J^{-1}\mathbf{r}$ to the primary variables.
5. __Update__: Update the primary variables $\mathbf{x} \gets \mathbf{x} + \Delta \mathbf{x}$ with a bit of logic to avoid overshoots and unphysical values

My notation skips over a lot of complexity, but the simulation itself can be divided into these five steps, with a time-stepping loop around it that handles time-step cuts, changes in controls, and so on.

### A small exampe

Let us consider a simple two-component, two-phase CO2-H2O model used for CO2 storage by geological sequestration (CCS) with thermal effects. An engineer would create a 3D model of an saline aquifer with certain geological properties, place one or more wells, and then simulate injection of CO2 for a time period of 30 days or 10,000 years to look at how the CO2 distributes in the aquifer model and how the pressure of the system changes.

In the reservoir simulator, the reservoir is divided (discretized) into a number of cells with known volume and connections to neighboring cells. The problem is then to predict how the species and energy moves, given operational constraints (a policy for injection of CO2) for a time period. In broad strokes, the number of cells and the amount of time to be simulated determines the actual runtime of the computer program.

### Governing equations

Advancing our CCS system through time amounts to solving three conservation equations for the transport in the reservoir:

1. Conservation of CO2 mass present in both phases in each cell
2. Conservation of H2O mass in both phases in each cell
3. Conservation of thermal energy in each cell as the sum of internal energy of the rock and the fluid phases present in the voidspace of the rock

In addition, there may be closure equations for thermodynamical equilibrium in each cell that determines how the species distribute between the phases, and a number of well equations. The well equations are the same type of conservation laws for the well-bore, coupled to the reservoir, as well as a number of equations for "facility constraints" that determine how the wells are operated. In this case, this would be how much CO2 gets injected at what times through the wells provided that the pressure build up in the well is within reasonable limits. The equations for geothermal energy, oil and gas recovery, hydrogen storage and other applications are from this vantage point very similar - the number of components and phases may change, but the types of equations are very much the same. Mathematicians who dip their toes in reservoir simulation like this form, as it unifies the description, and also makes it easy to solve the "easiest" variant and assume that generalizing solvers to the harder variants used in industry can be left as future work for some unlucky PhD students.

### Properties

If we now move from the high mathematical vantage points of governing equations to property evaluation, the situation becomes much more messy. Reservoir simulation is (perhaps uniquely) very data-intensive in terms of defining simulation problems. Any of the above applications have a large number of choices for different constitutive relationships, and the relationships themselves are often quite mathematically complex. For example, evaluating densities and phase distributions of species may require the solution of a local thermodynamic equilibrium, and there are countless options for different equations of state that require different solution strategies. Another example is the evaluation of relative permeabilities where you may have different choices for endpoint scaling, hysteresis, three-phase model and relative permeabilities for each phase pair. 

This is the part that is potentiallty very ugly when p


These are evaluated per cell


The largest cost in a forward simulation is typically the linear solver and this is a fairly self-contained

### Equations

### Linear solvers

Reservoir simulators running on GPUs is hardly a new development, so the reason for this blog post is instead to highlight that this port was done without altering the implementations of equations themselves. It is also 


### Convergence criteria, updates and miscellanious



## Jutul and JutulDarcy

I started what eventually became Jutul.jl back in 2020 with three goals:

1. Learn Julia well enough to confidently use it in ongoing projects at SINTEF
2. Explore the potential for fast automatic differentiation of PDEs using the many AD packages in the Julia ecosystem[^1]
3. Assess the potential for "write once, execute everywhere" GPU/CPU parallelism that was emerging through the the nascent `KernelAbstractions` package[^2].

### A small example
As an example, consider how we simulate geological sequestration of CO2.


GPUs have a lot of threads and memory bandwidth. One of the things GPUs are really good at is parallel processing on arrays that contain "number-like" things. In the Julia world, these are referred to as `isbitstypes`, which are immutable types that have a fixed size in memory. GPUs are not so good at execute heavily branching logic, allocating memory during execution or manage complex data types that have variable size in memory. The design of the code was written with these restrictions of GPUs in mind.


The KA code was eventually decided to be too brittle to keep maintaing.

The emphasis in JutulDarcy has so far been to be able to get high performance





 and potentially in the future in Apple Metal
##

## Motivation




- We had code already
- Primary variables, equations, etc, discuss
- Adapt

## Hello Sol

[^1]: There was many, now there

[^2]: [The first release of KernelAbstractions.jl was March 9th in 2020](https://github.com/JuliaGPU/KernelAbstractions.jl/releases/tag/v0.1.0). This is, for other reasons, the start of a period where a lot of people spent time indoors in front of computers.
