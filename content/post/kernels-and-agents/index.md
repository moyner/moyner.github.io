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
math: true
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

This blog post is not about the great performance offered by GPUs, which is well established in the literature, but rather how reservoir simulation can be executed in a fast manner on GPUs without making the code a "GPU-ified" code that has a lot of complexity and is tied to one particular vendor or execution mode. The ingredients we are going to use are:

1. The Jutul+JutulDarcy framework for automatic differentiation
1. KernelAbstractions for vendor-neutral parallelization
1. Julia package extensions for load-on-demand functionality
1. A bit of coding agents to fill in some gaps in the Julia linear solver ecosystem for our particular use case

## How does a reservoir simulator work?

Modern reservoir simulators predominantly use a fully or partially implicit scheme for solving the governing equations. For brevity, we will limit our selves to the fully implicit case in this blog post. A rough sketch of a time stepping loop [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) is as follows:

1. __Properties__: Evaluate constitutive laws based on current primary variables $\mathbf{x}$ and the primary variables at the previous time-step $\mathbf{x}_0$.
2. __Equation assembly__: Compute the residual equations $\mathbf{r}$ and the corresponding Jacobian matrix from the evaluated properties and current primary variables
3. __Convergence__: Check convergence by checking the magnitude of $\mathbf{r}$ since equations on residual form are solved when $\mathbf{r} = \mathbf{0}$.If the equations are converged, set $\mathbf{x}_0 \gets \mathbf{x}$ and go to the next time-step, starting from point 1.
4. __Linear solve__: If we have not yet converged, solve the linearized system to obtain an update $\Delta \mathbf{x} = -J^{-1}\mathbf{r}$ to the primary variables.
5. __Update__: Update the primary variables $\mathbf{x} \gets \mathbf{x} + \Delta \mathbf{x}$ with a bit of logic to avoid overshoots and unphysical values

My notation skips over a lot of complexity, but the simulation itself can be divided into these five steps, with a time-stepping loop around it that handles time-step cuts, changes in controls, and so on.

### A small example

Let us consider a simple two-component, two-phase CO2-H2O model used for CO2 storage by geological sequestration (CCS) with thermal effects. An engineer would create a 3D model of an saline aquifer with certain geological properties, place one or more wells, and then simulate injection of CO2 for a time period of 30 days or 10,000 years to look at how the CO2 distributes in the aquifer model and how the pressure of the system changes.

In the reservoir simulator, the reservoir is divided (discretized) into a number of cells with known volume and connections to neighboring cells. The problem is then to predict how the species and energy moves, given operational constraints (a policy for injection of CO2) for a time period. In broad strokes, the number of cells and the amount of time to be simulated determines the actual runtime of the computer program.

### Governing equations

Advancing our CCS system through time amounts to solving three conservation equations for the transport in the reservoir:

1. Conservation of CO2 mass present in both phases in each cell
2. Conservation of H2O mass in both phases in each cell
3. Conservation of thermal energy in each cell as the sum of internal energy of the rock and the fluid phases present in the voidspace of the rock

In addition, there may be closure equations for thermodynamical equilibrium in each cell that determines how the species distribute between the phases, and a number of well equations. The well equations are the same type of conservation laws for the well-bore, coupled to the reservoir, as well as a number of equations for "facility constraints" that determine how the wells are operated. In this case, this would be how much CO2 gets injected at what times through the wells provided that the pressure build up in the well is within reasonable limits. The equations for geothermal energy, oil and gas recovery, hydrogen storage and other applications are from this vantage point very similar - the number of components and phases may change, but the types of equations are very much the same. Mathematicians who dip their toes in reservoir simulation like this form, as it unifies the description, and also makes it easy to solve the "easiest" variant and assume that generalizing solvers to the harder variants used in industry can be left as future work for some unlucky PhD students.

### Properties

If we now move from the high mathematical vantage points of governing equations to property evaluation, the situation becomes much more messy. Reservoir simulation is (perhaps uniquely) very data-intensive in terms of defining simulation problems. Any of the above applications have a large number of choices for different constitutive relationships, and the relationships themselves are often quite mathematically complex. A few examples for our CCS example:

- Relative permeabilities model the change to Darcy's law under multiphase flow conditions. There are several types of relative permeability end-point scaling, hysteresis models. Typically, these functions are also different depending on the rock type a given cell is taken from.
- Evaluating phase distributions of species may require the solution of a local thermodynamic equilibrium, and there are countless options for different equations of state that require different solution strategies. This includes the black-oil model, equilibrium constant flashes (K-values) with dependence on state variables and different types of equations of state, all of which may be used to model dissolution of CO2 under changing conditions.
- The density of the mixtures depend on the pressure, temperature and compositions in each cell. This may be provided by an equation of state, or different correlations for each phase.

In terms of GPU execution, the good news is that properties are evaluated cell-by-cell and as each entry is independent from the others, they are massively parallel. The bad news is that there are an enormous number of them, with varying dependency relationships. In terms of number of line of code, this is usually the largest part of a reservoir simulator.

### Equations

The equations take the properties and variables and produce the residual equations, and their Jacobians. Practical reservoir simulation overwhelmingly uses two-point finite-volume schemes. The conservation equations for the reservoir are made up of the evaluation of a cell-wise accumulation term and the numerical fluxes between pairs of cells that share an interface. The residual equations, if cell-wise values are computed, are also independent of each other and suitable for parallelism, with the largest wrinkle being the simultanous computation of residual and Jacobian. We must also at this stage compute the coupling terms that connect the wells to the reservoir, which are conceptually similar to fluxes, albeit with yet another implementation per type of governing equation.

### Convergence

Checking the convergence of the system requires parallel reductions (e.g. a sum or maximum value) that are straightforward to parallelize. The convergence criteria can in practice be complicated expressions that depend on the model type. For our CCS problem, we would check that the well equations are solved in the Inf norm, that the sum of mass balance over all cells for each component is sufficiently small, and that the scaled maximum error is small (less than $10^-3$ in all cells).

### Linear solvers

Solving the resulting linear systems is typically done using Krylov-subspace method accelerated either with a pure smoother like ILU(0), or with a constrained-pressure-residual (CPR) preconditioner that combines algebraic multigrid (AMG) for the pressure with a second-stage smoother. This amounts to matrix-vector products (which are parallel for compressed sparse row (CSR) matrices), AMG cycles with diagonal smoothers (which are parallel) and triangular solves (which can be colored and parallelized).

### Updates

Finally, the primary variables are updated for each cell. This is trivially parallel, but there is again specific logic to each variable. For instance, pressure updates must be limited in both absolute and relative magnitude, saturation and composition fractions  after updates must be projected to sum to one and black-oil variable switching requires careful change of variable sets across phase boundaries.

## Jutul and JutulDarcy

I started what eventually became Jutul.jl and JutulDarcy.jl back in 2020 with three goals:

1. Learn Julia well enough to confidently use it in ongoing projects at SINTEF _(a success! I now use Julia a lot, as do many of my colleagues.)_
2. Explore the potential for fast automatic differentiation of PDEs using the many AD packages in the Julia ecosystem[^1] _(definitely a success, even if we had to write a lot of code to get there)_
3. Assess the potential for "write once, execute everywhere" GPU/CPU parallelism that was emerging through the the nascent `KernelAbstractions` package[^2] _(...it took six years)_

The initial version of the code contained AD solves for single and multiphase immiscible flow without any wells running on both CPU and GPU. Back in 2020, writing and testing kernels required a working GPU device, and the developer experience for kernel programming was very rough around the edges, with incomprehensible error messages and frequent crashes when launching kernels that gave errors. I made the decision to instead focus on differentiablity and high performance on the CPU. In the main paper ["JutulDarcy.jl - a fully differentiable high-performance reservoir simulator based on automatic differentiation"](https://link.springer.com/article/10.1007/s10596-025-10366-6), the GPU results are limited to the linear solvers for NVIDIA devices via CuSPARSE/AMGX vendor libraries.

### Array-based programming

I spent many years writing efficient MATLAB code. If I permit a generalization, in MATLAB, execution of user code is slow, but the compiler can vectorize many operations to call compiled libraries. There has been improvements to MATLAB's JIT compiler over the years, but for the longest time the following two code snippets would differ in runtime with a huge number of magnitude:

#### Looping MATLAB

```matlab
a = rand(N, N)
c = zeros(N, N)
for i = 1:N
    for j = 1:N
        c(i, j) = a(i, j)^2
    end
end
```

#### Vectorized MATLAB

```matlab
a = rand(N, N)
% Vectorized code
b = a.^2
```

Writing fast MATLAB code was an exercise of:

1. Avoiding loop.
2. If you cannot avoid a loop, loop over the smallest index
3. Angrily write a MEX C-extension to get a fast loop when I could not do 1 and 2.

Julia (and most other compiled languages) allows you to write fast loops, and you are generally free to use either loops or vectorized expressions based on personal preference. For me, the flexibility of Julia has been a joy to work with -- towards the end of my post-doc in 2018 I was spending far too much on point 3 in the above list over actually writing application code.

#### GPU programming in Julia

Going to GPU programming, however, means that you are not allowed to do scalar indexing on the CPU if you want fast code. In practical terms, this means that you either write kernels that perform the same bit of code many times in parallel, or you use vectorized functions on GPU-resident arrays to execute. It is was a bit of a "back to the future"-moment where I had to go back to the old MATLAB mental model for what code was performance safe.

GPUs have a lot of threads and memory bandwidth. One of the things GPUs are really good at is parallel processing on arrays that contain "number-like" things. In the Julia world, these are referred to as `isbitstypes`, which are immutable types that have a fixed size in memory. GPUs are not so good at execute heavily branching logic, allocating memory during execution or manage complex data types that have variable size in memory. The design of the code was written with these restrictions of GPUs in mind.


### A small example
As an example, consider how we simulate geological sequestration of CO2.




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
