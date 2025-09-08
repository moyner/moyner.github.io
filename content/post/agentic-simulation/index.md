---
title: Agentic reservoir simulation with JutulGPT
description: Using a agentic workflow to set up complex simulation models
slug: agentic-simulation
date: 2025-09-08
image: header.png
categories:
    - Julia
    - JutulDarcy
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

## Motivation

*This blog post is based on work together with Elling Svee (who did most of the coding!), Knut-Andreas Lie and Jakob Torben. [code is available on GitHub](https://github.com/ellingsvee/JutulGPT).*

There is a fair bit of hype around large language model (LLM) agents at the moment, and discussions about what they can be used for. LLM-based agents are, in the broadest sense, essentially LLMs that are given a set of tools that can be used to perform tasks. Tools are typically a function call together with metadata (description of usage, purpose, etc) and allows a LLM program to perform tasks by picking (or guessing) the appropriate tool when a text response is insufficient.

At the Applied Computational Science group, we work extensively with numerical simulation of physical processes. We are developing open source software for reservoir simulation (e.g. MRST, OPM Flow and JutulDarcy.jl) as well as software for battery simulation and many other applications. In this post, we will limit the discussion to reservoir simulation, but most of the lessions apply to other types of modelling as well.

We are interested in exploring how agents can be used to set up and interact with complex simulation models, examplified by a reservoir simulator. We can break down a basic reservoir simulation workflow into four steps:

1. Define the model itself: Setting up a mesh, populating it with static properties (e.g. permeability/porosity), defining a fluid model and setting up wells, aquifers and other types of boundary conditions.
2. Set up a scenario to simulate: Pick operating controls/limits for the wells, time-steps to solve before simulating with the approprioate numerical tolerances, linear solver and solution strategy.
3. Perform analysis on the results, for example by computing the total thermal energy produced over a given number of years, examining the amount of CO2 stored in a CCS model, and so on.
4. Based on the analysis and goal, go back and redo steps 1-3 to improve the setup by introducing additional features, increasing grid resolution, adding tracers, moving wells, etc.

Reservoir simulators are complex pieces of software, and iterating between these four steps requires detailed knowledge of reservoir engineering (*what physical process is being modelled?*), reservoir simulation expertise (*what is the right choice of physics and resolution to capture the process?*) and software know-how (*how do I make the available software do what I want?*). Each topic can be a daunting field of study in itself, so there exists a substantial challenge when it comes to making these modelling workflows accessible, especially for experts who want to test a new code for an existing application. This workflow sketch is also intentionally a bit simplistic, and most experienced reservoir simulation readers will spot some missing details (where are the ensembles? What about seismic interpretation? Where do you get the fluid data? What method do you use for populating the properties?).

I have personally no expectation that LLMs are going to be replacing "PhD level intelligence" any time soon - but they seem to have potential as a research partner that can perform time-consuming tasks and condense information from documentation and technical sources very quickly. The goal is not to stop writing code, but to spend the time on the code that really matters. For this reason, we have an internal activity targeted towards use of LLMs with simulators. The goal is to allow a LLM to help with setting up simulations, looking up documentation and performing task-specific coding in steps 1 to 4 above.

For more information on the high level goals of this activity, have a look at the SINTEF page on [Discussing with your simulator](https://www.sintef.no/en/digital/departments/department-of-mathematics-and-cybernetics/research-group-applied-computational-science/discussing-with-your-simulator/).

## Building an agent for reservoir simulation

The best way to get started with technology is to build something, so this year we allocated one of our many talented summer student towards building a prototype agent for reservoir simulation. Up front, we knew that two months were unlikely to build the agents of our dreams, so the scope was initially limited to steps 1/2, i.e. setting up simulation models rather than analysis of the results.

### Picking a code

We also had the issue of *what* code was to be used. We have three open source reservoir simulators that were candidates:

- MRST is a MATLAB-based toolbox for reservoir simulation and uses scripts to set up models, and has been around for a long time with a large number of different modules
- OPM Flow is a C++ based reservoir simulator that is run by industry standard input files
- JutulDarcy.jl is a Julia-based reservoir simulator that also uses scripts as the primary setup

We considered working directly on the industry standard .DATA files, as all three codes can set up models this way, but these files are both enormously complex, and are limited to the model setup as analysis is usually done in another language or GUI anyway. We selected JutulDarcy.jl for two reasons:

1. JutulDarcy.jl is easy to call from within an agent framework, since Julia is pretty easy to both install and call in a virtual machine.
2. MRST has a large number of modules that have been written over the last 15 years, with code styles, interfaces and documentation that varies from module to module. Although the wide feature set is very valuable to researchers, it is less useful to a LLM, which has a limited context window.

### Architecture

After a bit of testing with smaller tools (like LLMCall.jl), we ended up using [LangChain](https://github.com/langchain-ai/langchain) as the framework. Elling made two front-ends: One that is a terminal

The [code is available on GitHub](https://github.com/ellingsvee/JutulGPT) for those who want to dig into the details.

## Notes

- Locally hosted LLMs?
- Should you build your own agent setup? Maybe not. Example from Jakob
- MPI
