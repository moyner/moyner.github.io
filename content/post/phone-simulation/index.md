---
title: Running a reservoir simulation on my phone
description: Big data? No, small cellphone.
slug: phone-simulation
date: 2025-07-31
image: header.png
categories:
    - Julia
    - JutulDarcy
weight: 1       # You can add weight to some posts to override the default sorting (date descending)
---

The trend has been to move compute heavy workloads over into the cloud, but today I did the opposite and put it in my pocket! Using the new [Android 16 Linux terminal](https://www.androidpolice.com/android-15-linux-terminal-app/) on my Pixel, I’m running a reservoir simulation on my phone 😎

The Linux terminal is a proper Debian setup, so I naturally installed julia and JutulDarcy to test the computational performance. The setup was fairly straightforward: Install [juliaup](https://github.com/JuliaLang/juliaup) and add the latest release. I then added JutulDarcy and HYPRE to the default environment. The precompilation crashed towards the end of the setup - perhaps due to the screen going to sleep - but once I restarted Julia, everything ran nicely.

On a small SPE9 case, it is surprisingly performant with a runtime comparable to that of a laptop. This case only has 9000 cells and ~18k degrees of freedom, so I also ran the larger OLYMPUS 1 case (~200k cells, ~400k degrees of freedom), runtime was roughly 2x that of my laptop when both devices were using a single thread.

I don’t foresee that I’m going to switch to my phone for running simulations any time soon, but it is impressive to see how powerful phone CPUs have gotten. ARM support is also pretty good in Julia, probably a side effect of macOS support that now also uses ARM. If you want to get reasonable performance you will have to keep the screen on and the Linux app open - it appears that the Android process manager heavily priortizes the active application.

{{<video library="1" src="pixel-run.mp4" controls="yes">}}
