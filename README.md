# simgen
Generates, compiles, and runs a C program to simulate recurrent neural networks in a configuration dependent manner.

## What is this for?

This is code I wrote to simulate rate-based recurrent neural networks for my doctoral dissertation.
I was studying the effects of many different plasticity rules (this included intrinsic, local synaptic, heterosynaptic, 
diffusion-based, and even full gradient-based rules), in networks with a variety of connectivity configurations and 
stimulus types. Because I was interested in the homeostatic and functional implications of such plasticity rules, the 
networks needed to be simulated for very long periods of time.

These requirements meant I needed a simulator that was extremely fast, and could be easily reconfigured according
to the plasticity rule(s) being used and the configuration of the network. I decided to write Python code that
would generate a C-based simulator (that uses the Intel Math Kernel Library for vectorisation), which would then be 
compiled and executed from Python using its CFFI module.

## How do I use it?

To see an example of the C code that `simgen` produces, you can just look in `./output/inner_impl.c` where you'll
find code that was generated by running `main.py`. If you want to generate the code yourself, feel free to choose
different `InterneuronPlasticity` and `PlasticitySubtype` values in the `main.py` file and run it. The new
simulator will be written into `./output/inner_impl.c`.

If you want the simulator to be automatically compiled and executed, you'll need to have the Intel MKL SDK installed 
on your computer.

## Why is the code so complicated?

The C code can appear complicated because it relies extensively on the Intel MKL, and unless one is familiar
with that library, calls to its API can look arcane. The Python code (mostly contained in `inner.py`) *is* complicated
because of the many different configurations of the simulator that I had to support. I apologize that it is
not well documented. It was written for my personal use, and was never intended for use by others. For examples of
clean, well-documented code, please see my other projects, e.g. [pubfig](https://github.com/owenmackwood/pubfig) or 
[connn](https://github.com/owenmackwood/connn). 
