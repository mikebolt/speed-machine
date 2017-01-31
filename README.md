# Speed Machine - Fast GPGPU Turing Machine simulation.

This project lets you simulate lots of Turing Machines quickly on your GPU using CUDA.

### Limitations

- Every machine has limited memory, and if you want to run N machines in parallel then you need to have (N * tape_size) memory available to your GPU.
- The number of steps must be pre-determined. The machines are incapable of stopping when they halt, and they will always stop after the pre-determined number of steps even if the machine hasn't "halted".
- All states and symbols are represented as 16-bit unsigned integers (unsigned short int).
- There is no "print nothing" symbol yet.
- The starting state is always 0.
- The transition functions requires a lookup table witha size proportional to (# states * # symbols).
- I have only tested this with 1 very simple machine.
- You need Visual Studio with the CUDA toolkit and a CUDA-capable graphics card.
- You might have to mess with the project settings to get this to work on your computer.

### Notes

The movement ("L" or "R") is represented as a 16-bit signed integer (short int) and you don't have to limit it to -1 for L or +1 for R. You can have a movement of 103 steps to the right represented as +103.

You can simulate "input" by initializing the tape with your input symbols.

This thing is fast. I have a GTX 1060 and an i7-6700. Here are the times for running 1,280,000 machines, based on the number of steps per machine:

- 10 steps: 0.531 seconds, 41.484 nanoseconds per step

- 100 steps: 0.953 seconds, 7.445 nanoseconds per step

- 1000 steps: 5.016 seconds, 3.918 nanoseconds per step

- 10000 steps: 42.031 seconds, 3.283 nanoseconds per step

- 100000 steps: 410.531 seconds, 3.207 nanoseconds per step

Note that the number of tape cells is equal to the number of steps in these tests. This means that the slowdowns may be due to an increase in required memory.

For comparison, a single-threaded python implementation takes about 572 nanoseconds per step.
