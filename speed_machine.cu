/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <stdlib.h>
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "helper_cuda.h"


typedef unsigned short int uint_16; // Not exactly platform-independent
typedef short int int_16;

// Transitions are indexed by their "condition number".
// The condition number is condition_state * MAX_SYMBOLS + condition_symbol
typedef struct Transition {
	//uint_16 condition_state;
	//uint_16 condition_symbol;
	uint_16 print_symbol;
	int_16 movement; // -1 for left, 1 for right
	uint_16 target_state;
} Transition;

typedef struct MachineLocation {
	uint_16 base_transition_index;
	uint_16 num_transitions;
};



// For now TAPE_SIZE and MAX_STEPS are equal.
// This could be done differently, but this makes it easy
// to avoid going off the high end of the tape.
#define TAPE_SIZE 100
#define MAX_STEPS TAPE_SIZE

// For easy lookup, we want to keep the transitions in a table that acts as
// a map. By keeping these numbers fixed and small we can implement that
// table efficiently.
#define MAX_STATES 8
#define MAX_SYMBOLS 8

#define NUM_POSSIBLE_CONDITIONS (MAX_STATES * MAX_SYMBOLS)

#define CONDITION(state, symbol) ((state) * MAX_SYMBOLS + (symbol))


/**
* CUDA Kernel Device code
*
*/
__global__ void
simulate_machines(const MachineLocation *machine_locations,
		  const Transition *transitions_block,
		  uint_16 *tape_block,
		  int num_machines) {

	int machine_index = blockDim.x * blockIdx.x + threadIdx.x;

	// Don't run code with more cores than there are problem instances.
	if (machine_index < num_machines) {

		// Grab a reference to the MachineLocation.
		MachineLocation machine_location = machine_locations[machine_index];

		// Grab a reference to the base of the transitions array.
		const Transition *transitions =
				transitions_block + machine_location.base_transition_index;

		// Save the number of total transitions.
		uint_16 num_transitions = machine_location.num_transitions;

		// Save a reference to the start of this machine's section of tape.
		uint_16 *tape = tape_block + (machine_index * TAPE_SIZE);

		// Set the initial values.
		uint_16 state = 0;
		int tape_position = 0;

		uint_16 symbol = tape[0];

		int valid = 1; // opposite of "halted"
		int diff;

		Transition transition;

		for (int i = 0; i < MAX_STEPS; ++i) {

			// Lookup the symbol.
			symbol = tape[tape_position];

			// Lookup the transition.
			transition = transitions[CONDITION(state, symbol)];
			
			// Check if the transition exists, if not then halt.
			valid = valid && (transition.movement != 0);

			// Print the symbol.
			diff = transition.print_symbol - symbol;
			tape[tape_position] += valid * diff; // If valid, then change symbol, else remain.

			// Move the tape head.
			tape_position += transition.movement;
			valid = valid && (tape_position >= 0);

			// Transition to the next state.
			diff = transition.target_state - state;
			state += valid * diff;

			// Loop around for the next rule.
		}
	}
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	int num_machines = 1280; // Max cores for GTX 1060
	//int transitions_per_machine = 6; // lots

	// Allocate space for the MachineLocations
	MachineLocation *machine_locations =
			(MachineLocation *) malloc(num_machines * sizeof(MachineLocation));
	if (machine_locations == NULL) {
		fprintf(stderr, "malloc returned NULL for machine_locations.");
		exit(EXIT_FAILURE);
	}

	Transition *transitions_block =
			(Transition *) malloc(num_machines * NUM_POSSIBLE_CONDITIONS * sizeof(Transition));
	if (transitions_block == NULL) {
		fprintf(stderr, "malloc returned NULL for transitions_block.");
		exit(EXIT_FAILURE);
	}

	uint_16 *tape_block =
			(uint_16 *) calloc(num_machines * TAPE_SIZE, sizeof(uint_16));
	if (tape_block == NULL) {
		fprintf(stderr, "malloc returned NULL for tape_block.");
		exit(EXIT_FAILURE);
	}

	// initialize machines

	for (int i = 0; i < num_machines; ++i) {
		MachineLocation *machine_location = &machine_locations[i];
		machine_location->base_transition_index = i * NUM_POSSIBLE_CONDITIONS;
		machine_location->num_transitions = NUM_POSSIBLE_CONDITIONS;

		Transition *transition;
		
		transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(0, 0)];
		//transition->condition_state = 0;
		//transition->condition_symbol = 0;
		transition->print_symbol = 1;
		transition->movement = 1;
		transition->target_state = 1;

		transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(1, 0)];
		//transition->condition_state = 1;
		//transition->condition_symbol = 0;
		transition->print_symbol = 2;
		transition->movement = 1;
		transition->target_state = 0;
	}

	// Clear the host tape with zeros.

	MachineLocation *device_machine_locations = NULL;
	err = cudaMalloc((void **)&device_machine_locations, num_machines * sizeof(MachineLocation));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for machine_locations.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	Transition *device_transitions_block = NULL;
	err = cudaMalloc((void **)&device_transitions_block, num_machines * NUM_POSSIBLE_CONDITIONS * sizeof(Transition));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for transitions_block.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	uint_16 *device_tape_block = NULL;
	err = cudaMalloc((void **)&device_tape_block, num_machines * TAPE_SIZE * sizeof(uint_16));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for tape_block.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(device_machine_locations, machine_locations,
			num_machines * sizeof(MachineLocation), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy machine_locations to device. Error code %s.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(device_transitions_block, transitions_block,
			num_machines * NUM_POSSIBLE_CONDITIONS * sizeof(Transition), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy transitions_block to device. Error code %s.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(device_tape_block, tape_block,
			num_machines * TAPE_SIZE * sizeof(uint_16), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy tape_block to device. Error code %s.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	for (int batch = 0; batch < 1000; ++batch) {

		// Launch the Vector Add CUDA Kernel
		int threadsPerBlock = 256;
		int blocksPerGrid = (num_machines + threadsPerBlock - 1) / threadsPerBlock;
		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
		simulate_machines <<<blocksPerGrid, threadsPerBlock>>>
			(device_machine_locations,
				device_transitions_block,
				device_tape_block,
				num_machines);
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch simulate_machines kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		else {
			printf("Simulated batch #%i\n", batch + 1);
		}
	}

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(tape_block, device_tape_block, num_machines * TAPE_SIZE * sizeof(uint_16), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
	/*
    for (int i = 0; i < num_machines; ++i)
    {
		printf("%i: %i,%i,%i,%i,%i,%i,%i,%i\n", i,
			(int)tape_block[i * TAPE_SIZE + 0], (int)tape_block[i * TAPE_SIZE + 1],
			(int)tape_block[i * TAPE_SIZE + 2], (int)tape_block[i * TAPE_SIZE + 3],
			(int)tape_block[i * TAPE_SIZE + 4], (int)tape_block[i * TAPE_SIZE + 5],
			(int)tape_block[i * TAPE_SIZE + 6], (int)tape_block[i * TAPE_SIZE + 7]);
    }
	*/

    // Free device global memory
    err = cudaFree(device_machine_locations);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for machine_locations.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(device_transitions_block);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for transitions_block.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(device_tape_block);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory for tape_block.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(machine_locations);
    free(transitions_block);
    free(tape_block);

    printf("Done\n");
    return 0;
}

