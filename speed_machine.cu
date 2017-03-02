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

#include <Windows.h>

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
#define TAPE_SIZE 200
#define MAX_STEPS TAPE_SIZE

// For easy lookup, we want to keep the transitions in a table that acts as
// a map. By keeping these numbers fixed and small we can implement that
// table efficiently.
#define MAX_STATES 8
#define MAX_SYMBOLS 4

#define NUM_TRANSITIONS 32

#define NUM_POSSIBLE_CONDITIONS (MAX_STATES * MAX_SYMBOLS)

#define CONDITION(state, symbol) ((state) * MAX_SYMBOLS + (symbol))

#define EXAMPLE_SIZE 45


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
			tape_position += valid * transition.movement;
			valid = valid && (tape_position >= 0);

			// Transition to the next state.
			diff = transition.target_state - state;
			state += valid * diff;

			// Loop around for the next rule.
		}
	}
}


void simulate_machines_cpu(const MachineLocation *machine_locations,
		const Transition *transitions_block,
		uint_16 *tape_block,
		int num_machines) {

	int machine_index;

	for (machine_index = 0; machine_index < num_machines; ++machine_index) {

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
			tape_position += valid * transition.movement;
			valid = valid && (tape_position >= 0);

			// Transition to the next state.
			diff = transition.target_state - state;
			state += valid * diff;

			// Loop around for the next rule.
		}
	}
}

void allocate_host_memory(int num_machines, MachineLocation **machine_locations,
	Transition **transitions_block, uint_16 **tape_block);
void create_example_machines(int num_machines, MachineLocation *machine_locations,
	Transition *transitions_block, uint_16 *tape_block);
void generate_random_machines(int num_machines, int num_transitions, MachineLocation *machine_locations,
	Transition *transitions_block, uint_16 *tape_block);
void generate_random_connected_machines(int num_machines, int num_transitions, MachineLocation *machine_locations,
	Transition *transitions_block, uint_16 *tape_block);
void allocate_device_memory(int num_machines, MachineLocation **device_machine_locations,
	Transition **device_transitions_block, uint_16 **device_tape_block);
void simulate_machines(int num_machines, MachineLocation *machine_locations, MachineLocation *device_machine_locations,
	Transition *transitions_block, Transition *device_transitions_block, uint_16 *tape_block, uint_16 *device_tape_block);
void copy_memory_to_device(int num_machines, MachineLocation *machine_locations, MachineLocation *device_machine_locations,
	Transition *transitions_block, Transition *device_transitions_block, uint_16 *tape_block, uint_16 *device_tape_block);
void free_device_memory(MachineLocation *device_machine_locations,
	Transition *device_transitions_block, uint_16 *device_tape_block);
void free_host_memory(MachineLocation *machine_locations,
	Transition *transitions_block, uint_16 *tape_block);

/**
 * Host main routine
 */
int
main(void)
{

	srand(22); // arbitrary

	uint_16 example[EXAMPLE_SIZE] = { 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0 };

	int num_machines = 1280; // Max cores for GTX 1060

	MachineLocation *machine_locations = NULL;
	Transition *transitions_block = NULL;
	uint_16 *tape_block = NULL;

	allocate_host_memory(num_machines, &machine_locations, &transitions_block, &tape_block);

	MachineLocation *device_machine_locations = NULL;
	Transition *device_transitions_block = NULL;
	uint_16 *device_tape_block = NULL;

	allocate_device_memory(num_machines, &device_machine_locations, &device_transitions_block, &device_tape_block);

	DWORD before = GetTickCount();
	int num_batches = 10000; // 24750
	// CPU: 2281ms for 1280000 (single core)
	// GPU: 656ms for 1280000

	int best_distance = TAPE_SIZE + 1;
	MachineLocation best_machine;

	for (int batch = 0; batch < num_batches; ++batch) {

		// We want to generate a new set of machines for each batch. Very redundant otherwise.
		generate_random_connected_machines(num_machines, NUM_TRANSITIONS, machine_locations, transitions_block, tape_block);

		simulate_machines(num_machines, machine_locations, device_machine_locations,
			transitions_block, device_transitions_block, tape_block, device_tape_block);

		//simulate_machines_cpu(machine_locations, transitions_block, tape_block, num_machines);
		
		//printf("Simulated batch #%i\n", batch + 1);

		for (int i = 0; i < num_machines; ++i)
		{

			// TODO: this part is probably the bottleneck now.
			// I should time it and quantify that claim.
			// If it is the bottleneck, then multithreading and/or SIMD could speed it up.
			// I could also send it to the GPU. Several batches of results could be gathered first, depending on
			// the memory constraints, so that the GPU program switching isn't prohibitive.
			int distance = 0;
			for (int j = 0; j < min(EXAMPLE_SIZE, TAPE_SIZE); ++j) {
				if (example[j] != tape_block[i * TAPE_SIZE + j]) {
					++distance;
				}
			}

			if (distance < best_distance) {
				best_distance = distance;
				best_machine = machine_locations[i];

				printf("Found a machine with a distance of %d\n", best_distance);

				printf("T={");

				for (int k = 0; k < NUM_POSSIBLE_CONDITIONS; ++k) {
					Transition transition = transitions_block[best_machine.base_transition_index + k];

					if (transition.movement != 0) {
						printf("(%d/%d->%d/%d/%d),",
								(int) (k / MAX_SYMBOLS), (int) (k % MAX_SYMBOLS),
								(int) transition.print_symbol,
								(int) transition.movement,
								(int) transition.target_state);
					}
				}

				printf("}\n");

				printf("Tape: [");
				for (int l = 0; l < EXAMPLE_SIZE; ++l) {
					printf("%d,", (int)tape_block[i * TAPE_SIZE + l]);
				}
				printf("]\n");
			}

			/*
			printf("%i: %i,%i,%i,%i,%i,%i,%i,%i\n", i,
				(int)tape_block[i * TAPE_SIZE + 0], (int)tape_block[i * TAPE_SIZE + 1],
				(int)tape_block[i * TAPE_SIZE + 2], (int)tape_block[i * TAPE_SIZE + 3],
				(int)tape_block[i * TAPE_SIZE + 4], (int)tape_block[i * TAPE_SIZE + 5],
				(int)tape_block[i * TAPE_SIZE + 6], (int)tape_block[i * TAPE_SIZE + 7]);
			*/
		}
		

		// The final tapes are stored in tape_block now, so clear it for the next iteration.
		memset(tape_block, 0, num_machines * TAPE_SIZE * sizeof(uint_16));

		// Same for the machines, because new ones will be generated.
		memset(transitions_block, 0, num_machines * NUM_POSSIBLE_CONDITIONS * sizeof(Transition));
	}

	DWORD after = GetTickCount();
	printf("Simulating %ld machines took %ld milliseconds.\n", (long)(num_batches * num_machines), (long)(after - before));

	free_device_memory(device_machine_locations, device_transitions_block, device_tape_block);

	// Free host memory
	free_host_memory(machine_locations, transitions_block, tape_block);

	return 0;
}

void allocate_host_memory(int num_machines, MachineLocation **machine_locations,
						  Transition **transitions_block, uint_16 **tape_block) {
	// Allocate space for the MachineLocations
	*machine_locations = (MachineLocation *)malloc(num_machines * sizeof(MachineLocation));
	if (*machine_locations == NULL) {
		fprintf(stderr, "malloc returned NULL for machine_locations.");
		exit(EXIT_FAILURE);
	}

	*transitions_block = (Transition *)calloc(num_machines * NUM_POSSIBLE_CONDITIONS, sizeof(Transition));
	if (*transitions_block == NULL) {
		fprintf(stderr, "malloc returned NULL for transitions_block.");
		exit(EXIT_FAILURE);
	}

	*tape_block = (uint_16 *)calloc(num_machines * TAPE_SIZE, sizeof(uint_16));
	if (*tape_block == NULL) {
		fprintf(stderr, "malloc returned NULL for tape_block.");
		exit(EXIT_FAILURE);
	}
}

void create_example_machines(int num_machines, MachineLocation *machine_locations,
							 Transition *transitions_block, uint_16 *tape_block) {
	// Initialize the machines.
	for (int i = 0; i < num_machines; ++i) {
		MachineLocation *machine_location = &machine_locations[i];
		machine_location->base_transition_index = i * NUM_POSSIBLE_CONDITIONS;
		machine_location->num_transitions = NUM_POSSIBLE_CONDITIONS;

		Transition *transition;

		transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(0, 0)];
		transition->print_symbol = 1;
		transition->movement = 1;
		transition->target_state = 1;

		transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(1, 0)];
		transition->print_symbol = 2;
		transition->movement = 1;
		transition->target_state = 0;
	}
}

// TODO: add a filter for useless machines
void generate_random_machines(int num_machines, int num_transitions, MachineLocation *machine_locations,
							  Transition *transitions_block, uint_16 *tape_block) {

	for (int i = 0; i < num_machines; ++i) {
		MachineLocation *machine_location = &machine_locations[i];
		machine_location->base_transition_index = i * NUM_POSSIBLE_CONDITIONS;
		machine_location->num_transitions = NUM_POSSIBLE_CONDITIONS;

		Transition *transition;

		for (int j = 0; j < num_transitions; ++j) {
			uint_16 condition_state = (uint_16) (rand() % MAX_STATES);
			uint_16 condition_symbol = (uint_16) (rand() % MAX_SYMBOLS);

			transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(condition_state, condition_symbol)];
			transition->print_symbol = (uint_16) (rand() % MAX_SYMBOLS);
			transition->movement = (int_16) ((rand() % 2 == 0) ? -1 : 1);
			transition->target_state = (uint_16) (rand() % MAX_STATES);
		}
	}
}


void generate_random_connected_machines(int num_machines, int num_transitions, MachineLocation *machine_locations,
										Transition *transitions_block, uint_16 *tape_block) {

	int *connected_states = (int *) malloc(MAX_STATES * sizeof(int));
	int *is_connected = (int *) malloc(MAX_STATES * sizeof(int));
	
	for (int i = 0; i < num_machines; ++i) {
		MachineLocation *machine_location = &machine_locations[i];
		machine_location->base_transition_index = i * NUM_POSSIBLE_CONDITIONS;
		machine_location->num_transitions = NUM_POSSIBLE_CONDITIONS;

		int num_connected_states = 1; // Start with only the initial source state.
		connected_states[0] = 0;

		for (int state = 0; state < MAX_STATES; ++state) {
			is_connected[state] = 0; // 0 means unconnected
		}
		is_connected[0] = 1;

		uint_16 condition_state = 0;

		for (int j = 0; j < num_transitions; ++j) {
			uint_16 condition_symbol = (uint_16)(rand() % MAX_SYMBOLS);
			
			// Store the transition information.
			Transition *transition = &transitions_block[i * NUM_POSSIBLE_CONDITIONS + CONDITION(condition_state, condition_symbol)];
			transition->print_symbol = (uint_16)(rand() % MAX_SYMBOLS);
			transition->movement = (int_16)((rand() % 2 == 0) ? -1 : 1);
			transition->target_state = (uint_16) (rand() % MAX_STATES);

			// Update the connected state information.
			if (!is_connected[transition->target_state]) {
				is_connected[transition->target_state] = 1;
				connected_states[num_connected_states] = transition->target_state;
				++num_connected_states;
			}

			condition_state = connected_states[rand() % num_connected_states];
		}
	}
}


void allocate_device_memory(int num_machines, MachineLocation **device_machine_locations,
						    Transition **device_transitions_block, uint_16 **device_tape_block) {
	cudaError_t err = cudaSuccess;

	*device_machine_locations = NULL;
	err = cudaMalloc((void **)device_machine_locations, num_machines * sizeof(MachineLocation));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for machine_locations.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	*device_transitions_block = NULL;
	err = cudaMalloc((void **)device_transitions_block, num_machines * NUM_POSSIBLE_CONDITIONS * sizeof(Transition));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for transitions_block.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	*device_tape_block = NULL;
	err = cudaMalloc((void **)device_tape_block, num_machines * TAPE_SIZE * sizeof(uint_16));
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for tape_block.\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


void simulate_machines(int num_machines, MachineLocation *machine_locations, MachineLocation *device_machine_locations,
		               Transition *transitions_block, Transition *device_transitions_block, uint_16 *tape_block, uint_16 *device_tape_block) {

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

	copy_memory_to_device(num_machines, machine_locations, device_machine_locations,
			transitions_block, device_transitions_block, tape_block, device_tape_block);

	// Launch the CUDA Kernel.
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_machines + threadsPerBlock - 1) / threadsPerBlock;
	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
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

    // Copy the device result tape_block to the host tape_block.
    err = cudaMemcpy(tape_block, device_tape_block, num_machines * TAPE_SIZE * sizeof(uint_16), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void copy_memory_to_device(int num_machines, MachineLocation *machine_locations, MachineLocation *device_machine_locations,
		Transition *transitions_block, Transition *device_transitions_block, uint_16 *tape_block, uint_16 *device_tape_block) {
	cudaError_t err = cudaSuccess;

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
}

void free_device_memory(MachineLocation *device_machine_locations,
		Transition *device_transitions_block, uint_16 *device_tape_block) {
	cudaError_t err = cudaSuccess;

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
}

void free_host_memory(MachineLocation *machine_locations,
		Transition *transitions_block, uint_16 *tape_block) {
	free(machine_locations);
	free(transitions_block);
	free(tape_block);
}

