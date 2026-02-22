/// \file rng_export.c
/// \brief Export RNG sequence for comparison with Mojo rng_c_compat.
///
/// Usage: rng_export [seed] [count]
///   seed  = RNG seed (default 42)
///   count = number of rand_uint32 and randnf values to output (default 64)
///
/// Writes to stdout a parseable format:
///   seed <seed>
///   n <count>
///   u32
///   <count lines: one uint32 per line>
///   randnf
///   <count lines: one float per line>
///
/// Mojo test reads this file and compares with rng_c_compat.mojo.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "rng.h"

int main(int argc, char** argv)
{
	uint64_t seed = 42;
	uint64_t n = 64;
	if (argc >= 2) {
		seed = (uint64_t)strtoull(argv[1], NULL, 0);
	}
	if (argc >= 3) {
		n = (uint64_t)strtoull(argv[2], NULL, 0);
	}

	struct rng_state state;
	seed_rng_state(seed, &state);

	printf("seed %llu\n", (unsigned long long)seed);
	printf("n %llu\n", (unsigned long long)n);
	printf("u32\n");
	for (uint64_t i = 0; i < n; i++) {
		printf("%u\n", rand_uint32(&state));
	}
	printf("randnf\n");
	for (uint64_t i = 0; i < n; i++) {
		printf("%.9g\n", (double)randnf(&state));
	}
	return 0;
}
