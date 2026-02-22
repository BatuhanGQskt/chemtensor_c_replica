/**
 * perf_contractions.c — Timed benchmarks for MPO-MPO, MPS-MPO inner product,
 * MPS-MPO apply, and MPS-MPS contractions. Writes one JSONL record per operation
 * for comparison with Mojo. No HDF5; builds Ising MPO and random MPS in code.
 *
 * Usage: perf_contractions [nsites] [d] [chi_max] [output_path]
 *   nsites     default 6
 *   d          default 2
 *   chi_max    default 16
 *   output_path default "generated/perf/contraction_timings.jsonl" (append mode; run from build/)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hamiltonian.h"
#include "chain_ops.h"
#include "mpo.h"
#include "mps.h"
#include "aligned_memory.h"
#include "rng.h"
#include "timing.h"

static void write_jsonl_record(FILE* fp, const char* operation, int nsites, long d, long chi_max,
                               double time_seconds, double result, int runs)
{
	time_t now = time(NULL);
	fprintf(fp, "{\"backend\":\"c\",\"operation\":\"%s\",\"params\":{\"nsites\":%d,\"d\":%ld,\"chi_max\":%ld},"
	            "\"time_seconds\":%.9g,\"result\":%.15g,\"runs\":%d,\"timestamp\":%ld}\n",
	        operation, nsites, d, chi_max, time_seconds, result, runs, (long)now);
}

int main(int argc, char** argv)
{
	int nsites = 6;
	long d = 2;
	long chi_max = 16;
	const char* out_path = "generated/perf/contraction_timings.jsonl";

	if (argc >= 2) nsites = atoi(argv[1]);
	if (argc >= 3) d = (long)atol(argv[2]);
	if (argc >= 4) chi_max = (long)atol(argv[3]);
	if (argc >= 5) out_path = argv[4];

	if (nsites < 2 || d < 1 || chi_max < 1) {
		fprintf(stderr, "usage: perf_contractions [nsites] [d] [chi_max] [output_path]\n");
		return -1;
	}

	FILE* out = fopen(out_path, "a");
	if (!out) {
		fprintf(stderr, "perf_contractions: cannot open %s for append\n", out_path);
		return -1;
	}

	const double ticks_per_sec = (double)get_tick_resolution();
	const int num_runs = 3;

	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	for (long i = 0; i < d; i++) qsite[i] = 0;

	/* Ising MPO */
	struct mpo_assembly assembly;
	construct_ising_1d_mpo_assembly(nsites, 1.0, 0.0, 0.0, &assembly);
	struct mpo mpo;
	mpo_from_assembly(&assembly, &mpo);
	delete_mpo_assembly(&assembly);
	if (!mpo_is_consistent(&mpo)) {
		fprintf(stderr, "perf_contractions: MPO consistency failed\n");
		fclose(out);
		ct_free(qsite);
		return -1;
	}

	/* Random MPS (psi, chi for inner products) */
	struct rng_state rng;
	seed_rng_state(42, &rng);
	struct mps psi;
	construct_random_mps(CT_DOUBLE_REAL, nsites, d, qsite, 0, chi_max, &rng, &psi);
	struct mps chi;
	seed_rng_state(43, &rng);
	construct_random_mps(CT_DOUBLE_REAL, nsites, d, qsite, 0, chi_max, &rng, &chi);
	if (!mps_is_consistent(&psi) || !mps_is_consistent(&chi)) {
		fprintf(stderr, "perf_contractions: MPS consistency failed\n");
		delete_mps(&psi);
		delete_mps(&chi);
		delete_mpo(&mpo);
		fclose(out);
		ct_free(qsite);
		return -1;
	}

	/* 1) MPO-MPO: time mpo_to_matrix (full contraction) */
	{
		struct block_sparse_tensor mat;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++) {
			mpo_to_matrix(&mpo, &mat);
			if (r < num_runs - 1) delete_block_sparse_tensor(&mat);
		}
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		double nrm = block_sparse_tensor_norm2(&mat);
		delete_block_sparse_tensor(&mat);
		printf("mpo_mpo: %.6f s (result=%.9g)\n", sec, nrm);
		write_jsonl_record(out, "mpo_mpo", nsites, d, chi_max, sec, nrm, num_runs);
	}

	/* 2) MPS-MPO inner product <chi|op|psi> */
	{
		double s = 0.0;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++)
			mpo_inner_product(&chi, &mpo, &psi, &s);
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		printf("mps_mpo_inner: %.6f s (result=%.9g)\n", sec, s);
		write_jsonl_record(out, "mps_mpo_inner", nsites, d, chi_max, sec, s, num_runs);
	}

	/* 3) MPS-MPO apply: op|psi> -> op_psi */
	{
		struct mps op_psi;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++) {
			apply_mpo(&mpo, &psi, &op_psi);
			if (r < num_runs - 1) delete_mps(&op_psi);
		}
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		double nrm = mps_norm(&op_psi);
		delete_mps(&op_psi);
		printf("mps_mpo_apply: %.6f s (result=%.9g)\n", sec, nrm);
		write_jsonl_record(out, "mps_mpo_apply", nsites, d, chi_max, sec, nrm, num_runs);
	}

	/* 4) MPS-MPS overlap <chi|psi> */
	{
		double s = 0.0;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++)
			mps_vdot(&chi, &psi, &s);
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		printf("mps_mps: %.6f s (result=%.9g)\n", sec, s);
		write_jsonl_record(out, "mps_mps", nsites, d, chi_max, sec, s, num_runs);
	}

	delete_mps(&chi);
	delete_mps(&psi);
	delete_mpo(&mpo);
	ct_free(qsite);
	fclose(out);
	printf("Appended 4 records to %s\n", out_path);
	return 0;
}
