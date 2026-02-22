/**
 * perf_contractions.c — Timed benchmarks for MPO-MPO, MPS-MPO inner product,
 * MPS-MPO apply, and MPS-MPS contractions. Writes one JSONL record per operation
 * for comparison with Mojo. No HDF5; builds Ising MPO and random MPS in code.
 *
 * Uses CT_SINGLE_REAL (float) for MPO and MPS so results match Mojo float32.
 * The Ising MPO is built via the normal API then converted to single precision
 * here (no changes to src/).
 *
 * Reads shared config from ../../bench_config.json (relative to build/).
 * CLI args override config values.
 *
 * Usage: perf_contractions [config_path] [nsites] [d] [chi_max]
 *   config_path  default "../../bench_config.json"
 *   nsites       override from config
 *   d            override from config
 *   chi_max      override from config
 *
 * Output: generated/perf/contraction_timings_{nsites}_{d}_{chi_max}.jsonl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hamiltonian.h"
#include "chain_ops.h"
#include "mpo.h"
#include "mps.h"
#include "dense_tensor.h"
#include "aligned_memory.h"
#include "rng.h"
#include "timing.h"

#define DEFAULT_CONFIG_PATH "../../bench_config.json"
#define OUTPUT_DIR "generated/perf"

/** Minimal JSON config reader: extracts integer value for a given key. Returns -1 on miss. */
static long json_read_int(const char* json, const char* key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char* pos = strstr(json, pattern);
	if (!pos) return -1;
	pos += strlen(pattern);
	/* skip whitespace and colon */
	while (*pos == ' ' || *pos == '\t' || *pos == ':') pos++;
	return atol(pos);
}

/** Read bench_config.json into nsites/d/chi_max/num_runs. Returns 0 on success. */
static int read_bench_config(const char* path, int* nsites, long* d, long* chi_max, int* num_runs)
{
	FILE* f = fopen(path, "r");
	if (!f) return -1;
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (sz <= 0 || sz > 4096) { fclose(f); return -1; }
	char* buf = malloc((size_t)sz + 1);
	if (!buf) { fclose(f); return -1; }
	fread(buf, 1, (size_t)sz, f);
	buf[sz] = '\0';
	fclose(f);

	long v;
	v = json_read_int(buf, "nsites");   if (v > 0) *nsites   = (int)v;
	v = json_read_int(buf, "d");        if (v > 0) *d        = v;
	v = json_read_int(buf, "chi_max");  if (v > 0) *chi_max  = v;
	v = json_read_int(buf, "num_runs"); if (v > 0) *num_runs = (int)v;
	free(buf);
	return 0;
}

/** Return non-zero if the string is an integer (optional sign, digits). */
static int is_number(const char* s)
{
	if (!s || !*s) return 0;
	if (*s == '-' || *s == '+') s++;
	if (!*s) return 0;
	while (*s) {
		if (*s < '0' || *s > '9') return 0;
		s++;
	}
	return 1;
}

static void write_jsonl_record(FILE* fp, const char* operation, int nsites, long d, long chi_max,
                               double time_seconds, double result, int runs)
{
	time_t now = time(NULL);
	fprintf(fp, "{\"backend\":\"c\",\"operation\":\"%s\",\"params\":{\"nsites\":%d,\"d\":%ld,\"chi_max\":%ld},"
	            "\"time_seconds\":%.9g,\"result\":%.15g,\"runs\":%d,\"timestamp\":%ld}\n",
	        operation, nsites, d, chi_max, time_seconds, result, runs, (long)now);
}

/** Convert assembly from double to single precision in-place (opmap and coeffmap only). */
static void assembly_convert_to_single_real(struct mpo_assembly* assembly)
{
	const int n = assembly->num_local_ops;
	const long d = assembly->d;
	const long nelem = d * d;
	struct dense_tensor* new_opmap = ct_malloc((size_t)n * sizeof(struct dense_tensor));
	const long dim[2] = { d, d };
	for (int i = 0; i < n; i++) {
		allocate_dense_tensor(CT_SINGLE_REAL, 2, dim, &new_opmap[i]);
		const double* src = assembly->opmap[i].data;
		float* dst = new_opmap[i].data;
		for (long k = 0; k < nelem; k++)
			dst[k] = (float)src[k];
	}
	float* new_coeffmap = ct_malloc((size_t)assembly->num_coeffs * sizeof(float));
	const double* old_coeff = assembly->coeffmap;
	for (int i = 0; i < assembly->num_coeffs; i++)
		new_coeffmap[i] = (float)old_coeff[i];
	for (int i = 0; i < n; i++)
		delete_dense_tensor(&assembly->opmap[i]);
	ct_free(assembly->opmap);
	ct_free(assembly->coeffmap);
	assembly->opmap = new_opmap;
	assembly->coeffmap = new_coeffmap;
	assembly->dtype = CT_SINGLE_REAL;
}

int main(int argc, char** argv)
{
	int nsites = 6;
	long d = 2;
	long chi_max = 16;
	int num_runs = 3;
	const char* config_path = DEFAULT_CONFIG_PATH;

	/* CLI: optional config path first (non-numeric), else legacy positional nsites/d/chi_max */
	int argi = 1;
	if (argc >= 2 && !is_number(argv[1])) {
		config_path = argv[1];
		argi = 2;
	}

	/* Read shared config file (defaults if missing) */
	if (read_bench_config(config_path, &nsites, &d, &chi_max, &num_runs) == 0) {
		printf("Loaded config from %s: nsites=%d, d=%ld, chi_max=%ld, num_runs=%d\n",
		       config_path, nsites, d, chi_max, num_runs);
	} else {
		printf("Config %s not found, using defaults: nsites=%d, d=%ld, chi_max=%ld, num_runs=%d\n",
		       config_path, nsites, d, chi_max, num_runs);
	}

	/* Positional overrides (legacy): nsites d chi_max */
	if (argc > argi) nsites  = atoi(argv[argi++]);
	if (argc > argi) d       = (long)atol(argv[argi++]);
	if (argc > argi) chi_max = (long)atol(argv[argi++]);

	if (nsites < 2 || d < 1 || chi_max < 1) {
		fprintf(stderr, "usage: perf_contractions [config_path] [nsites] [d] [chi_max]\n");
		return -1;
	}

	/* Build parameterized output path: contraction_timings_{nsites}_{d}_{chi_max}.jsonl */
	char out_path[512];
	snprintf(out_path, sizeof(out_path), "%s/contraction_timings_%d_%ld_%ld.jsonl",
	         OUTPUT_DIR, nsites, d, chi_max);

	FILE* out = fopen(out_path, "a");
	if (!out) {
		fprintf(stderr, "perf_contractions: cannot open %s for append\n", out_path);
		return -1;
	}

	const double ticks_per_sec = (double)get_tick_resolution();

	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	for (long i = 0; i < d; i++) qsite[i] = 0;

	/* Ising MPO in single precision (match Mojo float32) */
	struct mpo_assembly assembly;
	construct_ising_1d_mpo_assembly(nsites, 1.0, 0.0, 0.0, &assembly);
	assembly_convert_to_single_real(&assembly);
	struct mpo mpo;
	mpo_from_assembly(&assembly, &mpo);
	delete_mpo_assembly(&assembly);
	if (!mpo_is_consistent(&mpo)) {
		fprintf(stderr, "perf_contractions: MPO consistency failed\n");
		fclose(out);
		ct_free(qsite);
		return -1;
	}

	/* Random MPS (psi, chi) in single precision */
	struct rng_state rng;
	seed_rng_state(42, &rng);
	struct mps psi;
	construct_random_mps(CT_SINGLE_REAL, nsites, d, qsite, 0, chi_max, &rng, &psi);
	struct mps chi;
	seed_rng_state(43, &rng);
	construct_random_mps(CT_SINGLE_REAL, nsites, d, qsite, 0, chi_max, &rng, &chi);
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

	/* 2) MPS-MPO inner product <chi|op|psi> (result is float for CT_SINGLE_REAL) */
	{
		float s_inner = 0.0f;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++)
			mpo_inner_product(&chi, &mpo, &psi, &s_inner);
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		printf("mps_mpo_inner: %.6f s (result=%.9g)\n", sec, (double)s_inner);
		write_jsonl_record(out, "mps_mpo_inner", nsites, d, chi_max, sec, (double)s_inner, num_runs);
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

	/* 4) MPS-MPS overlap <chi|psi> (result is float for CT_SINGLE_REAL) */
	{
		float s_mps = 0.0f;
		uint64_t t0 = get_time_ticks();
		for (int r = 0; r < num_runs; r++)
			mps_vdot(&chi, &psi, &s_mps);
		uint64_t t1 = get_time_ticks();
		double sec = (t1 - t0) / ticks_per_sec / num_runs;
		printf("mps_mps: %.6f s (result=%.9g)\n", sec, (double)s_mps);
		write_jsonl_record(out, "mps_mps", nsites, d, chi_max, sec, (double)s_mps, num_runs);
	}

	delete_mps(&chi);
	delete_mps(&psi);
	delete_mpo(&mpo);
	ct_free(qsite);
	fclose(out);
	printf("Appended 4 records to %s\n", out_path);
	return 0;
}
