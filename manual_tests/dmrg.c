/// \file dmrg.c
/// \brief Manual DMRG tests — mirrors test/algorithm/test_dmrg.c but prints
///        results instead of comparing against HDF5 reference data.
///
/// Uses the Heisenberg XXZ model (built in code via create_heisenberg_xxz_1d_mpo_tensors)
/// so no external data files are needed.  Both single-site and two-site DMRG
/// are exercised with the same overall flow as the original unit test.
///
/// The output is designed to be directly comparable with the Mojo test at
///   chemtensor_mojo/src/tests/algorithms/test_dmrg.mojo
///
/// Consistency checks: normalized MPS (~1) and finite final energy (same for all configs).
/// Export JSON to generated/dmrg/ (or copy to Mojo test_data/) for C vs Mojo comparison.

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>

static int g_dmrg_tests_failed = 0;
static const char* DMRG_CONFIG_PATH = "../../bench_config.json";

struct dmrg_bench_params
{
	int nsites;
	long d;
	double J;
	double D;
	double h;
	long chi_max;
	int num_sweeps;
	int maxiter_lanczos;
	double tol_split;
};

static struct dmrg_bench_params g_cfg_singlesite = {
	.nsites = 7, .d = 2, .J = 1.0, .D = 1.0, .h = 0.0,
	.chi_max = 16, .num_sweeps = 6, .maxiter_lanczos = 25, .tol_split = -1.0
};
static struct dmrg_bench_params g_cfg_twosite = {
	.nsites = 11, .d = 2, .J = 1.0, .D = 0.5, .h = 0.2,
	.chi_max = 32, .num_sweeps = 4, .maxiter_lanczos = 25, .tol_split = 1e-10
};

static long json_read_int(const char* json, const char* key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char* pos = strstr(json, pattern);
	if (!pos) return -1;
	pos += strlen(pattern);
	while (*pos == ' ' || *pos == '\t' || *pos == ':') pos++;
	return atol(pos);
}

static double json_read_double(const char* json, const char* key)
{
	char pattern[128];
	snprintf(pattern, sizeof(pattern), "\"%s\"", key);
	const char* pos = strstr(json, pattern);
	if (!pos) return NAN;
	pos += strlen(pattern);
	while (*pos == ' ' || *pos == '\t' || *pos == ':') pos++;
	return strtod(pos, NULL);
}

static void load_dmrg_bench_config(void)
{
	FILE* f = fopen(DMRG_CONFIG_PATH, "r");
	if (!f) {
		printf("DMRG config %s not found, using defaults\n", DMRG_CONFIG_PATH);
		return;
	}
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (sz <= 0 || sz > 16384) {
		fclose(f);
		return;
	}
	char* buf = malloc((size_t)sz + 1);
	if (!buf) {
		fclose(f);
		return;
	}
	fread(buf, 1, (size_t)sz, f);
	buf[sz] = '\0';
	fclose(f);

	long v;
	double x;

	v = json_read_int(buf, "dmrg_singlesite_nsites");         if (v > 1) g_cfg_singlesite.nsites = (int)v;
	v = json_read_int(buf, "dmrg_singlesite_d");              if (v > 0) g_cfg_singlesite.d = v;
	v = json_read_int(buf, "dmrg_singlesite_chi_max");        if (v > 0) g_cfg_singlesite.chi_max = v;
	v = json_read_int(buf, "dmrg_singlesite_num_sweeps");     if (v > 0) g_cfg_singlesite.num_sweeps = (int)v;
	v = json_read_int(buf, "dmrg_singlesite_maxiter_lanczos");if (v > 0) g_cfg_singlesite.maxiter_lanczos = (int)v;
	x = json_read_double(buf, "dmrg_singlesite_J");           if (!isnan(x)) g_cfg_singlesite.J = x;
	x = json_read_double(buf, "dmrg_singlesite_D");           if (!isnan(x)) g_cfg_singlesite.D = x;
	x = json_read_double(buf, "dmrg_singlesite_h");           if (!isnan(x)) g_cfg_singlesite.h = x;

	v = json_read_int(buf, "dmrg_twosite_nsites");            if (v > 1) g_cfg_twosite.nsites = (int)v;
	v = json_read_int(buf, "dmrg_twosite_d");                 if (v > 0) g_cfg_twosite.d = v;
	v = json_read_int(buf, "dmrg_twosite_chi_max");           if (v > 0) g_cfg_twosite.chi_max = v;
	v = json_read_int(buf, "dmrg_twosite_num_sweeps");        if (v > 0) g_cfg_twosite.num_sweeps = (int)v;
	v = json_read_int(buf, "dmrg_twosite_maxiter_lanczos");   if (v > 0) g_cfg_twosite.maxiter_lanczos = (int)v;
	x = json_read_double(buf, "dmrg_twosite_J");              if (!isnan(x)) g_cfg_twosite.J = x;
	x = json_read_double(buf, "dmrg_twosite_D");              if (!isnan(x)) g_cfg_twosite.D = x;
	x = json_read_double(buf, "dmrg_twosite_h");              if (!isnan(x)) g_cfg_twosite.h = x;
	x = json_read_double(buf, "dmrg_twosite_tol_split");      if (!isnan(x)) g_cfg_twosite.tol_split = x;

	free(buf);
}

static int ensure_parent_dirs_timing(const char* path)
{
	if (!path || path[0] == '\0')
		return 0;

	char buf[4096];
	size_t n = strlen(path);
	if (n >= sizeof(buf))
		return -1;
	memcpy(buf, path, n + 1);

	char* last_slash = strrchr(buf, '/');
	if (!last_slash)
		return 0;
	if (last_slash == buf)
		return 0;

	*last_slash = '\0';

	for (char* p = buf + 1; *p; p++) {
		if (*p != '/')
			continue;
		*p = '\0';
		if (mkdir(buf, 0777) != 0 && errno != EEXIST)
			return -1;
		*p = '/';
	}
	if (mkdir(buf, 0777) != 0 && errno != EEXIST)
		return -1;

	return 0;
}

static int append_dmrg_timing_jsonl(
	const char* path,
	const char* operation,
	int nsites,
	int d,
	int chi_max,
	int num_sweeps,
	int maxiter_lanczos,
	double J,
	double D,
	double h,
	double tol_split,
	double time_seconds,
	double result,
	int runs)
{
	if (ensure_parent_dirs_timing(path) != 0)
		return -1;

	FILE* f = fopen(path, "a");
	if (!f)
		return -1;

	time_t now = time(NULL);
	if (tol_split >= 0.0) {
		fprintf(f, "{\"backend\":\"c\",\"operation\":\"%s\",\"params\":{\"nsites\":%d,\"d\":%d,\"chi_max\":%d,\"num_sweeps\":%d,\"maxiter_lanczos\":%d,\"J\":%.15g,\"D\":%.15g,\"h\":%.15g,\"tol_split\":%.15g},\"time_seconds\":%.15g,\"result\":%.15g,\"runs\":%d,\"timestamp\":%ld}\n",
		        operation, nsites, d, chi_max, num_sweeps, maxiter_lanczos, J, D, h, tol_split, time_seconds, result, runs, (long)now);
	}
	else {
		fprintf(f, "{\"backend\":\"c\",\"operation\":\"%s\",\"params\":{\"nsites\":%d,\"d\":%d,\"chi_max\":%d,\"num_sweeps\":%d,\"maxiter_lanczos\":%d,\"J\":%.15g,\"D\":%.15g,\"h\":%.15g},\"time_seconds\":%.15g,\"result\":%.15g,\"runs\":%d,\"timestamp\":%ld}\n",
		        operation, nsites, d, chi_max, num_sweeps, maxiter_lanczos, J, D, h, time_seconds, result, runs, (long)now);
	}

	if (fclose(f) != 0)
		return -1;
	return 0;
}

static void make_dmrg_perf_path(
	char* out_path,
	size_t out_size,
	const char* name,
	int nsites,
	long d,
	long chi_max)
{
	snprintf(out_path, out_size, "generated/perf/%s_timings_%d_%ld_%ld.jsonl", name, nsites, d, chi_max);
}

static double get_time_seconds(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** Canonical post-DMRG checks: normalized state and finite ground-state energy (any model/parameters). */
static void dmrg_validate_result(const char* test_name, double norm, double E_final)
{
	const double norm_tol = 1e-5;

	if (fabs(norm - 1.0) > norm_tol) {
		fprintf(stderr, "FAIL [%s]: final MPS norm = %.12f (expected ~1.0, tol=%.1e)\n",
		        test_name, norm, norm_tol);
		g_dmrg_tests_failed = 1;
	} else {
		printf("  CHECK: norm = %.12f  PASS\n", norm);
	}

	if (!isfinite(E_final)) {
		fprintf(stderr, "FAIL [%s]: final energy is not finite (E = %.12g)\n", test_name, E_final);
		g_dmrg_tests_failed = 1;
	} else {
		printf("  CHECK: final energy finite (E = %.12f)  PASS\n", E_final);
	}
}
#include "../src/operator/mpo.h"
#include "../src/state/mps.h"
#include "../src/algorithm/dmrg.h"
#include "../src/util/rng.h"
#include "../src/util/util.h"  /* ipow */
#include "../src/aligned_memory.h"
#include "dmrg_results_json.h"

/* Forward declarations — implemented in mpo.c (included earlier via main.c) */
extern void create_heisenberg_xxz_1d_mpo_tensors(const int nsites, const double J, const double D,
                                                  const double h, struct mpo* mpo);

/* print_separator / print_subsection are defined in mpo.c */


//________________________________________________________________________________________________________________________
//
// Test 1 — Single-site DMRG  (mirrors test_dmrg_singlesite in test_dmrg.c)
//
// Original test: nsites=7, d=3 (Bose–Hubbard loaded from HDF5), 6 sweeps.
// Here we use the Heisenberg XXZ model (d=2, real) so no HDF5 is needed.
//
void test_dmrg_singlesite_manual(void)
{
	print_separator("DMRG Single-Site (Heisenberg XXZ, manual)");

	// ---- parameters ----
	const int    nsites          = g_cfg_singlesite.nsites;
	const long   d               = g_cfg_singlesite.d;
	const double J               = g_cfg_singlesite.J;
	const double D               = g_cfg_singlesite.D;
	const double h               = g_cfg_singlesite.h;
	const long   max_vdim        = g_cfg_singlesite.chi_max;
	const int    num_sweeps      = g_cfg_singlesite.num_sweeps;
	const int    maxiter_lanczos = g_cfg_singlesite.maxiter_lanczos;

	printf("Model        : Heisenberg XXZ (J=%.2f, D=%.2f, h=%.2f)\n", J, D, h);
	printf("nsites       : %d\n", nsites);
	printf("d            : %ld\n", d);
	printf("num_sweeps   : %d\n", num_sweeps);
	printf("maxiter_lanc : %d\n", maxiter_lanczos);
	printf("max_vdim     : %ld\n\n", max_vdim);

	// ---- Hamiltonian MPO ----
	struct mpo hamiltonian;
	create_heisenberg_xxz_1d_mpo_tensors(nsites, J, D, h, &hamiltonian);

	if (!mpo_is_consistent(&hamiltonian)) {
		fprintf(stderr, "ERROR: MPO consistency check failed\n");
		delete_mpo(&hamiltonian);
		return;
	}
	printf("MPO consistency: PASSED\n");

	// ---- physical quantum numbers (no symmetry) ----
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	for (long i = 0; i < d; i++) { qsite[i] = 0; }

	// ---- initial random MPS ----
	struct mps psi;
	struct rng_state rng;
	seed_rng_state(42, &rng);
	construct_random_mps(CT_DOUBLE_REAL, nsites, d, qsite, 0, max_vdim, &rng, &psi);

	if (!mps_is_consistent(&psi)) {
		fprintf(stderr, "ERROR: initial MPS consistency check failed\n");
		delete_mps(&psi);
		delete_mpo(&hamiltonian);
		ct_free(qsite);
		return;
	}

	printf("Initial MPS norm : %.12f\n\n", mps_norm(&psi));

	// ---- run DMRG (with timing) ----
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));

	double t_start = get_time_seconds();
	if (dmrg_singlesite(&hamiltonian, num_sweeps, maxiter_lanczos, &psi, en_sweeps) < 0) {
		fprintf(stderr, "ERROR: dmrg_singlesite failed\n");
		ct_free(en_sweeps);
		delete_mps(&psi);
		delete_mpo(&hamiltonian);
		ct_free(qsite);
		return;
	}
	double t_end = get_time_seconds();
	double dmrg_time = t_end - t_start;
	printf("\nDMRG single-site runtime: %.6f seconds\n", dmrg_time);

	// ---- print results ----
	print_subsection("Energy per sweep");
	for (int i = 0; i < num_sweeps; i++) {
		printf("  sweep %d : E = %.12f\n", i + 1, en_sweeps[i]);
	}

	double norm = mps_norm(&psi);
	printf("\nFinal MPS norm : %.12f  (should be ~1.0)\n", norm);
	dmrg_validate_result("singlesite", norm, en_sweeps[num_sweeps - 1]);

	printf("Bond dimensions: [");
	for (int i = 0; i <= nsites; i++) {
		printf("%ld%s", mps_bond_dim(&psi, i), i < nsites ? ", " : "");
	}
	printf("]\n");

	// ---- save results to JSON (for comparison with Mojo) ----
	{
		long* bond_dims = ct_malloc((size_t)(nsites + 1) * sizeof(long));
		for (int i = 0; i <= nsites; i++)
			bond_dims[i] = mps_bond_dim(&psi, i);
		struct dmrg_json_params params = {
			.nsites = nsites,
			.d = (int)d,
			.J = J, .D = D, .h = h,
			.num_sweeps = num_sweeps,
			.maxiter_lanczos = maxiter_lanczos,
			.chi_max = (int)max_vdim,
			.tol_split = -1.0   /* omit for single-site */
		};
		if (dmrg_results_to_json("generated/dmrg/dmrg_results_c_singlesite.json", "c", "heisenberg_xxz",
		                          &params, en_sweeps[num_sweeps - 1], en_sweeps, num_sweeps,
		                          NULL, bond_dims, nsites + 1, norm) == 0)
			printf("Results written to generated/dmrg/dmrg_results_c_singlesite.json\n");
		ct_free(bond_dims);
	}

	// ---- save timing to JSONL (for performance comparison with Mojo) ----
	char perf_path[512];
	make_dmrg_perf_path(perf_path, sizeof(perf_path), "dmrg_singlesite", nsites, d, max_vdim);
	if (append_dmrg_timing_jsonl(perf_path,
	                              "dmrg_singlesite", nsites, (int)d, (int)max_vdim, num_sweeps,
	                              maxiter_lanczos, J, D, h, -1.0,
	                              dmrg_time, en_sweeps[num_sweeps - 1], 1) == 0)
		printf("Timing written to %s\n", perf_path);

	// ---- clean up ----
	ct_free(en_sweeps);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);
	ct_free(qsite);

	printf("\nSingle-site DMRG test completed.\n");
}


//________________________________________________________________________________________________________________________
//
// Test 2 — Two-site DMRG  (mirrors test_dmrg_twosite in test_dmrg.c)
//
// Original test: nsites=11, d=2, real, 4 sweeps.
// We use the same Heisenberg XXZ model.
//
void test_dmrg_twosite_manual(void)
{
	print_separator("DMRG Two-Site (Heisenberg XXZ, manual)");

	// ---- parameters ----
	const int    nsites          = g_cfg_twosite.nsites;
	const long   d               = g_cfg_twosite.d;
	const double J               = g_cfg_twosite.J;
	const double D               = g_cfg_twosite.D;
	const double h               = g_cfg_twosite.h;
	const long   max_vdim        = g_cfg_twosite.chi_max;
	const int    num_sweeps      = g_cfg_twosite.num_sweeps;
	const int    maxiter_lanczos = g_cfg_twosite.maxiter_lanczos;
	const double tol_split       = g_cfg_twosite.tol_split;

	printf("Model        : Heisenberg XXZ (J=%.2f, D=%.2f, h=%.2f)\n", J, D, h);
	printf("nsites       : %d\n", nsites);
	printf("d            : %ld\n", d);
	printf("num_sweeps   : %d\n", num_sweeps);
	printf("maxiter_lanc : %d\n", maxiter_lanczos);
	printf("tol_split    : %.1e\n", tol_split);
	printf("max_vdim     : %ld\n\n", max_vdim);

	// ---- Hamiltonian MPO ----
	struct mpo hamiltonian;
	create_heisenberg_xxz_1d_mpo_tensors(nsites, J, D, h, &hamiltonian);

	if (!mpo_is_consistent(&hamiltonian)) {
		fprintf(stderr, "ERROR: MPO consistency check failed\n");
		delete_mpo(&hamiltonian);
		return;
	}
	printf("MPO consistency: PASSED\n");

	// ---- physical quantum numbers (no symmetry) ----
	qnumber* qsite = ct_malloc(d * sizeof(qnumber));
	for (long i = 0; i < d; i++) { qsite[i] = 0; }

	// ---- initial random MPS ----
	struct mps psi;
	struct rng_state rng;
	seed_rng_state(42, &rng);
	construct_random_mps(CT_DOUBLE_REAL, nsites, d, qsite, 0, max_vdim, &rng, &psi);

	if (!mps_is_consistent(&psi)) {
		fprintf(stderr, "ERROR: initial MPS consistency check failed\n");
		delete_mps(&psi);
		delete_mpo(&hamiltonian);
		ct_free(qsite);
		return;
	}

	printf("Initial MPS norm : %.12f\n\n", mps_norm(&psi));

	// ---- run DMRG (with timing) ----
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));

	double t_start = get_time_seconds();
	if (dmrg_twosite(&hamiltonian, num_sweeps, maxiter_lanczos, tol_split, max_vdim,
	                  &psi, en_sweeps, entropy) < 0)
	{
		fprintf(stderr, "ERROR: dmrg_twosite failed\n");
		ct_free(entropy);
		ct_free(en_sweeps);
		delete_mps(&psi);
		delete_mpo(&hamiltonian);
		ct_free(qsite);
		return;
	}
	double t_end = get_time_seconds();
	double dmrg_time = t_end - t_start;
	printf("\nDMRG two-site runtime: %.6f seconds\n", dmrg_time);

	// ---- print results ----
	print_subsection("Energy per sweep");
	for (int i = 0; i < num_sweeps; i++) {
		printf("  sweep %d : E = %.12f\n", i + 1, en_sweeps[i]);
	}

	print_subsection("Entanglement entropy (after final sweep)");
	for (int i = 0; i < nsites - 1; i++) {
		printf("  bond %2d : S = %.6e\n", i, entropy[i]);
	}

	double norm = mps_norm(&psi);
	printf("\nFinal MPS norm : %.12f  (should be ~1.0)\n", norm);
	dmrg_validate_result("twosite", norm, en_sweeps[num_sweeps - 1]);

	printf("Bond dimensions: [");
	for (int i = 0; i <= nsites; i++) {
		printf("%ld%s", mps_bond_dim(&psi, i), i < nsites ? ", " : "");
	}
	printf("]\n");

	// ---- save results to JSON (for comparison with Mojo) ----
	{
		long* bond_dims = ct_malloc((size_t)(nsites + 1) * sizeof(long));
		for (int i = 0; i <= nsites; i++)
			bond_dims[i] = mps_bond_dim(&psi, i);
		struct dmrg_json_params params = {
			.nsites = nsites,
			.d = (int)d,
			.J = J, .D = D, .h = h,
			.num_sweeps = num_sweeps,
			.maxiter_lanczos = maxiter_lanczos,
			.chi_max = (int)max_vdim,
			.tol_split = tol_split
		};
		if (dmrg_results_to_json("generated/dmrg/dmrg_results_c_twosite.json", "c", "heisenberg_xxz",
		                          &params, en_sweeps[num_sweeps - 1], en_sweeps, num_sweeps,
		                          entropy, bond_dims, nsites + 1, norm) == 0)
			printf("Results written to generated/dmrg/dmrg_results_c_twosite.json\n");
		ct_free(bond_dims);
	}

	// ---- save timing to JSONL (for performance comparison with Mojo) ----
	char perf_path[512];
	make_dmrg_perf_path(perf_path, sizeof(perf_path), "dmrg_twosite", nsites, d, max_vdim);
	if (append_dmrg_timing_jsonl(perf_path,
	                              "dmrg_twosite", nsites, (int)d, (int)max_vdim, num_sweeps,
	                              maxiter_lanczos, J, D, h, tol_split,
	                              dmrg_time, en_sweeps[num_sweeps - 1], 1) == 0)
		printf("Timing written to %s\n", perf_path);

	// ---- clean up ----
	ct_free(entropy);
	ct_free(en_sweeps);
	delete_mps(&psi);
	delete_mpo(&hamiltonian);
	ct_free(qsite);

	printf("\nTwo-site DMRG test completed.\n");
}


//________________________________________________________________________________________________________________________
//
// Entry point called from main.c
//
void dmrg_tests(void)
{
	print_separator("DMRG (Density Matrix Renormalization Group) Manual Tests");
	load_dmrg_bench_config();

	printf("Mirrors test/algorithm/test_dmrg.c but prints results\n");
	printf("instead of comparing against HDF5 reference data.\n");
	printf("Hamiltonian: Heisenberg XXZ, built in code.\n\n");

	test_dmrg_singlesite_manual();
	test_dmrg_twosite_manual();

	print_separator("All DMRG Manual Tests Completed");
	printf("\nSummary:\n");
	printf("  - Single-site DMRG (Heisenberg XXZ, params from bench_config.json)\n");
	printf("  - Two-site DMRG   (Heisenberg XXZ, params from bench_config.json)\n");
	if (g_dmrg_tests_failed) {
		fprintf(stderr, "\nDMRG tests FAILED (norm or final energy check)\n");
		exit(1);
	}
	printf("\nAll DMRG consistency checks PASSED.\n");
}
