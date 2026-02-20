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
/// Consistency checks (norm ~1, energy in expected range) mirror the
/// assertions in Mojo test_dmrg.mojo and test_dmrg_c_comparison.mojo.
/// Export JSON to generated/dmrg/ (or copy to Mojo test_data/) for C vs Mojo comparison.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int g_dmrg_tests_failed = 0;

static void dmrg_check_norm(const char* test_name, double norm, double tol)
{
	if (fabs(norm - 1.0) > tol) {
		fprintf(stderr, "FAIL [%s]: final MPS norm = %.12f (expected ~1.0, tol=%.1e)\n",
		        test_name, norm, tol);
		g_dmrg_tests_failed = 1;
	} else {
		printf("  CHECK: norm = %.12f  PASS\n", norm);
	}
}

static void dmrg_check_energy_range(const char* test_name, double E, double E_lo, double E_hi)
{
	if (E < E_lo || E > E_hi) {
		fprintf(stderr, "FAIL [%s]: energy = %.12f (expected in [%.2f, %.2f])\n",
		        test_name, E, E_lo, E_hi);
		g_dmrg_tests_failed = 1;
	} else {
		printf("  CHECK: energy in [%.2f, %.2f]  PASS\n", E_lo, E_hi);
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
	const int    nsites          = 7;
	const long   d               = 2;     // spin-1/2
	const double J               = 1.0;
	const double D               = 1.0;   // D=1 => isotropic XXX limit
	const double h               = 0.0;
	const long   max_vdim        = 16;
	const int    num_sweeps       = 6;
	const int    maxiter_lanczos  = 25;

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

	// ---- run DMRG ----
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));

	if (dmrg_singlesite(&hamiltonian, num_sweeps, maxiter_lanczos, &psi, en_sweeps) < 0) {
		fprintf(stderr, "ERROR: dmrg_singlesite failed\n");
		ct_free(en_sweeps);
		delete_mps(&psi);
		delete_mpo(&hamiltonian);
		ct_free(qsite);
		return;
	}

	// ---- print results ----
	print_subsection("Energy per sweep");
	for (int i = 0; i < num_sweeps; i++) {
		printf("  sweep %d : E = %.12f\n", i + 1, en_sweeps[i]);
	}

	double norm = mps_norm(&psi);
	printf("\nFinal MPS norm : %.12f  (should be ~1.0)\n", norm);
	dmrg_check_norm("singlesite", norm, 1e-5);
	dmrg_check_energy_range("singlesite", en_sweeps[num_sweeps - 1], -9.0, -7.0);

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
	const int    nsites          = 11;
	const long   d               = 2;     // spin-1/2
	const double J               = 1.0;
	const double D               = 0.5;   // anisotropy
	const double h               = 0.2;   // magnetic field
	const long   max_vdim        = ipow(d, nsites / 2);  // same as original test
	const int    num_sweeps       = 4;
	const int    maxiter_lanczos  = 25;
	const double tol_split        = 1e-10;

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

	// ---- run DMRG ----
	double* en_sweeps = ct_malloc(num_sweeps * sizeof(double));
	double* entropy   = ct_malloc((nsites - 1) * sizeof(double));

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
	dmrg_check_norm("twosite", norm, 1e-5);
	dmrg_check_energy_range("twosite", en_sweeps[num_sweeps - 1], -4.0, -2.0);

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

	printf("Mirrors test/algorithm/test_dmrg.c but prints results\n");
	printf("instead of comparing against HDF5 reference data.\n");
	printf("Hamiltonian: Heisenberg XXZ, built in code.\n\n");

	test_dmrg_singlesite_manual();
	test_dmrg_twosite_manual();

	print_separator("All DMRG Manual Tests Completed");
	printf("\nSummary:\n");
	printf("  - Single-site DMRG (Heisenberg XXX, 7 sites)\n");
	printf("  - Two-site DMRG   (Heisenberg XXZ, 11 sites)\n");
	if (g_dmrg_tests_failed) {
		fprintf(stderr, "\nDMRG tests FAILED (norm or energy range check)\n");
		exit(1);
	}
	printf("\nAll DMRG consistency checks PASSED.\n");
}
