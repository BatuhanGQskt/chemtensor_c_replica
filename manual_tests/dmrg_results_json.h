/// \file dmrg_results_json.h
/// \brief Helper to save DMRG results to a JSON file (C and Mojo share the same format).
///
/// See manual_tests/DMRG_RESULTS_JSON_SCHEMA.md for the schema.
/// Use the same format from Mojo so that dmrg_results_c.json and dmrg_results_mojo.json
/// can be compared (e.g. with diff or a tolerance-based script).

#pragma once

#include <stddef.h>

/** Parameters that are written to the "params" object in JSON. */
struct dmrg_json_params {
	int nsites;
	int d;
	double J;
	double D;
	double h;
	int num_sweeps;
	int maxiter_lanczos;
	int chi_max;
	double tol_split;  /**< Set to -1.0 to omit from JSON (e.g. single-site). */
};

/**
 * Write DMRG results to a JSON file.
 *
 * \param path Output file path (overwritten if exists).
 * \param impl String "c" or "mojo".
 * \param model String e.g. "heisenberg_xxz".
 * \param params Run parameters (tol_split < 0 omits "tol_split" in JSON).
 * \param energy_final Final ground-state energy.
 * \param en_sweeps Array of length num_sweeps (energy at end of each sweep).
 * \param num_sweeps Length of en_sweeps.
 * \param entropy Optional array of length (nsites - 1), or NULL to omit "entropy" key.
 * \param bond_dims Array of length (nsites + 1).
 * \param num_bond_dims Length of bond_dims (must be params.nsites + 1).
 * \param norm Final MPS norm.
 * \return 0 on success, -1 on file open/write error.
 */
int dmrg_results_to_json(
	const char* path,
	const char* impl,
	const char* model,
	const struct dmrg_json_params* params,
	double energy_final,
	const double* en_sweeps,
	int num_sweeps,
	const double* entropy,
	const long* bond_dims,
	int num_bond_dims,
	double norm);
