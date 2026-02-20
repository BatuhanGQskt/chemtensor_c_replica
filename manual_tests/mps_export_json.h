/// \file mps_export_json.h
/// \brief Export MPS observables (and state vector) to JSON for Mojo comparison.
/// Tests compare norms and overlaps only; no per-site tensor export.

#ifndef MPS_EXPORT_JSON_H
#define MPS_EXPORT_JSON_H

#include "../src/state/mps.h"

/// \brief Export MPS observables to JSON: nsites, d, bond_dims, norm, state_vector.
/// Mojo tests use this for implementation-agnostic comparison (assert_close on norm/overlap).
void export_mps_observables_to_json(const struct mps* mps, const char* filename);

#endif /* MPS_EXPORT_JSON_H */
