/// \file mpo_export_json.h
/// \brief Export MPO tensors to JSON format for comparison with Mojo implementation.

#ifndef MPO_EXPORT_JSON_H
#define MPO_EXPORT_JSON_H

#include "../src/operator/mpo.h"

/// \brief Export MPO site tensors to JSON file for Mojo comparison.
///
/// The JSON format includes metadata (nsites, physical dimension, bond dimensions)
/// and the dense tensor data for each site in row-major order.
void export_mpo_to_json(const struct mpo* mpo, const char* filename);

#endif // MPO_EXPORT_JSON_H
