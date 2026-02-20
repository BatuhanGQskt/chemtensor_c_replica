/// \file mps_export_json.c
/// \brief Export MPS observables to JSON for Mojo comparison.

#include <stdio.h>
#include <math.h>
#include "mps_export_json.h"
#include "../src/state/mps.h"
#include "../src/tensor/block_sparse_tensor.h"
#include "../src/tensor/dense_tensor.h"

void export_mps_observables_to_json(const struct mps* mps, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", filename);
        return;
    }

    double norm_val = mps_norm(mps);

    fprintf(f, "{\n");
    fprintf(f, "  \"nsites\": %d,\n", mps->nsites);
    fprintf(f, "  \"d\": %ld,\n", mps->d);

    fprintf(f, "  \"bond_dims\": [");
    for (int i = 0; i <= mps->nsites; i++) {
        fprintf(f, "%ld%s", mps_bond_dim(mps, i), i < mps->nsites ? ", " : "");
    }
    fprintf(f, "],\n");

    fprintf(f, "  \"norm\": %.15e,\n", norm_val);

    /* Export state vector for overlap comparison in Mojo */
    struct block_sparse_tensor vec;
    mps_to_statevector(mps, &vec);

    struct dense_tensor vec_dense;
    block_sparse_to_dense_tensor(&vec, &vec_dense);
    delete_block_sparse_tensor(&vec);

    long vec_len = 1;
    for (int i = 0; i < vec_dense.ndim; i++)
        vec_len *= vec_dense.dim[i];

    fprintf(f, "  \"state_vector\": [");
    if (vec_dense.dtype == CT_SINGLE_REAL) {
        float* data = (float*)vec_dense.data;
        for (long j = 0; j < vec_len; j++)
            fprintf(f, "%.15e%s", data[j], j < vec_len - 1 ? ", " : "");
    } else if (vec_dense.dtype == CT_DOUBLE_REAL) {
        double* data = (double*)vec_dense.data;
        for (long j = 0; j < vec_len; j++)
            fprintf(f, "%.15e%s", data[j], j < vec_len - 1 ? ", " : "");
    }
    fprintf(f, "]\n");
    fprintf(f, "}\n");

    delete_dense_tensor(&vec_dense);
    fclose(f);
    printf("Exported MPS observables to '%s'\n", filename);
}
