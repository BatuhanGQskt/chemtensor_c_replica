/// \file mpo_export_json.c
/// \brief Export MPO tensors to JSON format for comparison with Mojo implementation.

#include <stdio.h>
#include <string.h>
#include "mpo_export_json.h"
#include "../src/tensor/block_sparse_tensor.h"
#include "../src/tensor/dense_tensor.h"
#include "../src/operator/mpo.h"

void export_mpo_to_json(const struct mpo* mpo, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s' for writing\n", filename);
        return;
    }
    
    fprintf(f, "{\n");
    fprintf(f, "  \"nsites\": %d,\n", mpo->nsites);
    fprintf(f, "  \"d\": %ld,\n", mpo->d);
    
    // Export bond dimensions
    fprintf(f, "  \"bond_dims\": [");
    for (int i = 0; i <= mpo->nsites; i++) {
        fprintf(f, "%ld%s", mpo_bond_dim(mpo, i), i < mpo->nsites ? ", " : "");
    }
    fprintf(f, "],\n");
    
    // Export quantum numbers for physical sites
    fprintf(f, "  \"qsite\": [");
    for (long i = 0; i < mpo->d; i++) {
        fprintf(f, "%d%s", mpo->qsite[i], i < mpo->d - 1 ? ", " : "");
    }
    fprintf(f, "],\n");
    
    // Export each site tensor
    fprintf(f, "  \"sites\": [\n");
    
    for (int i = 0; i < mpo->nsites; i++) {
        // Convert block sparse tensor to dense for export
        struct dense_tensor site_dense;
        block_sparse_to_dense_tensor(&mpo->a[i], &site_dense);
        
        fprintf(f, "    {\n");
        fprintf(f, "      \"site\": %d,\n", i);
        fprintf(f, "      \"shape\": [%ld, %ld, %ld, %ld],\n", 
                site_dense.dim[0], site_dense.dim[1], 
                site_dense.dim[2], site_dense.dim[3]);
        
        // Export tensor data (real only)
        fprintf(f, "      \"dtype\": \"");
        switch (site_dense.dtype) {
            case CT_SINGLE_REAL: fprintf(f, "float32"); break;
            case CT_DOUBLE_REAL: fprintf(f, "float64"); break;
            default:             fprintf(f, "unknown"); break;
        }
        fprintf(f, "\",\n");
        
        fprintf(f, "      \"data\": [");
        
        long total = site_dense.dim[0] * site_dense.dim[1] * 
                     site_dense.dim[2] * site_dense.dim[3];
        
        if (site_dense.dtype == CT_SINGLE_REAL) {
            float* data = (float*)site_dense.data;
            for (long j = 0; j < total; j++) {
                fprintf(f, "%.15e%s", data[j], j < total - 1 ? ", " : "");
            }
        } else if (site_dense.dtype == CT_DOUBLE_REAL) {
            double* data = (double*)site_dense.data;
            for (long j = 0; j < total; j++) {
                fprintf(f, "%.15e%s", data[j], j < total - 1 ? ", " : "");
            }
        }
        
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", i < mpo->nsites - 1 ? "," : "");
        
        delete_dense_tensor(&site_dense);
    }
    
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    
    fclose(f);
    printf("Exported MPO to '%s'\n", filename);
}
