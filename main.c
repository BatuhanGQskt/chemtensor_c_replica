#include <stdio.h>
#include <string.h>
#include "./src/tensor/dense_tensor.h"
#include "./src/numeric.h"
#include "./manual_tests/mpo_export_json.h"
#include "./manual_tests/mps_export_json.h"
#include "./manual_tests/dense_tensor.c"
#include "./manual_tests/mpo.c"
#include "./manual_tests/mps.c"
#include "./manual_tests/dmrg.c"

int main(int argc, char** argv) {
    int skip_dmrg = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-dmrg") == 0) {
            skip_dmrg = 1;
        }
    }

    dense_tensor_create_test();
    dense_tensor_dot_test();
    mpo_tests();  // This will also export reference data
    mps_tests();  // Export MPS observables for Mojo comparison
    if (!skip_dmrg) {
        dmrg_tests();
    } else {
        printf("Skipping DMRG manual tests (--skip-dmrg).\n");
    }
    return 0;
}