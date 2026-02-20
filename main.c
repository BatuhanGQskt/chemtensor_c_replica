#include <stdio.h>
#include "./src/tensor/dense_tensor.h"
#include "./src/numeric.h"
#include "./manual_tests/mpo_export_json.h"
#include "./manual_tests/mps_export_json.h"
#include "./manual_tests/dense_tensor.c"
#include "./manual_tests/mpo.c"
#include "./manual_tests/mps.c"
#include "./manual_tests/dmrg.c"

int main() {
    dense_tensor_create_test();
    dense_tensor_dot_test();
    mpo_tests();  // This will also export reference data
    mps_tests();  // Export MPS observables for Mojo comparison
    // dmrg_tests();
    return 0;
}