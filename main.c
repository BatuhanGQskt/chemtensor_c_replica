#include <stdio.h>
#include "./src/tensor/dense_tensor.h"
#include "./src/numeric.h"
#include "./manual_tests/dense_tensor.c"
#include "./manual_tests/mpo.c"

int main() {
    //dense_tensor_create_test();
    // dense_tensor_dot_test();
    mpo_tests();
    return 0;
}