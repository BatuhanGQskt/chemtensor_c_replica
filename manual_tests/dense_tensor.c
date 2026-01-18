#include <stdio.h>
#include "../src/tensor/dense_tensor.h"
#include "../src/numeric.h"


void print_tensor_info(const char* name, struct dense_tensor* t) {
    printf("\n=== Tensor: %s ===\n", name);
    printf("Shape: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", t->dim[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("]\n");
    
    long nelem = dense_tensor_num_elements(t);
    printf("Total elements: %ld\n", nelem);
    printf("Data type: %s\n", 
           t->dtype == CT_SINGLE_REAL ? "float32" :
           t->dtype == CT_DOUBLE_REAL ? "float64" : "unknown");
    
    printf("Data (row-major): [");
    if (t->dtype == CT_SINGLE_REAL) {
        float* data = (float*)t->data;
        for (long i = 0; i < nelem; i++) {
            printf("%.6f%s", data[i], i < nelem - 1 ? ", " : "");
        }
    } else if (t->dtype == CT_DOUBLE_REAL) {
        double* data = (double*)t->data;
        for (long i = 0; i < nelem; i++) {
            printf("%.6f%s", data[i], i < nelem - 1 ? ", " : "");
        }
    }
    printf("]\n");
}

void dense_tensor_create_test() {
    printf("Creating test tensors for Mojo comparison\n");
    printf("==========================================\n");
    
    // Example 1: Simple 2D matrix (float32)
    struct dense_tensor matrix_2d;
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){3, 4}, &matrix_2d);
    
    // Fill with specific values: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    float matrix_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 12; i++) {
        ((float*)matrix_2d.data)[i] = matrix_data[i];
    }
    print_tensor_info("matrix_2d", &matrix_2d);
    
    // Example 2: 3D tensor (float32)
    struct dense_tensor tensor_3d;
    allocate_dense_tensor(CT_SINGLE_REAL, 3, (long[]){2, 3, 4}, &tensor_3d);
    
    // Fill with sequential values 0 to 23
    for (int i = 0; i < 24; i++) {
        ((float*)tensor_3d.data)[i] = (float)i;
    }
    print_tensor_info("tensor_3d", &tensor_3d);
    
    // Example 3: Vector (1D tensor)
    struct dense_tensor vector;
    allocate_dense_tensor(CT_SINGLE_REAL, 1, (long[]){5}, &vector);
    
    // Fill with specific values
    float vec_data[] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    for (int i = 0; i < 5; i++) {
        ((float*)vector.data)[i] = vec_data[i];
    }
    print_tensor_info("vector", &vector);
    
    // Example 4: Double precision matrix
    struct dense_tensor matrix_double;
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, (long[]){2, 3}, &matrix_double);
    
    double double_data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    for (int i = 0; i < 6; i++) {
        ((double*)matrix_double.data)[i] = double_data[i];
    }
    print_tensor_info("matrix_double", &matrix_double);
    
    // Example 5: Small square matrix for operations
    struct dense_tensor square_matrix;
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){3, 3}, &square_matrix);
    
    // Identity-like matrix with some variation
    float square_data[] = {1.0f, 0.5f, 0.0f,
                           0.5f, 1.0f, 0.5f,
                           0.0f, 0.5f, 1.0f};
    for (int i = 0; i < 9; i++) {
        ((float*)square_matrix.data)[i] = square_data[i];
    }
    print_tensor_info("square_matrix", &square_matrix);
    
    // Test some operations
    printf("\n=== Testing Operations ===\n");
    
    // Calculate norm of vector
    double norm = dense_tensor_norm2(&vector);
    printf("Norm of vector: %.6f\n", norm);
    
    // Scale the vector
    float scale_factor = 2.0f;
    scale_dense_tensor(&scale_factor, &vector);
    printf("After scaling by 2.0:\n");
    print_tensor_info("vector_scaled", &vector);
    
    // Clean up
    delete_dense_tensor(&matrix_2d);
    delete_dense_tensor(&tensor_3d);
    delete_dense_tensor(&vector);
    delete_dense_tensor(&matrix_double);
    delete_dense_tensor(&square_matrix);
    
    printf("\n=== All tests completed ===\n");
}

void dense_tensor_dot_test() {
    printf("\n\n");
    printf("=========================================\n");
    printf("Testing Tensor Dot Products\n");
    printf("=========================================\n");
    
    // Test 1: Simple Matrix-Matrix Multiplication (2x3) @ (3x2)
    printf("\n--- Test 1: Matrix-Matrix Multiplication ---\n");
    struct dense_tensor A, B, C;
    
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){2, 3}, &A);
    float A_data[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) {
        ((float*)A.data)[i] = A_data[i];
    }
    print_tensor_info("A", &A);
    
    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]  (3x2)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){3, 2}, &B);
    float B_data[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++) {
        ((float*)B.data)[i] = B_data[i];
    }
    print_tensor_info("B", &B);
    
    // C = A @ B (should be 2x2)
    // Expected: [[58, 64], [139, 154]]
    dense_tensor_dot(&A, TENSOR_AXIS_RANGE_TRAILING, &B, TENSOR_AXIS_RANGE_LEADING, 1, &C);
    printf("\nResult C = A @ B:\n");
    print_tensor_info("C", &C);
    printf("Expected: [[58.0, 64.0], [139.0, 154.0]]\n");
    
    // Test 2: Matrix-Vector Multiplication
    printf("\n--- Test 2: Matrix-Vector Multiplication ---\n");
    struct dense_tensor M, v, result;
    
    // M = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){2, 3}, &M);
    float M_data[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < 6; i++) {
        ((float*)M.data)[i] = M_data[i];
    }
    print_tensor_info("M", &M);
    
    // v = [10, 20, 30]  (3,)
    allocate_dense_tensor(CT_SINGLE_REAL, 1, (long[]){3}, &v);
    float v_data[] = {10, 20, 30};
    for (int i = 0; i < 3; i++) {
        ((float*)v.data)[i] = v_data[i];
    }
    print_tensor_info("v", &v);
    
    // result = M @ v (should be 2,)
    // Expected: [140, 320]
    dense_tensor_dot(&M, TENSOR_AXIS_RANGE_TRAILING, &v, TENSOR_AXIS_RANGE_LEADING, 1, &result);
    printf("\nResult = M @ v:\n");
    print_tensor_info("result", &result);
    printf("Expected: [140.0, 320.0]\n");
    
    // Test 3: Square Matrix Multiplication
    printf("\n--- Test 3: Square Matrix (3x3) @ (3x3) ---\n");
    struct dense_tensor X, Y, Z;
    
    // X = [[1, 0, 0],
    //      [0, 2, 0],
    //      [0, 0, 3]]  (3x3) - diagonal
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){3, 3}, &X);
    float X_data[] = {1, 0, 0, 0, 2, 0, 0, 0, 3};
    for (int i = 0; i < 9; i++) {
        ((float*)X.data)[i] = X_data[i];
    }
    print_tensor_info("X", &X);
    
    // Y = [[1, 2, 3],
    //      [4, 5, 6],
    //      [7, 8, 9]]  (3x3)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){3, 3}, &Y);
    float Y_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < 9; i++) {
        ((float*)Y.data)[i] = Y_data[i];
    }
    print_tensor_info("Y", &Y);
    
    // Z = X @ Y (should be 3x3)
    // Expected: [[1, 2, 3], [8, 10, 12], [21, 24, 27]]
    dense_tensor_dot(&X, TENSOR_AXIS_RANGE_TRAILING, &Y, TENSOR_AXIS_RANGE_LEADING, 1, &Z);
    printf("\nResult Z = X @ Y:\n");
    print_tensor_info("Z", &Z);
    printf("Expected: [[1.0, 2.0, 3.0], [8.0, 10.0, 12.0], [21.0, 24.0, 27.0]]\n");
    
    // Test 4: Small example for manual verification
    printf("\n--- Test 4: Small 2x2 @ 2x2 ---\n");
    struct dense_tensor P, Q, R;
    
    // P = [[1, 2],
    //      [3, 4]]  (2x2)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){2, 2}, &P);
    float P_data[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++) {
        ((float*)P.data)[i] = P_data[i];
    }
    print_tensor_info("P", &P);
    
    // Q = [[5, 6],
    //      [7, 8]]  (2x2)
    allocate_dense_tensor(CT_SINGLE_REAL, 2, (long[]){2, 2}, &Q);
    float Q_data[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        ((float*)Q.data)[i] = Q_data[i];
    }
    print_tensor_info("Q", &Q);
    
    // R = P @ Q (should be 2x2)
    // Manual: [1*5+2*7, 1*6+2*8] = [19, 22]
    //         [3*5+4*7, 3*6+4*8] = [43, 50]
    dense_tensor_dot(&P, TENSOR_AXIS_RANGE_TRAILING, &Q, TENSOR_AXIS_RANGE_LEADING, 1, &R);
    printf("\nResult R = P @ Q:\n");
    print_tensor_info("R", &R);
    printf("Expected: [[19.0, 22.0], [43.0, 50.0]]\n");
    
    // Test 5: Vector dot product (inner product)
    printf("\n--- Test 5: Vector Inner Product ---\n");
    struct dense_tensor u, w, dot_result;
    
    // u = [1, 2, 3, 4]
    allocate_dense_tensor(CT_SINGLE_REAL, 1, (long[]){4}, &u);
    float u_data[] = {1, 2, 3, 4};
    for (int i = 0; i < 4; i++) {
        ((float*)u.data)[i] = u_data[i];
    }
    print_tensor_info("u", &u);
    
    // w = [5, 6, 7, 8]
    allocate_dense_tensor(CT_SINGLE_REAL, 1, (long[]){4}, &w);
    float w_data[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; i++) {
        ((float*)w.data)[i] = w_data[i];
    }
    print_tensor_info("w", &w);
    
    // dot_result = u · w (should be scalar, represented as 0-d tensor)
    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    dense_tensor_dot(&u, TENSOR_AXIS_RANGE_TRAILING, &w, TENSOR_AXIS_RANGE_LEADING, 1, &dot_result);
    printf("\nResult = u · w:\n");
    print_tensor_info("dot_result", &dot_result);
    printf("Expected: 70.0\n");
    
    // Clean up
    delete_dense_tensor(&A);
    delete_dense_tensor(&B);
    delete_dense_tensor(&C);
    delete_dense_tensor(&M);
    delete_dense_tensor(&v);
    delete_dense_tensor(&result);
    delete_dense_tensor(&X);
    delete_dense_tensor(&Y);
    delete_dense_tensor(&Z);
    delete_dense_tensor(&P);
    delete_dense_tensor(&Q);
    delete_dense_tensor(&R);
    delete_dense_tensor(&u);
    delete_dense_tensor(&w);
    delete_dense_tensor(&dot_result);
    
    printf("\n=== All dot product tests completed ===\n");
}
