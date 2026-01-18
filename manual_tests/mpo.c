#include <stdio.h>
#include <complex.h>
#include "../src/operator/mpo.h"
#include "../src/operator/hamiltonian.h"
#include "../src/tensor/block_sparse_tensor.h"
#include "../src/numeric.h"


void print_separator(const char* title) {
    printf("\n");
    printf("=========================================\n");
    printf("%s\n", title);
    printf("=========================================\n");
}

void print_subsection(const char* title) {
    printf("\n--- %s ---\n", title);
}

void print_mpo_info(const char* name, const struct mpo* mpo) {
    print_subsection(name);
    
    printf("Number of sites: %d\n", mpo->nsites);
    printf("Physical dimension: %ld\n", mpo->d);
    
    printf("Physical quantum numbers (qsite): [");
    for (long i = 0; i < mpo->d; i++) {
        printf("%d%s", mpo->qsite[i], i < mpo->d - 1 ? ", " : "");
    }
    printf("]\n");
    
    printf("Virtual bond dimensions: [");
    for (int i = 0; i <= mpo->nsites; i++) {
        printf("%ld%s", mpo_bond_dim(mpo, i), i < mpo->nsites ? ", " : "");
    }
    printf("]\n");
    
    printf("\nDetailed tensor information per site:\n");
    for (int i = 0; i < mpo->nsites; i++) {
        printf("\n  Site %d:\n", i);
        printf("    Logical dimensions: [%ld, %ld, %ld, %ld]\n",
               mpo->a[i].dim_logical[0], mpo->a[i].dim_logical[1],
               mpo->a[i].dim_logical[2], mpo->a[i].dim_logical[3]);
        
        printf("    Axis directions: [");
        for (int j = 0; j < 4; j++) {
            const char* dir = (mpo->a[i].axis_dir[j] == TENSOR_AXIS_OUT) ? "OUT" : "IN";
            printf("%s%s", dir, j < 3 ? ", " : "");
        }
        printf("]\n");
        
        // Calculate total number of blocks
        long total_blocks = 1;
        for (int j = 0; j < mpo->a[i].ndim; j++) {
            total_blocks *= mpo->a[i].dim_blocks[j];
        }
        printf("    Total block slots: %ld\n", total_blocks);
        
        // Print quantum numbers for virtual bonds
        printf("    Left bond quantum numbers: [");
        for (long k = 0; k < mpo->a[i].dim_logical[0]; k++) {
            printf("%d%s", mpo->a[i].qnums_logical[0][k], k < mpo->a[i].dim_logical[0] - 1 ? ", " : "");
        }
        printf("]\n");
        
        printf("    Right bond quantum numbers: [");
        for (long k = 0; k < mpo->a[i].dim_logical[3]; k++) {
            printf("%d%s", mpo->a[i].qnums_logical[3][k], k < mpo->a[i].dim_logical[3] - 1 ? ", " : "");
        }
        printf("]\n");
    }
    
    printf("\nMPO consistency check: %s\n", mpo_is_consistent(mpo) ? "PASSED" : "FAILED");
}

void print_dense_tensor_data(const struct dense_tensor* t, const char* name) {
    printf("\n=== Dense Tensor: %s ===\n", name);
    printf("Shape: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%ld%s", t->dim[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("]\n");
    
    printf("Data type: ");
    switch (t->dtype) {
        case CT_SINGLE_REAL:    printf("float32\n"); break;
        case CT_DOUBLE_REAL:    printf("float64\n"); break;
        case CT_SINGLE_COMPLEX: printf("complex64\n"); break;
        case CT_DOUBLE_COMPLEX: printf("complex128\n"); break;
        default:                printf("unknown\n"); break;
    }
    
    long total_elements = 1;
    for (int i = 0; i < t->ndim; i++) {
        total_elements *= t->dim[i];
    }
    printf("Total elements: %ld\n", total_elements);
    
    // Print data (limit to reasonable size)
    long max_print = (total_elements < 100) ? total_elements : 100;
    printf("\nFirst %ld elements (row-major order):\n", max_print);
    
    if (t->dtype == CT_SINGLE_REAL) {
        float* data = (float*)t->data;
        for (long i = 0; i < max_print; i++) {
            printf("%.6f%s", data[i], (i + 1) % 10 == 0 ? "\n" : ", ");
        }
    } else if (t->dtype == CT_DOUBLE_REAL) {
        double* data = (double*)t->data;
        for (long i = 0; i < max_print; i++) {
            printf("%.6f%s", data[i], (i + 1) % 10 == 0 ? "\n" : ", ");
        }
    } else if (t->dtype == CT_SINGLE_COMPLEX) {
        scomplex* data = (scomplex*)t->data;
        for (long i = 0; i < max_print; i++) {
            printf("(%.4f%+.4fi)%s", crealf(data[i]), cimagf(data[i]), 
                   (i + 1) % 5 == 0 ? "\n" : ", ");
        }
    } else if (t->dtype == CT_DOUBLE_COMPLEX) {
        dcomplex* data = (dcomplex*)t->data;
        for (long i = 0; i < max_print; i++) {
            printf("(%.4f%+.4fi)%s", creal(data[i]), cimag(data[i]), 
                   (i + 1) % 5 == 0 ? "\n" : ", ");
        }
    }
    if (max_print % 10 != 0) printf("\n");
    if (total_elements > max_print) {
        printf("... (%ld more elements not shown)\n", total_elements - max_print);
    }
}

void test_ising_1d_mpo() {
    print_separator("Test 1: 1D Ising Model MPO");
    
    const int nsites = 4;
    const double J = 1.0;   // Coupling strength
    const double h = 0.5;   // Transverse field
    const double g = 0.3;   // Longitudinal field
    
    printf("Parameters:\n");
    printf("  Number of sites: %d\n", nsites);
    printf("  J (coupling): %.2f\n", J);
    printf("  h (transverse field): %.2f\n", h);
    printf("  g (longitudinal field): %.2f\n", g);
    printf("\nHamiltonian: H = -J * sum_i(Z_i * Z_{i+1}) - h * sum_i(X_i) - g * sum_i(Z_i)\n");
    
    // Construct the MPO assembly
    struct mpo_assembly assembly;
    construct_ising_1d_mpo_assembly(nsites, J, h, g, &assembly);
    
    printf("\nAssembly information:\n");
    printf("  Physical dimension: %ld\n", assembly.d);
    printf("  Number of local operators: %d\n", assembly.num_local_ops);
    printf("  Number of coefficients: %d\n", assembly.num_coeffs);
    printf("  Data type: %s\n", assembly.dtype == CT_DOUBLE_REAL ? "float64" : "other");
    
    // Construct MPO from assembly
    struct mpo mpo;
    mpo_from_assembly(&assembly, &mpo);
    
    print_mpo_info("Ising 1D MPO", &mpo);
    
    // Print a sample tensor from the middle site
    if (nsites > 1) {
        print_subsection("Sample: Middle site tensor (site 1)");
        struct dense_tensor middle_tensor;
        block_sparse_to_dense_tensor(&mpo.a[1], &middle_tensor);
        print_dense_tensor_data(&middle_tensor, "Site 1 Tensor");
        delete_dense_tensor(&middle_tensor);
    }
    
    // Convert to matrix representation
    print_subsection("Full Hamiltonian Matrix");
    struct block_sparse_tensor mat;
    mpo_to_matrix(&mpo, &mat);
    
    struct dense_tensor mat_dense;
    block_sparse_to_dense_tensor(&mat, &mat_dense);
    
    printf("Matrix dimensions: [%ld, %ld, %ld, %ld]\n",
           mat_dense.dim[0], mat_dense.dim[1], mat_dense.dim[2], mat_dense.dim[3]);
    printf("Physical space dimension: %ld x %ld\n", mat_dense.dim[1], mat_dense.dim[2]);
    
    // Flatten to get the actual Hamiltonian matrix
    long total_dim = mat_dense.dim[1];
    for (int i = 0; i < nsites - 1; i++) {
        total_dim *= mpo.d;
    }
    printf("Full Hilbert space dimension: %ld\n", total_dim);
    
    print_dense_tensor_data(&mat_dense, "Full Hamiltonian");
    
    delete_dense_tensor(&mat_dense);
    delete_block_sparse_tensor(&mat);
    delete_mpo(&mpo);
    delete_mpo_assembly(&assembly);
}

void test_heisenberg_xxz_1d_mpo() {
    print_separator("Test 2: 1D Heisenberg XXZ Model MPO");
    
    const int nsites = 3;
    const double J = 1.0;   // Exchange coupling
    const double D = 0.5;   // Anisotropy
    const double h = 0.2;   // Magnetic field
    
    printf("Parameters:\n");
    printf("  Number of sites: %d\n", nsites);
    printf("  J (exchange): %.2f\n", J);
    printf("  D (anisotropy): %.2f\n", D);
    printf("  h (magnetic field): %.2f\n", h);
    printf("\nHamiltonian: H = -J * sum_i[(X_i*X_{i+1} + Y_i*Y_{i+1} + D*Z_i*Z_{i+1})] - h * sum_i(Z_i)\n");
    
    // Construct the MPO assembly
    struct mpo_assembly assembly;
    construct_heisenberg_xxz_1d_mpo_assembly(nsites, J, D, h, &assembly);
    
    printf("\nAssembly information:\n");
    printf("  Physical dimension: %ld\n", assembly.d);
    printf("  Number of local operators: %d\n", assembly.num_local_ops);
    printf("  Number of coefficients: %d\n", assembly.num_coeffs);
    printf("  Data type: %s\n", assembly.dtype == CT_DOUBLE_COMPLEX ? "complex128" : "other");
    
    // Construct MPO from assembly
    struct mpo mpo;
    mpo_from_assembly(&assembly, &mpo);
    
    print_mpo_info("Heisenberg XXZ 1D MPO", &mpo);
    
    // Print the first site tensor
    print_subsection("First site tensor (site 0)");
    struct dense_tensor first_tensor;
    block_sparse_to_dense_tensor(&mpo.a[0], &first_tensor);
    print_dense_tensor_data(&first_tensor, "Site 0 Tensor");
    delete_dense_tensor(&first_tensor);
    
    // Convert to matrix representation
    print_subsection("Full Hamiltonian Matrix");
    struct block_sparse_tensor mat;
    mpo_to_matrix(&mpo, &mat);
    
    struct dense_tensor mat_dense;
    block_sparse_to_dense_tensor(&mat, &mat_dense);
    
    printf("Matrix dimensions: [%ld, %ld, %ld, %ld]\n",
           mat_dense.dim[0], mat_dense.dim[1], mat_dense.dim[2], mat_dense.dim[3]);
    
    print_dense_tensor_data(&mat_dense, "Full Hamiltonian");
    
    delete_dense_tensor(&mat_dense);
    delete_block_sparse_tensor(&mat);
    delete_mpo(&mpo);
    delete_mpo_assembly(&assembly);
}

void test_bose_hubbard_1d_mpo() {
    print_separator("Test 3: 1D Bose-Hubbard Model MPO");
    
    const int nsites = 3;
    const long d = 3;       // Physical dimension (max occupancy = d-1 = 2)
    const double t = 1.0;   // Hopping
    const double u = 2.0;   // On-site interaction
    const double mu = 0.5;  // Chemical potential
    
    printf("Parameters:\n");
    printf("  Number of sites: %d\n", nsites);
    printf("  Physical dimension: %ld (max occupancy: %ld)\n", d, d-1);
    printf("  t (hopping): %.2f\n", t);
    printf("  u (interaction): %.2f\n", u);
    printf("  mu (chemical potential): %.2f\n", mu);
    printf("\nHamiltonian: H = -t*sum_i(b^dag_i*b_{i+1} + h.c.) + (u/2)*sum_i(n_i(n_i-1)) - mu*sum_i(n_i)\n");
    
    // Construct the MPO assembly
    struct mpo_assembly assembly;
    construct_bose_hubbard_1d_mpo_assembly(nsites, d, t, u, mu, &assembly);
    
    printf("\nAssembly information:\n");
    printf("  Physical dimension: %ld\n", assembly.d);
    printf("  Number of local operators: %d\n", assembly.num_local_ops);
    printf("  Number of coefficients: %d\n", assembly.num_coeffs);
    printf("  Data type: %s\n", assembly.dtype == CT_DOUBLE_REAL ? "float64" : "other");
    
    // Construct MPO from assembly
    struct mpo mpo;
    mpo_from_assembly(&assembly, &mpo);
    
    print_mpo_info("Bose-Hubbard 1D MPO", &mpo);
    
    // Print the first site tensor
    print_subsection("First site tensor (site 0)");
    struct dense_tensor first_tensor;
    block_sparse_to_dense_tensor(&mpo.a[0], &first_tensor);
    print_dense_tensor_data(&first_tensor, "Site 0 Tensor");
    delete_dense_tensor(&first_tensor);
    
    // Convert to matrix representation
    print_subsection("Full Hamiltonian Matrix");
    struct block_sparse_tensor mat;
    mpo_to_matrix(&mpo, &mat);
    
    struct dense_tensor mat_dense;
    block_sparse_to_dense_tensor(&mat, &mat_dense);
    
    printf("Matrix dimensions: [%ld, %ld, %ld, %ld]\n",
           mat_dense.dim[0], mat_dense.dim[1], mat_dense.dim[2], mat_dense.dim[3]);
    
    print_dense_tensor_data(&mat_dense, "Full Hamiltonian");
    
    delete_dense_tensor(&mat_dense);
    delete_block_sparse_tensor(&mat);
    delete_mpo(&mpo);
    delete_mpo_assembly(&assembly);
}

void test_mpo_merge_operations() {
    print_separator("Test 4: MPO Tensor Merge Operations");
    
    const int nsites = 2;
    const double J = 1.0;
    const double h = 0.0;
    const double g = 0.0;
    
    printf("Creating a simple 2-site Ising MPO for merge test\n");
    printf("Parameters: nsites=%d, J=%.1f, h=%.1f, g=%.1f\n", nsites, J, h, g);
    
    struct mpo_assembly assembly;
    construct_ising_1d_mpo_assembly(nsites, J, h, g, &assembly);
    
    struct mpo mpo;
    mpo_from_assembly(&assembly, &mpo);
    
    print_mpo_info("Original 2-site MPO", &mpo);
    
    // Print individual site tensors
    print_subsection("Site 0 Tensor");
    struct dense_tensor tensor0;
    block_sparse_to_dense_tensor(&mpo.a[0], &tensor0);
    print_dense_tensor_data(&tensor0, "Site 0");
    
    print_subsection("Site 1 Tensor");
    struct dense_tensor tensor1;
    block_sparse_to_dense_tensor(&mpo.a[1], &tensor1);
    print_dense_tensor_data(&tensor1, "Site 1");
    
    // Merge the two tensors
    print_subsection("Merged Tensor");
    struct block_sparse_tensor merged;
    mpo_merge_tensor_pair(&mpo.a[0], &mpo.a[1], &merged);
    
    struct dense_tensor merged_dense;
    block_sparse_to_dense_tensor(&merged, &merged_dense);
    print_dense_tensor_data(&merged_dense, "Merged (Site 0 + Site 1)");
    
    printf("\nVerification: Merged tensor dimensions should be [1, 4, 4, 1]\n");
    printf("Actual dimensions: [%ld, %ld, %ld, %ld]\n",
           merged_dense.dim[0], merged_dense.dim[1], 
           merged_dense.dim[2], merged_dense.dim[3]);
    
    delete_dense_tensor(&merged_dense);
    delete_block_sparse_tensor(&merged);
    delete_dense_tensor(&tensor1);
    delete_dense_tensor(&tensor0);
    delete_mpo(&mpo);
    delete_mpo_assembly(&assembly);
}

void mpo_tests() {
    printf("=========================================\n");
    printf("MPO (Matrix Product Operator) Test Suite\n");
    printf("=========================================\n");
    printf("This test suite demonstrates MPO construction\n");
    printf("and operations for comparison with Mojo implementation.\n");
    
    test_ising_1d_mpo();
    test_heisenberg_xxz_1d_mpo();
    test_bose_hubbard_1d_mpo();
    test_mpo_merge_operations();
    
    print_separator("All MPO Tests Completed");
    printf("\nSummary:\n");
    printf("  - Created and analyzed multiple MPO models\n");
    printf("  - Verified MPO consistency checks\n");
    printf("  - Converted MPOs to matrix representations\n");
    printf("  - Tested tensor merge operations\n");
    printf("\nThese outputs can be used for comparison with Mojo implementation.\n");
}