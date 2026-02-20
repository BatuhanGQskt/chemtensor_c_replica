#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../src/operator/mpo.h"
#include "../src/tensor/block_sparse_tensor.h"
#include "../src/tensor/dense_tensor.h"
#include "../src/numeric.h"
#include "../src/aligned_memory.h"
#include "mpo_export_json.h"


static void print_separator(const char* title) {
    printf("\n");
    printf("=========================================\n");
    printf("%s\n", title);
    printf("=========================================\n");
}

static void print_subsection(const char* title) {
    printf("\n--- %s ---\n", title);
}

void print_mpo_info(const char* name, const struct mpo* mpo) {
    print_subsection(name);
    
    printf("Number of sites: %d\n", mpo->nsites);
    printf("Physical dimension: %ld\n", mpo->d);
    
    printf("Physical quantum numbers (qsite): [");
    for (long i = 0; i < mpo->d; i++) {
        printf("%d%s", mpo->qsite[i], (i < mpo->d - 1) ? ", " : "");
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
        case CT_SINGLE_REAL: printf("float32\n"); break;
        case CT_DOUBLE_REAL: printf("float64\n"); break;
        default:             printf("unknown\n"); break;
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
    }
    if (max_print % 10 != 0) printf("\n");
    if (total_elements > max_print) {
        printf("... (%ld more elements not shown)\n", total_elements - max_print);
    }
}

//________________________________________________________________________________________________________________________
//
// Helper functions to create MPO tensors directly
//

// Create Pauli matrices (as 2x2 dense tensors)
void create_pauli_x(struct dense_tensor* X) {
    const long dim[2] = { 2, 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, X);
    double* data = (double*)X->data;
    // X = [[0, 1], [1, 0]]
    data[0] = 0.0; data[1] = 1.0;
    data[2] = 1.0; data[3] = 0.0;
}

/* Raising operator S+ = |0><1| = [[0, 1], [0, 0]] (real) */
void create_raising_operator(struct dense_tensor* Sp) {
    const long dim[2] = { 2, 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, Sp);
    double* data = (double*)Sp->data;
    data[0] = 0.0; data[1] = 1.0;
    data[2] = 0.0; data[3] = 0.0;
}

/* Lowering operator S- = |1><0| = [[0, 0], [1, 0]] (real) */
void create_lowering_operator(struct dense_tensor* Sm) {
    const long dim[2] = { 2, 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, Sm);
    double* data = (double*)Sm->data;
    data[0] = 0.0; data[1] = 0.0;
    data[2] = 1.0; data[3] = 0.0;
}

void create_pauli_z(struct dense_tensor* Z) {
    const long dim[2] = { 2, 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, Z);
    double* data = (double*)Z->data;
    // Z = [[1, 0], [0, -1]]
    data[0] = 1.0;  data[1] = 0.0;
    data[2] = 0.0;  data[3] = -1.0;
}

void create_identity_2x2(struct dense_tensor* Id) {
    const long dim[2] = { 2, 2 };
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, dim, Id);
    double* data = (double*)Id->data;
    // Id = [[1, 0], [0, 1]]
    data[0] = 1.0; data[1] = 0.0;
    data[2] = 0.0; data[3] = 1.0;
}

// Helper to copy a 2x2 operator into a specific [Wl, p_in, p_out, Wr] position in the MPO tensor
void set_mpo_operator_at(struct dense_tensor* mpo_tensor, long wl, long wr, 
                         const struct dense_tensor* op, double scale) {
    assert(mpo_tensor->ndim == 4);
    assert(op->ndim == 2);
    assert(op->dim[0] == mpo_tensor->dim[1]);  // physical in
    assert(op->dim[1] == mpo_tensor->dim[2]);  // physical out
    
    double* mpo_data = (double*)mpo_tensor->data;
    const double* op_data = (const double*)op->data;
    
    const long d = op->dim[0];
    const long Wr = mpo_tensor->dim[3];
    
    // Add operator into position [wl, :, :, wr] (use += to properly sum multiple operators)
    for (long i = 0; i < d; i++) {
        for (long j = 0; j < d; j++) {
            long idx[4] = { wl, i, j, wr };
            long offset = tensor_index_to_offset(4, mpo_tensor->dim, idx);
            mpo_data[offset] += scale * op_data[i * d + j];
        }
    }
}

//________________________________________________________________________________________________________________________
//
// Create 1D Ising model MPO
// H = -J * sum_i(Z_i * Z_{i+1}) - h * sum_i(Z_i) - g * sum_i(X_i)
//

void create_ising_1d_mpo_tensors(const int nsites, const double J, const double h, const double g, 
                                  struct mpo* mpo) {
    const long d = 2;  // Physical dimension for qubits
    
    // Quantum numbers: [0, 0] for qubits (no symmetry)
    qnumber qsite[2] = { 0, 0 };
    
    // Bond dimensions: [1, 3, 3, ..., 3, 1]
    // W = [I, Z, -J*Z] for left boundary
    // W = [[  I  ],    for bulk
    //      [  Z  ],
    //      [-J*Z ]] on left, and [I, Z, -J*Z] on top for right
    // W = [[  I  ],    for right boundary
    //      [  Z  ],
    //      [-J*Z ]]
    
    long* dim_bonds = ct_malloc((nsites + 1) * sizeof(long));
    dim_bonds[0] = 1;
    for (int i = 1; i < nsites; i++) {
        dim_bonds[i] = 3;
    }
    dim_bonds[nsites] = 1;
    
    // Bond quantum numbers (all zeros for no symmetry)
    qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
    for (int i = 0; i <= nsites; i++) {
        qbonds[i] = ct_calloc(dim_bonds[i], sizeof(qnumber));
    }
    
    // Allocate MPO structure
    allocate_mpo(CT_DOUBLE_REAL, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mpo);
    
    // Create Pauli matrices
    struct dense_tensor X, Z, Id;
    create_pauli_x(&X);
    create_pauli_z(&Z);
    create_identity_2x2(&Id);
    
    // Fill in MPO tensors for each site
    for (int i = 0; i < nsites; i++) {
        struct dense_tensor site_tensor;
        const long site_dim[4] = { dim_bonds[i], d, d, dim_bonds[i+1] };
        allocate_dense_tensor(CT_DOUBLE_REAL, 4, site_dim, &site_tensor);
        
        // Initialize to zero
        double* data = (double*)site_tensor.data;
        long total_elements = 1;
        for (int j = 0; j < 4; j++) total_elements *= site_dim[j];
        memset(data, 0, total_elements * sizeof(double));
        
        if (i == 0) {
            // Left boundary: shape [1, d, d, 3]
            // W[0] = [ -h*Z - g*X,  Id,  Z ]
            set_mpo_operator_at(&site_tensor, 0, 0, &Z, -h);  // -h*Z at [0,0]
            set_mpo_operator_at(&site_tensor, 0, 0, &X, -g);  // -g*X at [0,0] (add)
            set_mpo_operator_at(&site_tensor, 0, 1, &Id, 1.0);  // Id at [0,1]
            set_mpo_operator_at(&site_tensor, 0, 2, &Z, 1.0);  // Z at [0,2]
        }
        else if (i == nsites - 1) {
            // Right boundary: shape [3, d, d, 1]
            // W[-1] = [ Id ]
            //         [ Z ]
            //         [-J*Z - h*Z - g*X ]
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);   // Id at [0,0]
            set_mpo_operator_at(&site_tensor, 1, 0, &Z, 1.0);   // Z at [1,0]
            set_mpo_operator_at(&site_tensor, 2, 0, &Z, -J);    // -J*Z at [2,0]
            set_mpo_operator_at(&site_tensor, 2, 0, &Z, -h);    // -h*Z at [2,0] (add)
            set_mpo_operator_at(&site_tensor, 2, 0, &X, -g);    // -g*X at [2,0] (add)
        }
        else {
            // Bulk sites: shape [3, d, d, 3]
            // W[i] = [ Id    0    0 ]
            //        [ Z    0    0 ]
            //        [-h*Z-g*X  Id  Z ]
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);   // Id at [0,0]
            set_mpo_operator_at(&site_tensor, 1, 0, &Z, 1.0);   // Z at [1,0]
            set_mpo_operator_at(&site_tensor, 2, 0, &Z, -h);    // -h*Z at [2,0]
            set_mpo_operator_at(&site_tensor, 2, 0, &X, -g);    // -g*X at [2,0] (add)
            set_mpo_operator_at(&site_tensor, 2, 1, &Id, 1.0);   // Id at [2,1]
            set_mpo_operator_at(&site_tensor, 2, 2, &Z, 1.0);   // Z at [2,2]
        }
        
        // Convert dense tensor to block sparse tensor
        dense_to_block_sparse_tensor_entries(&site_tensor, &mpo->a[i]);
        
        delete_dense_tensor(&site_tensor);
    }
    
    // Clean up
    delete_dense_tensor(&X);
    delete_dense_tensor(&Z);
    delete_dense_tensor(&Id);
    
    for (int i = 0; i <= nsites; i++) {
        ct_free(qbonds[i]);
    }
    ct_free(qbonds);
    ct_free(dim_bonds);
}

//________________________________________________________________________________________________________________________
//
// Create 1D Heisenberg XXZ model MPO (real only: S+, S- instead of X, Y)
// H = -J * sum_i[(X_i*X_{i+1} + Y_i*Y_{i+1} + D*Z_i*Z_{i+1})] - h * sum_i(Z_i)
// Uses XX + YY = 2*(S+ S- + S- S+) so bond 2 = S+, bond 3 = S- (matches Mojo).
//

void create_heisenberg_xxz_1d_mpo_tensors(const int nsites, const double J, const double D, const double h,
                                           struct mpo* mpo) {
    const long d = 2;
    qnumber qsite[2] = { 0, 0 };
    
    // Bond dimensions: [1, 5, 5, ..., 5, 1]
    long* dim_bonds = ct_malloc((nsites + 1) * sizeof(long));
    dim_bonds[0] = 1;
    for (int i = 1; i < nsites; i++) {
        dim_bonds[i] = 5;
    }
    dim_bonds[nsites] = 1;
    
    qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
    for (int i = 0; i <= nsites; i++) {
        qbonds[i] = ct_calloc(dim_bonds[i], sizeof(qnumber));
    }
    
    allocate_mpo(CT_DOUBLE_REAL, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mpo);
    
    struct dense_tensor Z, Id, Sp, Sm;
    create_pauli_z(&Z);
    create_identity_2x2(&Id);
    create_raising_operator(&Sp);
    create_lowering_operator(&Sm);
    
    for (int i = 0; i < nsites; i++) {
        struct dense_tensor site_tensor;
        const long site_dim[4] = { dim_bonds[i], d, d, dim_bonds[i+1] };
        allocate_dense_tensor(CT_DOUBLE_REAL, 4, site_dim, &site_tensor);
        
        double* data = (double*)site_tensor.data;
        long total_elements = 1;
        for (int j = 0; j < 4; j++) total_elements *= site_dim[j];
        memset(data, 0, total_elements * sizeof(double));
        
        if (i == 0) {
            // Left boundary: [1, d, d, 5]  W[0] = [ -h*Z,  I,  S+,  S-,  D*Z ]
            set_mpo_operator_at(&site_tensor, 0, 0, &Z, -h);
            set_mpo_operator_at(&site_tensor, 0, 1, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 0, 2, &Sp, 1.0);
            set_mpo_operator_at(&site_tensor, 0, 3, &Sm, 1.0);
            set_mpo_operator_at(&site_tensor, 0, 4, &Z, D);
        }
        else if (i == nsites - 1) {
            // Right boundary: [5, d, d, 1]  [ I, -2J*S-, -2J*S+, -J*Z, -h*Z ]^T
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 1, 0, &Sm, -2.0 * J);
            set_mpo_operator_at(&site_tensor, 2, 0, &Sp, -2.0 * J);
            set_mpo_operator_at(&site_tensor, 3, 0, &Z, -J);
            set_mpo_operator_at(&site_tensor, 4, 0, &Z, -h);
        }
        else {
            // Bulk: [5, d, d, 5]
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 1, 0, &Sm, -2.0 * J);
            set_mpo_operator_at(&site_tensor, 2, 0, &Sp, -2.0 * J);
            set_mpo_operator_at(&site_tensor, 3, 0, &Z, -J);
            set_mpo_operator_at(&site_tensor, 4, 0, &Z, -h);
            set_mpo_operator_at(&site_tensor, 4, 1, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 4, 2, &Sp, 1.0);
            set_mpo_operator_at(&site_tensor, 4, 3, &Sm, 1.0);
            set_mpo_operator_at(&site_tensor, 4, 4, &Z, D);
        }
        
        dense_to_block_sparse_tensor_entries(&site_tensor, &mpo->a[i]);
        delete_dense_tensor(&site_tensor);
    }
    
    delete_dense_tensor(&Z);
    delete_dense_tensor(&Id);
    delete_dense_tensor(&Sp);
    delete_dense_tensor(&Sm);
    
    for (int i = 0; i <= nsites; i++) {
        ct_free(qbonds[i]);
    }
    ct_free(qbonds);
    ct_free(dim_bonds);
}

//________________________________________________________________________________________________________________________
//
// Create 1D Bose-Hubbard model MPO
// H = -t*sum_i(b^dag_i*b_{i+1} + h.c.) + (u/2)*sum_i(n_i(n_i-1)) - mu*sum_i(n_i)
//

void create_bose_hubbard_1d_mpo_tensors(const int nsites, const long d, const double t, 
                                        const double u, const double mu, struct mpo* mpo) {
    // Quantum numbers: [0, 1, 2, ..., d-1]
    qnumber* qsite = ct_malloc(d * sizeof(qnumber));
    for (long i = 0; i < d; i++) {
        qsite[i] = 0;  // No symmetry for simplicity
    }
    
    // Bond dimensions: [1, 4, 4, ..., 4, 1]
    long* dim_bonds = ct_malloc((nsites + 1) * sizeof(long));
    dim_bonds[0] = 1;
    for (int i = 1; i < nsites; i++) {
        dim_bonds[i] = 4;
    }
    dim_bonds[nsites] = 1;
    
    qnumber** qbonds = ct_malloc((nsites + 1) * sizeof(qnumber*));
    for (int i = 0; i <= nsites; i++) {
        qbonds[i] = ct_calloc(dim_bonds[i], sizeof(qnumber));
    }
    
    allocate_mpo(CT_DOUBLE_REAL, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mpo);
    
    // Create bosonic operators
    struct dense_tensor b_dag, b, n, n_n_minus_1, Id;
    const long op_dim[2] = { d, d };
    
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, op_dim, &b_dag);
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, op_dim, &b);
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, op_dim, &n);
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, op_dim, &n_n_minus_1);
    allocate_dense_tensor(CT_DOUBLE_REAL, 2, op_dim, &Id);
    
    double* b_dag_data = (double*)b_dag.data;
    double* b_data = (double*)b.data;
    double* n_data = (double*)n.data;
    double* n_n_data = (double*)n_n_minus_1.data;
    double* Id_data = (double*)Id.data;
    
    memset(b_dag_data, 0, d*d*sizeof(double));
    memset(b_data, 0, d*d*sizeof(double));
    memset(n_data, 0, d*d*sizeof(double));
    memset(n_n_data, 0, d*d*sizeof(double));
    memset(Id_data, 0, d*d*sizeof(double));
    
    // b^dag[i,j] = sqrt(j) * delta_{i,j+1}
    // b[i,j] = sqrt(i) * delta_{i+1,j}
    // n[i,j] = i * delta_{i,j}
    // n(n-1)[i,j] = i*(i-1) * delta_{i,j}
    // Id[i,j] = delta_{i,j}
    
    for (long i = 0; i < d; i++) {
        for (long j = 0; j < d; j++) {
            if (j > 0 && i == j - 1) {
                b_dag_data[i * d + j] = sqrt((double)j);
            }
            if (i > 0 && j == i - 1) {
                b_data[i * d + j] = sqrt((double)i);
            }
            if (i == j) {
                n_data[i * d + j] = (double)i;
                n_n_data[i * d + j] = (double)(i * (i - 1));
                Id_data[i * d + j] = 1.0;
            }
        }
    }
    
    // Fill in MPO tensors
    for (int i = 0; i < nsites; i++) {
        struct dense_tensor site_tensor;
        const long site_dim[4] = { dim_bonds[i], d, d, dim_bonds[i+1] };
        allocate_dense_tensor(CT_DOUBLE_REAL, 4, site_dim, &site_tensor);
        
        double* data = (double*)site_tensor.data;
        long total_elements = 1;
        for (int j = 0; j < 4; j++) total_elements *= site_dim[j];
        memset(data, 0, total_elements * sizeof(double));
        
        if (i == 0) {
            // Left boundary: [1, d, d, 4]
            // W[0] = [ (u/2)*n(n-1) - mu*n,  Id,  b^dag,  b ]
            set_mpo_operator_at(&site_tensor, 0, 0, &n_n_minus_1, u/2.0);
            set_mpo_operator_at(&site_tensor, 0, 0, &n, -mu);
            set_mpo_operator_at(&site_tensor, 0, 1, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 0, 2, &b_dag, 1.0);
            set_mpo_operator_at(&site_tensor, 0, 3, &b, 1.0);
        }
        else if (i == nsites - 1) {
            // Right boundary: [4, d, d, 1]
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 1, 0, &b, -t);
            set_mpo_operator_at(&site_tensor, 2, 0, &b_dag, -t);
            set_mpo_operator_at(&site_tensor, 3, 0, &n_n_minus_1, u/2.0);
            set_mpo_operator_at(&site_tensor, 3, 0, &n, -mu);
        }
        else {
            // Bulk: [4, d, d, 4]
            set_mpo_operator_at(&site_tensor, 0, 0, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 1, 0, &b, -t);
            set_mpo_operator_at(&site_tensor, 2, 0, &b_dag, -t);
            set_mpo_operator_at(&site_tensor, 3, 0, &n_n_minus_1, u/2.0);
            set_mpo_operator_at(&site_tensor, 3, 0, &n, -mu);
            set_mpo_operator_at(&site_tensor, 3, 1, &Id, 1.0);
            set_mpo_operator_at(&site_tensor, 3, 2, &b_dag, 1.0);
            set_mpo_operator_at(&site_tensor, 3, 3, &b, 1.0);
        }
        
        dense_to_block_sparse_tensor_entries(&site_tensor, &mpo->a[i]);
        delete_dense_tensor(&site_tensor);
    }
    
    delete_dense_tensor(&b_dag);
    delete_dense_tensor(&b);
    delete_dense_tensor(&n);
    delete_dense_tensor(&n_n_minus_1);
    delete_dense_tensor(&Id);
    
    for (int i = 0; i <= nsites; i++) {
        ct_free(qbonds[i]);
    }
    ct_free(qbonds);
    ct_free(dim_bonds);
    ct_free(qsite);
}

//________________________________________________________________________________________________________________________
//
// Test functions
//

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
    printf("\nHamiltonian: H = -J * sum_i(Z_i * Z_{i+1}) - h * sum_i(Z_i) - g * sum_i(X_i)\n");
    printf("\nCreating MPO directly from dense tensors...\n");
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
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
    printf("\nCreating MPO directly from dense tensors...\n");
    
    struct mpo mpo;
    create_heisenberg_xxz_1d_mpo_tensors(nsites, J, D, h, &mpo);
    
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
    printf("\nCreating MPO directly from dense tensors...\n");
    
    struct mpo mpo;
    create_bose_hubbard_1d_mpo_tensors(nsites, d, t, u, mu, &mpo);
    
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
}

void test_mpo_merge_operations() {
    print_separator("Test 4: MPO Tensor Merge Operations");
    
    const int nsites = 2;
    const double J = 1.0;
    const double h = 0.0;
    const double g = 0.0;
    
    printf("Creating a simple 2-site Ising MPO for merge test\n");
    printf("Parameters: nsites=%d, J=%.1f, h=%.1f, g=%.1f\n", nsites, J, h, g);
    printf("\nCreating MPO directly from dense tensors...\n");
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
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
}

//________________________________________________________________________________________________________________________
//
// Test 5: MPO site tensor transpose (swap physical in/out)
// [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]  perm = [0, 2, 1, 3]
//
void test_mpo_site_transpose() {
    print_separator("Test 5: MPO Site Tensor Transpose");
    
    const int nsites = 2;
    const double J = 1.0, h = 0.0, g = 0.0;
    
    printf("Transpose site 0 tensor: [Wl, d_in, d_out, Wr] -> [Wl, d_out, d_in, Wr]\n");
    printf("Permutation: [0, 2, 1, 3] (swap physical in and out indices)\n\n");
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
    struct dense_tensor site0;
    block_sparse_to_dense_tensor(&mpo.a[0], &site0);
    
    print_subsection("Original Site 0 (before transpose)");
    print_dense_tensor_data(&site0, "Site 0 [1,2,2,3]");
    
    struct dense_tensor site0_transposed;
    const int perm[4] = { 0, 2, 1, 3 };  /* [Wl, d_out, d_in, Wr] */
    transpose_dense_tensor(perm, &site0, &site0_transposed);
    
    print_subsection("Site 0 Transposed [1,2,2,3] -> [1,2,2,3]");
    print_dense_tensor_data(&site0_transposed, "Site 0 transposed (perm 0,2,1,3)");
    
    delete_dense_tensor(&site0_transposed);
    delete_dense_tensor(&site0);
    delete_mpo(&mpo);
}

//________________________________________________________________________________________________________________________
//
// Test 6: MPO site tensor reshape
// [Wl, d_in, d_out, Wr] -> [Wl*d_in*d_out, Wr] for matrix representation
//
void test_mpo_site_reshape() {
    print_separator("Test 6: MPO Site Tensor Reshape");
    
    const int nsites = 2;
    const double J = 1.0, h = 0.0, g = 0.0;
    
    printf("Reshape site 0 tensor: [1, 2, 2, 3] -> [4, 3]\n");
    printf("Flatten dims 0,1,2 into rows; keep Wr as columns\n\n");
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
    struct dense_tensor site0;
    block_sparse_to_dense_tensor(&mpo.a[0], &site0);
    
    print_subsection("Original Site 0 [1,2,2,3]");
    print_dense_tensor_data(&site0, "Site 0");
    
    struct dense_tensor site0_reshaped;
    copy_dense_tensor(&site0, &site0_reshaped);
    const long new_dim[2] = { 4, 3 };  /* 1*2*2=4, 3 */
    reshape_dense_tensor(2, new_dim, &site0_reshaped);
    
    print_subsection("Site 0 Reshaped to [4, 3]");
    print_dense_tensor_data(&site0_reshaped, "Site 0 reshaped [4,3]");
    
    delete_dense_tensor(&site0_reshaped);
    delete_dense_tensor(&site0);
    delete_mpo(&mpo);
}

//________________________________________________________________________________________________________________________
//
// Test 7: MPO site tensor scale
//
void test_mpo_site_scale() {
    print_separator("Test 7: MPO Site Tensor Scale");
    
    const int nsites = 2;
    const double J = 1.0, h = 0.0, g = 0.0;
    const double scale_factor = 2.0;
    
    printf("Scale site 0 tensor by factor %.1f\n\n", scale_factor);
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
    struct dense_tensor site0;
    block_sparse_to_dense_tensor(&mpo.a[0], &site0);
    
    print_subsection("Original Site 0 (before scale)");
    print_dense_tensor_data(&site0, "Site 0");
    
    scale_dense_tensor(&scale_factor, &site0);
    
    print_subsection("Site 0 Scaled by 2.0");
    print_dense_tensor_data(&site0, "Site 0 scaled");
    
    delete_dense_tensor(&site0);
    delete_mpo(&mpo);
}

//________________________________________________________________________________________________________________________
//
// Test 8: Merged tensor reshape to matrix
// [1, 4, 4, 1] -> [4, 4] Hamiltonian matrix
//
void test_mpo_merged_reshape_to_matrix() {
    print_separator("Test 8: Merged MPO Reshape to Matrix");
    
    const int nsites = 2;
    const double J = 1.0, h = 0.0, g = 0.0;
    
    printf("Merge sites 0+1, then reshape [1,4,4,1] -> [4,4] Hamiltonian matrix\n\n");
    
    struct mpo mpo;
    create_ising_1d_mpo_tensors(nsites, J, h, g, &mpo);
    
    struct block_sparse_tensor merged;
    mpo_merge_tensor_pair(&mpo.a[0], &mpo.a[1], &merged);
    
    struct dense_tensor merged_dense;
    block_sparse_to_dense_tensor(&merged, &merged_dense);
    
    print_subsection("Merged Tensor [1,4,4,1]");
    print_dense_tensor_data(&merged_dense, "Merged");
    
    /* Reshape to [4, 4] matrix (1*4*4*1 = 16 elements) */
    const long mat_dim[2] = { 4, 4 };
    reshape_dense_tensor(2, mat_dim, &merged_dense);
    
    print_subsection("Reshaped to Hamiltonian Matrix [4,4]");
    print_dense_tensor_data(&merged_dense, "H [4x4]");
    
    delete_dense_tensor(&merged_dense);
    delete_block_sparse_tensor(&merged);
    delete_mpo(&mpo);
}

void export_reference_data() {
    print_separator("Exporting Reference Data for Mojo Comparison");
    
    // 1. 4-site Ising: J=1.0, h=0.5, g=0.3
    printf("\n1. Creating 4-site Ising MPO (J=1.0, h=0.5, g=0.3)...\n");
    struct mpo ising_4site;
    create_ising_1d_mpo_tensors(4, 1.0, 0.5, 0.3, &ising_4site);
    export_mpo_to_json(&ising_4site, "generated/mpo/ising_1d_J1.0_h0.5_g0.3_n4.json");
    delete_mpo(&ising_4site);
    
    // 2. 2-site Ising pure ZZ: J=1.0, h=0.0, g=0.0 (for analytical checks)
    printf("\n2. Creating 2-site Ising MPO (J=1.0, h=0.0, g=0.0) [pure ZZ]...\n");
    struct mpo ising_2site;
    create_ising_1d_mpo_tensors(2, 1.0, 0.0, 0.0, &ising_2site);
    export_mpo_to_json(&ising_2site, "generated/mpo/ising_1d_J1.0_h0.0_g0.0_n2.json");
    delete_mpo(&ising_2site);
    
    // 3. 3-site Heisenberg XXZ: J=1.0, D=0.5, h=0.2
    printf("\n3. Creating 3-site Heisenberg XXZ MPO (J=1.0, D=0.5, h=0.2)...\n");
    struct mpo heisenberg_xxz_3site;
    create_heisenberg_xxz_1d_mpo_tensors(3, 1.0, 0.5, 0.2, &heisenberg_xxz_3site);
    export_mpo_to_json(&heisenberg_xxz_3site, "generated/mpo/heisenberg_xxz_J1.0_D0.5_h0.2_n3.json");
    delete_mpo(&heisenberg_xxz_3site);
    
    // 4. 3-site Bose-Hubbard: d=3, t=1.0, u=2.0, mu=0.5
    printf("\n4. Creating 3-site Bose-Hubbard MPO (d=3, t=1.0, u=2.0, mu=0.5)...\n");
    struct mpo bose_hubbard_3site;
    create_bose_hubbard_1d_mpo_tensors(3, 3, 1.0, 2.0, 0.5, &bose_hubbard_3site);
    export_mpo_to_json(&bose_hubbard_3site, "generated/mpo/bose_hubbard_d3_t1.0_u2.0_mu0.5_n3.json");
    delete_mpo(&bose_hubbard_3site);
    
    printf("\n✅ All reference data files exported successfully!\n");
}

void mpo_tests() {
    printf("=========================================\n");
    printf("MPO (Matrix Product Operator) Test Suite\n");
    printf("=========================================\n");
    printf("This test suite demonstrates MPO construction\n");
    printf("using direct tensor creation (no assemblies).\n");
    printf("This approach matches the Mojo implementation.\n");
    
    test_ising_1d_mpo();
    test_heisenberg_xxz_1d_mpo();
    test_bose_hubbard_1d_mpo();
    test_mpo_merge_operations();
    test_mpo_site_transpose();
    test_mpo_site_reshape();
    test_mpo_site_scale();
    test_mpo_merged_reshape_to_matrix();
    
    print_separator("All MPO Tests Completed");
    printf("\nSummary:\n");
    printf("  - Created MPOs directly from dense tensors\n");
    printf("  - Verified MPO consistency checks\n");
    printf("  - Converted MPOs to matrix representations\n");
    printf("  - Tested tensor merge operations\n");
    printf("  - Tested transpose, reshape, scale on MPO site tensors\n");
    printf("\nThese outputs can be used for comparison with Mojo implementation.\n");
    
    // Export reference data
    export_reference_data();
}
