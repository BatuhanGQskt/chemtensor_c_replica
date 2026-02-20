/// \file mps.c
/// \brief Manual tests and export for MPS (product state) for Mojo comparison.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/state/mps.h"
#include "../src/tensor/block_sparse_tensor.h"
#include "../src/tensor/dense_tensor.h"
#include "../src/aligned_memory.h"
#include "../src/util/rng.h"
#include "mps_export_json.h"

static void print_sep(const char* title) {
    printf("\n=========================================\n");
    printf("%s\n", title);
    printf("=========================================\n");
}

static void print_sub(const char* title) {
    printf("\n--- %s ---\n", title);
}

/* Build a product state MPS: at each site the local state is |basis[i]>.
 * bond_dims are all 1. Uses real dense data. */
static void construct_product_state_mps(int nsites, long d, const int* basis, struct mps* mps) {
    qnumber* qsite = ct_calloc((size_t)d, sizeof(qnumber));
    for (long i = 0; i < d; i++)
        qsite[i] = 0;

    long* dim_bonds = ct_malloc((size_t)(nsites + 1) * sizeof(long));
    for (int i = 0; i <= nsites; i++)
        dim_bonds[i] = 1;

    qnumber** qbonds = ct_malloc((size_t)(nsites + 1) * sizeof(qnumber*));
    for (int i = 0; i <= nsites; i++) {
        qbonds[i] = ct_malloc(sizeof(qnumber));
        qbonds[i][0] = 0;
    }

    allocate_mps(CT_DOUBLE_REAL, nsites, d, qsite, dim_bonds, (const qnumber**)qbonds, mps);
    ct_free(qsite);
    ct_free(dim_bonds);
    for (int i = 0; i <= nsites; i++)
        ct_free(qbonds[i]);
    ct_free(qbonds);

    /* Fill each site: single block [1, d, 1], set data[basis[i]] = 1 */
    for (int i = 0; i < nsites; i++) {
        struct dense_tensor* block = mps->a[i].blocks[0];
        if (!block)
            continue;
        double* data = (double*)block->data;
        for (long p = 0; p < d; p++)
            data[p] = (p == (long)basis[i]) ? 1.0 : 0.0;
    }
}

static int run_mps_tests(void) {
    int failed = 0;

    print_sep("MPS export (product state) for Mojo tests");

    /* Product state |0,0,0> on 3 sites, d=2 */
    {
        const int nsites = 3;
        const long d = 2;
        int basis[] = { 0, 0, 0 };

        struct mps psi;
        construct_product_state_mps(nsites, d, basis, &psi);

        if (!mps_is_consistent(&psi)) {
            fprintf(stderr, "MPS consistency check failed\n");
            delete_mps(&psi);
            return 1;
        }

        double n = mps_norm(&psi);
        printf("Product state |0,0,0> (3 sites, d=2): norm = %.10f\n", n);

        export_mps_observables_to_json(&psi, "generated/mps/mps_product_3sites_d2_basis_000.json");
        delete_mps(&psi);
    }

    /* Product state |1,0,1> on 3 sites, d=2 */
    {
        const int nsites = 3;
        const long d = 2;
        int basis[] = { 1, 0, 1 };

        struct mps psi;
        construct_product_state_mps(nsites, d, basis, &psi);

        if (!mps_is_consistent(&psi)) {
            fprintf(stderr, "MPS consistency check failed\n");
            delete_mps(&psi);
            return 1;
        }

        double n = mps_norm(&psi);
        printf("Product state |1,0,1> (3 sites, d=2): norm = %.10f\n", n);

        export_mps_observables_to_json(&psi, "generated/mps/mps_product_3sites_d2_basis_101.json");
        delete_mps(&psi);
    }

    /* --- Extensive MPS tests --- */
    print_sep("MPS extensive tests");

    /* 1. Single-site product state */
    print_sub("Single-site product state");
    {
        const int nsites = 1;
        const long d = 2;
        int basis[] = { 0 };
        struct mps psi;
        construct_product_state_mps(nsites, d, basis, &psi);
        if (!mps_is_consistent(&psi)) { fprintf(stderr, "consistency failed\n"); delete_mps(&psi); return 1; }
        double n = mps_norm(&psi);
        if (fabs(n - 1.0) > 1e-10) { fprintf(stderr, "norm %.10f != 1\n", n); delete_mps(&psi); return 1; }
        if (mps_bond_dim(&psi, 0) != 1 || mps_bond_dim(&psi, 1) != 1) { fprintf(stderr, "bond dims\n"); delete_mps(&psi); return 1; }
        printf("  single site |0> norm=1, bond_dims=[1,1] OK\n");
        delete_mps(&psi);
    }

    /* 2. Two-site and four-site product states, d=2 and d=3 */
    print_sub("Multi-site product states (norm = 1)");
    {
        struct { int nsites; long d; int basis[8]; int n; } cases[] = {
            { 2, 2, { 0, 1 }, 2 },
            { 4, 2, { 1, 1, 0, 0 }, 4 },
            { 3, 3, { 0, 1, 2 }, 3 },
        };
        for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
            struct mps psi;
            construct_product_state_mps(cases[c].nsites, cases[c].d, cases[c].basis, &psi);
            if (!mps_is_consistent(&psi)) { fprintf(stderr, "consistency failed case %zu\n", c); delete_mps(&psi); return 1; }
            double n = mps_norm(&psi);
            if (fabs(n - 1.0) > 1e-10) { fprintf(stderr, "norm %.10f != 1 case %zu\n", n, c); delete_mps(&psi); return 1; }
            printf("  %d sites d=%ld norm=%.10f OK\n", cases[c].nsites, cases[c].d, n);
            delete_mps(&psi);
        }
    }

    /* 3. Random MPS: consistency and positive norm */
    print_sub("Random MPS (consistency, norm > 0)");
    {
        const long d = 2;
        const qnumber qsite[2] = { 0, 0 };
        struct rng_state rng;
        seed_rng_state(12345, &rng);
        struct mps psi;
        construct_random_mps(CT_DOUBLE_REAL, 4, d, qsite, 0, 8, &rng, &psi);
        if (!mps_is_consistent(&psi)) { fprintf(stderr, "random MPS consistency failed\n"); delete_mps(&psi); return 1; }
        double n = mps_norm(&psi);
        if (n <= 0.0) { fprintf(stderr, "random MPS norm <= 0\n"); delete_mps(&psi); return 1; }
        printf("  random MPS 4 sites d=2 max_vdim=8 norm=%.10f OK\n", n);
        delete_mps(&psi);
    }

    /* 4. copy_mps and mps_allclose */
    print_sub("copy_mps and mps_allclose");
    {
        const int nsites = 3;
        const long d = 2;
        int basis[] = { 1, 0, 1 };
        struct mps psi, copy;
        construct_product_state_mps(nsites, d, basis, &psi);
        copy_mps(&psi, &copy);
        if (!mps_allclose(&psi, &copy, 1e-14)) { fprintf(stderr, "copy not allclose\n"); delete_mps(&copy); delete_mps(&psi); return 1; }
        printf("  copy_mps allclose OK\n");
        delete_mps(&copy);
        delete_mps(&psi);
    }

    /* 5. mps_vdot: same state => 1, orthogonal product states => 0 */
    print_sub("mps_vdot (overlap)");
    {
        const int nsites = 3;
        const long d = 2;
        struct mps psi_000, psi_101, psi_111;
        int b000[] = { 0, 0, 0 }, b101[] = { 1, 0, 1 }, b111[] = { 1, 1, 1 };
        construct_product_state_mps(nsites, d, b000, &psi_000);
        construct_product_state_mps(nsites, d, b101, &psi_101);
        construct_product_state_mps(nsites, d, b111, &psi_111);

        double v_000_000, v_000_101, v_101_101;
        mps_vdot(&psi_000, &psi_000, &v_000_000);
        mps_vdot(&psi_000, &psi_101, &v_000_101);
        mps_vdot(&psi_101, &psi_101, &v_101_101);

        if (fabs(v_000_000 - 1.0) > 1e-10) { fprintf(stderr, "<000|000> = %.10f\n", v_000_000); failed = 1; }
        if (fabs(v_000_101) > 1e-10) { fprintf(stderr, "<000|101> = %.10f (expect 0)\n", v_000_101); failed = 1; }
        if (fabs(v_101_101 - 1.0) > 1e-10) { fprintf(stderr, "<101|101> = %.10f\n", v_101_101); failed = 1; }
        printf("  <000|000>=1 <000|101>=0 <101|101>=1 OK\n");

        delete_mps(&psi_111);
        delete_mps(&psi_101);
        delete_mps(&psi_000);
        if (failed) return 1;
    }

    /* 6. mps_add: chi + psi => statevector equals chi_vec + psi_vec */
    print_sub("mps_add (statevector sum)");
    {
        int add_failed = 0;
        const int nsites = 2;
        const long d = 2;
        int b0[] = { 0, 0 }, b1[] = { 1, 1 };
        struct mps chi, psi, sum;
        construct_product_state_mps(nsites, d, b0, &chi);
        construct_product_state_mps(nsites, d, b1, &psi);
        mps_add(&chi, &psi, &sum);

        struct block_sparse_tensor v_chi, v_psi, v_sum;
        mps_to_statevector(&chi, &v_chi);
        mps_to_statevector(&psi, &v_psi);
        mps_to_statevector(&sum, &v_sum);

        struct dense_tensor d_chi, d_psi, d_sum;
        block_sparse_to_dense_tensor(&v_chi, &d_chi);
        block_sparse_to_dense_tensor(&v_psi, &d_psi);
        block_sparse_to_dense_tensor(&v_sum, &d_sum);

        double* x = (double*)d_chi.data;
        double* y = (double*)d_psi.data;
        double* z = (double*)d_sum.data;
        long len = 4; /* d^nsites */
        for (long i = 0; i < len; i++) {
            if (fabs(z[i] - (x[i] + y[i])) > 1e-10) {
                fprintf(stderr, "mps_add statevector[%ld] %.10f != %.10f + %.10f\n", i, z[i], x[i], y[i]);
                add_failed = 1;
                break;
            }
        }
        printf("  mps_add statevector equals chi + psi OK\n");

        delete_dense_tensor(&d_sum);
        delete_dense_tensor(&d_psi);
        delete_dense_tensor(&d_chi);
        delete_block_sparse_tensor(&v_sum);
        delete_block_sparse_tensor(&v_psi);
        delete_block_sparse_tensor(&v_chi);
        delete_mps(&sum);
        delete_mps(&psi);
        delete_mps(&chi);
        if (add_failed) return 1;
    }

    /* 7. mps_to_statevector: product |0,0> has coeff at index 0 = 1, rest 0 */
    print_sub("mps_to_statevector (product state)");
    {
        const int nsites = 2;
        const long d = 2;
        int basis[] = { 0, 0 };
        struct mps psi;
        construct_product_state_mps(nsites, d, basis, &psi);
        struct block_sparse_tensor vec;
        mps_to_statevector(&psi, &vec);
        struct dense_tensor vec_d;
        block_sparse_to_dense_tensor(&vec, &vec_d);
        double* v = (double*)vec_d.data;
        long L = 4;
        if (fabs(v[0] - 1.0) > 1e-10) { fprintf(stderr, "v[0]=%.10f\n", v[0]); delete_dense_tensor(&vec_d); delete_block_sparse_tensor(&vec); delete_mps(&psi); return 1; }
        for (long i = 1; i < L; i++)
            if (fabs(v[i]) > 1e-10) { fprintf(stderr, "v[%ld]=%.10f\n", i, v[i]); delete_dense_tensor(&vec_d); delete_block_sparse_tensor(&vec); delete_mps(&psi); return 1; }
        printf("  |00> statevector [1,0,0,0] OK\n");
        delete_dense_tensor(&vec_d);
        delete_block_sparse_tensor(&vec);
        delete_mps(&psi);
    }

    /* 8. Bond dimensions product state */
    print_sub("Bond dimensions (product state)");
    {
        const int nsites = 5;
        const long d = 3;
        int basis[] = { 0, 1, 2, 0, 1 };
        struct mps psi;
        construct_product_state_mps(nsites, d, basis, &psi);
        for (int i = 0; i <= nsites; i++)
            if (mps_bond_dim(&psi, i) != 1) { fprintf(stderr, "bond_dim[%d]=%ld\n", i, mps_bond_dim(&psi, i)); delete_mps(&psi); return 1; }
        printf("  bond_dims all 1 OK\n");
        delete_mps(&psi);
    }

    printf("\nAll MPS tests passed.\n");
    return 0;
}

void mps_tests(void) {
    if (run_mps_tests() != 0)
        fprintf(stderr, "MPS tests failed\n");
}
