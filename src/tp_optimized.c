// optimized_tp_mem.c
// Comparaison avec l'ancien code (AoS + ASCII + 2 passes) :
//  1)First Optim:
// - Avant: cplx {re,im} (AoS). 
// -Maintenant: deux tableaux re[]/im[] (SoA) donc mieux pour SIMD et le cache.
// 2)second Optim:
// - Avant: génération + lecture en ASCII (fprintf/fscanf) ce qui fait que c'etait très lent. 
// -Maintenant: génération directe en mémoire (pas d'I/O=> gain enorme).
// 3)third Optim:
// - Avant: une passe pour calculer une w puis une 2e passe pour l'énergie (compute_energy) -> 2 parcours mémoire.
//   Maintenant: je calcule l'énergie dans la même boucle -> 1 seul parcours.
// 4)fourth Optim:
// - Avant: indices via IDX() et accès à u[IDX(...)] en struct.
// - Maintenant: id = row + j et accès scalaires ce qui fait qu'il y'a moins d'indirections.
// 5)fifth Optim:
// - Avant: pas d'alignement garanti. 
// -Maintenant: allocations alignées + assume_aligned ==> aide la vectorisation.
// 6)Sixth Optim:
// - Avant: pas d'info d'aliasing. 
// -Maintenant: restrict -> le compilateur peut vectoriser plus agressivement.
// 7)seventh Optim:
// - Avant: swap de pointeurs sur cplx*. 
// -Maintenant: swap sur 2 pointeurs (re et im) -> même idée, coût O(1).

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// Allocation alignée (contrairement à malloc classique de l'ancien code)
static inline void* xaligned_malloc(size_t alignment, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
}

static inline double frand01(void) {
    return (double)rand() / (double)RAND_MAX;
}

// Remplace generate_input_ascii + load_field_ascii :
static void generate_input_mem(double *re, double *im, int N) {
    srand(0);
    const size_t NN = (size_t)N * (size_t)N;
    for (size_t k = 0; k < NN; k++) {
        re[k] = frand01();
        im[k] = frand01();
    }
}

static inline size_t IDX(int i, int j, int N) {
    return (size_t)i * (size_t)N + (size_t)j;
}

void timestep_optimized(double * restrict ure,
                        double * restrict uim,
                        double * restrict unewre,
                        double * restrict unewim,
                        int N, int steps,
                        double alpha, double beta, double gamma)
{
    // Nouveau par rapport à l'ancien: j'indique l'alignement -> facilite les loads/stores SIMD.
    ure    = (double*)__builtin_assume_aligned(ure,    64);
    uim    = (double*)__builtin_assume_aligned(uim,    64);
    unewre = (double*)__builtin_assume_aligned(unewre, 64);
    unewim = (double*)__builtin_assume_aligned(unewim, 64);

    const double a = alpha, b = beta, g = gamma;

    for (int t = 0; t < steps; t++) {

        double E = 0.0;

        // Contrairement à l'ancien (j puis i), je mets j en interne pour du stride-1 en C (row-major).
        for (int i = 1; i < N-1; i++) {
            const size_t row = (size_t)i * (size_t)N;

            // Nouveau: directive SIMD + réduction (dans l'ancien code, boucle scalaire)
            #pragma omp simd reduction(+:E)
            for (int j = 1; j < N-1; j++) {

                const size_t id = row + (size_t)j;

                // Avant: cplx v = u[id]; maintenant: deux scalaires -> plus simple à vectoriser
                const double vre = ure[id];
                const double vim = uim[id];

                const double upre = ure[id - (size_t)N];
                const double upim = uim[id - (size_t)N];

                const double dnre = ure[id + (size_t)N];
                const double dnim = uim[id + (size_t)N];

                const double lfre = ure[id - 1];
                const double lfim = uim[id - 1];

                const double rgre = ure[id + 1];
                const double rgim = uim[id + 1];

                const double lap_re = upre + dnre + lfre + rgre - 4.0 * vre;
                const double lap_im = upim + dnim + lfim + rgim - 4.0 * vim;

                const double r2 = vre*vre + vim*vim;
                const double f  = exp(-g * r2);

                const double nre = vre + a * lap_re + b * f * vre;
                const double nim = vim + a * lap_im + b * f * vim;

                unewre[id] = nre;
                unewim[id] = nim;

                // Avant: compute_energy(unew) faisait un 2e parcours mémoire.
                // Maintenant: je calcule E en même temps -> moins de trafic mémoire.
                E += nre*nre + nim*nim;
            }
        }

        printf("step %d energy = %.10e\n", t, E);

        // Même logique que l'ancien: swap, mais sur deux tableaux au lieu d'un tableau de struct.
        double *tmp;
        tmp = ure; ure = unewre; unewre = tmp;
        tmp = uim; uim = unewim; unewim = tmp;
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("usage: %s N steps\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    const int steps = atoi(argv[2]);

    const double alpha = 0.17, beta = 0.031, gamma = 0.92;
    const size_t NN = (size_t)N * (size_t)N;

    // Avant: malloc(N*N*sizeof(cplx)) en AoS. Maintenant: 4 malloc alignés en SoA.
    double *ure    = (double*)xaligned_malloc(64, NN * sizeof(double));
    double *uim    = (double*)xaligned_malloc(64, NN * sizeof(double));
    double *unewre = (double*)xaligned_malloc(64, NN * sizeof(double));
    double *unewim = (double*)xaligned_malloc(64, NN * sizeof(double));

    if (!ure || !uim || !unewre || !unewim) {
        fprintf(stderr, "aligned malloc failed\n");
        return 1;
    }

    // Avant: generate_input_ascii("input.dat") puis load_field_ascii("input.dat", u).
    // Maintenant: génération directe en mémoire.
    printf("Generating random input in memory\n");
    generate_input_mem(ure, uim, N);

    printf("Computing %d time steps\n", steps);
    timestep_optimized(ure, uim, unewre, unewim, N, steps, alpha, beta, gamma);

    free(ure); free(uim); free(unewre); free(unewim);
    return 0;
}
