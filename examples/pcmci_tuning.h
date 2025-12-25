/**
 * @file pcmci_tuning.h
 * @brief CPU and BLAS tuning for consistent, low-latency performance
 *
 * Call pcmci_tuning_init() at the start of main() before any BLAS calls.
 *
 * Key optimizations:
 *   1. Flush denormals to zero (100x speedup for edge cases)
 *   2. Fixed thread count (no dynamic scaling jitter)
 *   3. Thread affinity hints for hybrid CPUs
 *   4. Windows process priority and timer resolution
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef PCMCI_TUNING_H
#define PCMCI_TUNING_H

#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* MKL-specific includes */
#ifdef PCMCI_USE_MKL
#include <mkl.h>
#endif

/* SSE/AVX control for denormals */
#ifdef _MSC_VER
#include <intrin.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#else
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * DENORMAL FLUSHING
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Flush denormals to zero - critical for consistent performance!
     * Denormal operations can be 100x slower than normal floats.
     */
    static inline void pcmci_tuning_flush_denormals(void)
    {
        /* FTZ: Flush To Zero - denormal results become zero */
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

        /* DAZ: Denormals Are Zero - denormal inputs treated as zero */
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * THREAD CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Configure threading for hybrid CPUs (P-cores + E-cores)
     *
     * @param n_threads Number of threads (0 = auto-detect)
     *
     * For Intel 12th/13th/14th gen hybrid CPUs:
     *   - Set to P-core count for lowest latency
     *   - i9-14900KF: 8 P-cores
     *   - i7-13700K:  8 P-cores
     *   - i5-13600K:  6 P-cores
     */
    static inline void pcmci_tuning_set_threads(int n_threads)
    {
        int num_threads;

        if (n_threads > 0)
        {
            num_threads = n_threads;
        }
        else
        {
#ifdef _OPENMP
            num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
        }

#ifdef PCMCI_USE_MKL
        mkl_set_num_threads(num_threads);
#endif

#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * MKL CONFIGURATION
     *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef PCMCI_USE_MKL
    /**
     * Configure MKL for maximum performance
     */
    static inline void pcmci_tuning_configure_mkl(void)
    {
        /* Disable MKL's internal dynamic thread scaling */
        mkl_set_dynamic(0);

        /* CBWR: Conditional Bitwise Reproducibility
         * Ensures identical results across runs */
        mkl_cbwr_set(MKL_CBWR_AVX2);
    }
#endif

    /*═══════════════════════════════════════════════════════════════════════════════
     * WINDOWS-SPECIFIC TUNING
     *═══════════════════════════════════════════════════════════════════════════════*/

#ifdef _MSC_VER

    static inline void pcmci_tuning_windows_priority(void)
    {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }

    static inline void pcmci_tuning_timer_begin(void)
    {
        timeBeginPeriod(1);
    }

    static inline void pcmci_tuning_timer_end(void)
    {
        timeEndPeriod(1);
    }

    static inline void pcmci_tuning_windows_init(void)
    {
        pcmci_tuning_windows_priority();
        pcmci_tuning_timer_begin();
    }

#endif /* _MSC_VER */

    /*═══════════════════════════════════════════════════════════════════════════════
     * MAIN INIT FUNCTION
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Initialize all tuning settings
     *
     * @param n_threads Number of threads (0 = auto, or set to P-core count)
     * @param verbose   Print configuration info
     *
     * Usage:
     *   int main(void) {
     *       pcmci_tuning_init(8, 1);  // 8 threads, verbose
     *       // ... rest of program
     *       pcmci_tuning_cleanup();
     *       return 0;
     *   }
     */
    static inline void pcmci_tuning_init(int n_threads, int verbose)
    {
        /* 1. Flush denormals (do this first!) */
        pcmci_tuning_flush_denormals();

        /* 2. Set thread count */
        pcmci_tuning_set_threads(n_threads);

#ifdef PCMCI_USE_MKL
        /* 3. Configure MKL */
        pcmci_tuning_configure_mkl();
#endif

#ifdef _MSC_VER
        /* 4. Windows-specific tuning */
        pcmci_tuning_windows_init();
#endif

        if (verbose)
        {
            printf("PCMCI+ Tuning Configuration\n");
            printf("===========================\n");
            printf("  Denormals:     FLUSH TO ZERO (FTZ+DAZ)\n");
#ifdef PCMCI_USE_MKL
            printf("  BLAS backend:  Intel MKL\n");
            printf("  MKL threads:   %d\n", mkl_get_max_threads());
            printf("  MKL dynamic:   %s\n", mkl_get_dynamic() ? "ON" : "OFF");

            MKLVersion version;
            mkl_get_version(&version);
            printf("  MKL version:   %d.%d.%d\n", version.MajorVersion,
                   version.MinorVersion, version.UpdateVersion);
#else
        printf("  BLAS backend:  OpenBLAS\n");
#endif
#ifdef _OPENMP
            printf("  OMP threads:   %d\n", omp_get_max_threads());
#endif
#ifdef _MSC_VER
            printf("  Windows:       HIGH priority, timer=1ms\n");
#endif
            printf("\n");
        }
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * CLEANUP
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Cleanup function - call before program exit
     */
    static inline void pcmci_tuning_cleanup(void)
    {
#ifdef _MSC_VER
        pcmci_tuning_timer_end();
#endif
    }

    /*═══════════════════════════════════════════════════════════════════════════════
     * AFFINITY HINTS
     *═══════════════════════════════════════════════════════════════════════════════*/

    /**
     * Print thread affinity hints for hybrid Intel CPUs
     */
    static inline void pcmci_tuning_print_hints(void)
    {
        printf("\n");
        printf("Thread Affinity for Intel Hybrid CPUs (12th/13th/14th Gen)\n");
        printf("==========================================================\n");
        printf("\n");
        printf("Windows (PowerShell, before running):\n");
        printf("  $env:KMP_AFFINITY=\"granularity=fine,compact,1,0\"\n");
        printf("  $env:KMP_BLOCKTIME=\"0\"\n");
        printf("  $env:MKL_NUM_THREADS=\"8\"   # Set to P-core count\n");
        printf("  $env:OMP_NUM_THREADS=\"8\"\n");
        printf("\n");
        printf("Linux:\n");
        printf("  export KMP_AFFINITY=granularity=fine,compact,1,0\n");
        printf("  export KMP_BLOCKTIME=0\n");
        printf("  taskset -c 0-7 ./pcmci_benchmark\n");
        printf("\n");
        printf("For lowest latency on i9-14900KF:\n");
        printf("  - Disable E-cores and Hyper-Threading in BIOS\n");
        printf("  - Set static CPU frequency (no P-state transitions)\n");
        printf("  - Disable C-states deeper than C1\n");
        printf("\n");
    }

#ifdef __cplusplus
}
#endif

#endif /* PCMCI_TUNING_H */