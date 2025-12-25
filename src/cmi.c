/**
 * @file cmi.c
 * @brief Conditional Mutual Information using KSG k-NN estimator
 *
 * Implements the Kraskov-Stögbauer-Grassberger (KSG) estimator for CMI.
 * Uses a KD-tree for efficient k-nearest neighbor queries.
 *
 * Algorithm (KSG Type 1):
 *   1. Normalize all variables to [0,1] range
 *   2. For each point i, find k-th NN distance ε_i in joint space (X,Y,Z)
 *   3. Count neighbors within ε_i in marginal spaces
 *   4. CMI = ψ(k) - <ψ(n_xz+1) + ψ(n_yz+1) - ψ(n_z+1)>
 *
 * Safety improvements:
 *   - No alloca() in recursive functions
 *   - Iterative tree traversal to avoid stack overflow
 *   - NULL checks on all allocations
 *   - Proper workspace management
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Must be before math.h for M_PI on Windows */
#define _USE_MATH_DEFINES

#include "pcmci_cmi.h"
#include "pcmci_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

/* Fallback M_PI definition if still not defined */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*============================================================================
 * Digamma Function
 *============================================================================*/

/**
 * Digamma function using asymptotic expansion + recurrence
 *
 * For x >= 6: use asymptotic series
 * For x < 6: use ψ(x) = ψ(x+1) - 1/x
 */
double pcmci_digamma(double x)
{
    /* Handle special cases */
    if (x <= 0.0)
    {
        if (x == (int)x)
            return -INFINITY; /* Pole */
        /* Reflection formula for negative x */
        return pcmci_digamma(1.0 - x) - M_PI / tan(M_PI * x);
    }

    double result = 0.0;

    /* Use recurrence to shift x >= 6 */
    while (x < 6.0)
    {
        result -= 1.0 / x;
        x += 1.0;
    }

    /* Asymptotic expansion for x >= 6 */
    double x2 = 1.0 / (x * x);
    result += log(x) - 0.5 / x - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 252.0)));

    return result;
}

/* Pre-computed digamma values for small integers (speed optimization) */
static double digamma_cache[1024];
static int digamma_cache_init = 0;

static void init_digamma_cache(void)
{
    if (digamma_cache_init)
        return;

#pragma omp critical
    {
        if (!digamma_cache_init)
        {
            for (int i = 1; i < 1024; i++)
            {
                digamma_cache[i] = pcmci_digamma((double)i);
            }
            digamma_cache_init = 1;
        }
    }
}

static inline double fast_digamma(int n)
{
    if (n > 0 && n < 1024)
    {
        return digamma_cache[n];
    }
    return pcmci_digamma((double)n);
}

/*============================================================================
 * KD-Tree Implementation (Iterative traversal to avoid stack overflow)
 *============================================================================*/

/** KD-tree node */
typedef struct kdnode
{
    int32_t idx;       /* Point index (-1 for internal nodes) */
    int32_t split_dim; /* Split dimension */
    double split_val;  /* Split value */
    struct kdnode *left;
    struct kdnode *right;
} kdnode_t;

/** KD-tree structure */
struct pcmci_kdtree
{
    kdnode_t *root;
    const double *data; /* Reference to original data */
    int32_t n;
    int32_t dim;
    int32_t max_depth; /* Track tree depth for stack allocation */
};

/* Quick select for median finding */
static int32_t partition(int32_t *indices, const double *vals, int32_t left, int32_t right, int32_t pivot_idx)
{
    double pivot_val = vals[indices[pivot_idx]];

    /* Move pivot to end */
    int32_t tmp = indices[pivot_idx];
    indices[pivot_idx] = indices[right];
    indices[right] = tmp;

    int32_t store_idx = left;
    for (int32_t i = left; i < right; i++)
    {
        if (vals[indices[i]] < pivot_val)
        {
            tmp = indices[store_idx];
            indices[store_idx] = indices[i];
            indices[i] = tmp;
            store_idx++;
        }
    }

    tmp = indices[store_idx];
    indices[store_idx] = indices[right];
    indices[right] = tmp;

    return store_idx;
}

static int32_t quickselect_median(int32_t *indices, const double *vals, int32_t left, int32_t right)
{
    int32_t k = (left + right) / 2;

    while (left < right)
    {
        int32_t pivot_idx = (left + right) / 2;
        pivot_idx = partition(indices, vals, left, right, pivot_idx);

        if (k == pivot_idx)
        {
            return k;
        }
        else if (k < pivot_idx)
        {
            right = pivot_idx - 1;
        }
        else
        {
            left = pivot_idx + 1;
        }
    }
    return left;
}

/* Find dimension with maximum spread */
static int32_t find_split_dim(const double *data, const int32_t *indices,
                              int32_t n_pts, int32_t dim, int32_t n_total)
{
    int32_t best_dim = 0;
    double best_spread = -1.0;

    for (int32_t d = 0; d < dim; d++)
    {
        double min_val = DBL_MAX, max_val = -DBL_MAX;
        for (int32_t i = 0; i < n_pts; i++)
        {
            double v = data[indices[i] + d * n_total];
            if (v < min_val)
                min_val = v;
            if (v > max_val)
                max_val = v;
        }
        double spread = max_val - min_val;
        if (spread > best_spread)
        {
            best_spread = spread;
            best_dim = d;
        }
    }
    return best_dim;
}

/* Recursive tree building - depth tracked */
static kdnode_t *kdtree_build_recursive(const double *data, int32_t *indices,
                                        int32_t n_pts, int32_t dim, int32_t n_total,
                                        int32_t depth, int32_t *max_depth)
{
    if (n_pts == 0)
        return NULL;

    if (depth > *max_depth)
        *max_depth = depth;

    kdnode_t *node = (kdnode_t *)malloc(sizeof(kdnode_t));
    if (!node)
        return NULL; /* Allocation check */

    if (n_pts == 1)
    {
        /* Leaf node */
        node->idx = indices[0];
        node->split_dim = -1;
        node->left = NULL;
        node->right = NULL;
        return node;
    }

    /* Find best split dimension */
    int32_t split_dim = find_split_dim(data, indices, n_pts, dim, n_total);

    /* Get values for this dimension */
    const double *dim_data = data + split_dim * n_total;

    /* Find median */
    int32_t median_idx = quickselect_median(indices, dim_data, 0, n_pts - 1);

    node->idx = -1; /* Internal node */
    node->split_dim = split_dim;
    node->split_val = dim_data[indices[median_idx]];

    /* Recursively build subtrees */
    node->left = kdtree_build_recursive(data, indices, median_idx, dim, n_total,
                                        depth + 1, max_depth);
    node->right = kdtree_build_recursive(data, indices + median_idx + 1,
                                         n_pts - median_idx - 1, dim, n_total,
                                         depth + 1, max_depth);

    return node;
}

pcmci_kdtree_t *pcmci_kdtree_build(const double *data, int32_t n, int32_t dim)
{
    pcmci_kdtree_t *tree = (pcmci_kdtree_t *)malloc(sizeof(pcmci_kdtree_t));
    if (!tree)
        return NULL;

    tree->data = data;
    tree->n = n;
    tree->dim = dim;
    tree->max_depth = 0;

    /* Create index array */
    int32_t *indices = (int32_t *)malloc(n * sizeof(int32_t));
    if (!indices)
    {
        free(tree);
        return NULL;
    }

    for (int32_t i = 0; i < n; i++)
    {
        indices[i] = i;
    }

    tree->root = kdtree_build_recursive(data, indices, n, dim, n, 0, &tree->max_depth);

    free(indices);

    if (!tree->root)
    {
        free(tree);
        return NULL;
    }

    return tree;
}

static void kdtree_free_recursive(kdnode_t *node)
{
    if (!node)
        return;
    kdtree_free_recursive(node->left);
    kdtree_free_recursive(node->right);
    free(node);
}

void pcmci_kdtree_free(pcmci_kdtree_t *tree)
{
    if (!tree)
        return;
    kdtree_free_recursive(tree->root);
    free(tree);
}

/*============================================================================
 * k-NN Query - ITERATIVE with explicit stack (no recursion)
 *============================================================================*/

/* Chebyshev (max/infinity) distance - inline for column-major data */
static inline double chebyshev_dist_colmajor(const double *query, const double *data,
                                             int32_t idx, int32_t n, int32_t dim)
{
    double max_diff = 0.0;
    for (int32_t d = 0; d < dim; d++)
    {
        double diff = fabs(query[d] - data[idx + d * n]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

/* Priority queue (max-heap) for k-NN */
typedef struct
{
    int32_t *indices;
    double *distances;
    int32_t k;
    int32_t count;
} knn_heap_t;

static void heap_push(knn_heap_t *h, int32_t idx, double dist)
{
    if (h->count < h->k)
    {
        /* Heap not full, add element */
        int32_t i = h->count++;
        h->indices[i] = idx;
        h->distances[i] = dist;

        /* Bubble up (max-heap) */
        while (i > 0)
        {
            int32_t parent = (i - 1) / 2;
            if (h->distances[parent] < h->distances[i])
            {
                double tmp_d = h->distances[parent];
                h->distances[parent] = h->distances[i];
                h->distances[i] = tmp_d;
                int32_t tmp_i = h->indices[parent];
                h->indices[parent] = h->indices[i];
                h->indices[i] = tmp_i;
                i = parent;
            }
            else
                break;
        }
    }
    else if (dist < h->distances[0])
    {
        /* Replace max element */
        h->indices[0] = idx;
        h->distances[0] = dist;

        /* Bubble down */
        int32_t i = 0;
        while (1)
        {
            int32_t left = 2 * i + 1;
            int32_t right = 2 * i + 2;
            int32_t largest = i;

            if (left < h->k && h->distances[left] > h->distances[largest])
                largest = left;
            if (right < h->k && h->distances[right] > h->distances[largest])
                largest = right;

            if (largest != i)
            {
                double tmp_d = h->distances[i];
                h->distances[i] = h->distances[largest];
                h->distances[largest] = tmp_d;
                int32_t tmp_i = h->indices[i];
                h->indices[i] = h->indices[largest];
                h->indices[largest] = tmp_i;
                i = largest;
            }
            else
                break;
        }
    }
}

/* Stack entry for iterative traversal */
typedef struct
{
    kdnode_t *node;
    int8_t state; /* 0=first visit, 1=visited left, 2=done */
} stack_entry_t;

/**
 * Iterative k-NN search using explicit stack
 * Avoids recursion to prevent stack overflow with degenerate trees
 */
void pcmci_kdtree_knn(const pcmci_kdtree_t *tree, const double *query,
                      int32_t k, int32_t *indices, double *distances)
{
    if (!tree || !tree->root || !query || !indices || !distances)
        return;

    knn_heap_t heap = {indices, distances, k, 0};

    /* Initialize with infinity */
    for (int32_t i = 0; i < k; i++)
    {
        indices[i] = -1;
        distances[i] = DBL_MAX;
    }

    /* Allocate traversal stack - max size is 2 * tree depth */
    int32_t stack_size = (tree->max_depth + 2) * 2;
    stack_entry_t *stack = (stack_entry_t *)malloc(stack_size * sizeof(stack_entry_t));
    if (!stack)
        return;

    int32_t sp = 0; /* Stack pointer */

    /* Push root */
    stack[sp].node = tree->root;
    stack[sp].state = 0;
    sp++;

    while (sp > 0)
    {
        stack_entry_t *top = &stack[sp - 1];
        kdnode_t *node = top->node;

        if (!node)
        {
            sp--;
            continue;
        }

        if (node->idx >= 0)
        {
            /* Leaf node */
            double dist = chebyshev_dist_colmajor(query, tree->data, node->idx,
                                                  tree->n, tree->dim);
            heap_push(&heap, node->idx, dist);
            sp--;
            continue;
        }

        /* Internal node */
        double split_val = node->split_val;
        double query_val = query[node->split_dim];
        double diff = query_val - split_val;

        if (top->state == 0)
        {
            /* First visit - go to closer child first */
            top->state = 1;
            kdnode_t *first = (diff <= 0) ? node->left : node->right;
            if (first)
            {
                stack[sp].node = first;
                stack[sp].state = 0;
                sp++;
            }
        }
        else if (top->state == 1)
        {
            /* Visited closer child - maybe visit farther */
            top->state = 2;
            double max_dist = (heap.count < heap.k) ? DBL_MAX : heap.distances[0];

            if (fabs(diff) < max_dist)
            {
                kdnode_t *second = (diff <= 0) ? node->right : node->left;
                if (second)
                {
                    stack[sp].node = second;
                    stack[sp].state = 0;
                    sp++;
                }
            }
        }
        else
        {
            /* Done with this node */
            sp--;
        }
    }

    free(stack);
}

/*============================================================================
 * Radius Count Query - ITERATIVE
 *============================================================================*/

/**
 * Iterative radius count using explicit stack
 * @param stack Pre-allocated stack buffer (caller provides)
 * @param stack_size Size of stack buffer
 */
static int32_t kdtree_count_radius_iter(const pcmci_kdtree_t *tree, const double *query,
                                        double radius, int32_t exclude_idx,
                                        stack_entry_t *stack, int32_t stack_size)
{
    if (!tree || !tree->root || !stack)
        return 0;

    int32_t count = 0;
    int32_t sp = 0;

    /* Push root */
    stack[sp].node = tree->root;
    stack[sp].state = 0;
    sp++;

    while (sp > 0)
    {
        stack_entry_t *top = &stack[sp - 1];
        kdnode_t *node = top->node;

        if (!node)
        {
            sp--;
            continue;
        }

        if (node->idx >= 0)
        {
            /* Leaf node */
            if (node->idx != exclude_idx)
            {
                double dist = chebyshev_dist_colmajor(query, tree->data, node->idx,
                                                      tree->n, tree->dim);
                if (dist < radius)
                    count++;
            }
            sp--;
            continue;
        }

        /* Internal node */
        double split_val = node->split_val;
        double query_val = query[node->split_dim];

        if (top->state == 0)
        {
            /* First visit - check if we should visit left subtree */
            top->state = 1;
            if (query_val - radius <= split_val && node->left)
            {
                stack[sp].node = node->left;
                stack[sp].state = 0;
                sp++;
            }
        }
        else if (top->state == 1)
        {
            /* Check if we should visit right subtree */
            top->state = 2;
            if (query_val + radius > split_val && node->right)
            {
                stack[sp].node = node->right;
                stack[sp].state = 0;
                sp++;
            }
        }
        else
        {
            sp--;
        }
    }

    return count;
}

int32_t pcmci_kdtree_count_radius(const pcmci_kdtree_t *tree, const double *query, double radius)
{
    if (!tree)
        return 0;

    int32_t stack_size = (tree->max_depth + 2) * 2;
    stack_entry_t *stack = (stack_entry_t *)malloc(stack_size * sizeof(stack_entry_t));
    if (!stack)
        return 0;

    int32_t count = kdtree_count_radius_iter(tree, query, radius, -1, stack, stack_size);

    free(stack);
    return count;
}

/*============================================================================
 * Data Normalization
 *============================================================================*/

void pcmci_normalize_data(double *data, int32_t n, int32_t dim)
{
    if (!data)
        return;

    for (int32_t d = 0; d < dim; d++)
    {
        double *col = data + d * n;

        /* Find min/max */
        double min_val = col[0], max_val = col[0];
        for (int32_t i = 1; i < n; i++)
        {
            if (col[i] < min_val)
                min_val = col[i];
            if (col[i] > max_val)
                max_val = col[i];
        }

        /* Normalize to [0, 1] */
        double range = max_val - min_val;
        if (range > 1e-10)
        {
            double inv_range = 1.0 / range;
            for (int32_t i = 0; i < n; i++)
            {
                col[i] = (col[i] - min_val) * inv_range;
            }
        }
        else
        {
            /* Constant column - set to 0.5 */
            for (int32_t i = 0; i < n; i++)
            {
                col[i] = 0.5;
            }
        }
    }
}

/*============================================================================
 * KSG CMI Estimator
 *============================================================================*/

pcmci_cmi_config_t pcmci_cmi_default_config(void)
{
    pcmci_cmi_config_t config = {
        .k = 5,
        .n_perm = 100,
        .n_threads = 0,
        .use_chebyshev = true,
        .seed = 0};
    return config;
}

/**
 * Thread-local workspace for CMI computation
 */
typedef struct
{
    int32_t *nn_idx;
    double *nn_dist;
    double *query_xyz;
    double *query_xz;
    double *query_yz;
    double *query_z;
    stack_entry_t *stack;
    int32_t stack_size;
} cmi_workspace_t;

static cmi_workspace_t *cmi_workspace_alloc(int32_t k, int32_t dim_xyz, int32_t dim_xz,
                                            int32_t dim_yz, int32_t dim_z, int32_t max_depth)
{
    cmi_workspace_t *ws = (cmi_workspace_t *)malloc(sizeof(cmi_workspace_t));
    if (!ws)
        return NULL;

    ws->stack_size = (max_depth + 2) * 2;

    ws->nn_idx = (int32_t *)malloc((k + 1) * sizeof(int32_t));
    ws->nn_dist = (double *)malloc((k + 1) * sizeof(double));
    ws->query_xyz = (double *)malloc(dim_xyz * sizeof(double));
    ws->query_xz = (double *)malloc(dim_xz * sizeof(double));
    ws->query_yz = (double *)malloc(dim_yz * sizeof(double));
    ws->query_z = (dim_z > 0) ? (double *)malloc(dim_z * sizeof(double)) : NULL;
    ws->stack = (stack_entry_t *)malloc(ws->stack_size * sizeof(stack_entry_t));

    /* Check all allocations */
    if (!ws->nn_idx || !ws->nn_dist || !ws->query_xyz ||
        !ws->query_xz || !ws->query_yz || !ws->stack ||
        (dim_z > 0 && !ws->query_z))
    {
        /* Cleanup on failure */
        free(ws->nn_idx);
        free(ws->nn_dist);
        free(ws->query_xyz);
        free(ws->query_xz);
        free(ws->query_yz);
        free(ws->query_z);
        free(ws->stack);
        free(ws);
        return NULL;
    }

    return ws;
}

static void cmi_workspace_free(cmi_workspace_t *ws)
{
    if (!ws)
        return;
    free(ws->nn_idx);
    free(ws->nn_dist);
    free(ws->query_xyz);
    free(ws->query_xz);
    free(ws->query_yz);
    free(ws->query_z);
    free(ws->stack);
    free(ws);
}

/**
 * KSG estimator for CMI(X; Y | Z)
 *
 * Using KSG Type 1 estimator:
 *   CMI = ψ(k) - <ψ(n_xz + 1) + ψ(n_yz + 1) - ψ(n_z + 1)>
 */
double pcmci_cmi_value(const double *X, const double *Y, const double *Z,
                       int32_t n, int32_t dim_z, int32_t k)
{
    if (!X || !Y || n < k + 1)
        return 0.0;

    init_digamma_cache();

    int32_t dim_xyz = 1 + 1 + dim_z; /* X, Y, Z dimensions */
    int32_t dim_xz = 1 + dim_z;
    int32_t dim_yz = 1 + dim_z;

    /* Allocate data matrices */
    double *data_xyz = (double *)pcmci_malloc(n * dim_xyz * sizeof(double));
    double *data_xz = (double *)pcmci_malloc(n * dim_xz * sizeof(double));
    double *data_yz = (double *)pcmci_malloc(n * dim_yz * sizeof(double));
    double *data_z = NULL;

    if (!data_xyz || !data_xz || !data_yz)
    {
        pcmci_mkl_free(data_xyz);
        pcmci_mkl_free(data_xz);
        pcmci_mkl_free(data_yz);
        return 0.0;
    }

    /* Copy X */
    memcpy(data_xyz, X, n * sizeof(double));
    memcpy(data_xz, X, n * sizeof(double));

    /* Copy Y */
    memcpy(data_xyz + n, Y, n * sizeof(double));
    memcpy(data_yz, Y, n * sizeof(double));

    /* Copy Z (if present) */
    if (dim_z > 0 && Z)
    {
        memcpy(data_xyz + 2 * n, Z, n * dim_z * sizeof(double));
        memcpy(data_xz + n, Z, n * dim_z * sizeof(double));
        memcpy(data_yz + n, Z, n * dim_z * sizeof(double));

        data_z = (double *)pcmci_malloc(n * dim_z * sizeof(double));
        if (!data_z)
        {
            pcmci_mkl_free(data_xyz);
            pcmci_mkl_free(data_xz);
            pcmci_mkl_free(data_yz);
            return 0.0;
        }
        memcpy(data_z, Z, n * dim_z * sizeof(double));
    }

    /* Normalize all data to [0, 1] */
    pcmci_normalize_data(data_xyz, n, dim_xyz);
    pcmci_normalize_data(data_xz, n, dim_xz);
    pcmci_normalize_data(data_yz, n, dim_yz);
    if (data_z)
    {
        pcmci_normalize_data(data_z, n, dim_z);
    }

    /* Build KD-trees */
    pcmci_kdtree_t *tree_xyz = pcmci_kdtree_build(data_xyz, n, dim_xyz);
    pcmci_kdtree_t *tree_xz = pcmci_kdtree_build(data_xz, n, dim_xz);
    pcmci_kdtree_t *tree_yz = pcmci_kdtree_build(data_yz, n, dim_yz);
    pcmci_kdtree_t *tree_z = NULL;

    if (!tree_xyz || !tree_xz || !tree_yz)
    {
        pcmci_kdtree_free(tree_xyz);
        pcmci_kdtree_free(tree_xz);
        pcmci_kdtree_free(tree_yz);
        pcmci_mkl_free(data_xyz);
        pcmci_mkl_free(data_xz);
        pcmci_mkl_free(data_yz);
        pcmci_mkl_free(data_z);
        return 0.0;
    }

    if (data_z)
    {
        tree_z = pcmci_kdtree_build(data_z, n, dim_z);
        if (!tree_z)
        {
            pcmci_kdtree_free(tree_xyz);
            pcmci_kdtree_free(tree_xz);
            pcmci_kdtree_free(tree_yz);
            pcmci_mkl_free(data_xyz);
            pcmci_mkl_free(data_xz);
            pcmci_mkl_free(data_yz);
            pcmci_mkl_free(data_z);
            return 0.0;
        }
    }

    /* Get max depth across all trees for stack allocation */
    int32_t max_depth = tree_xyz->max_depth;
    if (tree_xz->max_depth > max_depth)
        max_depth = tree_xz->max_depth;
    if (tree_yz->max_depth > max_depth)
        max_depth = tree_yz->max_depth;
    if (tree_z && tree_z->max_depth > max_depth)
        max_depth = tree_z->max_depth;

    /* For each point, find k-th NN in joint space and count in marginals */
    double sum_digamma = 0.0;
    int alloc_failed = 0;

#pragma omp parallel reduction(+ : sum_digamma) reduction(| : alloc_failed)
    {
        /* Allocate thread-local workspace */
        cmi_workspace_t *ws = cmi_workspace_alloc(k, dim_xyz, dim_xz, dim_yz,
                                                  dim_z, max_depth);
        if (!ws)
        {
            alloc_failed = 1;
        }

        if (ws)
        {
#pragma omp for schedule(dynamic, 64)
            for (int32_t i = 0; i < n; i++)
            {
                /* Build query points */
                for (int32_t d = 0; d < dim_xyz; d++)
                {
                    ws->query_xyz[d] = data_xyz[i + d * n];
                }
                for (int32_t d = 0; d < dim_xz; d++)
                {
                    ws->query_xz[d] = data_xz[i + d * n];
                }
                for (int32_t d = 0; d < dim_yz; d++)
                {
                    ws->query_yz[d] = data_yz[i + d * n];
                }
                if (data_z)
                {
                    for (int32_t d = 0; d < dim_z; d++)
                    {
                        ws->query_z[d] = data_z[i + d * n];
                    }
                }

                /* Find k-th NN distance in joint space */
                pcmci_kdtree_knn(tree_xyz, ws->query_xyz, k + 1, ws->nn_idx, ws->nn_dist);

                /* k-th NN distance (k+1 because we include self) */
                double eps = ws->nn_dist[0]; /* Max-heap, so [0] is largest = k-th */

                /* Add small epsilon to handle ties */
                eps += 1e-10;

                /* Count neighbors in marginal spaces */
                int32_t n_xz = kdtree_count_radius_iter(tree_xz, ws->query_xz, eps, i,
                                                        ws->stack, ws->stack_size);
                int32_t n_yz = kdtree_count_radius_iter(tree_yz, ws->query_yz, eps, i,
                                                        ws->stack, ws->stack_size);
                int32_t n_z = 0;
                if (tree_z)
                {
                    n_z = kdtree_count_radius_iter(tree_z, ws->query_z, eps, i,
                                                   ws->stack, ws->stack_size);
                }

                /* Accumulate digamma terms */
                if (dim_z > 0)
                {
                    sum_digamma += fast_digamma(n_xz + 1) + fast_digamma(n_yz + 1) - fast_digamma(n_z + 1);
                }
                else
                {
                    /* MI case: no Z */
                    sum_digamma += fast_digamma(n_xz + 1) + fast_digamma(n_yz + 1);
                }
            }

            cmi_workspace_free(ws);
        }
    }

    /* Compute CMI */
    double cmi = 0.0;
    if (!alloc_failed)
    {
        if (dim_z > 0)
        {
            cmi = fast_digamma(k) - sum_digamma / n;
        }
        else
        {
            /* MI(X;Y) = ψ(k) + ψ(n) - <ψ(n_x+1) + ψ(n_y+1)> */
            cmi = fast_digamma(k) + fast_digamma(n) - sum_digamma / n;
        }
    }

    /* Cleanup */
    pcmci_kdtree_free(tree_xyz);
    pcmci_kdtree_free(tree_xz);
    pcmci_kdtree_free(tree_yz);
    if (tree_z)
        pcmci_kdtree_free(tree_z);

    pcmci_mkl_free(data_xyz);
    pcmci_mkl_free(data_xz);
    pcmci_mkl_free(data_yz);
    if (data_z)
        pcmci_mkl_free(data_z);

    return cmi;
}

double pcmci_mi_value(const double *X, const double *Y, int32_t n, int32_t k)
{
    return pcmci_cmi_value(X, Y, NULL, n, 0, k);
}

/*============================================================================
 * Permutation Test for P-Value
 *============================================================================*/

/* Fisher-Yates shuffle */
static void shuffle_array(double *arr, int32_t n, uint64_t *rng_state)
{
    for (int32_t i = n - 1; i > 0; i--)
    {
        /* Simple xorshift RNG */
        *rng_state ^= *rng_state << 13;
        *rng_state ^= *rng_state >> 7;
        *rng_state ^= *rng_state << 17;

        int32_t j = *rng_state % (i + 1);

        double tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

pcmci_cmi_result_t pcmci_cmi_test(const double *X, const double *Y, const double *Z,
                                  int32_t n, int32_t dim_z, const pcmci_cmi_config_t *config)
{
    pcmci_cmi_config_t cfg = config ? *config : pcmci_cmi_default_config();

    pcmci_cmi_result_t result = {0};
    result.k = cfg.k;
    result.n_perm = cfg.n_perm;

    /* Compute actual CMI value */
    result.cmi = pcmci_cmi_value(X, Y, Z, n, dim_z, cfg.k);
    result.stat = result.cmi;

    if (cfg.n_perm <= 0)
    {
        /* No permutation test, return CMI only */
        result.pvalue = -1.0; /* Undefined */
        return result;
    }

    /* Permutation test */
    double *X_perm = (double *)pcmci_malloc(n * sizeof(double));
    if (!X_perm)
    {
        result.pvalue = -1.0;
        return result;
    }
    memcpy(X_perm, X, n * sizeof(double));

    uint64_t rng_state = cfg.seed ? (uint64_t)cfg.seed : (uint64_t)time(NULL);

    int32_t n_greater = 0;

    for (int32_t p = 0; p < cfg.n_perm; p++)
    {
        /* Shuffle X */
        shuffle_array(X_perm, n, &rng_state);

        /* Compute CMI under null */
        double cmi_null = pcmci_cmi_value(X_perm, Y, Z, n, dim_z, cfg.k);

        if (cmi_null >= result.cmi)
        {
            n_greater++;
        }
    }

    result.pvalue = (double)(n_greater + 1) / (cfg.n_perm + 1);

    pcmci_mkl_free(X_perm);

    return result;
}

/*============================================================================
 * Integration with PCMCI+ (test wrapper matching parcorr interface)
 *============================================================================*/

/**
 * CMI test wrapper compatible with PCMCI+ test interface
 */
pcmci_ci_result_t pcmci_cmi_ci_test(const double *X, const double *Y,
                                    const double *Z, int32_t n, int32_t dim_z)
{
    pcmci_cmi_config_t config = pcmci_cmi_default_config();
    config.n_perm = 100; /* Default permutations */

    pcmci_cmi_result_t cmi_result = pcmci_cmi_test(X, Y, Z, n, dim_z, &config);

    pcmci_ci_result_t result = {
        .val = cmi_result.cmi,
        .pvalue = cmi_result.pvalue,
        .stat = cmi_result.stat,
        .df = n - dim_z - 2};

    return result;
}