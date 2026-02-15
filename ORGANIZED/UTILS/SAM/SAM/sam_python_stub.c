/*
 * Enhanced SAM-NN stub library for Python integration
 * Provides essential function signatures for ctypes binding
 * Enhanced with hard invariant tracking and morphogenetic primitives
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Constants */
#define MIN_UNCERTAINTY 0.001
#define MIN_IDENTITY_OVERLAP 0.7
#define MIN_EPISTEMIC_RANK 0.01

/* Invariant violation codes */
typedef enum {
    INVARIANT_OK = 0,
    INVARIANT_SIGMA_VANISHED = 1,
    INVARIANT_UNCERTAINTY_DEPLETED = 2,
    INVARIANT_EPISTEMIC_RANK_LOW = 3,
    INVARIANT_IDENTITY_LOST = 4
} InvariantViolation;

/* Minimal SAM state structure with enhanced tracking */
typedef struct {
    int latent_dim;
    double* S;              /* Latent state */
    double* theta;          /* Parameters */
    double* phi;            /* Meta-parameters */
    double* Sigma;          /* Identity manifold */
    double U;               /* Unsolvability budget */
    double identity_overlap; /* Current identity overlap */
    double epistemic_rank;   /* Current epistemic rank */
    int step_count;
    int growth_count;        /* Number of growth operations */
    int invariant_violations; /* Count of violations */
} SAMStateStub;

/* Growth primitives */
typedef enum {
    GROWTH_EXPAND = 0,
    GROWTH_COMPRESS = 1,
    GROWTH_BRANCH = 2,
    GROWTH_MERGE = 3,
    GROWTH_FREEZE = 4,
    GROWTH_UNFREEZE = 5,
    GROWTH_PRUNE = 6
} GrowthPrimitive;

/* Morphogenetic pressure signals */
typedef enum {
    PRESSURE_PERPLEXITY = 0,
    PRESSURE_GRADIENT_NORM = 1,
    PRESSURE_LATENCY = 2,
    PRESSURE_ENTROPY = 3
} PressureSignal;

/* SAM Functions */
SAMStateStub* sam_create(int latent_dim) {
    SAMStateStub* sam = (SAMStateStub*)malloc(sizeof(SAMStateStub));
    sam->latent_dim = latent_dim;
    sam->S = (double*)calloc(latent_dim, sizeof(double));
    sam->theta = (double*)calloc(latent_dim, sizeof(double));
    sam->phi = (double*)calloc(latent_dim, sizeof(double));
    sam->Sigma = (double*)calloc(latent_dim * latent_dim, sizeof(double));
    
    /* Initialize Sigma as identity */
    for (int i = 0; i < latent_dim; i++) {
        sam->Sigma[i * latent_dim + i] = 1.0;
    }
    
    /* Initialize latent state with small random values */
    for (int i = 0; i < latent_dim; i++) {
        sam->S[i] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
    
    sam->U = 0.1;                    /* Initial uncertainty budget */
    sam->identity_overlap = 1.0;      /* Start with perfect overlap */
    sam->epistemic_rank = 1.0;        /* Start with full rank */
    sam->step_count = 0;
    sam->growth_count = 0;
    sam->invariant_violations = 0;
    
    return sam;
}

void sam_destroy(SAMStateStub* sam) {
    if (sam) {
        free(sam->S);
        free(sam->theta);
        free(sam->phi);
        free(sam->Sigma);
        free(sam);
    }
}

/* Calculate dot product */
static double dot_product(double* a, double* b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

/* Calculate vector norm */
static double vector_norm(double* a, int n) {
    return sqrt(dot_product(a, a, n));
}

/* Update identity overlap */
static void update_identity_overlap(SAMStateStub* sam) {
    /* Calculate overlap between S and Sigma diagonal */
    double s_norm = vector_norm(sam->S, sam->latent_dim);
    double sigma_diag_norm = sqrt((double)sam->latent_dim);  /* Norm of identity */
    
    if (s_norm > 0 && sigma_diag_norm > 0) {
        double overlap = 0.0;
        for (int i = 0; i < sam->latent_dim; i++) {
            overlap += sam->S[i] * sam->Sigma[i * sam->latent_dim + i];
        }
        overlap /= (s_norm * sigma_diag_norm);
        sam->identity_overlap = fabs(overlap);
    }
}

/* Update epistemic rank (simplified) */
static void update_epistemic_rank(SAMStateStub* sam) {
    /* Estimate rank based on variance of S */
    double mean = 0.0;
    for (int i = 0; i < sam->latent_dim; i++) {
        mean += sam->S[i];
    }
    mean /= sam->latent_dim;
    
    double variance = 0.0;
    for (int i = 0; i < sam->latent_dim; i++) {
        variance += (sam->S[i] - mean) * (sam->S[i] - mean);
    }
    variance /= sam->latent_dim;
    
    /* Higher variance = higher effective rank */
    sam->epistemic_rank = fmin(variance * 10.0, 1.0);
}

void sam_step(SAMStateStub* sam, double* observation, double reward) {
    sam->step_count++;
    
    /* Update latent state based on observation */
    double learning_rate = 0.01;
    for (int i = 0; i < sam->latent_dim; i++) {
        double obs_val = (i < 64) ? observation[i] : 0.0;  /* Handle variable input */
        sam->S[i] = (1.0 - learning_rate) * sam->S[i] + learning_rate * obs_val;
    }
    
    /* Update theta based on reward (gradient proxy) */
    for (int i = 0; i < sam->latent_dim; i++) {
        sam->theta[i] += 0.001 * reward * sam->S[i];
    }
    
    /* Decay uncertainty */
    sam->U *= 0.999;
    sam->U = fmax(sam->U, MIN_UNCERTAINTY);
    
    /* Update derived metrics */
    update_identity_overlap(sam);
    update_epistemic_rank(sam);
}

/* Enhanced Hard Invariant Checking */
int sam_check_invariants(SAMStateStub* sam) {
    /* Check 1: Identity manifold non-vanishing (Sigma not zero) */
    double sigma_trace = 0.0;
    for (int i = 0; i < sam->latent_dim && i < 10; i++) {
        sigma_trace += sam->Sigma[i * sam->latent_dim + i];
    }
    if (sigma_trace < 0.01) {
        sam->invariant_violations++;
        return INVARIANT_SIGMA_VANISHED;
    }
    
    /* Check 2: Uncertainty budget positive (Hard invariant: U > 0) */
    if (sam->U <= MIN_UNCERTAINTY) {
        sam->invariant_violations++;
        return INVARIANT_UNCERTAINTY_DEPLETED;
    }
    
    /* Check 3: Epistemic rank minimum (Cov[s] >= delta) */
    if (sam->epistemic_rank < MIN_EPISTEMIC_RANK) {
        sam->invariant_violations++;
        return INVARIANT_EPISTEMIC_RANK_LOW;
    }
    
    /* Check 4: Identity continuity (overlap > 0.7) */
    if (sam->identity_overlap < MIN_IDENTITY_OVERLAP) {
        sam->invariant_violations++;
        return INVARIANT_IDENTITY_LOST;
    }
    
    return INVARIANT_OK;
}

/* Get invariant violation string */
const char* sam_get_invariant_violation_str(int violation_code) {
    switch (violation_code) {
        case INVARIANT_OK: return "OK";
        case INVARIANT_SIGMA_VANISHED: return "SIGMA_VANISHED";
        case INVARIANT_UNCERTAINTY_DEPLETED: return "UNCERTAINTY_DEPLETED";
        case INVARIANT_EPISTEMIC_RANK_LOW: return "EPISTEMIC_RANK_LOW";
        case INVARIANT_IDENTITY_LOST: return "IDENTITY_LOST";
        default: return "UNKNOWN";
    }
}

/* Morphogenetic Functions */
int sam_grow(SAMStateStub* sam, int primitive) {
    sam->growth_count++;
    
    switch (primitive) {
        case GROWTH_EXPAND:
            /* EXPAND - increase capacity */
            sam->latent_dim += 8;
            /* Reallocate arrays */
            sam->S = (double*)realloc(sam->S, sam->latent_dim * sizeof(double));
            sam->theta = (double*)realloc(sam->theta, sam->latent_dim * sizeof(double));
            sam->phi = (double*)realloc(sam->phi, sam->latent_dim * sizeof(double));
            /* Initialize new elements */
            for (int i = sam->latent_dim - 8; i < sam->latent_dim; i++) {
                sam->S[i] = 0.0;
                sam->theta[i] = 0.0;
                sam->phi[i] = 0.0;
            }
            break;
            
        case GROWTH_COMPRESS:
            /* COMPRESS - reduce dimensionality (simplified) */
            if (sam->latent_dim > 16) {
                sam->latent_dim -= 8;
            }
            break;
            
        case GROWTH_FREEZE:
            /* FREEZE - lock parameters (mark in phi) */
            for (int i = 0; i < sam->latent_dim; i++) {
                sam->phi[i] = 1.0;  /* Frozen marker */
            }
            break;
            
        case GROWTH_UNFREEZE:
            /* UNFREEZE - enable learning */
            for (int i = 0; i < sam->latent_dim; i++) {
                sam->phi[i] = 0.0;
            }
            break;
            
        default:
            return -1;  /* Unknown primitive */
    }
    
    return 0;  /* Success */
}

/* Calculate pressure signal */
double sam_get_pressure(SAMStateStub* sam, int signal_type) {
    switch (signal_type) {
        case PRESSURE_PERPLEXITY:
            /* Proxy: inverse of certainty */
            return 1.0 - sam->identity_overlap;
            
        case PRESSURE_GRADIENT_NORM:
            /* Proxy: magnitude of theta */
            return vector_norm(sam->theta, sam->latent_dim) / sam->latent_dim;
            
        case PRESSURE_LATENCY:
            /* Proxy: based on step count */
            return fmin(sam->step_count / 1000.0, 1.0);
            
        case PRESSURE_ENTROPY:
            /* Proxy: variance of S */
            double mean = 0.0;
            for (int i = 0; i < sam->latent_dim; i++) {
                mean += sam->S[i];
            }
            mean /= sam->latent_dim;
            
            double variance = 0.0;
            for (int i = 0; i < sam->latent_dim; i++) {
                variance += (sam->S[i] - mean) * (sam->S[i] - mean);
            }
            variance /= sam->latent_dim;
            
            return fmin(variance, 1.0);
            
        default:
            return 0.0;
    }
}

/* Utility Functions */
double sam_get_uncertainty(SAMStateStub* sam) {
    return sam->U;
}

double sam_get_identity_overlap(SAMStateStub* sam) {
    return sam->identity_overlap;
}

double sam_get_epistemic_rank(SAMStateStub* sam) {
    return sam->epistemic_rank;
}

int sam_get_step_count(SAMStateStub* sam) {
    return sam->step_count;
}

int sam_get_growth_count(SAMStateStub* sam) {
    return sam->growth_count;
}

int sam_get_invariant_violations(SAMStateStub* sam) {
    return sam->invariant_violations;
}

void sam_get_latent(SAMStateStub* sam, double* out, int n) {
    int copy_n = n < sam->latent_dim ? n : sam->latent_dim;
    memcpy(out, sam->S, copy_n * sizeof(double));
}

void sam_get_theta(SAMStateStub* sam, double* out, int n) {
    int copy_n = n < sam->latent_dim ? n : sam->latent_dim;
    memcpy(out, sam->theta, copy_n * sizeof(double));
}

/* Reset invariants (emergency recovery) */
void sam_reset_invariants(SAMStateStub* sam) {
    /* Reset to safe state */
    sam->U = 0.1;
    sam->identity_overlap = 1.0;
    sam->epistemic_rank = 1.0;
    
    /* Reset Sigma to identity */
    memset(sam->Sigma, 0, sam->latent_dim * sam->latent_dim * sizeof(double));
    for (int i = 0; i < sam->latent_dim; i++) {
        sam->Sigma[i * sam->latent_dim + i] = 1.0;
    }
    
    sam->invariant_violations = 0;
}
