/*
 * operational_factor_model.c
 *
 * Generic C99 factor normalization and aggregation for three operational
 * domains: people management, software, and hardware. Includes generalized
 * weighted sums/means and an optional compute-resource adjustment based on
 * operation count, memory, runtime, and energy.
 *
 * Compile as a reusable object:
 *   cc -std=c99 -Wall -Wextra -pedantic -O2 -c cases-study/operational_factor_model.c -o operational_factor_model.o
 *
 * API functions are intentionally defined here so the file can be linked into
 * a larger application without a demo executable or third-party dependency.
 * Scores are descriptive only; they do not authorize access, rank individuals,
 * or establish causal effects. Example bounds are placeholders requiring local
 * review.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define OFM_MAX_FACTORS 16
#define OFM_DOMAIN_COUNT 3
#define OFM_COMPUTE_FACTOR_COUNT 4

typedef enum {
    OFM_BENEFIT = 0, /* larger raw value maps to a larger score */
    OFM_COST = 1,    /* smaller raw value maps to a larger score */
    OFM_TARGET = 2   /* values closer to target map to a larger score */
} OfmDirection;

typedef enum {
    OFM_SUM = 0,
    OFM_WEIGHTED_SUM = 1,
    OFM_MEAN = 2,
    OFM_WEIGHTED_MEAN = 3
} OfmAggregate;

typedef struct {
    const char *name;
    double minimum;
    double maximum;
    double target;
    double weight;
    OfmDirection direction;
} OfmFactorSpec;

typedef struct {
    const char *name;
    const OfmFactorSpec *factors;
    size_t factor_count;
} OfmDomainSpec;

typedef struct {
    double normalized[OFM_MAX_FACTORS];
    unsigned char present[OFM_MAX_FACTORS];
    size_t factor_count;
    size_t factors_used;
    double coverage;
    double sum;
    double weighted_sum;
    double mean;
    double weighted_mean;
} OfmGroupScore;

typedef struct {
    double operation_count;
    double memory_bytes;
    double runtime_ms;
    double energy_mj;
} OfmComputeUsage;

typedef struct {
    double operation_budget;
    double memory_budget_bytes;
    double runtime_budget_ms;
    double energy_budget_mj;
    double penalty_strength; /* bounded to [0, 1] */
} OfmComputeBudget;

typedef struct {
    double normalized[OFM_COMPUTE_FACTOR_COUNT];
    double mean_cost;
    double adjustment_multiplier;
    double adjusted_score;
    size_t factors_used;
} OfmComputeScore;

/* Illustrative values only. Replace with reviewed, context-specific limits. */
static const OfmFactorSpec PEOPLE_FACTORS[] = {
    {"staffing_coverage",       0.0, 1.0,    0.0, 1.0, OFM_BENEFIT},
    {"workload_ratio",          0.25, 2.0,   1.0, 1.0, OFM_TARGET},
    {"skill_match",              0.0, 1.0,    0.0, 1.0, OFM_BENEFIT},
    {"team_retention_rate",      0.0, 1.0,    0.0, 1.0, OFM_BENEFIT}
};

static const OfmFactorSpec SOFTWARE_FACTORS[] = {
    {"service_availability",    0.0, 1.0,    0.0, 1.0, OFM_BENEFIT},
    {"latency_ms",               0.0, 1000.0, 0.0, 1.0, OFM_COST},
    {"error_rate",               0.0, 1.0,    0.0, 1.0, OFM_COST},
    {"throughput_rps",           0.0, 10000.0,0.0, 1.0, OFM_BENEFIT}
};

static const OfmFactorSpec HARDWARE_FACTORS[] = {
    {"hardware_availability",    0.0, 1.0,    0.0, 1.0, OFM_BENEFIT},
    {"cpu_utilization_pct",      0.0, 100.0, 70.0, 1.0, OFM_TARGET},
    {"memory_utilization_pct",   0.0, 100.0, 70.0, 1.0, OFM_TARGET},
    {"temperature_c",           20.0, 100.0, 0.0, 1.0, OFM_COST}
};

static const OfmDomainSpec DEFAULT_DOMAINS[OFM_DOMAIN_COUNT] = {
    {"people_management", PEOPLE_FACTORS, sizeof(PEOPLE_FACTORS) / sizeof(PEOPLE_FACTORS[0])},
    {"software", SOFTWARE_FACTORS, sizeof(SOFTWARE_FACTORS) / sizeof(SOFTWARE_FACTORS[0])},
    {"hardware", HARDWARE_FACTORS, sizeof(HARDWARE_FACTORS) / sizeof(HARDWARE_FACTORS[0])}
};

static const double COMPUTE_MINIMUM[OFM_COMPUTE_FACTOR_COUNT] = {0.0, 0.0, 0.0, 0.0};
static const char *COMPUTE_NAMES[OFM_COMPUTE_FACTOR_COUNT] = {
    "operation_count", "memory_bytes", "runtime_ms", "energy_mj"
};

const OfmDomainSpec *ofm_default_domains(size_t *count) {
    if (count != NULL) *count = OFM_DOMAIN_COUNT;
    return DEFAULT_DOMAINS;
}

static int finite_number(double value) {
    return isfinite(value) != 0;
}

static int validate_factor(const OfmFactorSpec *spec) {
    if (spec == NULL || spec->name == NULL || spec->name[0] == '\0') return 0;
    if (!finite_number(spec->minimum) || !finite_number(spec->maximum) ||
        !finite_number(spec->weight) || spec->maximum <= spec->minimum || spec->weight < 0.0) return 0;
    if (spec->direction != OFM_BENEFIT && spec->direction != OFM_COST &&
        spec->direction != OFM_TARGET) return 0;
    if (spec->direction == OFM_TARGET &&
        (!finite_number(spec->target) || spec->target < spec->minimum || spec->target > spec->maximum)) return 0;
    return 1;
}

/* Returns 0 on success; rejects non-finite inputs and invalid specs. */
int ofm_normalize(const OfmFactorSpec *spec, double raw_value, double *score, int *was_clipped) {
    double bounded;
    double normalized;
    if (!validate_factor(spec) || score == NULL || !finite_number(raw_value)) return -1;
    bounded = raw_value < spec->minimum ? spec->minimum : raw_value;
    bounded = bounded > spec->maximum ? spec->maximum : bounded;
    if (was_clipped != NULL) *was_clipped = (raw_value < spec->minimum || raw_value > spec->maximum);

    if (spec->direction == OFM_BENEFIT) {
        normalized = (bounded - spec->minimum) / (spec->maximum - spec->minimum);
    } else if (spec->direction == OFM_COST) {
        normalized = (spec->maximum - bounded) / (spec->maximum - spec->minimum);
    } else {
        double furthest = fmax(spec->target - spec->minimum, spec->maximum - spec->target);
        normalized = furthest == 0.0 ? 1.0 : 1.0 - fabs(bounded - spec->target) / furthest;
    }
    *score = fmin(1.0, fmax(0.0, normalized));
    return 0;
}

/*
 * Score values aligned by index with domain->factors. `present[i] == 0`
 * means missing; missing factors are skipped, with coverage reported. Returns
 * 0 on success, -1 for invalid arguments/specifications/values.
 */
int ofm_score_group(const OfmDomainSpec *domain,
                    const double *raw_values,
                    const unsigned char *present,
                    OfmGroupScore *out) {
    double weight_total = 0.0;
    double weighted_total = 0.0;
    double total = 0.0;
    size_t used = 0;
    if (domain == NULL || out == NULL || domain->factors == NULL ||
        raw_values == NULL || present == NULL || domain->factor_count == 0 ||
        domain->factor_count > OFM_MAX_FACTORS) return -1;

    memset(out, 0, sizeof(*out));
    out->factor_count = domain->factor_count;
    for (size_t i = 0; i < domain->factor_count; ++i) {
        double normalized;
        if (!validate_factor(&domain->factors[i])) return -1;
        if (!present[i]) continue;
        if (ofm_normalize(&domain->factors[i], raw_values[i], &normalized, NULL) != 0) return -1;
        out->normalized[i] = normalized;
        out->present[i] = 1;
        total += normalized;
        weighted_total += normalized * domain->factors[i].weight;
        weight_total += domain->factors[i].weight;
        ++used;
    }
    out->factors_used = used;
    out->coverage = (double)used / (double)domain->factor_count;
    out->sum = total;
    out->weighted_sum = weighted_total;
    out->mean = used ? total / (double)used : NAN;
    out->weighted_mean = weight_total > 0.0 ? weighted_total / weight_total : NAN;
    return 0;
}

/*
 * Generalized aggregation for already-normalized values.
 * Missing entries are omitted. Weights must be finite and nonnegative.
 * SUM/MEAN ignore weights; WEIGHTED_SUM/WEIGHTED_MEAN use them.
 * Returns NAN for an empty effective input, or -1 on invalid mode/data.
 */
double ofm_aggregate(const double *values,
                     const double *weights,
                     const unsigned char *present,
                     size_t count,
                     OfmAggregate mode) {
    double sum = 0.0;
    double weight_sum = 0.0;
    size_t used = 0;
    if (values == NULL || present == NULL || count == 0 ||
        (mode != OFM_SUM && mode != OFM_WEIGHTED_SUM &&
         mode != OFM_MEAN && mode != OFM_WEIGHTED_MEAN)) return NAN;

    for (size_t i = 0; i < count; ++i) {
        double weight = weights != NULL ? weights[i] : 1.0;
        if (!present[i]) continue;
        if (!finite_number(values[i]) || !finite_number(weight) || weight < 0.0) return NAN;
        if (mode == OFM_SUM || mode == OFM_MEAN) {
            sum += values[i];
            weight_sum += 1.0;
        } else {
            sum += values[i] * weight;
            weight_sum += weight;
        }
        ++used;
    }
    if (used == 0) return NAN;
    if (mode == OFM_MEAN || mode == OFM_WEIGHTED_MEAN) {
        if (weight_sum <= 0.0) return NAN;
        return sum / weight_sum;
    }
    return sum;
}

/*
 * Compute-resource-aware adjustment. Each usage quantity is treated as a cost
 * relative to its positive budget and clipped to [0,1]. Missing dimensions are
 * omitted. `base_score` must be in [0,1]. The multiplier is
 * 1 - penalty_strength * mean(normalized resource costs), also in [0,1].
 */
int ofm_adjust_for_compute(double base_score,
                           const OfmComputeUsage *usage,
                           const OfmComputeBudget *budget,
                           const unsigned char present[OFM_COMPUTE_FACTOR_COUNT],
                           OfmComputeScore *out) {
    double quantities[OFM_COMPUTE_FACTOR_COUNT];
    double budgets[OFM_COMPUTE_FACTOR_COUNT];
    double sum = 0.0;
    size_t used = 0;
    if (usage == NULL || budget == NULL || present == NULL || out == NULL ||
        !finite_number(base_score) || base_score < 0.0 || base_score > 1.0 ||
        !finite_number(budget->penalty_strength) || budget->penalty_strength < 0.0 ||
        budget->penalty_strength > 1.0) return -1;

    quantities[0] = usage->operation_count;
    quantities[1] = usage->memory_bytes;
    quantities[2] = usage->runtime_ms;
    quantities[3] = usage->energy_mj;
    budgets[0] = budget->operation_budget;
    budgets[1] = budget->memory_budget_bytes;
    budgets[2] = budget->runtime_budget_ms;
    budgets[3] = budget->energy_budget_mj;
    (void)COMPUTE_MINIMUM;
    (void)COMPUTE_NAMES;
    memset(out, 0, sizeof(*out));

    for (size_t i = 0; i < OFM_COMPUTE_FACTOR_COUNT; ++i) {
        if (!present[i]) continue;
        if (!finite_number(quantities[i]) || quantities[i] < 0.0 ||
            !finite_number(budgets[i]) || budgets[i] <= 0.0) return -1;
        out->normalized[i] = fmin(1.0, quantities[i] / budgets[i]);
        sum += out->normalized[i];
        ++used;
    }
    out->factors_used = used;
    out->mean_cost = used ? sum / (double)used : 0.0;
    out->adjustment_multiplier = 1.0 - budget->penalty_strength * out->mean_cost;
    out->adjusted_score = base_score * out->adjustment_multiplier;
    return 0;
}

/* Return stable factor names for introspection; NULL for invalid index. */
const char *ofm_compute_factor_name(size_t index) {
    return index < OFM_COMPUTE_FACTOR_COUNT ? COMPUTE_NAMES[index] : NULL;
}

/* Public dimensions for callers that allocate input arrays. */
size_t ofm_compute_factor_count(void) {
    return OFM_COMPUTE_FACTOR_COUNT;
}
