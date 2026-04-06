# Day 53: A/B Testing for ML Models

## Learning Objectives

- Design A/B tests to compare ML model versions with proper sample size calculations
- Apply statistical significance testing (t-test, chi-squared, bootstrap) to model experiments
- Understand multi-armed bandit strategies as an adaptive alternative to fixed A/B tests
- Plan canary deployments that gradually shift traffic from the old model to the new one
- Avoid common pitfalls: peeking, Simpson's paradox, novelty effects, and interference

## Key Concepts

Offline evaluation metrics (accuracy, F1, AUC) do not always predict real-world
performance. A model that scores higher on a test set may perform worse in production
due to distributional differences, user behavior changes, or feedback loops. A/B testing
provides causal evidence by randomly assigning users to either the control (current
model) or treatment (new model) and measuring the impact on a business metric (click
rate, conversion, revenue, time-to-resolution). The randomization ensures that any
observed difference is attributable to the model change and not to confounding factors.

Before running a test, you must determine the required sample size based on the
minimum detectable effect (the smallest improvement worth detecting), baseline metric
value, and desired statistical power (typically 80%). Running a test without sufficient
samples leads to underpowered experiments that miss real improvements or, worse,
declare insignificant noise as a win. After collecting data, a hypothesis test (usually
a two-sample t-test for continuous metrics or chi-squared for proportions) determines
whether the observed difference is statistically significant at your chosen confidence
level (typically 95%).

Multi-armed bandits offer an alternative to fixed-split A/B tests by dynamically
allocating more traffic to the better-performing variant during the experiment. Thompson
Sampling and Upper Confidence Bound (UCB) are popular bandit algorithms. They reduce
the opportunity cost of showing the worse model to users but provide weaker statistical
guarantees. Canary deployments use a different strategy: route a small percentage of
traffic (1-5%) to the new model, monitor for regressions (errors, latency spikes), and
gradually increase traffic if metrics look healthy. This limits blast radius if the new
model has unexpected issues.

## Practical Example

```python
"""
A/B testing framework for ML model comparison.
"""
import numpy as np
from scipy import stats

np.random.seed(42)

# Simulate conversion rates for two models
# Control (current model): 5.0% conversion
# Treatment (new model): 5.5% conversion (10% relative improvement)
n_control = 10000
n_treatment = 10000
control_conversions = np.random.binomial(1, 0.050, n_control)
treatment_conversions = np.random.binomial(1, 0.055, n_treatment)

# Step 1: Sample size calculation (before running the test)
def required_sample_size(p_control, min_detectable_effect, alpha=0.05, power=0.80):
    p_treatment = p_control + min_detectable_effect
    p_avg = (p_control + p_treatment) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
          z_beta * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment)))
         / min_detectable_effect) ** 2
    return int(np.ceil(n))

n_required = required_sample_size(0.05, 0.005)
print(f"Required sample size per group: {n_required:,}")

# Step 2: Analyze results
rate_control = control_conversions.mean()
rate_treatment = treatment_conversions.mean()
lift = (rate_treatment - rate_control) / rate_control

# Chi-squared test for proportions
contingency = np.array([
    [control_conversions.sum(), n_control - control_conversions.sum()],
    [treatment_conversions.sum(), n_treatment - treatment_conversions.sum()],
])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

print(f"\nControl rate:   {rate_control:.4f}")
print(f"Treatment rate: {rate_treatment:.4f}")
print(f"Relative lift:  {lift:+.2%}")
print(f"Chi-squared:    {chi2:.4f}")
print(f"P-value:        {p_value:.4f}")
print(f"Significant:    {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")

# Step 3: Bootstrap confidence interval for the lift
def bootstrap_lift(control, treatment, n_bootstrap=10000):
    lifts = []
    for _ in range(n_bootstrap):
        c = np.random.choice(control, size=len(control), replace=True).mean()
        t = np.random.choice(treatment, size=len(treatment), replace=True).mean()
        lifts.append((t - c) / c if c > 0 else 0)
    return np.percentile(lifts, [2.5, 97.5])

ci_low, ci_high = bootstrap_lift(control_conversions, treatment_conversions)
print(f"95% CI for lift: [{ci_low:+.2%}, {ci_high:+.2%}]")

# Step 4: Simple Thompson Sampling bandit
class ThompsonSampling:
    def __init__(self, n_arms):
        self.successes = np.ones(n_arms)  # Beta prior alpha
        self.failures = np.ones(n_arms)   # Beta prior beta

    def select_arm(self):
        samples = [np.random.beta(s, f) for s, f in zip(self.successes, self.failures)]
        return int(np.argmax(samples))

    def update(self, arm, reward):
        if reward:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

bandit = ThompsonSampling(n_arms=2)
for _ in range(1000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, [0.05, 0.055][arm])
    bandit.update(arm, reward)

print(f"\nBandit arm selection rates after 1000 rounds:")
total = bandit.successes + bandit.failures - 2
print(f"  Model A: {total[0]:.0f} pulls, Model B: {total[1]:.0f} pulls")
```

## Resources

- [Trustworthy Online Controlled Experiments (Kohavi, Tang, Xu)](https://experimentguide.com/)
- [Multi-armed Bandits and the Stitch Fix Experimentation Platform](https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/)
- [Evan Miller's A/B testing sample size calculator](https://www.evanmiller.org/ab-testing/sample-size.html)

## Next Day Preview

Day 54 starts Project 10: Enterprise Document Classification, building a production-grade multi-class document categorization system.
