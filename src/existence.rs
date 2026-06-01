//! Existence and uniqueness verification for mean-field games.
//!
//! Lasry-Lions conditions:
//! - Existence: monotone coupling + continuity → fixed point exists
//! - Uniqueness: Lasry-Lions monotonicity condition
//!   ∫(f(x,m₁) - f(x,m₂))(m₁ - m₂) dx ≥ 0 (anti-monotone cost)

use crate::types::{Density, MFGParams};
use crate::wasserstein::wasserstein_1;

/// Check the Lasry-Lions monotonicity condition:
/// The cost functional f(x,m) should satisfy:
/// ∫(f(x,m₁) - f(x,m₂))(m₁(x) - m₂(x)) dx ≥ 0
pub fn check_monotonicity(
    m1: &Density,
    m2: &Density,
    crowd_aversion: f64,
) -> (bool, f64) {
    assert_eq!(m1.n(), m2.n());
    let n = m1.n();
    let dx = m1.dx;

    let mut integral = 0.0;
    for i in 0..n {
        let f1 = crowd_aversion * m1.values[i];
        let f2 = crowd_aversion * m2.values[i];
        integral += (f1 - f2) * (m1.values[i] - m2.values[i]) * dx;
    }

    (integral >= -1e-10, integral)
}

/// Check contraction mapping condition for the FP-HJB system.
/// If the mapping T: m ↦ Φ(m) is a contraction in W_2, then unique fixed point exists.
pub fn check_contraction(
    m1: &Density,
    m2: &Density,
    phi_m1: &Density,
    phi_m2: &Density,
) -> (bool, f64) {
    let w_input = wasserstein_1(m1, m2);
    let w_output = wasserstein_1(phi_m1, phi_m2);

    if w_input < 1e-15 {
        return (true, 0.0);
    }

    let lipschitz = w_output / w_input;
    (lipschitz < 1.0, lipschitz)
}

/// Verify existence via Schauder fixed-point theorem conditions.
/// Need: compact convex set, continuous self-map.
pub fn verify_existence_conditions(
    density: &Density,
    params: &MFGParams,
) -> Vec<(String, bool)> {
    let mut checks = Vec::new();

    // 1. Density is a probability measure
    let mass_ok = (density.total_mass() - 1.0).abs() < 0.01;
    checks.push(("Mass conservation".into(), mass_ok));

    // 2. Non-negativity
    let nonneg_ok = density.is_nonnegative();
    checks.push(("Non-negativity".into(), nonneg_ok));

    // 3. Finite variance (compact support approximation)
    let var = density.variance();
    let var_ok = var.is_finite() && var < 1000.0;
    checks.push(("Finite variance".into(), var_ok));

    // 4. Crowd aversion is non-negative (monotonicity condition)
    let monotone_ok = params.crowd_aversion >= 0.0;
    checks.push(("Non-negative crowd aversion".into(), monotone_ok));

    // 5. Diffusion is positive (ensures smoothing)
    let diffusion_ok = params.sigma > 0.0;
    checks.push(("Positive diffusion".into(), diffusion_ok));

    // 6. Time horizon positive
    let time_ok = params.time_horizon > 0.0;
    checks.push(("Positive time horizon".into(), time_ok));

    checks
}

/// Estimate uniqueness via the Lasry-Lions condition.
/// For quadratic Hamiltonian and monotone cost, uniqueness holds.
pub fn verify_uniqueness(params: &MFGParams) -> (bool, String) {
    if params.crowd_aversion < 0.0 {
        return (false, "Crowd aversion is negative — monotonicity may fail".into());
    }
    if params.sigma <= 0.0 {
        return (false, "Non-positive diffusion — regularization fails".into());
    }
    if params.running_cost_weight <= 0.0 {
        return (false, "Non-positive running cost weight".into());
    }

    // For quadratic Hamiltonian + monotone crowd cost: unique
    (true, "Quadratic Hamiltonian + monotone cost → unique MFE".into())
}

/// Compute a stability estimate: Lipschitz constant of the fixed-point map.
pub fn stability_estimate(params: &MFGParams) -> f64 {
    // Rough estimate: L ~ crowd_aversion / (sigma^2 * running_cost_weight)
    let sigma2 = params.sigma * params.sigma;
    if sigma2 * params.running_cost_weight < 1e-15 {
        return f64::INFINITY;
    }
    params.crowd_aversion / (sigma2 * params.running_cost_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_monotonicity_identical() {
        let m = Density::uniform(50, 0.2);
        let (ok, val) = check_monotonicity(&m, &m, 1.0);
        assert!(ok);
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_monotonicity_different() {
        let mut v1 = vec![0.0; 50];
        let mut v2 = vec![0.0; 50];
        v1[10] = 10.0;
        v2[30] = 10.0;
        let m1 = Density::new(v1, 0.2);
        let m2 = Density::new(v2, 0.2);
        let (ok, val) = check_monotonicity(&m1, &m2, 1.0);
        assert!(ok); // Should be monotone (positive integral)
        assert!(val >= 0.0);
    }

    #[test]
    fn test_monotonicity_positive_aversion() {
        let m1 = Density::uniform(30, 0.2);
        let mut v2 = vec![0.0; 30];
        v2[15] = 5.0;
        let m2 = Density::new(v2, 0.2);
        let (ok, _) = check_monotonicity(&m1, &m2, 1.0);
        assert!(ok);
    }

    #[test]
    fn test_contraction_identical() {
        let m = Density::uniform(30, 0.2);
        let (ok, _) = check_contraction(&m, &m, &m, &m);
        assert!(ok);
    }

    #[test]
    fn test_contraction_different() {
        let m1 = Density::uniform(30, 0.2);
        let m2 = Density::uniform(30, 0.2);
        let mut v3 = vec![0.0; 30]; v3[15] = 5.0;
        let m3 = Density::new(v3, 0.2);
        let (ok, lip) = check_contraction(&m1, &m2, &m3, &m3);
        // m1 == m2, so w_input ~ 0, should be ok
        assert!(ok);
    }

    #[test]
    fn test_existence_all_pass() {
        let m = Density::uniform(50, 0.2);
        let params = MFGParams::default();
        let checks = verify_existence_conditions(&m, &params);
        for (name, ok) in &checks {
            assert!(ok, "Failed: {}", name);
        }
    }

    #[test]
    fn test_existence_negative_sigma_fails() {
        let m = Density::uniform(50, 0.2);
        let params = MFGParams { sigma: -1.0, ..Default::default() };
        let checks = verify_existence_conditions(&m, &params);
        let diff_check = checks.iter().find(|(n, _)| n.contains("diffusion")).unwrap();
        assert!(!diff_check.1);
    }

    #[test]
    fn test_uniqueness_standard() {
        let params = MFGParams::default();
        let (ok, msg) = verify_uniqueness(&params);
        assert!(ok);
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_uniqueness_negative_aversion() {
        let params = MFGParams { crowd_aversion: -1.0, ..Default::default() };
        let (ok, _) = verify_uniqueness(&params);
        assert!(!ok);
    }

    #[test]
    fn test_stability_estimate_finite() {
        let params = MFGParams::default();
        let est = stability_estimate(&params);
        assert!(est.is_finite());
        assert!(est >= 0.0);
    }

    #[test]
    fn test_stability_estimate_zero_sigma() {
        let params = MFGParams { sigma: 0.0, ..Default::default() };
        let est = stability_estimate(&params);
        assert!(est.is_infinite());
    }

    #[test]
    fn test_uniqueness_zero_running_cost() {
        let params = MFGParams { running_cost_weight: 0.0, ..Default::default() };
        let (ok, _) = verify_uniqueness(&params);
        assert!(!ok);
    }

    #[test]
    fn test_existence_checks_count() {
        let m = Density::uniform(30, 0.2);
        let params = MFGParams::default();
        let checks = verify_existence_conditions(&m, &params);
        assert_eq!(checks.len(), 6);
    }
}
