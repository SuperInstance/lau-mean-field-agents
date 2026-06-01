//! Nash equilibrium computation.
//!
//! In the mean-field game setting, the Nash equilibrium is a fixed point
//! where no individual agent benefits from deviating.
//!
//! The mean-field Nash equilibrium (u*, m*) satisfies:
//! - u* is the value function of the representative agent facing density m*
//! - m* is the law of the optimal trajectory under u*

use crate::types::{ConvergenceConfig, Density, MFGParams, MFGSolution, ValueFunction};
use crate::lasry_lions::solve_mfg_lasry_lions;
use crate::hjb::{hamiltonian, optimal_control, terminal_cost};
use crate::fokker_planck::fokker_planck_step;
use crate::wasserstein::wasserstein_1;

/// Best response: given a population density, compute the optimal control.
pub fn best_response(
    density: &Density,
    params: &MFGParams,
) -> Vec<f64> {
    let n = density.n();
    let dx = density.dx;

    // Simple best response: gradient of cost-to-go approximation
    let mut grad = vec![0.0; n];
    for i in 1..n - 1 {
        let cost_right = running_cost_integrated(i as f64 * dx, density, params);
        let cost_left = running_cost_integrated((i - 1) as f64 * dx, density, params);
        grad[i] = (cost_right - cost_left) / dx;
    }

    grad.iter().map(|&p| optimal_control(p)).collect()
}

fn running_cost_integrated(x: f64, density: &Density, params: &MFGParams) -> f64 {
    let idx = (x / density.dx) as usize;
    let local_density = if idx < density.n() { density.values[idx] } else { 0.0 };
    params.running_cost_weight * x * x + params.crowd_aversion * local_density
}

/// Check if (u, m) is an approximate Nash equilibrium.
pub fn is_nash_equilibrium(
    u: &ValueFunction,
    m: &Density,
    params: &MFGParams,
    tolerance: f64,
) -> bool {
    // Verify: the control from u is optimal given m
    let grad = u.gradient();
    let alpha: Vec<f64> = grad.iter().map(|&p| optimal_control(p)).collect();

    // Check that small perturbations of the control don't improve cost
    let base_cost = compute_total_cost(&alpha, m, params);

    // Try small perturbation
    let mut perturbed = alpha.clone();
    for v in &mut perturbed {
        *v += 0.01;
    }
    let perturbed_cost = compute_total_cost(&perturbed, m, params);

    perturbed_cost >= base_cost - tolerance
}

/// Compute total expected cost under a control policy.
pub fn compute_total_cost(alpha: &[f64], m: &Density, params: &MFGParams) -> f64 {
    let n = alpha.len().min(m.n());
    let mut cost = 0.0;
    for i in 0..n {
        let x = i as f64 * m.dx;
        cost += (0.5 * alpha[i].powi(2) * params.running_cost_weight
            + params.crowd_aversion * m.values[i])
            * m.values[i]
            * m.dx;
    }
    cost
}

/// Compute the Nash equilibrium social cost.
pub fn social_cost(u: &ValueFunction, m: &Density) -> f64 {
    let mut cost = 0.0;
    for i in 0..u.n().min(m.n()) {
        cost += u.values[i] * m.values[i] * m.dx;
    }
    cost
}

/// Compute the price of anarchy: ratio of Nash social cost to optimal social cost.
pub fn price_of_anarchy(
    nash_solution: &MFGSolution,
    cooperative_cost: f64,
) -> f64 {
    let nash_cost = social_cost(&nash_solution.value_function, &nash_solution.density);
    if cooperative_cost.abs() < 1e-15 {
        return 1.0;
    }
    nash_cost / cooperative_cost
}

/// Compute cooperative (central planner) solution cost.
pub fn cooperative_cost(params: &MFGParams, n_grid: usize) -> f64 {
    let dx = 2.0 * params.domain_half / (n_grid as f64 - 1.0);
    let m = Density::uniform(n_grid, dx);
    // Central planner minimizes total cost
    let mut cost = 0.0;
    for i in 0..n_grid {
        let x = -params.domain_half + i as f64 * dx;
        cost += (params.running_cost_weight * x * x) * m.values[i] * dx;
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_best_response_length() {
        let m = Density::uniform(30, 0.2);
        let params = MFGParams::default();
        let br = best_response(&m, &params);
        assert_eq!(br.len(), 30);
    }

    #[test]
    fn test_compute_total_cost_positive() {
        let m = Density::uniform(30, 0.2);
        let alpha = vec![1.0; 30];
        let params = MFGParams::default();
        let cost = compute_total_cost(&alpha, &m, &params);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_compute_total_cost_zero_control() {
        let m = Density::uniform(30, 0.2);
        let alpha = vec![0.0; 30];
        let params = MFGParams {
            crowd_aversion: 0.0,
            ..Default::default()
        };
        let cost = compute_total_cost(&alpha, &m, &params);
        assert_abs_diff_eq!(cost, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_social_cost_finite() {
        let u = ValueFunction::new(vec![1.0; 20], 0.2);
        let m = Density::uniform(20, 0.2);
        let sc = social_cost(&u, &m);
        assert!(sc.is_finite());
    }

    #[test]
    fn test_social_cost_positive() {
        let u = ValueFunction::new(vec![1.0; 20], 0.2);
        let m = Density::uniform(20, 0.2);
        let sc = social_cost(&u, &m);
        assert!(sc > 0.0);
    }

    #[test]
    fn test_price_of_anarchy_finite() {
        let sol = MFGSolution {
            value_function: ValueFunction::new(vec![1.0; 20], 0.2),
            density: Density::uniform(20, 0.2),
            converged: true,
            iterations: 5,
            final_wasserstein: 0.001,
        };
        let poa = price_of_anarchy(&sol, 0.5);
        assert!(poa.is_finite());
        assert!(poa >= 0.0);
    }

    #[test]
    fn test_price_of_anarchy_one_when_equal() {
        let sol = MFGSolution {
            value_function: ValueFunction::new(vec![0.5; 20], 0.2),
            density: Density::uniform(20, 0.2),
            converged: true,
            iterations: 5,
            final_wasserstein: 0.001,
        };
        let sc = social_cost(&sol.value_function, &sol.density);
        let poa = price_of_anarchy(&sol, sc);
        assert_abs_diff_eq!(poa, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cooperative_cost_positive() {
        let params = MFGParams::default();
        let cost = cooperative_cost(&params, 30);
        assert!(cost >= 0.0);
    }

    #[test]
    fn test_is_nash_equilibrium_runs() {
        let u = ValueFunction::zeros(30, 0.2);
        let m = Density::uniform(30, 0.2);
        let params = MFGParams::default();
        // Just check it doesn't panic
        let _ = is_nash_equilibrium(&u, &m, &params, 1.0);
    }
}
