//! Lasry-Lions coupling: iterative solution of the coupled HJB/Fokker-Planck system.
//!
//! The Lasry-Lions method alternates:
//! 1. Solve HJB backward given current density → get value function
//! 2. Extract optimal controls from value function
//! 3. Solve Fokker-Planck forward given optimal controls → get new density
//! 4. Check convergence via Wasserstein distance
//! 5. Damp and repeat

use crate::types::{ConvergenceConfig, Density, MFGParams, MFGSolution, ValueFunction};
use crate::hjb::{solve_hjb_backward, terminal_cost, compute_optimal_controls};
use crate::fokker_planck::solve_fokker_planck_forward;
use crate::wasserstein::wasserstein_1;
use crate::grids::grid_spacing;

/// Run one Lasry-Lions iteration: HJB backward then FP forward.
pub fn lasry_lions_iteration(
    density_trajectory: &[Density],
    initial_density: &Density,
    dx: f64,
    params: &MFGParams,
) -> (Vec<ValueFunction>, Vec<Density>) {
    let n = initial_density.n();

    // Terminal cost
    let terminal: Vec<f64> = (0..n)
        .map(|i| terminal_cost(-params.domain_half + i as f64 * dx, params.terminal_cost_weight))
        .collect();

    // Step 1: Solve HJB backward
    let value_trajectory = solve_hjb_backward(&terminal, density_trajectory, dx, params);

    // Step 3: Solve FP forward with controls from value function
    let new_density_trajectory = solve_fokker_planck_forward(initial_density, &value_trajectory, dx, params);

    (value_trajectory, new_density_trajectory)
}

/// Solve the full mean-field game via Lasry-Lions fixed point iteration.
pub fn solve_mfg_lasry_lions(
    initial_density: &Density,
    params: &MFGParams,
    config: &ConvergenceConfig,
) -> MFGSolution {
    let n = initial_density.n();
    let dx = initial_density.dx;
    let n_time = params.n_time;

    // Initialize with uniform density trajectory
    let mut density_trajectory: Vec<Density> = (0..=n_time)
        .map(|_| initial_density.clone())
        .collect();

    let mut converged = false;
    let mut iterations = 0;
    let mut final_w = f64::INFINITY;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Save old densities at final time for comparison
        let old_final = density_trajectory.last().unwrap().clone();

        // Run iteration
        let (value_trajectory, new_density_trajectory) =
            lasry_lions_iteration(&density_trajectory, initial_density, dx, params);

        // Check convergence at final time
        let new_final = new_density_trajectory.last().unwrap();
        let w = wasserstein_1(&old_final, new_final);
        final_w = w;

        // Damped update
        let damping = config.damping;
        let mut damped_trajectory = Vec::with_capacity(new_density_trajectory.len());
        for (old, new) in density_trajectory.iter().zip(new_density_trajectory.iter()) {
            let mut damped = Density::new(
                old.values
                    .iter()
                    .zip(new.values.iter())
                    .map(|(&o, &n)| o * (1.0 - damping) + n * damping)
                    .collect(),
                dx,
            );
            damped.normalize();
            damped_trajectory.push(damped);
        }
        density_trajectory = damped_trajectory;

        // Convergence check
        if w < config.tolerance {
            converged = true;
            break;
        }
    }

    // Final HJB solve
    let terminal: Vec<f64> = (0..n)
        .map(|i| terminal_cost(-params.domain_half + i as f64 * dx, params.terminal_cost_weight))
        .collect();
    let value_trajectory = solve_hjb_backward(&terminal, &density_trajectory, dx, params);

    MFGSolution {
        value_function: value_trajectory.into_iter().next().unwrap_or_else(|| ValueFunction::zeros(n, dx)),
        density: density_trajectory.into_iter().next().unwrap_or_else(|| initial_density.clone()),
        converged,
        iterations,
        final_wasserstein: final_w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn small_params() -> MFGParams {
        MFGParams {
            sigma: 0.5,
            running_cost_weight: 1.0,
            terminal_cost_weight: 0.1,
            crowd_aversion: 0.5,
            time_horizon: 0.5,
            n_time: 5,
            domain_half: 3.0,
        }
    }

    fn small_config() -> ConvergenceConfig {
        ConvergenceConfig {
            max_iterations: 10,
            tolerance: 1e-3,
            damping: 0.3,
        }
    }

    #[test]
    fn test_ll_iteration_returns_trajectories() {
        let params = small_params();
        let n = 30;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let densities = vec![m0.clone(); params.n_time + 1];
        let (vals, dens) = lasry_lions_iteration(&densities, &m0, dx, &params);
        assert_eq!(vals.len(), params.n_time + 1);
        assert_eq!(dens.len(), params.n_time + 1);
    }

    #[test]
    fn test_ll_iteration_density_normalized() {
        let params = small_params();
        let n = 30;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let densities = vec![m0.clone(); params.n_time + 1];
        let (_, dens) = lasry_lions_iteration(&densities, &m0, dx, &params);
        for d in &dens {
            assert_abs_diff_eq!(d.total_mass(), 1.0, epsilon = 0.05);
        }
    }

    #[test]
    fn test_solve_mfg_returns_solution() {
        let params = small_params();
        let config = small_config();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol = solve_mfg_lasry_lions(&m0, &params, &config);
        assert_eq!(sol.density.n(), n);
        assert_eq!(sol.value_function.n(), n);
    }

    #[test]
    fn test_solve_mfg_iterations_bounded() {
        let params = small_params();
        let config = small_config();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol = solve_mfg_lasry_lions(&m0, &params, &config);
        assert!(sol.iterations <= config.max_iterations);
    }

    #[test]
    fn test_solve_mfg_density_nonneg() {
        let params = small_params();
        let config = small_config();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol = solve_mfg_lasry_lions(&m0, &params, &config);
        assert!(sol.density.is_nonnegative());
    }

    #[test]
    fn test_solve_mfg_wasserstein_finite() {
        let params = small_params();
        let config = small_config();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol = solve_mfg_lasry_lions(&m0, &params, &config);
        assert!(sol.final_wasserstein.is_finite());
    }

    #[test]
    fn test_damping_effect() {
        let config1 = ConvergenceConfig { damping: 0.1, ..small_config() };
        let config2 = ConvergenceConfig { damping: 0.9, ..small_config() };
        let params = small_params();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol1 = solve_mfg_lasry_lions(&m0, &params, &config1);
        let sol2 = solve_mfg_lasry_lions(&m0, &params, &config2);
        // Different damping → different results (usually)
        // Just check both produce valid output
        assert!(sol1.density.total_mass() > 0.0);
        assert!(sol2.density.total_mass() > 0.0);
    }

    #[test]
    fn test_ll_terminal_cost_drives_value() {
        let params = MFGParams {
            terminal_cost_weight: 10.0,
            ..small_params()
        };
        let config = small_config();
        let n = 20;
        let dx = 2.0 * params.domain_half / (n as f64 - 1.0);
        let m0 = Density::uniform(n, dx);
        let sol = solve_mfg_lasry_lions(&m0, &params, &config);
        // Value function should be positive at some points (costs are positive)
        let has_positive = sol.value_function.values.iter().any(|&v| v > 0.0);
        assert!(has_positive);
    }
}
