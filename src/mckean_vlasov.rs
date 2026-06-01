//! McKean-Vlasov fixed point: N agents → continuum → one PDE.
//!
//! In the mean-field limit, the interaction between N agents reduces to
//! a single representative agent interacting with the population distribution.
//! The McKean-Vlasov PDE is the self-consistent equation:
//!
//!   dX_t = α(t, X_t) dt + σ dW_t
//!   α* = argmin E[∫ L(X_t, α_t, μ_t) dt + g(X_T)]
//!   μ_t = Law(X_t)

use crate::types::{ConvergenceConfig, Density, MFGParams, MFGSolution, ValueFunction};
use crate::lasry_lions::solve_mfg_lasry_lions;
use crate::wasserstein::wasserstein_1;

/// Simulate N-agent system and compute empirical distribution.
pub fn simulate_n_agents(
    n_agents: usize,
    n_steps: usize,
    dt: f64,
    sigma: f64,
    controls: &[Vec<f64>], // control[i][t]
    initial_positions: &[f64],
) -> Vec<Vec<f64>> {
    let mut positions: Vec<Vec<f64>> = Vec::with_capacity(n_agents);
    for agent in 0..n_agents {
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(initial_positions[agent]);
        let mut x = initial_positions[agent];
        for t in 0..n_steps {
            let alpha = if agent < controls.len() && t < controls[agent].len() {
                controls[agent][t]
            } else {
                0.0
            };
            // Euler-Maruyama step (deterministic for reproducibility)
            x += alpha * dt + sigma * (dt.sqrt()) * simple_noise(agent, t);
            path.push(x);
        }
        positions.push(path);
    }
    positions
}

/// Simple deterministic noise (hash-based, for reproducibility).
fn simple_noise(agent: usize, step: usize) -> f64 {
    // Box-Muller-ish from hash
    let seed = (agent * 10007 + step * 7919 + 12345) as u64;
    let x = ((seed.wrapping_mul(6364136223846793005)).wrapping_add(1) >> 33) as f64 / (1u64 << 31) as f64;
    2.0 * x - 1.0 // in [-1, 1]
}

/// Compute empirical distribution from agent positions.
pub fn empirical_distribution(positions: &[f64], n_bins: usize, dx: f64) -> Density {
    let mut vals = vec![0.0; n_bins];
    let half = n_bins as f64 * dx / 2.0;

    for &x in positions {
        let bin = ((x + half) / dx) as isize;
        if bin >= 0 && (bin as usize) < n_bins {
            vals[bin as usize] += 1.0;
        }
    }

    let mut d = Density::new(vals, dx);
    d.normalize();
    d
}

/// Verify the McKean-Vlasov limit: as N → ∞, the empirical distribution
/// converges to the mean-field solution.
pub fn verify_mckean_vlasov_convergence(
    n_agents_list: &[usize],
    params: &MFGParams,
    config: &ConvergenceConfig,
) -> Vec<(usize, f64)> {
    let n_grid = 50;
    let dx = 2.0 * params.domain_half / (n_grid as f64 - 1.0);
    let m0 = Density::uniform(n_grid, dx);

    // Solve mean-field
    let mfg_sol = solve_mfg_lasry_lions(&m0, params, config);

    let mut results = Vec::new();

    for &n_agents in n_agents_list {
        // Simulate agents with mean-field optimal control
        let initial_positions: Vec<f64> = (0..n_agents)
            .map(|i| -params.domain_half + 2.0 * params.domain_half * i as f64 / n_agents as f64)
            .collect();

        let grad = mfg_sol.value_function.gradient();
        let controls: Vec<Vec<f64>> = (0..n_agents)
            .map(|_| grad.iter().map(|&p| -p).collect())
            .collect();

        let paths = simulate_n_agents(
            n_agents,
            params.n_time,
            params.time_horizon / params.n_time as f64,
            params.sigma,
            &controls,
            &initial_positions,
        );

        let final_positions: Vec<f64> = paths.iter().map(|p| *p.last().unwrap()).collect();
        let empirical = empirical_distribution(&final_positions, n_grid, dx);

        let dist = wasserstein_1(&empirical, &mfg_sol.density);
        results.push((n_agents, dist));
    }

    results
}

/// Compute the propagation of chaos: for independent agents, the joint
/// distribution factorizes as N copies of the marginal (mean-field).
pub fn propagation_of_chaos_rate(n: usize) -> f64 {
    // Rate is O(1/√N)
    1.0 / (n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simulate_n_agents_length() {
        let controls = vec![vec![0.0; 5]; 10];
        let init = vec![0.0; 10];
        let paths = simulate_n_agents(10, 5, 0.1, 1.0, &controls, &init);
        assert_eq!(paths.len(), 10);
        assert_eq!(paths[0].len(), 6);
    }

    #[test]
    fn test_simulate_zero_control_stationary() {
        let controls = vec![vec![0.0; 10]; 5];
        let init = vec![0.0; 5];
        let paths = simulate_n_agents(5, 10, 0.1, 0.0, &controls, &init);
        for path in &paths {
            assert_abs_diff_eq!(path[0], 0.0);
            // With zero sigma and zero control, should stay at origin
            for &x in path {
                assert_abs_diff_eq!(x, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_empirical_distribution_mass() {
        let positions = vec![0.0, 0.1, -0.1, 0.05, -0.05];
        let d = empirical_distribution(&positions, 20, 0.2);
        assert!(d.total_mass() > 0.0);
        assert!(d.is_nonnegative());
    }

    #[test]
    fn test_empirical_distribution_single_point() {
        let positions = vec![0.0];
        let d = empirical_distribution(&positions, 101, 0.1);
        assert_abs_diff_eq!(d.total_mass(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_propagation_of_chaos_rate() {
        let r1 = propagation_of_chaos_rate(100);
        let r2 = propagation_of_chaos_rate(10000);
        assert!(r2 < r1);
    }

    #[test]
    fn test_propagation_of_chaos_decay() {
        assert_abs_diff_eq!(propagation_of_chaos_rate(1), 1.0);
        assert_abs_diff_eq!(propagation_of_chaos_rate(4), 0.5);
    }

    #[test]
    fn test_simple_noise_bounded() {
        for agent in 0..10 {
            for step in 0..10 {
                let n = simple_noise(agent, step);
                assert!(n >= -1.0 && n <= 1.0);
            }
        }
    }

    #[test]
    fn test_simulate_with_control() {
        let controls = vec![vec![1.0; 5]; 1];
        let init = vec![0.0];
        let paths = simulate_n_agents(1, 5, 0.1, 0.0, &controls, &init);
        // With positive control and zero noise, should drift right
        assert!(paths[0][5] > paths[0][0]);
    }

    #[test]
    fn test_verify_mv_returns_results() {
        let params = MFGParams {
            n_time: 3,
            domain_half: 2.0,
            sigma: 0.3,
            time_horizon: 0.3,
            ..Default::default()
        };
        let config = ConvergenceConfig {
            max_iterations: 3,
            tolerance: 0.1,
            damping: 0.5,
        };
        let results = verify_mckean_vlasov_convergence(&[10, 20], &params, &config);
        assert_eq!(results.len(), 2);
        for (_, d) in &results {
            assert!(d.is_finite());
        }
    }

    #[test]
    fn test_empirical_distribution_normalizes() {
        let positions: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let d = empirical_distribution(&positions, 30, 0.2);
        assert_abs_diff_eq!(d.total_mass(), 1.0, epsilon = 0.1);
    }
}
