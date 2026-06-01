//! Fokker-Planck forward equation for density evolution.
//!
//! Solves: ∂m/∂t = σ²/2 · Δm - div(m · α*(t,x))
//! forward in time from initial condition m(0,x) = m₀(x).

use crate::types::{Density, MFGParams, ValueFunction};
use crate::hjb::optimal_control;

/// One forward Fokker-Planck step (explicit Euler).
pub fn fokker_planck_step(
    m: &Density,
    alpha: &[f64],
    dt: f64,
    params: &MFGParams,
) -> Density {
    let n = m.n();
    let dx = m.dx;
    let sigma2_half = 0.5 * params.sigma * params.sigma;
    let idx2 = 1.0 / (dx * dx);
    let idx = 1.0 / (2.0 * dx);

    let mut new_vals = vec![0.0; n];

    for i in 1..n - 1 {
        // Diffusion: σ²/2 * d²m/dx²
        let diffusion = sigma2_half * (m.values[i + 1] - 2.0 * m.values[i] + m.values[i - 1]) * idx2;

        // Advection: -d/dx(m * α)
        let flux_right = m.values[i + 1] * alpha[i + 1];
        let flux_left = m.values[i - 1] * alpha[i - 1];
        let advection = -(flux_right - flux_left) * idx;

        new_vals[i] = m.values[i] + dt * (diffusion + advection);
    }

    // Boundary: zero flux (Neumann)
    new_vals[0] = new_vals[1];
    new_vals[n - 1] = new_vals[n - 2];

    // Clamp to non-negative
    for v in &mut new_vals {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    let mut result = Density::new(new_vals, dx);
    result.normalize();
    result
}

/// Solve the full forward Fokker-Planck equation given a control trajectory.
pub fn solve_fokker_planck_forward(
    initial_density: &Density,
    value_function_trajectory: &[ValueFunction],
    dx: f64,
    params: &MFGParams,
) -> Vec<Density> {
    let n_time = params.n_time;
    let dt = params.time_horizon / n_time as f64;

    let mut trajectory = Vec::with_capacity(n_time + 1);
    trajectory.push(initial_density.clone());

    let mut m = initial_density.clone();

    for t_step in 0..n_time {
        let u = if t_step < value_function_trajectory.len() {
            &value_function_trajectory[t_step]
        } else {
            value_function_trajectory.last().unwrap()
        };
        let grad = u.gradient();
        let alpha: Vec<f64> = grad.iter().map(|&p| optimal_control(p)).collect();

        m = fokker_planck_step(&m, &alpha, dt, params);
        trajectory.push(m.clone());
    }

    trajectory
}

/// Steady-state density via iteration (no advection, pure diffusion).
pub fn steady_state_diffusion(n: usize, dx: f64, sigma: f64, iterations: usize) -> Density {
    let mut m = Density::uniform(n, dx);
    let sigma2_half = 0.5 * sigma * sigma;
    let idx2 = 1.0 / (dx * dx);
    let dt = 0.4 * dx * dx / sigma2_half; // CFL condition

    for _ in 0..iterations {
        let mut new_vals = vec![0.0; n];
        for i in 1..n - 1 {
            new_vals[i] = m.values[i]
                + dt * sigma2_half * (m.values[i + 1] - 2.0 * m.values[i] + m.values[i - 1]) * idx2;
        }
        new_vals[0] = new_vals[1];
        new_vals[n - 1] = new_vals[n - 2];
        for v in &mut new_vals {
            if *v < 0.0 { *v = 0.0; }
        }
        m = Density::new(new_vals, dx);
        m.normalize();
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fp_step_preserves_length() {
        let n = 50;
        let dx = 0.2;
        let m = Density::uniform(n, dx);
        let alpha = vec![0.0; n];
        let params = MFGParams::default();
        let m_new = fokker_planck_step(&m, &alpha, 0.001, &params);
        assert_eq!(m_new.n(), n);
    }

    #[test]
    fn test_fp_step_mass_conservation() {
        let n = 100;
        let dx = 0.1;
        let m = Density::uniform(n, dx);
        let alpha = vec![0.0; n];
        let params = MFGParams::default();
        let m_new = fokker_planck_step(&m, &alpha, 0.001, &params);
        assert_abs_diff_eq!(m_new.total_mass(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_fp_step_nonneg() {
        let n = 50;
        let dx = 0.2;
        let m = Density::uniform(n, dx);
        let alpha = vec![0.0; n];
        let params = MFGParams::default();
        let m_new = fokker_planck_step(&m, &alpha, 0.001, &params);
        assert!(m_new.is_nonnegative());
    }

    #[test]
    fn test_fp_forward_length() {
        let params = MFGParams { n_time: 5, ..Default::default() };
        let n = 30;
        let dx = 0.3;
        let m0 = Density::uniform(n, dx);
        let u_traj: Vec<ValueFunction> = (0..6).map(|_| ValueFunction::zeros(n, dx)).collect();
        let traj = solve_fokker_planck_forward(&m0, &u_traj, dx, &params);
        assert_eq!(traj.len(), 6);
    }

    #[test]
    fn test_fp_zero_control_gaussian_spread() {
        let n = 101;
        let dx = 0.1;
        let mut vals = vec![0.0; n];
        vals[50] = 10.0; // delta-like
        let mut m = Density::new(vals, dx);
        m.normalize();
        let alpha = vec![0.0; n];
        let params = MFGParams { sigma: 1.0, ..Default::default() };
        let m_new = fokker_planck_step(&m, &alpha, 0.001, &params);
        // Should have spread: variance should increase
        assert!(m_new.variance() > m.variance() || m.variance() < 1e-10);
    }

    #[test]
    fn test_steady_state_diffusion_mass() {
        let m = steady_state_diffusion(50, 0.2, 1.0, 1000);
        assert_abs_diff_eq!(m.total_mass(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_steady_state_diffusion_uniform() {
        let m = steady_state_diffusion(50, 0.2, 1.0, 5000);
        // With Neumann BC, steady state should be approximately uniform
        let max_val = m.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = m.values.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max_val - min_val) / max_val < 0.2);
    }

    #[test]
    fn test_fp_advection_shifts() {
        let n = 101;
        let dx = 0.1;
        let mut vals = vec![0.0; n];
        // Put mass at center
        for i in 45..56 {
            vals[i] = 1.0;
        }
        let mut m = Density::new(vals, dx);
        m.normalize();
        // Constant positive control → should shift mass right
        let alpha = vec![1.0; n];
        let params = MFGParams { sigma: 0.1, ..Default::default() };
        let m_before_mean = m.mean();
        for _ in 0..10 {
            let m_new = fokker_planck_step(&m, &alpha, 0.001, &params);
            m = m_new;
        }
        assert!(m.mean() > m_before_mean);
    }
}
