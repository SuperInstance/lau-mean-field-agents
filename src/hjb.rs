//! Hamilton-Jacobi-Bellman (HJB) backward equation for cost-to-go.
//!
//! Solves: -∂u/∂t = σ²/2 · Δu + H(∇u) + f(x, m(t,x))
//! backward in time from terminal condition u(T,x) = g(x).

use crate::types::{Density, MFGParams, ValueFunction};

/// Terminal cost: quadratic around origin.
pub fn terminal_cost(x: f64, weight: f64) -> f64 {
    weight * x * x
}

/// Running cost: L(x, α) = |α|²/2 + crowd_aversion * m(x).
pub fn running_cost(x: f64, alpha: f64, local_density: f64, params: &MFGParams) -> f64 {
    0.5 * alpha * alpha * params.running_cost_weight
        + params.crowd_aversion * local_density
}

/// Hamiltonian: H(p) = p²/2 (quadratic).
pub fn hamiltonian(p: f64) -> f64 {
    0.5 * p * p
}

/// Optimal control from value function gradient: α* = -∇u.
pub fn optimal_control(grad_u: f64) -> f64 {
    -grad_u
}

/// Solve one backward HJB step (implicit Euler in time, central differences in space).
pub fn hjb_backward_step(
    u: &ValueFunction,
    density: &Density,
    dt: f64,
    params: &MFGParams,
) -> ValueFunction {
    let n = u.n();
    let dx = u.dx;
    let sigma2_half = 0.5 * params.sigma * params.sigma;
    let idx2 = 1.0 / (dx * dx);

    let grad = u.gradient();
    let mut new_vals = vec![0.0; n];

    // Interior: implicit scheme solved explicitly (small dt approximation)
    for i in 1..n - 1 {
        let d2u = (u.values[i + 1] - 2.0 * u.values[i] + u.values[i - 1]) * idx2;
        let du = grad[i];
        let h = hamiltonian(du);
        let crowd_cost = params.crowd_aversion * density.values[i];
        // -du/dt = sigma^2/2 * d2u + H(du) + crowd
        // u^{n-1} = u^n + dt * (sigma^2/2 * d2u + H + crowd)
        new_vals[i] = u.values[i] + dt * (sigma2_half * d2u + h + crowd_cost);
    }

    // Boundary conditions (Dirichlet: u = 0 at edges)
    new_vals[0] = 0.0;
    new_vals[n - 1] = 0.0;

    ValueFunction::new(new_vals, dx)
}

/// Solve the full backward HJB equation.
pub fn solve_hjb_backward(
    terminal_values: &[f64],
    density_trajectory: &[Density],
    dx: f64,
    params: &MFGParams,
) -> Vec<ValueFunction> {
    let n_time = params.n_time;
    let dt = params.time_horizon / n_time as f64;
    let n = terminal_values.len();

    let mut trajectory = Vec::with_capacity(n_time + 1);

    // Terminal condition
    let u_t = ValueFunction::new(terminal_values.to_vec(), dx);
    trajectory.push(u_t.clone());

    let mut u = u_t;
    // March backward
    for t_step in (0..n_time).rev() {
        let density = if t_step < density_trajectory.len() {
            &density_trajectory[t_step]
        } else {
            density_trajectory.last().unwrap()
        };
        u = hjb_backward_step(&u, density, dt, params);
        trajectory.push(u.clone());
    }

    trajectory.reverse();
    trajectory
}

/// Compute the optimal control field from a value function.
pub fn compute_optimal_controls(u: &ValueFunction) -> Vec<f64> {
    u.gradient().iter().map(|&p| optimal_control(p)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_terminal_cost_zero() {
        assert_abs_diff_eq!(terminal_cost(0.0, 1.0), 0.0);
    }

    #[test]
    fn test_terminal_cost_positive() {
        assert!(terminal_cost(2.0, 1.0) > 0.0);
    }

    #[test]
    fn test_running_cost_zero_control() {
        let params = MFGParams::default();
        let rc = running_cost(0.0, 0.0, 0.0, &params);
        assert_abs_diff_eq!(rc, 0.0);
    }

    #[test]
    fn test_hamiltonian_zero() {
        assert_abs_diff_eq!(hamiltonian(0.0), 0.0);
    }

    #[test]
    fn test_hamiltonian_positive() {
        assert!(hamiltonian(3.0) > 0.0);
    }

    #[test]
    fn test_optimal_control_negates() {
        assert_abs_diff_eq!(optimal_control(2.0), -2.0);
    }

    #[test]
    fn test_hjb_step_preserves_len() {
        let u = ValueFunction::zeros(50, 0.1);
        let d = Density::uniform(50, 0.1);
        let params = MFGParams::default();
        let dt = 0.01;
        let u_new = hjb_backward_step(&u, &d, dt, &params);
        assert_eq!(u_new.n(), 50);
    }

    #[test]
    fn test_hjb_boundary_conditions() {
        let n = 50;
        let mut vals = vec![1.0; n];
        vals[25] = 10.0;
        let u = ValueFunction::new(vals, 0.1);
        let d = Density::uniform(n, 0.1);
        let params = MFGParams::default();
        let u_new = hjb_backward_step(&u, &d, 0.01, &params);
        assert_abs_diff_eq!(u_new.values[0], 0.0);
        assert_abs_diff_eq!(u_new.values[n - 1], 0.0);
    }

    #[test]
    fn test_solve_hjb_backward_length() {
        let params = MFGParams { n_time: 10, ..Default::default() };
        let n = 50;
        let dx = 0.2;
        let terminal: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();
        let density = Density::uniform(n, dx);
        let densities = vec![density; 11];
        let traj = solve_hjb_backward(&terminal, &densities, dx, &params);
        assert_eq!(traj.len(), 11);
    }

    #[test]
    fn test_compute_optimal_controls_length() {
        let u = ValueFunction::zeros(30, 0.1);
        let ctrl = compute_optimal_controls(&u);
        assert_eq!(ctrl.len(), 30);
    }

    #[test]
    fn test_hamiltonian_quadratic() {
        assert_abs_diff_eq!(hamiltonian(1.0), 0.5);
        assert_abs_diff_eq!(hamiltonian(-2.0), 2.0);
    }

    #[test]
    fn test_running_cost_with_crowd() {
        let params = MFGParams { crowd_aversion: 2.0, ..Default::default() };
        let rc = running_cost(0.0, 0.0, 1.5, &params);
        assert_abs_diff_eq!(rc, 3.0);
    }

    #[test]
    fn test_terminal_cost_symmetric() {
        assert_abs_diff_eq!(terminal_cost(1.0, 1.0), terminal_cost(-1.0, 1.0));
    }
}
