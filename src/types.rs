//! Common types for mean-field game computations.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A 1D density profile over a grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Density {
    /// Probability mass at each grid point.
    pub values: Vec<f64>,
    /// Grid spacing.
    pub dx: f64,
}

impl Density {
    pub fn new(values: Vec<f64>, dx: f64) -> Self {
        Self { values, dx }
    }

    pub fn zeros(n: usize, dx: f64) -> Self {
        Self { values: vec![0.0; n], dx }
    }

    pub fn uniform(n: usize, dx: f64) -> Self {
        let v = 1.0 / (n as f64 * dx);
        Self { values: vec![v; n], dx }
    }

    /// Normalize so that integral = 1.
    pub fn normalize(&mut self) {
        let total: f64 = self.values.iter().sum::<f64>() * self.dx;
        if total > 1e-15 {
            for v in &mut self.values {
                *v /= total;
            }
        }
    }

    /// Total mass (integral).
    pub fn total_mass(&self) -> f64 {
        self.values.iter().sum::<f64>() * self.dx
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    /// First moment (mean).
    pub fn mean(&self) -> f64 {
        let mut m = 0.0;
        for (i, &v) in self.values.iter().enumerate() {
            let x = i as f64 * self.dx;
            m += x * v;
        }
        m * self.dx
    }

    /// Second central moment (variance).
    pub fn variance(&self) -> f64 {
        let mu = self.mean();
        let mut v = 0.0;
        for (i, &val) in self.values.iter().enumerate() {
            let x = i as f64 * self.dx - mu;
            v += x * x * val;
        }
        v * self.dx
    }

    pub fn to_dvector(&self) -> DVector<f64> {
        DVector::from_vec(self.values.clone())
    }

    /// Check non-negativity.
    pub fn is_nonnegative(&self) -> bool {
        self.values.iter().all(|&v| v >= -1e-12)
    }
}

/// A value function defined on a grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueFunction {
    pub values: Vec<f64>,
    pub dx: f64,
}

impl ValueFunction {
    pub fn new(values: Vec<f64>, dx: f64) -> Self {
        Self { values, dx }
    }

    pub fn zeros(n: usize, dx: f64) -> Self {
        Self { values: vec![0.0; n], dx }
    }

    pub fn n(&self) -> usize {
        self.values.len()
    }

    /// Compute gradient via central differences.
    pub fn gradient(&self) -> Vec<f64> {
        let n = self.values.len();
        let mut grad = vec![0.0; n];
        if n < 2 {
            return grad;
        }
        grad[0] = (self.values[1] - self.values[0]) / self.dx;
        for i in 1..n - 1 {
            grad[i] = (self.values[i + 1] - self.values[i - 1]) / (2.0 * self.dx);
        }
        grad[n - 1] = (self.values[n - 1] - self.values[n - 2]) / self.dx;
        grad
    }

    /// Compute Laplacian via central differences.
    pub fn laplacian(&self) -> Vec<f64> {
        let n = self.values.len();
        let mut lap = vec![0.0; n];
        let idx2 = 1.0 / (self.dx * self.dx);
        for i in 1..n - 1 {
            lap[i] = (self.values[i + 1] - 2.0 * self.values[i] + self.values[i - 1]) * idx2;
        }
        lap
    }

    pub fn to_dvector(&self) -> DVector<f64> {
        DVector::from_vec(self.values.clone())
    }
}

/// Parameters for a mean-field game.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MFGParams {
    /// Diffusion coefficient.
    pub sigma: f64,
    /// Running cost weight.
    pub running_cost_weight: f64,
    /// Terminal cost weight.
    pub terminal_cost_weight: f64,
    /// Crowd aversion parameter.
    pub crowd_aversion: f64,
    /// Time horizon.
    pub time_horizon: f64,
    /// Number of time steps.
    pub n_time: usize,
    /// Domain half-length.
    pub domain_half: f64,
}

impl Default for MFGParams {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            running_cost_weight: 1.0,
            terminal_cost_weight: 1.0,
            crowd_aversion: 1.0,
            time_horizon: 1.0,
            n_time: 100,
            domain_half: 5.0,
        }
    }
}

/// Result of a mean-field game solve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MFGSolution {
    pub value_function: ValueFunction,
    pub density: Density,
    pub converged: bool,
    pub iterations: usize,
    pub final_wasserstein: f64,
}

/// Convergence configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub damping: f64,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            damping: 0.5,
        }
    }
}
