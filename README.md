# lau-mean-field-agents

**Mean-field games for agent populations — Lasry-Lions coupled HJB/Fokker-Planck fixed point.**

A Rust implementation of mean-field game (MFG) theory: the mathematical framework where a continuum of identical agents each solve an optimal control problem, coupled through the population distribution. The solution is a Nash equilibrium where no individual agent benefits from deviating.

## What This Does

This crate implements the full MFG pipeline:

| Module | Role | Key Equations |
|---|---|---|
| **HJB** | Backward value function (cost-to-go) | −∂u/∂t = σ²/2·Δu + H(∇u) + f(x, m) |
| **Fokker-Planck** | Forward density evolution | ∂m/∂t = σ²/2·Δm − div(m·α*) |
| **Lasry-Lions** | Coupled fixed-point iteration | Solve HJB → extract controls → solve FP → repeat |
| **McKean-Vlasov** | N-agent → continuum limit | μₜ = Law(Xₜ), propagation of chaos |
| **Nash equilibrium** | Best response, social cost | Price of anarchy, cooperative vs competitive |
| **Wasserstein** | Distance between distributions | W₁ (earth mover), W₂ (quantile-based) |
| **Existence** | Verification of MFG conditions | Lasry-Lions monotonicity, contraction |

## Key Idea

In a mean-field game, each agent faces this problem: minimize a cost that depends on the *population distribution*, not on any specific other agent. When the number of agents N → ∞, the N-player game collapses into a single **representative agent** interacting with the distribution μₜ. This is the **McKean-Vlasov limit**.

The equilibrium `(u*, m*)` satisfies a **coupled PDE system**:
- **HJB backward**: solve for the value function `u*` given density `m*`
- **Fokker-Planck forward**: evolve density `m*` given optimal controls from `u*`
- **Fixed point**: `(u*, m*)` must be self-consistent

The Lasry-Lions method iterates this until convergence, checked via Wasserstein distance.

## Install

```toml
[dependencies]
lau-mean-field-agents = "0.1.0"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-mean-field-agents.git
cd lau-mean-field-agents
cargo test
```

### Dependencies

- `nalgebra` 0.33 — linear algebra
- `serde` / `serde_json` — serialization

## Quick Start

```rust
use lau_mean_field_agents::{
    types::{Density, MFGParams, ConvergenceConfig},
    lasry_lions::solve_mfg_lasry_lions,
};

// Set up parameters
let params = MFGParams {
    sigma: 0.5,                    // diffusion
    running_cost_weight: 1.0,      // control cost
    terminal_cost_weight: 0.1,     // final cost
    crowd_aversion: 0.5,           // monotone coupling
    time_horizon: 0.5,             // T
    n_time: 5,                     // time steps
    domain_half: 3.0,              // spatial domain [-3, 3]
};

// Initial distribution (uniform)
let n_grid = 30;
let dx = 2.0 * params.domain_half / (n_grid as f64 - 1.0);
let m0 = Density::uniform(n_grid, dx);

// Convergence config
let config = ConvergenceConfig {
    max_iterations: 10,
    tolerance: 1e-3,
    damping: 0.3,
};

// Solve the mean-field game
let solution = solve_mfg_lasry_lions(&m0, &params, &config);

println!("Converged: {}", solution.converged);
println!("Iterations: {}", solution.iterations);
println!("Final W₁: {:.6}", solution.final_wasserstein);
```

## API Reference

### Core Types

#### `Density` — 1D probability distribution on a grid

```rust
let m = Density::uniform(100, 0.1);    // uniform on grid
m.total_mass();                          // ≈ 1.0
m.mean();                                // first moment
m.variance();                            // second central moment
m.is_nonnegative();                      // physical constraint
```

#### `ValueFunction` — Cost-to-go on a grid

```rust
let u = ValueFunction::zeros(50, 0.2);
let grad = u.gradient();                 // central differences
let lap = u.laplacian();                 // second differences
```

#### `MFGParams` — Game parameters

| Field | Meaning | Default |
|---|---|---|
| `sigma` | Diffusion coefficient | 1.0 |
| `running_cost_weight` | Control cost scaling | 1.0 |
| `terminal_cost_weight` | Final cost scaling | 1.0 |
| `crowd_aversion` | Monotone coupling strength | 1.0 |
| `time_horizon` | Time T | 1.0 |
| `n_time` | Number of time steps | 100 |
| `domain_half` | Spatial half-domain | 5.0 |

#### `MFGSolution` — Solve result

```rust
solution.value_function;          // optimal cost-to-go
solution.density;                 // equilibrium distribution
solution.converged;               // did it converge?
solution.iterations;              // Lasry-Lions iterations
solution.final_wasserstein;       // convergence measure
```

### HJB (Hamilton-Jacobi-Bellman)

```rust
use lau_mean_field_agents::hjb::*;

let ham = hamiltonian(2.0);           // H(p) = p²/2 = 2.0
let alpha = optimal_control(2.0);     // α* = −∇u = −2.0
let rc = running_cost(x, alpha, density, &params);
let u_new = hjb_backward_step(&u, &density, dt, &params);
```

### Fokker-Planck

```rust
use lau_mean_field_agents::fokker_planck::*;

let m_new = fokker_planck_step(&m, &alpha, dt, &params);
let trajectory = solve_fokker_planck_forward(&m0, &u_traj, dx, &params);
let steady = steady_state_diffusion(50, 0.2, 1.0, 1000);
```

### Lasry-Lions Coupling

```rust
use lau_mean_field_agents::lasry_lions::*;

let (values, densities) = lasry_lions_iteration(&dens, &m0, dx, &params);
let solution = solve_mfg_lasry_lions(&m0, &params, &config);
```

### McKean-Vlasov

```rust
use lau_mean_field_agents::mckean_vlasov::*;

let paths = simulate_n_agents(100, 50, 0.01, 1.0, &controls, &init_pos);
let empirical = empirical_distribution(&final_positions, 50, 0.2);
let rate = propagation_of_chaos_rate(1000);  // O(1/√N) ≈ 0.032
```

### Nash Equilibrium

```rust
use lau_mean_field_agents::nash::*;

let br = best_response(&density, &params);
let is_ne = is_nash_equilibrium(&u, &m, &params, 0.01);
let poa = price_of_anarchy(&solution, cooperative_cost);
let sc = social_cost(&u, &m);
```

### Wasserstein Distance

```rust
use lau_mean_field_agents::wasserstein::*;

let w1 = wasserstein_1(&m1, &m2);           // Earth Mover's Distance
let w2 = wasserstein_2(&m1, &m2);           // 2-Wasserstein
let matrix = wasserstein_distance_matrix(&densities);
```

### Existence & Uniqueness

```rust
use lau_mean_field_agents::existence::*;

let (is_monotone, val) = check_monotonicity(&m1, &m2, crowd_aversion);
let checks = verify_existence_conditions(&density, &params);
let (is_unique, reason) = verify_uniqueness(&params);
let lipschitz = stability_estimate(&params);
```

## How It Works

### Numerical Method

1. **Spatial discretization**: 1D grid on `[-L, L]` with `n` points, spacing `dx`
2. **HJB backward**: explicit Euler in time, central differences in space, Dirichlet boundary conditions
3. **Fokker-Planck forward**: explicit Euler with diffusion + advection, Neumann (zero-flux) boundary, clamped non-negative, renormalized
4. **Lasry-Lions iteration**:
   - Start with uniform density trajectory
   - Solve HJB backward → get value functions
   - Extract optimal controls α* = −∇u
   - Solve FP forward → get new density trajectory
   - Damped update: m_new = (1−η)·m_old + η·m_computed
   - Check convergence via W₁ at final time
5. **Convergence**: guaranteed by Lasry-Lions monotonicity (crowd aversion ≥ 0) and positive diffusion

### CFL Condition

The explicit Euler time step satisfies `dt ≤ 0.4·dx²/σ²` for numerical stability.

## The Math

### Mean-Field Game System

The coupled PDE system for the representative agent:

```
     -∂u/∂t = σ²/2 · Δu + H(∇u) + f(x, m(t,·))    (HJB, backward)
      ∂m/∂t = σ²/2 · Δm − div(m · ∂H*/∂p)           (FP, forward)
      u(T,x) = g(x)                                     (terminal cost)
      m(0,x) = m₀(x)                                    (initial distribution)
```

With quadratic Hamiltonian `H(p) = p²/2`, the optimal control is `α* = −∇u`.

### Lasry-Lions Monotonicity

Uniqueness requires the **Lasry-Lions monotonicity condition**:

```
∫ (f(x,m₁) − f(x,m₂)) · (m₁(x) − m₂(x)) dx ≥ 0
```

For crowd aversion `f(x,m) = β·m(x)` with `β ≥ 0`, this is automatically satisfied.

### Propagation of Chaos

For N agents with pairwise interaction strength `O(1/N)`, as N → ∞:

```
Law(X₁ᵢ, ..., Xₙᵢ) → μ ⊗ ... ⊗ μ    (tensorization)
```

The rate of convergence is `O(1/√N)`.

### Wasserstein Distance

- **W₁(μ, ν)**: `∫ |F(x) − G(x)| dx` where F, G are CDFs (Earth Mover's Distance)
- **W₂(μ, ν)**: `(∫₀¹ (F⁻¹(t) − G⁻¹(t))² dt)^{1/2}` via quantile functions

## Test Coverage

**76 tests** covering:
- HJB: terminal/running costs, Hamiltonian, optimal control, backward solve, boundary conditions (14 tests)
- Fokker-Planck: mass conservation, non-negativity, Gaussian spreading, advection, steady state (9 tests)
- Lasry-Lions: iteration outputs, normalization, full solve, damping effects (8 tests)
- McKean-Vlasov: N-agent simulation, empirical distribution, propagation of chaos (10 tests)
- Nash: best response, social cost, price of anarchy, equilibrium checking (9 tests)
- Wasserstein: identity, symmetry, triangle inequality, known values, distance matrix (11 tests)
- Existence: monotonicity, contraction, existence conditions, uniqueness, stability (15 tests)

## License

MIT
