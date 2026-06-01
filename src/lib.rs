//! # lau-mean-field-agents
//!
//! Mean-field games for agent populations — Lasry-Lions coupled HJB/Fokker-Planck fixed point.
//!
//! Core components:
//! - HJB backward equation (cost-to-go)
//! - Fokker-Planck forward equation (density evolution)
//! - Lasry-Lions coupling (iterative solve until convergence)
//! - McKean-Vlasov fixed point (N agents → continuum → one PDE)
//! - Nash equilibrium computation
//! - Wasserstein distance between density iterates
//! - Existence/uniqueness verification

pub mod hjb;
pub mod fokker_planck;
pub mod lasry_lions;
pub mod mckean_vlasov;
pub mod nash;
pub mod wasserstein;
pub mod existence;
pub mod grids;
pub mod types;

pub use types::*;
