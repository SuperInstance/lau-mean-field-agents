//! Wasserstein distance between density iterates.
//!
//! Implements 1-Wasserstein (Earth Mover's Distance) and 2-Wasserstein
//! for 1D probability distributions.

use crate::types::Density;

/// 1-Wasserstein distance (Earth Mover's Distance) for 1D distributions.
/// W_1(μ, ν) = integral |F(x) - G(x)| dx where F, G are CDFs.
pub fn wasserstein_1(m1: &Density, m2: &Density) -> f64 {
    assert_eq!(m1.n(), m2.n(), "Densities must have same grid size");
    assert!((m1.dx - m2.dx).abs() < 1e-12, "Densities must have same dx");

    let n = m1.n();
    let dx = m1.dx;

    // Build CDFs
    let mut cdf1 = vec![0.0; n];
    let mut cdf2 = vec![0.0; n];
    cdf1[0] = m1.values[0] * dx;
    cdf2[0] = m2.values[0] * dx;
    for i in 1..n {
        cdf1[i] = cdf1[i - 1] + m1.values[i] * dx;
        cdf2[i] = cdf2[i - 1] + m2.values[i] * dx;
    }

    // W_1 = sum |cdf1 - cdf2| * dx
    let mut dist = 0.0;
    for i in 0..n {
        dist += (cdf1[i] - cdf2[i]).abs() * dx;
    }
    dist
}

/// 2-Wasserstein distance for 1D distributions (closed form via quantiles).
/// W_2² = integral_0^1 (F^{-1}(t) - G^{-1}(t))² dt
pub fn wasserstein_2_squared(m1: &Density, m2: &Density) -> f64 {
    assert_eq!(m1.n(), m2.n());

    let n = m1.n();
    let dx = m1.dx;

    // Build quantile functions by inverting CDFs
    let nq = 1000;
    let mut w2sq = 0.0;
    let dt = 1.0 / nq as f64;

    for k in 0..nq {
        let t = (k as f64 + 0.5) / nq as f64;
        let q1 = quantile(m1, t);
        let q2 = quantile(m2, t);
        w2sq += (q1 - q2).powi(2) * dt;
    }
    w2sq
}

/// 2-Wasserstein distance.
pub fn wasserstein_2(m1: &Density, m2: &Density) -> f64 {
    wasserstein_2_squared(m1, m2).sqrt()
}

/// Compute quantile (inverse CDF) for a density.
fn quantile(m: &Density, t: f64) -> f64 {
    let n = m.n();
    let dx = m.dx;
    let mut cumsum = 0.0;
    for i in 0..n {
        cumsum += m.values[i] * dx;
        if cumsum >= t {
            return i as f64 * dx;
        }
    }
    (n - 1) as f64 * dx
}

/// Compute the Wasserstein distance matrix between a set of densities.
pub fn wasserstein_distance_matrix(densities: &[Density]) -> Vec<Vec<f64>> {
    let k = densities.len();
    let mut matrix = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in (i + 1)..k {
            let d = wasserstein_1(&densities[i], &densities[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_w1_identical() {
        let m = Density::uniform(50, 0.2);
        assert_abs_diff_eq!(wasserstein_1(&m, &m), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_w2_identical() {
        let m = Density::uniform(50, 0.2);
        assert_abs_diff_eq!(wasserstein_2(&m, &m), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_w1_symmetric() {
        let mut v1 = vec![0.0; 50];
        let mut v2 = vec![0.0; 50];
        v1[10] = 5.0; v1[11] = 5.0;
        v2[30] = 5.0; v2[31] = 5.0;
        let mut m1 = Density::new(v1, 0.2);
        let mut m2 = Density::new(v2, 0.2);
        m1.normalize();
        m2.normalize();
        let d12 = wasserstein_1(&m1, &m2);
        let d21 = wasserstein_1(&m2, &m1);
        assert_abs_diff_eq!(d12, d21, epsilon = 1e-10);
    }

    #[test]
    fn test_w1_positive_for_different() {
        let mut v1 = vec![0.0; 50];
        let mut v2 = vec![0.0; 50];
        v1[10] = 10.0;
        v2[40] = 10.0;
        let m1 = Density::new(v1, 0.2);
        let m2 = Density::new(v2, 0.2);
        assert!(wasserstein_1(&m1, &m2) > 0.0);
    }

    #[test]
    fn test_w1_triangle_inequality() {
        let mut v1 = vec![0.0; 100];
        let mut v2 = vec![0.0; 100];
        let mut v3 = vec![0.0; 100];
        v1[20] = 10.0; v2[50] = 10.0; v3[80] = 10.0;
        let m1 = Density::new(v1, 0.1);
        let m2 = Density::new(v2, 0.1);
        let m3 = Density::new(v3, 0.1);
        let d12 = wasserstein_1(&m1, &m2);
        let d23 = wasserstein_1(&m2, &m3);
        let d13 = wasserstein_1(&m1, &m3);
        assert!(d13 <= d12 + d23 + 1e-8);
    }

    #[test]
    fn test_w2_positive_for_different() {
        let mut v1 = vec![0.0; 50];
        let mut v2 = vec![0.0; 50];
        v1[10] = 10.0;
        v2[40] = 10.0;
        let m1 = Density::new(v1, 0.2);
        let m2 = Density::new(v2, 0.2);
        assert!(wasserstein_2(&m1, &m2) > 0.0);
    }

    #[test]
    fn test_distance_matrix_shape() {
        let d1 = Density::uniform(30, 0.2);
        let d2 = Density::uniform(30, 0.2);
        let d3 = Density::uniform(30, 0.2);
        let mat = wasserstein_distance_matrix(&[d1, d2, d3]);
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
    }

    #[test]
    fn test_distance_matrix_diagonal_zero() {
        let d1 = Density::uniform(30, 0.2);
        let d2 = Density::uniform(30, 0.2);
        let mat = wasserstein_distance_matrix(&[d1, d2]);
        assert_abs_diff_eq!(mat[0][0], 0.0);
        assert_abs_diff_eq!(mat[1][1], 0.0);
    }

    #[test]
    fn test_distance_matrix_symmetric() {
        let mut v1 = vec![0.0; 50];
        let mut v2 = vec![0.0; 50];
        v1[10] = 5.0; v2[30] = 5.0;
        let d1 = Density::new(v1, 0.2);
        let d2 = Density::new(v2, 0.2);
        let mat = wasserstein_distance_matrix(&[d1, d2]);
        assert_abs_diff_eq!(mat[0][1], mat[1][0]);
    }

    #[test]
    fn test_w1_known_value() {
        // Two delta distributions separated by 5 grid points, dx=1.0
        let mut v1 = vec![0.0; 20];
        let mut v2 = vec![0.0; 20];
        v1[5] = 1.0;
        v2[10] = 1.0;
        let m1 = Density::new(v1, 1.0);
        let m2 = Density::new(v2, 1.0);
        // W_1 should be 5.0
        assert_abs_diff_eq!(wasserstein_1(&m1, &m2), 5.0, epsilon = 0.1);
    }
}
