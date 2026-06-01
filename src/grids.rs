//! Grid construction utilities.

/// Build a uniform 1D grid over [-L, L] with n points.
pub fn uniform_grid_1d(n: usize, half_length: f64) -> Vec<f64> {
    let dx = 2.0 * half_length / (n as f64 - 1.0);
    (0..n).map(|i| -half_length + i as f64 * dx).collect()
}

/// Grid spacing for n points over [-L, L].
pub fn grid_spacing(n: usize, half_length: f64) -> f64 {
    2.0 * half_length / (n as f64 - 1.0)
}

/// Build an index map for grid points.
pub fn grid_index_map(n: usize) -> Vec<usize> {
    (0..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_uniform_grid_length() {
        let g = uniform_grid_1d(101, 5.0);
        assert_eq!(g.len(), 101);
    }

    #[test]
    fn test_uniform_grid_bounds() {
        let g = uniform_grid_1d(101, 5.0);
        assert_abs_diff_eq!(g[0], -5.0);
        assert_abs_diff_eq!(g[100], 5.0);
    }

    #[test]
    fn test_grid_spacing_value() {
        let dx = grid_spacing(101, 5.0);
        assert_abs_diff_eq!(dx, 10.0 / 100.0);
    }

    #[test]
    fn test_grid_symmetry() {
        let g = uniform_grid_1d(101, 5.0);
        let mid = 50;
        for i in 0..50 {
            assert_abs_diff_eq!(g[mid - i], -g[mid + i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_grid_index_map() {
        let m = grid_index_map(5);
        assert_eq!(m, vec![0, 1, 2, 3, 4]);
    }
}
