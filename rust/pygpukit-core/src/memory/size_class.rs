//! Size class allocation strategy
//!
//! Provides power-of-2 size classes for efficient memory reuse.
//! Matches Python implementation exactly.

/// Size classes for block allocation (powers of 2)
/// Range: 256 bytes to 256 MB
pub const SIZE_CLASSES: [usize; 11] = [
    256,        // 256 B
    1024,       // 1 KB
    4096,       // 4 KB
    16384,      // 16 KB
    65536,      // 64 KB
    262144,     // 256 KB
    1048576,    // 1 MB
    4194304,    // 4 MB
    16777216,   // 16 MB
    67108864,   // 64 MB
    268435456,  // 256 MB
];

/// Get the appropriate size class for a given size.
///
/// Returns the smallest size class that can fit the requested size.
/// For sizes larger than the largest class, rounds up to 1MB boundary.
///
/// # Examples
///
/// ```
/// use pygpukit_core::memory::get_size_class;
///
/// assert_eq!(get_size_class(100), 256);
/// assert_eq!(get_size_class(1000), 1024);
/// assert_eq!(get_size_class(5000), 16384);
/// ```
#[inline]
pub fn get_size_class(size: usize) -> usize {
    for &sc in &SIZE_CLASSES {
        if size <= sc {
            return sc;
        }
    }
    // Larger than any size class - round up to 1MB boundary
    ((size + 1048575) / 1048576) * 1048576
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class_small() {
        assert_eq!(get_size_class(1), 256);
        assert_eq!(get_size_class(100), 256);
        assert_eq!(get_size_class(256), 256);
    }

    #[test]
    fn test_size_class_boundaries() {
        assert_eq!(get_size_class(257), 1024);
        assert_eq!(get_size_class(1024), 1024);
        assert_eq!(get_size_class(1025), 4096);
    }

    #[test]
    fn test_size_class_large() {
        // Larger than 256MB - rounds to 1MB boundary
        let size = 300 * 1024 * 1024; // 300 MB
        let expected = 300 * 1024 * 1024; // Already on 1MB boundary
        assert_eq!(get_size_class(size), expected);
    }

    #[test]
    fn test_size_class_round_up() {
        // 257 MB should round to 257 MB (1MB boundary)
        let size = 257 * 1024 * 1024 + 1;
        let expected = 258 * 1024 * 1024;
        assert_eq!(get_size_class(size), expected);
    }
}
