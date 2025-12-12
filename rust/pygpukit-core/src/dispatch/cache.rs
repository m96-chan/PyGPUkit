//! Kernel Cache
//!
//! Caches compiled CUDA kernels to avoid repeated NVRTC compilation.
//! Kernels are identified by a hash of their source code and compile options.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Compile options that affect kernel output
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    /// Compute capability (e.g., "sm_75")
    pub compute_capability: String,
    /// Additional compiler flags
    pub flags: Vec<String>,
    /// Define macros
    pub defines: Vec<(String, String)>,
    /// Include paths
    pub include_paths: Vec<String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            compute_capability: "sm_75".into(),
            flags: Vec::new(),
            defines: Vec::new(),
            include_paths: Vec::new(),
        }
    }
}

impl CompileOptions {
    /// Create with compute capability
    pub fn with_compute(compute: &str) -> Self {
        Self {
            compute_capability: compute.into(),
            ..Default::default()
        }
    }

    /// Add a flag
    pub fn flag(mut self, flag: &str) -> Self {
        self.flags.push(flag.into());
        self
    }

    /// Add a define macro
    pub fn define(mut self, name: &str, value: &str) -> Self {
        self.defines.push((name.into(), value.into()));
        self
    }

    /// Add an include path
    pub fn include(mut self, path: &str) -> Self {
        self.include_paths.push(path.into());
        self
    }
}

/// Cached kernel entry
#[derive(Debug, Clone)]
pub struct CachedKernel {
    /// Cache key (hash)
    pub key: u64,
    /// Kernel name
    pub name: String,
    /// PTX code
    pub ptx: String,
    /// CUmodule handle (set after loading)
    pub module_handle: Option<u64>,
    /// CUfunction handle (set after loading)
    pub function_handle: Option<u64>,
    /// Compile options used
    pub options: CompileOptions,
    /// Creation timestamp
    pub created_at: f64,
    /// Last access timestamp
    pub last_access: f64,
    /// Access count
    pub access_count: usize,
    /// Source code hash (for verification)
    pub source_hash: u64,
}

impl CachedKernel {
    /// Create a new cached kernel
    pub fn new(key: u64, name: String, ptx: String, options: CompileOptions, source_hash: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Self {
            key,
            name,
            ptx,
            module_handle: None,
            function_handle: None,
            options,
            created_at: now,
            last_access: now,
            access_count: 1,
            source_hash,
        }
    }

    /// Set module and function handles
    pub fn set_handles(&mut self, module: u64, function: u64) {
        self.module_handle = Some(module);
        self.function_handle = Some(function);
    }

    /// Touch to update access time
    pub fn touch(&mut self) {
        self.last_access = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.access_count += 1;
    }

    /// Check if loaded
    pub fn is_loaded(&self) -> bool {
        self.function_handle.is_some()
    }
}

/// Kernel cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache entries
    pub max_entries: usize,
    /// Maximum PTX size in bytes
    pub max_ptx_size: usize,
    /// Enable LRU eviction
    pub enable_eviction: bool,
    /// TTL in seconds (0 = infinite)
    pub ttl_seconds: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_ptx_size: 256 * 1024 * 1024, // 256MB
            enable_eviction: true,
            ttl_seconds: 0.0, // No TTL by default
        }
    }
}

impl CacheConfig {
    /// Create with max entries
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            max_entries,
            ..Default::default()
        }
    }

    /// Set max PTX size
    pub fn max_ptx_size(mut self, bytes: usize) -> Self {
        self.max_ptx_size = bytes;
        self
    }

    /// Set TTL
    pub fn ttl(mut self, seconds: f64) -> Self {
        self.ttl_seconds = seconds;
        self
    }
}

/// Kernel cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Total entries
    pub entries: usize,
    /// Total PTX size in bytes
    pub ptx_size: usize,
    /// Evictions due to capacity
    pub evictions: usize,
    /// Evictions due to TTL
    pub ttl_evictions: usize,
    /// Loaded kernels (with function handles)
    pub loaded_count: usize,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

/// Kernel cache
///
/// Caches compiled CUDA kernels to avoid repeated NVRTC compilation.
#[derive(Debug)]
pub struct KernelCache {
    config: CacheConfig,
    /// Cached kernels by key
    cache: HashMap<u64, CachedKernel>,
    /// Name to key mapping for lookups
    name_to_key: HashMap<String, Vec<u64>>,
    /// Statistics
    hits: usize,
    misses: usize,
    evictions: usize,
    ttl_evictions: usize,
    total_ptx_size: usize,
}

impl KernelCache {
    /// Create a new kernel cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            name_to_key: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            ttl_evictions: 0,
            total_ptx_size: 0,
        }
    }

    /// Create with defaults
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Compute cache key from source and options
    pub fn compute_key(source: &str, name: &str, options: &CompileOptions) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        name.hash(&mut hasher);
        options.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute source hash only
    pub fn hash_source(source: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cached kernel by key
    pub fn get(&mut self, key: u64) -> Option<&CachedKernel> {
        // Check TTL first
        if self.config.ttl_seconds > 0.0 {
            if let Some(entry) = self.cache.get(&key) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                if now - entry.created_at > self.config.ttl_seconds {
                    // TTL expired - remove
                    self.remove(key);
                    self.ttl_evictions += 1;
                    self.misses += 1;
                    return None;
                }
            }
        }

        if let Some(entry) = self.cache.get_mut(&key) {
            entry.touch();
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get cached kernel by name and options
    pub fn get_by_name(&mut self, name: &str, options: &CompileOptions) -> Option<&CachedKernel> {
        // Find keys for this name
        let keys = self.name_to_key.get(name)?;

        // Find matching options
        for &key in keys {
            if let Some(entry) = self.cache.get(&key) {
                if &entry.options == options {
                    // Touch and return
                    self.hits += 1;
                    if let Some(entry) = self.cache.get_mut(&key) {
                        entry.touch();
                    }
                    return self.cache.get(&key);
                }
            }
        }

        self.misses += 1;
        None
    }

    /// Insert a compiled kernel
    pub fn insert(&mut self, source: &str, name: &str, ptx: String, options: CompileOptions) -> u64 {
        let key = Self::compute_key(source, name, &options);
        let source_hash = Self::hash_source(source);

        // Check if already exists
        if self.cache.contains_key(&key) {
            if let Some(entry) = self.cache.get_mut(&key) {
                entry.touch();
            }
            return key;
        }

        // Evict if necessary
        self.evict_if_needed(ptx.len());

        // Insert
        let entry = CachedKernel::new(key, name.into(), ptx.clone(), options, source_hash);
        self.total_ptx_size += ptx.len();
        self.cache.insert(key, entry);

        // Update name mapping
        self.name_to_key
            .entry(name.into())
            .or_insert_with(Vec::new)
            .push(key);

        key
    }

    /// Update handles for a cached kernel
    pub fn set_handles(&mut self, key: u64, module: u64, function: u64) -> bool {
        if let Some(entry) = self.cache.get_mut(&key) {
            entry.set_handles(module, function);
            true
        } else {
            false
        }
    }

    /// Remove a kernel from cache
    pub fn remove(&mut self, key: u64) -> Option<CachedKernel> {
        if let Some(entry) = self.cache.remove(&key) {
            self.total_ptx_size = self.total_ptx_size.saturating_sub(entry.ptx.len());

            // Remove from name mapping
            if let Some(keys) = self.name_to_key.get_mut(&entry.name) {
                keys.retain(|&k| k != key);
                if keys.is_empty() {
                    self.name_to_key.remove(&entry.name);
                }
            }

            Some(entry)
        } else {
            None
        }
    }

    /// Evict entries if needed
    fn evict_if_needed(&mut self, new_size: usize) {
        if !self.config.enable_eviction {
            return;
        }

        // Evict by entry count
        while self.cache.len() >= self.config.max_entries {
            self.evict_lru();
        }

        // Evict by size
        while self.total_ptx_size + new_size > self.config.max_ptx_size && !self.cache.is_empty() {
            self.evict_lru();
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        // Find LRU entry
        let lru_key = self.cache
            .iter()
            .min_by(|a, b| a.1.last_access.partial_cmp(&b.1.last_access).unwrap())
            .map(|(&k, _)| k);

        if let Some(key) = lru_key {
            self.remove(key);
            self.evictions += 1;
        }
    }

    /// Clear expired entries (TTL)
    pub fn clear_expired(&mut self) -> usize {
        if self.config.ttl_seconds <= 0.0 {
            return 0;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let expired: Vec<u64> = self.cache
            .iter()
            .filter(|(_, v)| now - v.created_at > self.config.ttl_seconds)
            .map(|(&k, _)| k)
            .collect();

        let count = expired.len();
        for key in expired {
            self.remove(key);
            self.ttl_evictions += 1;
        }

        count
    }

    /// Get statistics
    pub fn stats(&self) -> CacheStats {
        let loaded_count = self.cache.values().filter(|e| e.is_loaded()).count();
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.cache.len(),
            ptx_size: self.total_ptx_size,
            evictions: self.evictions,
            ttl_evictions: self.ttl_evictions,
            loaded_count,
        }
    }

    /// Check if kernel is cached
    pub fn contains(&self, key: u64) -> bool {
        self.cache.contains_key(&key)
    }

    /// Get all cached kernel names
    pub fn kernel_names(&self) -> Vec<&str> {
        self.name_to_key.keys().map(|s| s.as_str()).collect()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear all cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.name_to_key.clear();
        self.total_ptx_size = 0;
    }

    /// Get config
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_options() {
        let opts = CompileOptions::with_compute("sm_80")
            .flag("-lineinfo")
            .define("BLOCK_SIZE", "256");

        assert_eq!(opts.compute_capability, "sm_80");
        assert_eq!(opts.flags.len(), 1);
        assert_eq!(opts.defines.len(), 1);
    }

    #[test]
    fn test_compute_key() {
        let source = "__global__ void foo() {}";
        let opts = CompileOptions::default();

        let key1 = KernelCache::compute_key(source, "foo", &opts);
        let key2 = KernelCache::compute_key(source, "foo", &opts);
        assert_eq!(key1, key2);

        // Different name = different key
        let key3 = KernelCache::compute_key(source, "bar", &opts);
        assert_ne!(key1, key3);

        // Different options = different key
        let opts2 = CompileOptions::with_compute("sm_80");
        let key4 = KernelCache::compute_key(source, "foo", &opts2);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_cache_insert_get() {
        let mut cache = KernelCache::with_defaults();

        let source = "__global__ void test_kernel() {}";
        let ptx = "// PTX code here";
        let opts = CompileOptions::default();

        let key = cache.insert(source, "test_kernel", ptx.into(), opts.clone());

        // Get should hit
        let entry = cache.get(key);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().name, "test_kernel");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = KernelCache::with_defaults();

        let result = cache.get(12345);
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_get_by_name() {
        let mut cache = KernelCache::with_defaults();

        let source = "__global__ void my_kernel() {}";
        let opts = CompileOptions::default();

        cache.insert(source, "my_kernel", "ptx".into(), opts.clone());

        let entry = cache.get_by_name("my_kernel", &opts);
        assert!(entry.is_some());

        // Different options should miss
        let opts2 = CompileOptions::with_compute("sm_80");
        let entry2 = cache.get_by_name("my_kernel", &opts2);
        assert!(entry2.is_none());
    }

    #[test]
    fn test_set_handles() {
        let mut cache = KernelCache::with_defaults();

        let key = cache.insert("source", "kernel", "ptx".into(), CompileOptions::default());

        assert!(!cache.get(key).unwrap().is_loaded());

        cache.set_handles(key, 0xABCD, 0x1234);

        let entry = cache.get(key).unwrap();
        assert!(entry.is_loaded());
        assert_eq!(entry.module_handle, Some(0xABCD));
        assert_eq!(entry.function_handle, Some(0x1234));
    }

    #[test]
    fn test_eviction() {
        let config = CacheConfig::with_max_entries(2);
        let mut cache = KernelCache::new(config);

        cache.insert("src1", "k1", "ptx1".into(), CompileOptions::default());
        cache.insert("src2", "k2", "ptx2".into(), CompileOptions::default());

        // Access k2 to make k1 the LRU
        let key2 = KernelCache::compute_key("src2", "k2", &CompileOptions::default());
        cache.get(key2);

        // Insert third - should evict k1
        cache.insert("src3", "k3", "ptx3".into(), CompileOptions::default());

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(KernelCache::compute_key("src1", "k1", &CompileOptions::default())));

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_remove() {
        let mut cache = KernelCache::with_defaults();

        let key = cache.insert("source", "kernel", "ptx".into(), CompileOptions::default());
        assert_eq!(cache.len(), 1);

        let removed = cache.remove(key);
        assert!(removed.is_some());
        assert_eq!(cache.len(), 0);
        assert!(cache.kernel_names().is_empty());
    }

    #[test]
    fn test_clear() {
        let mut cache = KernelCache::with_defaults();

        cache.insert("src1", "k1", "ptx1".into(), CompileOptions::default());
        cache.insert("src2", "k2", "ptx2".into(), CompileOptions::default());

        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().ptx_size, 0);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = KernelCache::with_defaults();

        let key = cache.insert("source", "kernel", "ptx".into(), CompileOptions::default());

        // 2 hits
        cache.get(key);
        cache.get(key);

        // 1 miss
        cache.get(99999);

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }
}
