//! Persistent Kernel Cache
//!
//! Extends the in-memory kernel cache with disk persistence.
//! Compiled PTX is saved to `~/.pygpukit/kernel_cache/` for reuse across sessions.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use serde::{Deserialize, Serialize};

/// GPU architecture fingerprint for cache key generation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArchFingerprint {
    /// SM version (e.g., 86 for SM 8.6)
    pub sm_version: u32,
    /// Total global memory in bytes
    pub global_memory: u64,
    /// Shared memory per SM in bytes
    pub shared_memory_per_sm: u32,
    /// Max registers per block
    pub max_registers_per_block: u32,
    /// L2 cache size in bytes
    pub l2_cache_size: u32,
    /// CUDA driver version (MAJOR*1000 + MINOR*10)
    pub driver_version: u32,
}

impl ArchFingerprint {
    /// Create a new architecture fingerprint
    pub fn new(
        sm_version: u32,
        global_memory: u64,
        shared_memory_per_sm: u32,
        max_registers_per_block: u32,
        l2_cache_size: u32,
        driver_version: u32,
    ) -> Self {
        Self {
            sm_version,
            global_memory,
            shared_memory_per_sm,
            max_registers_per_block,
            l2_cache_size,
            driver_version,
        }
    }

    /// Compute hash of fingerprint
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        Hash::hash(self, &mut hasher);
        hasher.finish()
    }

    /// Check if this fingerprint is compatible with another
    /// (same SM version and driver version are minimum requirements)
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.sm_version == other.sm_version && self.driver_version == other.driver_version
    }
}

impl Default for ArchFingerprint {
    fn default() -> Self {
        Self {
            sm_version: 80,
            global_memory: 0,
            shared_memory_per_sm: 0,
            max_registers_per_block: 0,
            l2_cache_size: 0,
            driver_version: 11000,
        }
    }
}

/// Persistent cache entry (stored on disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentEntry {
    /// Source code hash
    pub source_hash: u64,
    /// Kernel name
    pub name: String,
    /// Compile options hash
    pub options_hash: u64,
    /// Architecture fingerprint
    pub arch_fingerprint: ArchFingerprint,
    /// PTX code
    pub ptx: String,
    /// Creation timestamp (Unix epoch)
    pub created_at: f64,
    /// Last access timestamp
    pub last_access: f64,
    /// Access count
    pub access_count: usize,
}

impl PersistentEntry {
    /// Create a new entry
    pub fn new(
        source_hash: u64,
        name: String,
        options_hash: u64,
        arch_fingerprint: ArchFingerprint,
        ptx: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        Self {
            source_hash,
            name,
            options_hash,
            arch_fingerprint,
            ptx,
            created_at: now,
            last_access: now,
            access_count: 1,
        }
    }

    /// Touch to update access time
    pub fn touch(&mut self) {
        self.last_access = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.access_count += 1;
    }

    /// Get PTX size in bytes
    pub fn ptx_size(&self) -> usize {
        self.ptx.len()
    }
}

/// Cache index (stored separately for quick lookup)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheIndex {
    /// Version of the cache format
    pub version: u32,
    /// Architecture fingerprint this index was built for
    pub arch_fingerprint: ArchFingerprint,
    /// Map of cache key to filename
    pub entries: HashMap<u64, String>,
    /// Total size of all cached PTX
    pub total_size: usize,
    /// Last cleanup timestamp
    pub last_cleanup: f64,
}

impl CacheIndex {
    /// Current cache format version
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new index
    pub fn new(arch_fingerprint: ArchFingerprint) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            arch_fingerprint,
            entries: HashMap::new(),
            total_size: 0,
            last_cleanup: 0.0,
        }
    }

    /// Check if index is compatible with current arch
    pub fn is_compatible(&self, arch: &ArchFingerprint) -> bool {
        self.version == Self::CURRENT_VERSION && self.arch_fingerprint.is_compatible(arch)
    }
}

/// Persistent cache configuration
#[derive(Debug, Clone)]
pub struct PersistentCacheConfig {
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Maximum total cache size in bytes
    pub max_size: usize,
    /// Maximum number of entries
    pub max_entries: usize,
    /// Enable auto-cleanup on startup
    pub auto_cleanup: bool,
    /// Entry TTL in seconds (0 = infinite)
    pub ttl_seconds: f64,
}

impl Default for PersistentCacheConfig {
    fn default() -> Self {
        let cache_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".pygpukit")
            .join("kernel_cache");

        Self {
            cache_dir,
            max_size: 512 * 1024 * 1024, // 512MB
            max_entries: 4096,
            auto_cleanup: true,
            ttl_seconds: 0.0, // No TTL by default
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistentCacheStats {
    /// Number of entries
    pub entries: usize,
    /// Total size in bytes
    pub total_size: usize,
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// Evictions
    pub evictions: usize,
    /// Load errors
    pub load_errors: usize,
    /// Save errors
    pub save_errors: usize,
}

impl PersistentCacheStats {
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

/// Persistent kernel cache
pub struct PersistentCache {
    config: PersistentCacheConfig,
    arch_fingerprint: ArchFingerprint,
    index: CacheIndex,
    stats: PersistentCacheStats,
    initialized: bool,
}

impl PersistentCache {
    /// Create a new persistent cache
    pub fn new(config: PersistentCacheConfig, arch_fingerprint: ArchFingerprint) -> Self {
        let index = CacheIndex::new(arch_fingerprint.clone());
        Self {
            config,
            arch_fingerprint,
            index,
            stats: PersistentCacheStats::default(),
            initialized: false,
        }
    }

    /// Create with defaults
    pub fn with_defaults(arch_fingerprint: ArchFingerprint) -> Self {
        Self::new(PersistentCacheConfig::default(), arch_fingerprint)
    }

    /// Initialize the cache (load index, create directories)
    pub fn initialize(&mut self) -> Result<(), CacheError> {
        if self.initialized {
            return Ok(());
        }

        // Create cache directory
        fs::create_dir_all(&self.config.cache_dir).map_err(|e| CacheError::Io(e.to_string()))?;

        // Load or create index
        let index_path = self.index_path();
        if index_path.exists() {
            match self.load_index() {
                Ok(index) => {
                    if index.is_compatible(&self.arch_fingerprint) {
                        self.index = index;
                        self.stats.entries = self.index.entries.len();
                        self.stats.total_size = self.index.total_size;
                    } else {
                        // Incompatible index - clear cache
                        self.clear()?;
                    }
                }
                Err(_) => {
                    // Corrupted index - clear cache
                    self.clear()?;
                }
            }
        }

        // Auto cleanup if enabled
        if self.config.auto_cleanup {
            let _ = self.cleanup();
        }

        self.initialized = true;
        Ok(())
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &PathBuf {
        &self.config.cache_dir
    }

    /// Get index file path
    fn index_path(&self) -> PathBuf {
        self.config.cache_dir.join("index.json")
    }

    /// Get entry file path
    fn entry_path(&self, key: u64) -> PathBuf {
        self.config.cache_dir.join(format!("{:016x}.ptx.json", key))
    }

    /// Compute cache key
    pub fn compute_key(source_hash: u64, name: &str, options_hash: u64, arch_hash: u64) -> u64 {
        let mut hasher = DefaultHasher::new();
        source_hash.hash(&mut hasher);
        name.hash(&mut hasher);
        options_hash.hash(&mut hasher);
        arch_hash.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash source code
    pub fn hash_source(source: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash compile options
    pub fn hash_options(options: &[String]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for opt in options {
            opt.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get cached entry
    pub fn get(&mut self, key: u64) -> Result<Option<PersistentEntry>, CacheError> {
        if !self.initialized {
            self.initialize()?;
        }

        if !self.index.entries.contains_key(&key) {
            self.stats.misses += 1;
            return Ok(None);
        }

        // Load entry from disk
        let entry_path = self.entry_path(key);
        match self.load_entry(&entry_path) {
            Ok(mut entry) => {
                // Check TTL
                if self.config.ttl_seconds > 0.0 {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0);
                    if now - entry.created_at > self.config.ttl_seconds {
                        // TTL expired - remove
                        let _ = self.remove(key);
                        self.stats.misses += 1;
                        return Ok(None);
                    }
                }

                // Update access time
                entry.touch();
                let _ = self.save_entry(key, &entry);

                self.stats.hits += 1;
                Ok(Some(entry))
            }
            Err(_) => {
                // Remove corrupted entry
                self.index.entries.remove(&key);
                let _ = self.save_index();
                self.stats.misses += 1;
                self.stats.load_errors += 1;
                Ok(None)
            }
        }
    }

    /// Insert entry
    pub fn insert(
        &mut self,
        source: &str,
        name: &str,
        options: &[String],
        ptx: String,
    ) -> Result<u64, CacheError> {
        if !self.initialized {
            self.initialize()?;
        }

        let source_hash = Self::hash_source(source);
        let options_hash = Self::hash_options(options);
        let arch_hash = self.arch_fingerprint.hash();
        let key = Self::compute_key(source_hash, name, options_hash, arch_hash);

        // Check if already exists
        if self.index.entries.contains_key(&key) {
            // Update access time
            if let Ok(Some(mut entry)) = self.get(key) {
                entry.touch();
                let _ = self.save_entry(key, &entry);
            }
            return Ok(key);
        }

        // Evict if necessary
        let ptx_size = ptx.len();
        self.evict_if_needed(ptx_size)?;

        // Create entry
        let entry = PersistentEntry::new(
            source_hash,
            name.to_string(),
            options_hash,
            self.arch_fingerprint.clone(),
            ptx,
        );

        // Save entry
        self.save_entry(key, &entry)?;

        // Update index
        self.index.entries.insert(key, format!("{:016x}.ptx.json", key));
        self.index.total_size += ptx_size;
        self.save_index()?;

        self.stats.entries = self.index.entries.len();
        self.stats.total_size = self.index.total_size;

        Ok(key)
    }

    /// Remove entry
    pub fn remove(&mut self, key: u64) -> Result<bool, CacheError> {
        if !self.index.entries.contains_key(&key) {
            return Ok(false);
        }

        // Get entry size before removing
        let entry_path = self.entry_path(key);
        let size = if let Ok(entry) = self.load_entry(&entry_path) {
            entry.ptx_size()
        } else {
            0
        };

        // Remove file
        let _ = fs::remove_file(&entry_path);

        // Update index
        self.index.entries.remove(&key);
        self.index.total_size = self.index.total_size.saturating_sub(size);
        self.save_index()?;

        self.stats.entries = self.index.entries.len();
        self.stats.total_size = self.index.total_size;

        Ok(true)
    }

    /// Evict entries if needed
    fn evict_if_needed(&mut self, new_size: usize) -> Result<(), CacheError> {
        // Evict by entry count
        while self.index.entries.len() >= self.config.max_entries {
            if !self.evict_lru()? {
                break; // No more entries to evict
            }
        }

        // Evict by size
        while self.index.total_size + new_size > self.config.max_size
            && !self.index.entries.is_empty()
        {
            if !self.evict_lru()? {
                break; // No more entries to evict
            }
        }

        Ok(())
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) -> Result<bool, CacheError> {
        // Load all entries to find LRU
        let mut lru_key: Option<u64> = None;
        let mut lru_time = f64::MAX;

        // Collect keys to avoid borrow issues
        let keys: Vec<u64> = self.index.entries.keys().copied().collect();

        for key in keys {
            let entry_path = self.entry_path(key);
            if let Ok(entry) = self.load_entry(&entry_path) {
                if entry.last_access < lru_time {
                    lru_time = entry.last_access;
                    lru_key = Some(key);
                }
            } else {
                // If we can't load the entry, it's a candidate for removal (orphaned index entry)
                lru_key = Some(key);
                break;
            }
        }

        if let Some(key) = lru_key {
            self.remove(key)?;
            self.stats.evictions += 1;
            Ok(true)
        } else {
            Ok(false) // No entry to evict
        }
    }

    /// Cleanup expired entries and orphaned files
    pub fn cleanup(&mut self) -> Result<usize, CacheError> {
        if !self.initialized {
            self.initialize()?;
        }

        let mut removed = 0;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        // Collect expired entries
        let mut to_remove = Vec::new();

        if self.config.ttl_seconds > 0.0 {
            for &key in self.index.entries.keys() {
                let entry_path = self.entry_path(key);
                if let Ok(entry) = self.load_entry(&entry_path) {
                    if now - entry.created_at > self.config.ttl_seconds {
                        to_remove.push(key);
                    }
                }
            }
        }

        // Remove expired entries
        for key in to_remove {
            self.remove(key)?;
            removed += 1;
        }

        // Update last cleanup time
        self.index.last_cleanup = now;
        self.save_index()?;

        Ok(removed)
    }

    /// Clear all cache
    pub fn clear(&mut self) -> Result<(), CacheError> {
        // Remove all entry files
        if self.config.cache_dir.exists() {
            for entry in fs::read_dir(&self.config.cache_dir)
                .map_err(|e| CacheError::Io(e.to_string()))?
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "json") {
                        let _ = fs::remove_file(path);
                    }
                }
            }
        }

        // Reset index
        self.index = CacheIndex::new(self.arch_fingerprint.clone());
        self.stats = PersistentCacheStats::default();

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> &PersistentCacheStats {
        &self.stats
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.index.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.index.entries.is_empty()
    }

    /// Check if key exists
    pub fn contains(&self, key: u64) -> bool {
        self.index.entries.contains_key(&key)
    }

    /// Load index from disk
    fn load_index(&self) -> Result<CacheIndex, CacheError> {
        let file = File::open(self.index_path()).map_err(|e| CacheError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| CacheError::Serialization(e.to_string()))
    }

    /// Save index to disk
    fn save_index(&self) -> Result<(), CacheError> {
        let file =
            File::create(self.index_path()).map_err(|e| CacheError::Io(e.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.index)
            .map_err(|e| CacheError::Serialization(e.to_string()))
    }

    /// Load entry from disk
    fn load_entry(&self, path: &PathBuf) -> Result<PersistentEntry, CacheError> {
        let file = File::open(path).map_err(|e| CacheError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).map_err(|e| CacheError::Serialization(e.to_string()))
    }

    /// Save entry to disk
    fn save_entry(&mut self, key: u64, entry: &PersistentEntry) -> Result<(), CacheError> {
        let path = self.entry_path(key);
        let file = File::create(&path).map_err(|e| CacheError::Io(e.to_string()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, entry).map_err(|e| {
            self.stats.save_errors += 1;
            CacheError::Serialization(e.to_string())
        })
    }
}

/// Cache error types
#[derive(Debug, Clone)]
pub enum CacheError {
    /// I/O error
    Io(String),
    /// Serialization error
    Serialization(String),
    /// Not initialized
    NotInitialized,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::Io(s) => write!(f, "I/O error: {}", s),
            CacheError::Serialization(s) => write!(f, "Serialization error: {}", s),
            CacheError::NotInitialized => write!(f, "Cache not initialized"),
        }
    }
}

impl std::error::Error for CacheError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Unique counter for test directories
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn test_config() -> PersistentCacheConfig {
        // Generate unique directory for each test
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let thread_id = std::thread::current().id();
        let temp_dir = env::temp_dir().join(format!(
            "pygpukit_test_cache_{:?}_{}",
            thread_id,
            id
        ));
        PersistentCacheConfig {
            cache_dir: temp_dir,
            max_size: 1024 * 1024, // 1MB
            max_entries: 10,
            auto_cleanup: false,
            ttl_seconds: 0.0,
        }
    }

    fn test_arch() -> ArchFingerprint {
        ArchFingerprint::new(86, 24 * 1024 * 1024 * 1024, 100 * 1024, 65536, 6 * 1024 * 1024, 12040)
    }

    #[test]
    fn test_arch_fingerprint() {
        let arch1 = test_arch();
        let arch2 = test_arch();

        assert!(arch1.is_compatible(&arch2));

        let arch3 = ArchFingerprint::new(80, 0, 0, 0, 0, 12040);
        assert!(!arch1.is_compatible(&arch3));
    }

    #[test]
    fn test_cache_key() {
        let source_hash = PersistentCache::hash_source("__global__ void foo() {}");
        let options_hash = PersistentCache::hash_options(&["-O3".to_string()]);
        let arch_hash = test_arch().hash();

        let key1 = PersistentCache::compute_key(source_hash, "foo", options_hash, arch_hash);
        let key2 = PersistentCache::compute_key(source_hash, "foo", options_hash, arch_hash);
        assert_eq!(key1, key2);

        let key3 = PersistentCache::compute_key(source_hash, "bar", options_hash, arch_hash);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_persistent_cache_basic() {
        let config = test_config();
        let arch = test_arch();

        // Clean up first
        let _ = fs::remove_dir_all(&config.cache_dir);

        let mut cache = PersistentCache::new(config.clone(), arch);
        cache.initialize().unwrap();

        // Insert
        let key = cache
            .insert(
                "__global__ void test() {}",
                "test",
                &[],
                "// PTX code".to_string(),
            )
            .unwrap();

        assert!(cache.contains(key));
        assert_eq!(cache.len(), 1);

        // Get
        let entry = cache.get(key).unwrap().unwrap();
        assert_eq!(entry.name, "test");
        assert_eq!(entry.ptx, "// PTX code");

        // Clean up
        let _ = fs::remove_dir_all(&config.cache_dir);
    }

    #[test]
    fn test_persistent_cache_eviction() {
        let mut config = test_config();
        config.max_entries = 2;
        let arch = test_arch();

        // Clean up first
        let _ = fs::remove_dir_all(&config.cache_dir);

        let mut cache = PersistentCache::new(config.clone(), arch.clone());
        cache.initialize().unwrap();

        // Insert 3 entries
        cache
            .insert("src1", "k1", &[], "ptx1".to_string())
            .unwrap();
        cache
            .insert("src2", "k2", &[], "ptx2".to_string())
            .unwrap();

        // Access k2 to make k1 the LRU
        let key2 = PersistentCache::compute_key(
            PersistentCache::hash_source("src2"),
            "k2",
            PersistentCache::hash_options(&[]),
            arch.hash(),
        );
        cache.get(key2).unwrap();

        // Insert third - should evict k1
        cache
            .insert("src3", "k3", &[], "ptx3".to_string())
            .unwrap();

        assert_eq!(cache.len(), 2);
        assert!(cache.stats().evictions >= 1);

        // Clean up
        let _ = fs::remove_dir_all(&config.cache_dir);
    }

    #[test]
    fn test_persistent_cache_clear() {
        let config = test_config();
        let arch = test_arch();

        // Clean up first
        let _ = fs::remove_dir_all(&config.cache_dir);

        let mut cache = PersistentCache::new(config.clone(), arch);
        cache.initialize().unwrap();

        cache
            .insert("src1", "k1", &[], "ptx1".to_string())
            .unwrap();
        cache
            .insert("src2", "k2", &[], "ptx2".to_string())
            .unwrap();

        cache.clear().unwrap();
        assert!(cache.is_empty());

        // Clean up
        let _ = fs::remove_dir_all(&config.cache_dir);
    }
}
