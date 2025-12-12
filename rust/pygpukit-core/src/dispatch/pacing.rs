//! Kernel Pacing Engine
//!
//! Throttles kernel launches based on allocated bandwidth.
//! Implements time-based pacing to prevent GPU saturation.

use std::time::{Instant, Duration};
use std::collections::HashMap;

/// Pacing configuration
#[derive(Debug, Clone)]
pub struct PacingConfig {
    /// Total bandwidth available (0.0 - 1.0)
    pub total_bandwidth: f64,
    /// Pacing window duration in milliseconds
    pub window_ms: f64,
    /// Minimum interval between launches in milliseconds
    pub min_interval_ms: f64,
    /// Enable adaptive pacing based on actual utilization
    pub adaptive: bool,
}

impl Default for PacingConfig {
    fn default() -> Self {
        Self {
            total_bandwidth: 1.0,
            window_ms: 100.0,
            min_interval_ms: 0.1,
            adaptive: true,
        }
    }
}

impl PacingConfig {
    /// Create with total bandwidth
    pub fn with_bandwidth(bandwidth: f64) -> Self {
        Self {
            total_bandwidth: bandwidth,
            ..Default::default()
        }
    }

    /// Set window duration
    pub fn window(mut self, window_ms: f64) -> Self {
        self.window_ms = window_ms;
        self
    }

    /// Set minimum interval
    pub fn min_interval(mut self, min_ms: f64) -> Self {
        self.min_interval_ms = min_ms;
        self
    }

    /// Enable/disable adaptive pacing
    pub fn adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }
}

/// Per-stream pacing state
#[derive(Debug)]
struct StreamPacing {
    /// Last launch timestamp
    last_launch: Option<Instant>,
    /// Allocated bandwidth for this stream
    bandwidth: f64,
    /// Launches in current window
    launches_in_window: usize,
    /// Window start time
    window_start: Instant,
    /// Total launches
    total_launches: usize,
    /// Total throttled requests
    throttled_count: usize,
}

impl StreamPacing {
    fn new(bandwidth: f64) -> Self {
        Self {
            last_launch: None,
            bandwidth,
            launches_in_window: 0,
            window_start: Instant::now(),
            total_launches: 0,
            throttled_count: 0,
        }
    }
}

/// Pacing decision result
#[derive(Debug, Clone, PartialEq)]
pub enum PacingDecision {
    /// Launch immediately
    Launch,
    /// Wait for the specified duration before launching
    Wait { delay_ms: f64 },
    /// Throttled - exceed bandwidth allocation
    Throttle { reason: String },
}

impl PacingDecision {
    /// Check if immediate launch is allowed
    pub fn can_launch(&self) -> bool {
        matches!(self, Self::Launch)
    }

    /// Check if throttled
    pub fn is_throttled(&self) -> bool {
        matches!(self, Self::Throttle { .. })
    }

    /// Get wait time in milliseconds (0.0 if can launch immediately)
    pub fn wait_ms(&self) -> f64 {
        match self {
            Self::Launch => 0.0,
            Self::Wait { delay_ms } => *delay_ms,
            Self::Throttle { .. } => f64::INFINITY,
        }
    }
}

/// Kernel pacing engine
#[derive(Debug)]
pub struct KernelPacingEngine {
    config: PacingConfig,
    /// Per-stream pacing state
    streams: HashMap<u64, StreamPacing>,
    /// Global bandwidth usage tracking
    used_bandwidth: f64,
    /// Statistics
    total_launches: usize,
    total_throttled: usize,
    total_waited: usize,
}

impl KernelPacingEngine {
    /// Create a new pacing engine
    pub fn new(config: PacingConfig) -> Self {
        Self {
            config,
            streams: HashMap::new(),
            used_bandwidth: 0.0,
            total_launches: 0,
            total_throttled: 0,
            total_waited: 0,
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(PacingConfig::default())
    }

    /// Allocate bandwidth for a stream
    pub fn allocate_stream(&mut self, stream_id: u64, bandwidth: f64) -> bool {
        let available = self.config.total_bandwidth - self.used_bandwidth;
        if bandwidth > available {
            return false;
        }

        self.used_bandwidth += bandwidth;
        self.streams.insert(stream_id, StreamPacing::new(bandwidth));
        true
    }

    /// Release bandwidth for a stream
    pub fn release_stream(&mut self, stream_id: u64) {
        if let Some(pacing) = self.streams.remove(&stream_id) {
            self.used_bandwidth = (self.used_bandwidth - pacing.bandwidth).max(0.0);
        }
    }

    /// Check if a kernel launch should proceed
    ///
    /// Returns a pacing decision indicating whether to launch,
    /// wait, or throttle.
    pub fn should_launch(&self, stream_id: u64) -> PacingDecision {
        let stream = match self.streams.get(&stream_id) {
            Some(s) => s,
            None => return PacingDecision::Launch, // Unknown stream, allow
        };

        let _now = Instant::now();

        // Check minimum interval
        if let Some(last) = stream.last_launch {
            let elapsed_ms = last.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms < self.config.min_interval_ms {
                let delay = self.config.min_interval_ms - elapsed_ms;
                return PacingDecision::Wait { delay_ms: delay };
            }
        }

        // Calculate pacing interval based on bandwidth allocation
        // Higher bandwidth = shorter interval = more launches allowed
        let pacing_interval_ms = if stream.bandwidth > 0.0 {
            self.config.window_ms * (1.0 - stream.bandwidth)
        } else {
            self.config.window_ms // No bandwidth = must wait full window
        };

        // Check if we're within pacing interval
        if let Some(last) = stream.last_launch {
            let elapsed_ms = last.elapsed().as_secs_f64() * 1000.0;
            if elapsed_ms < pacing_interval_ms {
                let delay = pacing_interval_ms - elapsed_ms;
                return PacingDecision::Wait { delay_ms: delay };
            }
        }

        // Check window-based throttling
        let window_elapsed = stream.window_start.elapsed();
        if window_elapsed < Duration::from_secs_f64(self.config.window_ms / 1000.0) {
            // Still in window - check launch count
            let max_launches_per_window = (stream.bandwidth * 100.0).ceil() as usize;
            if stream.launches_in_window >= max_launches_per_window.max(1) {
                return PacingDecision::Throttle {
                    reason: format!(
                        "Exceeded {} launches in {:.1}ms window",
                        max_launches_per_window, self.config.window_ms
                    ),
                };
            }
        }

        PacingDecision::Launch
    }

    /// Record a kernel launch
    pub fn record_launch(&mut self, stream_id: u64) {
        let now = Instant::now();

        if let Some(stream) = self.streams.get_mut(&stream_id) {
            // Reset window if expired
            let window_duration = Duration::from_secs_f64(self.config.window_ms / 1000.0);
            if stream.window_start.elapsed() >= window_duration {
                stream.window_start = now;
                stream.launches_in_window = 0;
            }

            stream.last_launch = Some(now);
            stream.launches_in_window += 1;
            stream.total_launches += 1;
        }

        self.total_launches += 1;
    }

    /// Record a throttled request
    pub fn record_throttle(&mut self, stream_id: u64) {
        if let Some(stream) = self.streams.get_mut(&stream_id) {
            stream.throttled_count += 1;
        }
        self.total_throttled += 1;
    }

    /// Record a waited request
    pub fn record_wait(&mut self) {
        self.total_waited += 1;
    }

    /// Get stream statistics
    pub fn stream_stats(&self, stream_id: u64) -> Option<StreamPacingStats> {
        self.streams.get(&stream_id).map(|s| StreamPacingStats {
            stream_id,
            bandwidth: s.bandwidth,
            launches_in_window: s.launches_in_window,
            total_launches: s.total_launches,
            throttled_count: s.throttled_count,
        })
    }

    /// Get global statistics
    pub fn stats(&self) -> PacingStats {
        PacingStats {
            stream_count: self.streams.len(),
            used_bandwidth: self.used_bandwidth,
            available_bandwidth: self.config.total_bandwidth - self.used_bandwidth,
            total_launches: self.total_launches,
            total_throttled: self.total_throttled,
            total_waited: self.total_waited,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PacingConfig {
        &self.config
    }

    /// Reset all pacing state
    pub fn reset(&mut self) {
        self.streams.clear();
        self.used_bandwidth = 0.0;
        self.total_launches = 0;
        self.total_throttled = 0;
        self.total_waited = 0;
    }
}

/// Per-stream pacing statistics
#[derive(Debug, Clone, Default)]
pub struct StreamPacingStats {
    /// Stream ID
    pub stream_id: u64,
    /// Allocated bandwidth
    pub bandwidth: f64,
    /// Launches in current window
    pub launches_in_window: usize,
    /// Total launches
    pub total_launches: usize,
    /// Throttled count
    pub throttled_count: usize,
}

/// Global pacing statistics
#[derive(Debug, Clone, Default)]
pub struct PacingStats {
    /// Number of active streams
    pub stream_count: usize,
    /// Used bandwidth
    pub used_bandwidth: f64,
    /// Available bandwidth
    pub available_bandwidth: f64,
    /// Total launches
    pub total_launches: usize,
    /// Total throttled requests
    pub total_throttled: usize,
    /// Total waited requests
    pub total_waited: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pacing_engine_creation() {
        let engine = KernelPacingEngine::with_defaults();
        assert_eq!(engine.stats().stream_count, 0);
        assert!((engine.config().total_bandwidth - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stream_allocation() {
        let mut engine = KernelPacingEngine::with_defaults();

        assert!(engine.allocate_stream(0, 0.5));
        assert!(engine.allocate_stream(1, 0.3));
        assert!(!engine.allocate_stream(2, 0.3)); // Would exceed 1.0

        let stats = engine.stats();
        assert_eq!(stats.stream_count, 2);
        assert!((stats.used_bandwidth - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_launch_decision() {
        let config = PacingConfig::default()
            .window(10.0)
            .min_interval(0.0);
        let mut engine = KernelPacingEngine::new(config);
        engine.allocate_stream(0, 0.5);

        // First launch should succeed
        let decision = engine.should_launch(0);
        assert!(decision.can_launch());
    }

    #[test]
    fn test_pacing_interval() {
        let config = PacingConfig::default()
            .window(100.0)
            .min_interval(5.0);
        let mut engine = KernelPacingEngine::new(config);
        engine.allocate_stream(0, 0.5);

        // First launch
        engine.record_launch(0);

        // Second launch should wait
        let decision = engine.should_launch(0);
        match decision {
            PacingDecision::Wait { delay_ms } => {
                assert!(delay_ms > 0.0);
            }
            _ => panic!("Expected Wait decision"),
        }
    }

    #[test]
    fn test_stream_release() {
        let mut engine = KernelPacingEngine::with_defaults();
        engine.allocate_stream(0, 0.5);
        engine.release_stream(0);

        let stats = engine.stats();
        assert_eq!(stats.stream_count, 0);
        assert!((stats.used_bandwidth).abs() < 0.001);
    }

    #[test]
    fn test_unknown_stream() {
        let engine = KernelPacingEngine::with_defaults();

        // Unknown stream should be allowed
        let decision = engine.should_launch(999);
        assert!(decision.can_launch());
    }
}
