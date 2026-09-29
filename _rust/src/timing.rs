use std::time::{Duration, Instant};

#[inline]
pub fn debug_enabled() -> bool {
    // Read in each call to allow dynamic change
    let debug: once_cell::sync::Lazy<bool> = once_cell::sync::Lazy::new(|| {
        std::env::var("SKRUB_RUST_DEBUG_TIMING")
            .map(|v| matches!(v.to_lowercase().as_str(), "1"))
            .unwrap_or(false)
    });

    *debug
}

#[inline]
pub fn start_timing() -> Option<Instant> {
    if debug_enabled() {
        Some(Instant::now())
    } else {
        None
    }
}

#[inline]
pub fn print_timing(msg: &str, start: Option<Instant>) {
    match start {
        Some(t0) => eprintln!("[rust] {msg}: {}ms", t0.elapsed().as_millis()),
        None => { /*do nothing*/ }
    }
}

// Accumulates many short internal phases without printing inside hot loops.
// Callers capture the debug flag once, so disabled counters add only a branch.
pub(crate) struct TimingCounter {
    enabled: bool,
    elapsed: Duration,
    calls: u64,
}

impl TimingCounter {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            elapsed: Duration::ZERO,
            calls: 0,
        }
    }

    #[inline]
    pub(crate) fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[inline]
    pub(crate) fn record(&mut self, start: Option<Instant>) {
        if let Some(start) = start {
            self.elapsed += start.elapsed();
            self.calls += 1;
        }
    }

    pub(crate) fn print(&self, msg: &str) {
        if self.enabled {
            eprintln!(
                "[rust] {msg}: {:.3}ms across {} calls",
                self.elapsed.as_secs_f64() * 1_000.0,
                self.calls
            );
        }
    }
}
