use once_cell::sync::Lazy;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// Thread-safe check on the first use
static NUM_THREADS: Lazy<usize> = Lazy::new(|| {
    match std::env::var("SKRUB_RUST_THREADS") {
        Ok(num) => num.parse().unwrap_or(0),
        _ => 0,
    }
});

#[inline]
fn get_num_threads() -> usize {
    *NUM_THREADS
}

// Thread-safe one time creation of thread pool
static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();

// Create rayon thread pool
fn create_thread_pool() -> Option<ThreadPool> {
    let num_threads = get_num_threads();
    let pool: Option<ThreadPool> = if num_threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .ok()
    }
    else { //num_threads = 0. Use global threadpool
        None
    };
    pool
}

pub fn get_thread_pool() -> Option<&'static ThreadPool> {
    let cached_pool = POOL.get_or_init(create_thread_pool);
    cached_pool.as_ref()
}

// Explicitly budgeted pools are cached by size. Unlike the legacy pool above,
// the first native call cannot freeze a later operation to the wrong budget.
static BOUNDED_POOLS: Lazy<Mutex<HashMap<usize, Arc<ThreadPool>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub(crate) fn get_bounded_thread_pool(workers: usize) -> Result<Arc<ThreadPool>, String> {
    if workers == 0 {
        return Err("native worker budget must be positive".into());
    }
    let mut pools = BOUNDED_POOLS
        .lock()
        .map_err(|_| "native thread-pool registry lock is poisoned".to_string())?;
    if let Some(pool) = pools.get(&workers) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(move |index| format!("stratum-{workers}-{index}"))
            .build()
            .map_err(|error| format!("failed to create native thread pool: {error}"))?,
    );
    pools.insert(workers, Arc::clone(&pool));
    Ok(pool)
}
