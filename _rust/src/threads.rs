use std::sync::OnceLock;
use rayon::{ThreadPool, ThreadPoolBuilder};
use crate::util::get_num_threads;

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