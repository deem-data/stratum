use ahash::AHasher; //fast non-cryptographic hasher
use std::hash::{Hash, Hasher};

pub fn bucket_id(token: &str, n_features: usize) -> usize {
    let mut hasher = AHasher::default();
    token.hash(&mut hasher);
    (hasher.finish() as usize) % n_features
}