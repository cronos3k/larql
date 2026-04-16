//! HNSW (Hierarchical Navigable Small World) index for gate vector search.
//!
//! Replaces brute-force gate KNN (O(N) comparisons per query) with
//! approximate nearest neighbor search via graph traversal (O(log N)).
//!
//! Uses random projection to reduce dimensionality during graph construction
//! and search traversal. Final candidates are scored with exact dot products
//! by the caller. This makes the build practical at dim=2560.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Max-heap element (best score first).
#[derive(Clone, Copy)]
struct MaxScored { score: f32, id: u32 }
impl PartialEq for MaxScored { fn eq(&self, o: &Self) -> bool { self.id == o.id } }
impl Eq for MaxScored {}
impl PartialOrd for MaxScored { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MaxScored {
    fn cmp(&self, o: &Self) -> Ordering { self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal) }
}

/// Min-heap element (worst score first — for eviction).
#[derive(Clone, Copy)]
struct MinScored { score: f32, id: u32 }
impl PartialEq for MinScored { fn eq(&self, o: &Self) -> bool { self.id == o.id } }
impl Eq for MinScored {}
impl PartialOrd for MinScored { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MinScored {
    fn cmp(&self, o: &Self) -> Ordering { o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal) }
}

/// Projected dimension for graph construction.
/// Full-dim dot products are only done for final candidate scoring.
const PROJ_DIM: usize = 64;

/// HNSW index for a single layer's gate vectors.
///
/// The graph is built and traversed using random-projected vectors (dim=64).
/// This makes build O(N log N) at dim=64 instead of dim=2560 — ~40x faster.
/// Search returns candidate IDs; the caller does exact scoring on the originals.
pub struct HnswLayer {
    num_vectors: usize,
    m: usize,
    m_max0: usize,
    max_level: usize,
    entry_point: usize,
    node_levels: Vec<u8>,
    level0: Vec<u32>,
    upper: Vec<Vec<u32>>,
    /// Random projection matrix: [dim, PROJ_DIM] for query projection.
    proj_matrix: Array2<f32>,
    /// Projected vectors: [num_vectors, PROJ_DIM] for fast graph traversal.
    projected: Array2<f32>,
}

impl HnswLayer {
    /// Build an HNSW index from gate vectors.
    ///
    /// `vectors`: [num_vectors, dim] matrix (used for random projection).
    /// `m`: max connections per node (8-16 typical for 10K vectors).
    /// `ef_construction`: beam width during build (32-100 typical).
    pub fn build(vectors: &ArrayView2<f32>, m: usize, ef_construction: usize) -> Self {
        let n = vectors.shape()[0];
        let dim = vectors.shape()[1];
        let m_max0 = m * 2;
        let ml = 1.0 / (m as f64).ln();

        if n == 0 {
            return Self {
                num_vectors: 0, m, m_max0, max_level: 0,
                entry_point: 0, node_levels: vec![],
                level0: vec![], upper: vec![],
                proj_matrix: Array2::zeros((0, PROJ_DIM)),
                projected: Array2::zeros((0, PROJ_DIM)),
            };
        }

        // Random projection: dim -> PROJ_DIM
        let proj_matrix = Self::random_projection_matrix(dim, PROJ_DIM);
        let cpu = larql_compute::default_backend();
        use larql_compute::ComputeBackend;
        let projected = cpu.matmul(vectors.view(), proj_matrix.view());

        // Assign random levels
        let mut node_levels = vec![0u8; n];
        let mut max_level = 0usize;
        let mut rng = 42u64;
        for nl in node_levels.iter_mut().take(n) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (rng >> 33) as f64 / (1u64 << 31) as f64;
            let level = ((-u.max(1e-12).ln() * ml).floor() as usize).min(12);
            *nl = level as u8;
            if level > max_level { max_level = level; }
        }

        let level0 = vec![u32::MAX; n * m_max0];
        let upper: Vec<Vec<u32>> = (0..max_level).map(|_| vec![u32::MAX; n * m]).collect();

        let entry_point = node_levels.iter().enumerate()
            .max_by_key(|(_, &l)| l).map(|(i, _)| i).unwrap_or(0);

        let mut index = Self {
            num_vectors: n, m, m_max0, max_level,
            entry_point, node_levels, level0, upper,
            proj_matrix, projected,
        };

        // Build graph using projected vectors (dim=64, fast).
        // Clone projected to avoid borrow conflict with mutable index methods.
        let proj = index.projected.clone();
        let proj_view = proj.view();
        for id in 0..n {
            if id == entry_point && id == 0 { continue; }
            let q = proj_view.row(id);
            let node_level = index.node_levels[id] as usize;

            let mut ep = index.entry_point;
            for lev in (node_level.saturating_add(1)..=index.max_level).rev() {
                ep = index.greedy_closest(&proj_view, &q, ep, lev);
            }

            for lev in (0..=node_level.min(index.max_level)).rev() {
                let max_conn = if lev == 0 { m_max0 } else { m };
                let candidates = index.search_level(&proj_view, &q, ep, ef_construction, lev);

                let selected: Vec<u32> = candidates.iter()
                    .take(max_conn)
                    .map(|s| s.id)
                    .collect();

                index.set_neighbors(id, lev, &selected);

                for &nb in &selected {
                    index.add_connection(nb as usize, lev, id as u32, max_conn, &proj_view);
                }

                if let Some(closest) = selected.first() {
                    ep = *closest as usize;
                }
            }

            if node_level > index.node_levels[index.entry_point] as usize {
                index.entry_point = id;
            }
        }

        index
    }

    /// Search for top-K nearest neighbors.
    ///
    /// Uses projected vectors for graph traversal, then scores final candidates
    /// with exact full-dimensional dot products against `vectors`.
    ///
    /// Returns (feature_index, exact_score) sorted by score descending.
    pub fn search(
        &self,
        vectors: &ArrayView2<f32>,
        query: &Array1<f32>,
        top_k: usize,
        ef_search: usize,
    ) -> Vec<(usize, f32)> {
        if self.num_vectors == 0 { return vec![]; }

        let ef = ef_search.max(top_k);

        // Project query to low-dim (PROJ_DIM) for fast graph traversal
        let proj_view = self.projected.view();
        let cpu = larql_compute::default_backend();
        use larql_compute::ComputeBackend;
        let x = query.view().into_shape_with_order((1, query.len())).unwrap();
        let proj_2d = cpu.matmul(x, self.proj_matrix.view());
        let proj_query = Array1::from_vec(proj_2d.into_raw_vec_and_offset().0);

        // Upper levels: greedy descent using projected vectors (dim=64, fast)
        let mut ep = self.entry_point;
        for lev in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&proj_view, &proj_query.view(), ep, lev);
        }

        // Level 0: beam search using projected vectors (ef comparisons at dim=64)
        let candidates = self.search_level(&proj_view, &proj_query.view(), ep, ef, 0);

        // Re-score final candidates with exact full-dim dot products
        let mut results: Vec<(usize, f32)> = candidates.into_iter()
            .map(|s| {
                let exact_score = Self::dot(&vectors.row(s.id as usize), &query.view());
                (s.id as usize, exact_score)
            })
            .collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Generate a random projection matrix [dim, proj_dim].
    /// Uses the same deterministic RNG for reproducibility.
    fn random_projection_matrix(dim: usize, proj_dim: usize) -> Array2<f32> {
        let scale = 1.0 / (proj_dim as f32).sqrt();
        let mut rng = 123456789u64;
        Array2::from_shape_fn((dim, proj_dim), |_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (rng >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
            u * scale
        })
    }

    // ── Internals ──

    #[inline(always)]
    fn dot(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        larql_compute::dot(a, b)
    }

    fn greedy_closest(&self, vectors: &ArrayView2<f32>, query: &ArrayView1<f32>, mut ep: usize, level: usize) -> usize {
        let mut best = Self::dot(&vectors.row(ep), query);
        loop {
            let mut changed = false;
            for &nb in self.neighbors(ep, level) {
                if nb == u32::MAX { break; }
                let s = Self::dot(&vectors.row(nb as usize), query);
                if s > best { best = s; ep = nb as usize; changed = true; }
            }
            if !changed { break; }
        }
        ep
    }

    fn search_level(
        &self,
        vectors: &ArrayView2<f32>,
        query: &ArrayView1<f32>,
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<MaxScored> {
        let mut visited = vec![false; self.num_vectors];
        visited[entry] = true;

        let entry_score = Self::dot(&vectors.row(entry), query);

        let mut candidates: BinaryHeap<MaxScored> = BinaryHeap::new();
        candidates.push(MaxScored { score: entry_score, id: entry as u32 });

        let mut results: BinaryHeap<MinScored> = BinaryHeap::new();
        results.push(MinScored { score: entry_score, id: entry as u32 });

        while let Some(current) = candidates.pop() {
            let worst = results.peek().map(|s| s.score).unwrap_or(f32::NEG_INFINITY);
            if current.score < worst && results.len() >= ef {
                break;
            }

            for &nb in self.neighbors(current.id as usize, level) {
                if nb == u32::MAX { break; }
                let nid = nb as usize;
                if nid >= self.num_vectors || visited[nid] { continue; }
                visited[nid] = true;

                let score = Self::dot(&vectors.row(nid), query);
                let worst = results.peek().map(|s| s.score).unwrap_or(f32::NEG_INFINITY);

                if score > worst || results.len() < ef {
                    candidates.push(MaxScored { score, id: nb });
                    results.push(MinScored { score, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<MaxScored> = results.into_iter()
            .map(|s| MaxScored { score: s.score, id: s.id })
            .collect();
        out.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        out
    }

    fn neighbors(&self, node: usize, level: usize) -> &[u32] {
        if level == 0 {
            let s = node * self.m_max0;
            &self.level0[s..s + self.m_max0]
        } else if level <= self.upper.len() {
            let s = node * self.m;
            let arr = &self.upper[level - 1];
            if s + self.m <= arr.len() { &arr[s..s + self.m] } else { &[] }
        } else {
            &[]
        }
    }

    fn set_neighbors(&mut self, node: usize, level: usize, nbs: &[u32]) {
        if level == 0 {
            let s = node * self.m_max0;
            for (i, &n) in nbs.iter().take(self.m_max0).enumerate() {
                self.level0[s + i] = n;
            }
        } else if level <= self.upper.len() {
            let s = node * self.m;
            let arr = &mut self.upper[level - 1];
            for (i, &n) in nbs.iter().take(self.m).enumerate() {
                arr[s + i] = n;
            }
        }
    }

    fn add_connection(&mut self, node: usize, level: usize, new_nb: u32, max_conn: usize, vectors: &ArrayView2<f32>) {
        let (arr, start, cap) = if level == 0 {
            (&mut self.level0 as &mut Vec<u32>, node * self.m_max0, self.m_max0.min(max_conn))
        } else if level <= self.upper.len() {
            (&mut self.upper[level - 1] as &mut Vec<u32>, node * self.m, self.m.min(max_conn))
        } else {
            return;
        };

        if start + cap > arr.len() { return; }
        let slot = &mut arr[start..start + cap];

        for s in slot.iter_mut().take(cap) {
            if *s == u32::MAX { *s = new_nb; return; }
            if *s == new_nb { return; }
        }

        // Evict worst neighbor if new one is better
        let node_vec = vectors.row(node);
        let new_score = Self::dot(&node_vec, &vectors.row(new_nb as usize));
        let mut worst_i = 0;
        let mut worst_s = f32::MAX;
        for (i, &nb) in slot.iter().enumerate().take(cap) {
            let s = Self::dot(&node_vec, &vectors.row(nb as usize));
            if s < worst_s { worst_s = s; worst_i = i; }
        }
        if new_score > worst_s {
            slot[worst_i] = new_nb;
        }
    }

    pub fn len(&self) -> usize { self.num_vectors }
    pub fn is_empty(&self) -> bool { self.num_vectors == 0 }
}
