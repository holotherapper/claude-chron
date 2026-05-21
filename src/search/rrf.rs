//! Reciprocal Rank Fusion.

use std::collections::HashMap;

/// Default RRF rank constant.
pub const RRF_K: f64 = 60.0;

/// A fused ranking score for one chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct RrfScore {
    /// Chunk row id.
    pub chunk_id: i64,
    /// One-based rank from dense vector search.
    pub dense_rank: Option<usize>,
    /// One-based rank from sparse FTS search.
    pub sparse_rank: Option<usize>,
    /// Fused score, higher is better.
    pub score: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct RankState {
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
}

/// Merges dense and sparse ranked chunk ids using Reciprocal Rank Fusion.
pub fn rrf_merge(dense_ids: &[i64], sparse_ids: &[i64], limit: usize) -> Vec<RrfScore> {
    if limit == 0 {
        return Vec::new();
    }

    let mut ranks = HashMap::<i64, RankState>::new();
    for (index, chunk_id) in dense_ids.iter().enumerate() {
        ranks.entry(*chunk_id).or_default().dense_rank = Some(index + 1);
    }
    for (index, chunk_id) in sparse_ids.iter().enumerate() {
        ranks.entry(*chunk_id).or_default().sparse_rank = Some(index + 1);
    }

    let mut scores = ranks
        .into_iter()
        .map(|(chunk_id, rank_state)| {
            let dense_score = rank_state
                .dense_rank
                .map_or(0.0, |rank| 1.0 / (RRF_K + rank as f64));
            let sparse_score = rank_state
                .sparse_rank
                .map_or(0.0, |rank| 1.0 / (RRF_K + rank as f64));
            RrfScore {
                chunk_id,
                dense_rank: rank_state.dense_rank,
                sparse_rank: rank_state.sparse_rank,
                score: dense_score + sparse_score,
            }
        })
        .collect::<Vec<_>>();

    scores.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| best_rank(left).cmp(&best_rank(right)))
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    scores.truncate(limit);
    scores
}

fn best_rank(score: &RrfScore) -> usize {
    score
        .dense_rank
        .into_iter()
        .chain(score.sparse_rank)
        .min()
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_merge_should_boost_results_seen_by_both_rankers() {
        let scores = rrf_merge(&[1, 2, 3], &[3, 4, 1], 4);

        assert_eq!(scores[0].chunk_id, 1);
        assert_eq!(scores[1].chunk_id, 3);
    }

    #[test]
    fn rrf_merge_should_respect_limit() {
        let scores = rrf_merge(&[1, 2, 3], &[4, 5, 6], 2);

        assert_eq!(scores.len(), 2);
    }
}
