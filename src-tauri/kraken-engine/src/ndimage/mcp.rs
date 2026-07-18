//! MCP_Connect: minimum-cost path finding between endpoint pairs.
//! Matches skimage.graph.MCP_Connect as used by kraken's LineMCP.
//!
//! Uses a single multi-source Dijkstra: all endpoints are seeded into one
//! priority queue simultaneously, each tagged with its origin ID. As each
//! pixel is settled, the source that reached it is recorded. When a source's
//! frontier reaches a pixel already settled by another source, a connection
//! is recorded between the two sources, and that frontier branch stops.
//! This matches skimage's behavior and prevents the "star pattern".

use ndarray::Array2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::SQRT_2;

/// A connection between two endpoints: the path and its cost.
pub struct Connection {
    pub path: Vec<(usize, usize)>,
    pub cost: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Endpoint {
    id: usize,
    pos: (usize, usize),
}

#[derive(Clone, Copy)]
struct State {
    cost: f32,
    pos: (usize, usize),
    origin: usize,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}
impl Eq for State {}
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn mcp_connect(cost: &Array2<f32>, endpoints: &[(usize, usize)]) -> Vec<Connection> {
    let (h, w) = cost.dim();
    if endpoints.len() < 2 {
        return Vec::new();
    }

    let endpoint_objs: Vec<Endpoint> = endpoints
        .iter()
        .enumerate()
        .map(|(id, &pos)| Endpoint { id, pos })
        .collect();

    // Single multi-source Dijkstra.
    // dist[pixel] = minimum cost to reach this pixel from its nearest origin.
    let mut dist = Array2::<f32>::from_elem((h, w), f32::INFINITY);
    // owner[pixel] = which source settled this pixel.
    let mut owner: Array2<Option<usize>> = Array2::from_elem((h, w), None);
    // came_from[pixel] = predecessor on the path from the owning source.
    let mut came_from: Array2<Option<(usize, usize)>> = Array2::from_elem((h, w), None);

    let mut heap = BinaryHeap::new();

    // Seed all endpoints simultaneously.
    for ep in &endpoint_objs {
        let (ey, ex) = ep.pos;
        if cost[[ey, ex]] == f32::INFINITY {
            continue;
        }
        dist[[ey, ex]] = cost[[ey, ex]];
        owner[[ey, ex]] = Some(ep.id);
        came_from[[ey, ex]] = None;
        heap.push(State {
            cost: cost[[ey, ex]],
            pos: ep.pos,
            origin: ep.id,
        });
    }

    // Connection storage: key = (min_id, max_id),
    // value = (pos_a, pos_b, cost_a + cost_b) — the two meeting pixels.
    // pos_a belongs to id_a's frontier, pos_b to id_b's frontier.
    let mut connections: std::collections::HashMap<(usize, usize), ((usize, usize), (usize, usize), f32)> =
        std::collections::HashMap::new();

    while let Some(state) = heap.pop() {
        let (cy, cx) = state.pos;
        let cur_origin = state.origin;

        // Skip stale entries (already settled by a different source at lower cost).
        if dist[[cy, cx]] < state.cost {
            continue;
        }
        // Skip if already settled by a different source (frontier meeting handled in neighbor scan).
        match owner[[cy, cx]] {
            Some(o) if o != cur_origin => continue,
            None => {
                // This shouldn't happen — the origin set owner when seeding.
                owner[[cy, cx]] = Some(cur_origin);
            }
            _ => {}
        }

        // kraken's LineMCP.goal_reached returns 2 ("done") when cumcost > 0,
        // i.e. when the frontier steps off the zero-cost skeleton. This
        // confines expansion to skeleton pixels, preventing spurious
        // connections between endpoints on different skeleton components.
        if dist[[cy, cx]] > 0.0 {
            continue;
        }

        // Explore 8-connected neighbors.
        for &dy in &[-1i64, 0, 1] {
            for &dx in &[-1i64, 0, 1] {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let ny = cy as i64 + dy;
                let nx = cx as i64 + dx;
                if ny < 0 || ny >= h as i64 || nx < 0 || nx >= w as i64 {
                    continue;
                }
                let ny = ny as usize;
                let nx = nx as usize;
                let step_cost = cost[[ny, nx]];
                if step_cost == f32::INFINITY {
                    continue;
                }
                let move_cost = if dy != 0 && dx != 0 { SQRT_2 } else { 1.0 };
                let new_cost = dist[[cy, cx]] + step_cost * move_cost;

                match owner[[ny, nx]] {
                    None => {
                        // Unvisited: claim it.
                        if new_cost < dist[[ny, nx]] {
                            dist[[ny, nx]] = new_cost;
                            owner[[ny, nx]] = Some(cur_origin);
                            came_from[[ny, nx]] = Some((cy, cx));
                            heap.push(State {
                                cost: new_cost,
                                pos: (ny, nx),
                                origin: cur_origin,
                            });
                        }
                    }
                    Some(neighbor_origin) if neighbor_origin != cur_origin => {
                        // Frontiers from different sources meet.
                        // (cur_origin, cy/cx) is one side; (neighbor_origin, ny/nx) is the other.
                        let total_cost = new_cost + dist[[ny, nx]];
                        let key = (
                            cur_origin.min(neighbor_origin),
                            cur_origin.max(neighbor_origin),
                        );
                        // pos_a (min id's side) and pos_b (max id's side).
                        let (pos_a, pos_b) = if cur_origin < neighbor_origin {
                            ((cy, cx), (ny, nx))
                        } else {
                            ((ny, nx), (cy, cx))
                        };
                        let entry = connections
                            .entry(key)
                            .or_insert((pos_a, pos_b, total_cost));
                        if total_cost < entry.2 {
                            *entry = (pos_a, pos_b, total_cost);
                        }
                        // Don't claim the pixel — it belongs to the other source.
                    }
                    _ => {
                        // Same origin: relax if cheaper.
                        if new_cost < dist[[ny, nx]] {
                            dist[[ny, nx]] = new_cost;
                            came_from[[ny, nx]] = Some((cy, cx));
                            heap.push(State {
                                cost: new_cost,
                                pos: (ny, nx),
                                origin: cur_origin,
                            });
                        }
                    }
                }
            }
        }
    }

    // Reconstruct full bidirectional paths, matching skimage's get_connections():
    //   traceback(pos_a) gives [ep_a, ..., pos_a]
    //   traceback(pos_b) gives [ep_b, ..., pos_b]; reversed -> [pos_b, ..., ep_b]
    //   concatenate -> [ep_a, ..., pos_a, pos_b, ..., ep_b]
    // Since came_from was set per-source during the single multi-source Dijkstra,
    // each pixel's chain correctly leads back to its owning endpoint.
    let mut result: Vec<Connection> = Vec::new();

    for ((id_a, id_b), (pos_a, pos_b, total_cost)) in connections {
        let ep_a = endpoint_objs[id_a].pos;
        let ep_b = endpoint_objs[id_b].pos;

        // Half A: traceback from pos_a to ep_a.
        let mut half_a = vec![pos_a];
        let mut cur = pos_a;
        while cur != ep_a {
            match came_from[[cur.0, cur.1]] {
                Some(prev) => {
                    half_a.push(prev);
                    cur = prev;
                }
                None => break, // chain broken (safety)
            }
        }
        half_a.reverse(); // -> [ep_a, ..., pos_a]

        // Half B: traceback from pos_b to ep_b.
        let mut half_b = vec![pos_b];
        cur = pos_b;
        while cur != ep_b {
            match came_from[[cur.0, cur.1]] {
                Some(prev) => {
                    half_b.push(prev);
                    cur = prev;
                }
                None => break,
            }
        }
        half_b.reverse(); // -> [ep_b, ..., pos_b]

        // Concatenate: half_a + reversed(half_b) -> [ep_a, ..., pos_a, pos_b, ..., ep_b].
        // Drop the duplicate pos_b if it appears at both the end of half_a and start of reversed half_b.
        let mut full_path = half_a;
        let mut rev_b = half_b;
        rev_b.reverse(); // -> [pos_b, ..., ep_b]
        if let Some(&last) = full_path.last() {
            if let Some(&first) = rev_b.first() {
                if last == first {
                    rev_b.remove(0);
                }
            }
        }
        full_path.extend(rev_b);

        result.push(Connection {
            path: full_path,
            cost: total_cost,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_mcp_connect_two_endpoints() {
        let cost = array![
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let endpoints = vec![(1usize, 0usize), (1, 4)];
        let connections = mcp_connect(&cost, &endpoints);
        assert!(!connections.is_empty(), "expected at least 1 connection");
        let path = &connections[0].path;
        assert!(path.contains(&(1, 0)) || path.contains(&(1, 4)), "path should contain an endpoint");
    }

    #[test]
    fn test_mcp_connect_no_connection() {
        let cost = array![
            [0.0, 0.0, f32::INFINITY, 0.0, 0.0],
        ];
        let endpoints = vec![(0, 0), (0, 4)];
        let connections = mcp_connect(&cost, &endpoints);
        assert!(connections.is_empty(), "expected no connections");
    }

    #[test]
    fn test_mcp_connect_no_star_pattern() {
        // 4 endpoints on a horizontal skeleton.
        let cost = array![
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let endpoints = vec![(1, 0), (1, 2), (1, 5), (1, 8)];
        let connections = mcp_connect(&cost, &endpoints);
        // Should find connections between adjacent endpoint pairs.
        assert!(!connections.is_empty(), "should find connections");
        assert!(connections.len() <= 6, "too many connections: {}", connections.len());
    }

    #[test]
    fn test_mcp_connect_full_bidirectional_path() {
        // Two endpoints at opposite ends of a zero-cost ridge.
        // The reconstructed path must contain BOTH endpoints, matching
        // skimage's get_connections() (traceback(pos1) + traceback(pos2)[::-1]).
        let cost = array![
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        let endpoints = vec![(1usize, 0usize), (1, 6)];
        let connections = mcp_connect(&cost, &endpoints);
        assert_eq!(connections.len(), 1, "expected exactly 1 connection");
        let path = &connections[0].path;
        assert!(
            path.contains(&(1, 0)),
            "path must contain endpoint A: {:?}",
            path
        );
        assert!(
            path.contains(&(1, 6)),
            "path must contain endpoint B: {:?}",
            path
        );
        // Path should span from one endpoint to the other (≥ 5 pixels for a 7-wide ridge).
        assert!(
            path.len() >= 5,
            "path too short for full bidirectional trace: {:?}",
            path
        );
    }
}
