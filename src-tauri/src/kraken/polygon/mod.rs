//! Lightweight polygon geometry: types, simplify, point-in-polygon.

pub mod boolean;

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A polyline: ordered sequence of points.
#[derive(Debug, Clone)]
pub struct PolyLine {
    pub points: Vec<Point>,
}

/// A polygon: exterior ring (and optionally holes, not needed for kraken's use).
#[derive(Debug, Clone)]
pub struct Polygon {
    pub exterior: Vec<Point>,
}

impl Polygon {
    pub fn new(exterior: Vec<Point>) -> Self {
        Self { exterior }
    }

    pub fn from_tuples(coords: &[(f64, f64)]) -> Self {
        Self {
            exterior: coords.iter().map(|&(x, y)| Point::new(x, y)).collect(),
        }
    }

    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &self.exterior {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        (min_x, min_y, max_x, max_y)
    }
}

/// Douglas-Peucker polyline/polygon simplification.
pub fn simplify(points: &[Point], tolerance: f64) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    simplify_recursive(points, &mut keep, 0, points.len() - 1, tolerance);
    let mut result = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k {
            result.push(points[i]);
        }
    }
    result
}

fn simplify_recursive(points: &[Point], keep: &mut [bool], start: usize, end: usize, tol: f64) {
    if end <= start + 1 {
        return;
    }
    let (mut max_dist, mut max_idx) = (0.0f64, start);
    let s = &points[start];
    let e = &points[end];
    for i in (start + 1)..end {
        let d = perpendicular_distance(&points[i], s, e);
        if d > max_dist {
            max_dist = d;
            max_idx = i;
        }
    }
    if max_dist > tol {
        keep[max_idx] = true;
        simplify_recursive(points, keep, start, max_idx, tol);
        simplify_recursive(points, keep, max_idx, end, tol);
    }
}

fn perpendicular_distance(p: &Point, s: &Point, e: &Point) -> f64 {
    let dx = e.x - s.x;
    let dy = e.y - s.y;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        let ddx = p.x - s.x;
        let ddy = p.y - s.y;
        return (ddx * ddx + ddy * ddy).sqrt();
    }
    let t = ((p.x - s.x) * dx + (p.y - s.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = s.x + t * dx;
    let proj_y = s.y + t * dy;
    let ddx = p.x - proj_x;
    let ddy = p.y - proj_y;
    (ddx * ddx + ddy * ddy).sqrt()
}

/// Ray-casting point-in-polygon test.
pub fn point_in_polygon(p: &Point, polygon: &Polygon) -> bool {
    let verts = &polygon.exterior;
    let n = verts.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let vi = &verts[i];
        let vj = &verts[j];
        if (vi.y > p.y) != (vj.y > p.y)
            && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y + 1e-12) + vi.x)
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_straight_line() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(4.0, 0.0),
        ];
        let simplified = simplify(&points, 0.5);
        assert_eq!(simplified.len(), 2);
    }

    #[test]
    fn test_simplify_preserves_corners() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
            Point::new(0.0, 0.0),
        ];
        let simplified = simplify(&points, 0.01);
        assert!(simplified.len() >= 4);
    }

    #[test]
    fn test_point_in_polygon() {
        let square =
            Polygon::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        assert!(point_in_polygon(&Point::new(5.0, 5.0), &square));
        assert!(!point_in_polygon(&Point::new(-1.0, 5.0), &square));
        assert!(!point_in_polygon(&Point::new(15.0, 5.0), &square));
    }
}
