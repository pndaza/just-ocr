//! Lightweight polygon boolean operations: unary_union, intersects, contains.

use super::{point_in_polygon, Point, Polygon};

pub fn bbox_intersects(a: &Polygon, b: &Polygon) -> bool {
    let (ax1, ay1, ax2, ay2) = a.bounds();
    let (bx1, by1, bx2, by2) = b.bounds();
    ax1 < bx2 && ax2 > bx1 && ay1 < by2 && ay2 > by1
}

pub fn intersects(a: &Polygon, b: &Polygon) -> bool {
    if !bbox_intersects(a, b) {
        return false;
    }
    for p in &a.exterior {
        if point_in_polygon(p, b) {
            return true;
        }
    }
    for p in &b.exterior {
        if point_in_polygon(p, a) {
            return true;
        }
    }
    false
}

pub fn contains(a: &Polygon, b: &Polygon) -> bool {
    for p in &b.exterior {
        if !point_in_polygon(p, a) {
            return false;
        }
    }
    true
}

pub fn unary_union(polygons: &[Polygon]) -> Vec<Polygon> {
    if polygons.is_empty() {
        return Vec::new();
    }
    if polygons.len() == 1 {
        return polygons.to_vec();
    }
    let mut merged: Vec<Polygon> = Vec::new();
    let mut consumed = vec![false; polygons.len()];
    for i in 0..polygons.len() {
        if consumed[i] {
            continue;
        }
        let mut group: Vec<Point> = polygons[i].exterior.clone();
        consumed[i] = true;
        for j in (i + 1)..polygons.len() {
            if consumed[j] {
                continue;
            }
            if intersects(&polygons[i], &polygons[j]) {
                group.extend(polygons[j].exterior.iter().cloned());
                consumed[j] = true;
            }
        }
        if group.len() > 2 {
            let hull = convex_hull(&group);
            merged.push(Polygon::new(hull));
        }
    }
    merged
}

pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut pts: Vec<Point> = points.to_vec();
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut lower: Vec<Point> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2
            && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], &p) <= 0.0
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<Point> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2
            && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], &p) <= 0.0
        {
            upper.pop();
        }
        upper.push(p);
    }
    let mut hull = lower;
    hull.pop();
    hull.extend(upper);
    hull.pop();
    hull
}

fn cross(a: &Point, b: &Point, c: &Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersects_overlapping() {
        let a = Polygon::from_tuples(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        let b = Polygon::from_tuples(&[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]);
        assert!(intersects(&a, &b));
    }

    #[test]
    fn test_intersects_disjoint() {
        let a = Polygon::from_tuples(&[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]);
        let b = Polygon::from_tuples(&[
            (10.0, 10.0),
            (15.0, 10.0),
            (15.0, 15.0),
            (10.0, 15.0),
        ]);
        assert!(!intersects(&a, &b));
    }

    #[test]
    fn test_contains() {
        let outer = Polygon::from_tuples(&[(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]);
        let inner = Polygon::from_tuples(&[(5.0, 5.0), (10.0, 5.0), (10.0, 10.0), (5.0, 10.0)]);
        assert!(contains(&outer, &inner));
        assert!(!contains(&inner, &outer));
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(4.0, 0.0),
            Point::new(4.0, 4.0),
            Point::new(0.0, 4.0),
            Point::new(2.0, 2.0),
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }
}
