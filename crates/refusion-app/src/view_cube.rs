//! Sharp cube faces with invisible face/edge/corner hit zones on their surfaces.
use eframe::egui::Pos2;
use glam::{DMat3, DQuat, DVec3};

pub struct Region {
    pub direction: DVec3,
    pub polygon: Vec<Pos2>,
    pub label: String,
}

impl Region {
    pub fn contains(&self, p: Pos2) -> bool {
        let cross: Vec<_> = (0..self.polygon.len())
            .map(|i| {
                let a = self.polygon[i];
                let b = self.polygon[(i + 1) % self.polygon.len()];
                (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            })
            .collect();
        cross.iter().all(|&v| v >= -0.001) || cross.iter().all(|&v| v <= 0.001)
    }

    pub fn center(&self) -> Pos2 {
        let n = self.polygon.len() as f32;
        Pos2::new(
            self.polygon.iter().map(|p| p.x).sum::<f32>() / n,
            self.polygon.iter().map(|p| p.y).sum::<f32>() / n,
        )
    }
}

/// Camera rotation looking from a cube face, edge or corner toward the target.
pub fn rotation(direction: DVec3) -> DQuat {
    let z = direction.normalize();
    let up = if z.y.abs() > 0.999 {
        if z.y > 0.0 {
            -DVec3::Z
        } else {
            DVec3::Z
        }
    } else {
        DVec3::Y
    };
    let x = up.cross(z).normalize();
    let y = z.cross(x);
    DQuat::from_mat3(&DMat3::from_cols(x, y, z))
}

fn label(d: DVec3) -> String {
    [
        (d.y, "Top", "Bottom"),
        (d.z, "Front", "Back"),
        (d.x, "Right", "Left"),
    ]
    .into_iter()
    .filter_map(|(v, pos, neg)| {
        if v == 0.0 {
            None
        } else {
            Some(if v > 0.0 { pos } else { neg })
        }
    })
    .collect::<Vec<_>>()
    .join(" / ")
}

fn patches(rot: DQuat, center: Pos2, scale: f64, split: bool) -> Vec<Region> {
    let inv = rot.inverse();
    let mut out = Vec::new();
    let bands: &[(f64, f64, f64)] = if split {
        &[(-1.0, -0.72, -1.0), (-0.72, 0.72, 0.0), (0.72, 1.0, 1.0)]
    } else {
        &[(-1.0, 1.0, 0.0)]
    };
    for axis in 0..3 {
        for sign in [-1.0, 1.0] {
            let mut normal = DVec3::ZERO;
            normal[axis] = sign;
            if (inv * normal).z <= 1e-6 {
                continue;
            }
            let free: Vec<_> = (0..3).filter(|&i| i != axis).collect();
            for &(a0, a1, da) in bands {
                for &(b0, b1, db) in bands {
                    let mut direction = normal;
                    direction[free[0]] = da;
                    direction[free[1]] = db;
                    let polygon = [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]
                        .into_iter()
                        .map(|(a, b)| {
                            let mut v = normal;
                            v[free[0]] = a;
                            v[free[1]] = b;
                            let p = inv * v;
                            Pos2::new(
                                center.x + (p.x * scale) as f32,
                                center.y - (p.y * scale) as f32,
                            )
                        })
                        .collect();
                    out.push(Region {
                        direction,
                        polygon,
                        label: label(direction),
                    });
                }
            }
        }
    }
    out
}

/// Full, unchamfered faces used to paint the cube.
pub fn faces(rot: DQuat, center: Pos2, scale: f64) -> Vec<Region> {
    patches(rot, center, scale, false)
}

/// Surface patches provide corner/edge targets without changing the silhouette.
/// A corner has a patch on each visible adjacent face; all select the same view.
pub fn regions(rot: DQuat, center: Pos2, scale: f64) -> Vec<Region> {
    patches(rot, center, scale, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_face_edge_and_corner_is_clickable_from_its_view() {
        let mut counts = [0; 3];
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let d = DVec3::new(x as f64, y as f64, z as f64);
                    if d == DVec3::ZERO {
                        continue;
                    }
                    counts[d.abs().element_sum() as usize - 1] += 1;
                    let rot = rotation(d);
                    assert!((rot * DVec3::Z - d.normalize()).length() < 1e-10);
                    assert!((rot * DVec3::Y).is_finite());
                    let regions = regions(rot, Pos2::ZERO, 28.0);
                    let region = regions.iter().find(|r| r.direction == d).unwrap();
                    let picked: Vec<_> = regions
                        .iter()
                        .filter(|r| r.contains(region.center()))
                        .collect();
                    assert_eq!(picked.len(), 1, "{d:?}");
                    assert_eq!(picked[0].direction, d);
                }
            }
        }
        assert_eq!(counts, [6, 12, 8]);
    }

    #[test]
    fn sharp_face_corners_remain_inside_the_corner_click_targets() {
        let faces = faces(DQuat::IDENTITY, Pos2::ZERO, 28.0);
        assert_eq!(faces.len(), 1);
        assert_eq!(
            faces[0].polygon,
            vec![
                Pos2::new(-28.0, 28.0),
                Pos2::new(28.0, 28.0),
                Pos2::new(28.0, -28.0),
                Pos2::new(-28.0, -28.0)
            ]
        );
        let zones = regions(DQuat::IDENTITY, Pos2::ZERO, 28.0);
        let picked: Vec<_> = zones
            .iter()
            .filter(|r| r.contains(Pos2::new(27.0, 27.0)))
            .collect();
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].direction, DVec3::new(1.0, -1.0, 1.0));
    }

    #[test]
    fn face_center_and_outside_are_not_corner_targets() {
        let regions = regions(DQuat::IDENTITY, Pos2::ZERO, 28.0);
        let picked: Vec<_> = regions.iter().filter(|r| r.contains(Pos2::ZERO)).collect();
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].direction, DVec3::Z);
        assert!(!regions.iter().any(|r| r.contains(Pos2::new(50.0, 50.0))));
        assert!(!regions.iter().any(|r| r.direction.z < 0.0));
    }
}
#[cfg(test)]
mod highlight_tests {
    use super::*;
    #[test]
    fn corner_and_edge_targets_have_patches_on_every_visible_adjacent_face() {
        for (direction, count) in [
            (DVec3::new(1.0, 1.0, 1.0), 3),
            (DVec3::new(1.0, 1.0, 0.0), 2),
            (DVec3::Z, 1),
        ] {
            let regions = regions(rotation(direction), Pos2::ZERO, 28.0);
            let patches: Vec<_> = regions
                .iter()
                .filter(|r| r.direction == direction)
                .collect();
            assert_eq!(patches.len(), count);
            assert!(patches.iter().all(|r| regions
                .iter()
                .any(|v| v.contains(r.center()) && v.direction == direction)));
        }
    }
}
