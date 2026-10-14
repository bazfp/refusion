//! Original, resolution-independent CAD tool artwork. Shared by egui and SVG export.
use eframe::egui::{self, Color32, Pos2, Rect, Stroke};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Icon {
    Sketch,
    Extrude,
    Revolve,
    Hole,
    PressPull,
    Fillet,
    Chamfer,
    Shell,
    Mirror,
    RectPattern,
    CircularPattern,
    OffsetPlane,
    AngledPlane,
    Measure,
    Insert,
    Select,
    WindowSelect,
    Params,
    Finish,
    Line,
    Rectangle,
    CenterRectangle,
    Circle,
    Arc,
    ThreePointArc,
    TangentArc,
    Point,
    Dimension,
    Project,
    Trim,
    Extend,
    Offset,
    HorizontalVertical,
    Coincident,
    Tangent,
    Equal,
    Parallel,
    Perpendicular,
    Fix,
    Midpoint,
    Concentric,
    Collinear,
    Symmetry,
}
impl Icon {
    #[cfg(test)]
    pub const ALL: [Self; 43] = [
        Self::Sketch,
        Self::Extrude,
        Self::Revolve,
        Self::Hole,
        Self::PressPull,
        Self::Fillet,
        Self::Chamfer,
        Self::Shell,
        Self::Mirror,
        Self::RectPattern,
        Self::CircularPattern,
        Self::OffsetPlane,
        Self::AngledPlane,
        Self::Measure,
        Self::Insert,
        Self::Select,
        Self::WindowSelect,
        Self::Params,
        Self::Finish,
        Self::Line,
        Self::Rectangle,
        Self::CenterRectangle,
        Self::Circle,
        Self::Arc,
        Self::ThreePointArc,
        Self::TangentArc,
        Self::Point,
        Self::Dimension,
        Self::Project,
        Self::Trim,
        Self::Extend,
        Self::Offset,
        Self::HorizontalVertical,
        Self::Coincident,
        Self::Tangent,
        Self::Equal,
        Self::Parallel,
        Self::Perpendicular,
        Self::Fix,
        Self::Midpoint,
        Self::Concentric,
        Self::Collinear,
        Self::Symmetry,
    ];
    #[cfg(test)]
    pub fn name(self) -> String {
        format!("{self:?}")
    }
}
const INK: Color32 = Color32::from_rgb(68, 88, 112);
const BLUE: Color32 = Color32::from_rgb(89, 167, 225);
const LIGHT: Color32 = Color32::from_rgb(199, 219, 234);
const SIDE: Color32 = Color32::from_rgb(119, 155, 184);
const ARROW_EDGE: Color32 = Color32::from_rgb(27, 43, 61);
const CAVITY: Color32 = Color32::from_rgb(37, 65, 88);
const WHITE: Color32 = Color32::from_rgb(247, 250, 253);
const GOLD: Color32 = Color32::from_rgb(240, 183, 74);
const RED: Color32 = Color32::from_rgb(239, 135, 104);
const GREEN: Color32 = Color32::from_rgb(97, 204, 160);
#[derive(Clone)]
struct Path {
    points: Vec<[f32; 2]>,
    fill: Option<Color32>,
    stroke: Color32,
    width: f32,
    closed: bool,
}
struct Art(Vec<Path>);
impl Art {
    fn path(
        &mut self,
        points: &[[f32; 2]],
        fill: Option<Color32>,
        stroke: Color32,
        width: f32,
        closed: bool,
    ) {
        self.0.push(Path {
            points: points.to_vec(),
            fill,
            stroke,
            width,
            closed,
        });
    }
    fn line(&mut self, points: &[[f32; 2]], color: Color32) {
        self.path(points, None, color, 1.5, false);
    }
    fn polygon(&mut self, points: &[[f32; 2]], fill: Color32) {
        self.path(points, Some(fill), INK, 1., true);
    }
    fn arc(
        &mut self,
        c: [f32; 2],
        r: [f32; 2],
        start: f32,
        end: f32,
        color: Color32,
        fill: Option<Color32>,
    ) {
        let points: Vec<_> = (0..=40)
            .map(|i| {
                let t = start + (end - start) * i as f32 / 40.;
                [c[0] + r[0] * t.cos(), c[1] + r[1] * t.sin()]
            })
            .collect();
        self.path(
            &points,
            fill,
            color,
            1.5,
            (end - start).abs() >= std::f32::consts::TAU - 0.01,
        );
    }
    fn circle(&mut self, c: [f32; 2], r: f32, color: Color32) {
        self.arc(c, [r, r], 0., std::f32::consts::TAU, color, None);
    }
    fn dot(&mut self, p: [f32; 2], color: Color32) {
        self.arc(
            p,
            [1.25, 1.25],
            0.,
            std::f32::consts::TAU,
            color,
            Some(color),
        );
    }
    /// All direction/dimension arrows use one palette and keyline treatment.
    fn arrow_path(&mut self, points: &[[f32; 2]], closed: bool) {
        self.path(
            points,
            if closed { Some(ARROW_EDGE) } else { None },
            ARROW_EDGE,
            3.2,
            closed,
        );
        self.path(
            points,
            if closed { Some(WHITE) } else { None },
            WHITE,
            if closed { 0.8 } else { 1.5 },
            closed,
        );
    }
    fn arrow_head(&mut self, from: [f32; 2], tip: [f32; 2]) {
        let (dx, dy) = (tip[0] - from[0], tip[1] - from[1]);
        let l = (dx * dx + dy * dy).sqrt();
        let (x, y) = (dx / l, dy / l);
        self.arrow_path(
            &[
                [tip[0] - 3. * x + 1.7 * y, tip[1] - 3. * y - 1.7 * x],
                tip,
                [tip[0] - 3. * x - 1.7 * y, tip[1] - 3. * y + 1.7 * x],
            ],
            true,
        );
    }
    fn arrow(&mut self, from: [f32; 2], tip: [f32; 2]) {
        self.arrow_path(&[from, tip], false);
        self.arrow_head(from, tip);
    }
    fn curved_arrow(&mut self, c: [f32; 2], r: [f32; 2], start: f32, end: f32) {
        let mut points: Vec<_> = (0..=32)
            .map(|i| {
                let t = start + (end - start) * i as f32 / 32.;
                [c[0] + r[0] * t.cos(), c[1] + r[1] * t.sin()]
            })
            .collect();
        let tip = *points.last().unwrap();
        // Reserve the last three units for the head, then join its shoulders
        // directly to the shaft outline. One fill/keyline avoids a dark seam.
        let mut remaining = 3.;
        while points.len() > 2 {
            let end = points.pop().unwrap();
            let previous = *points.last().unwrap();
            let length = ((end[0] - previous[0]).powi(2) + (end[1] - previous[1]).powi(2)).sqrt();
            if length >= remaining {
                points.push([
                    end[0] + (previous[0] - end[0]) * remaining / length,
                    end[1] + (previous[1] - end[1]) * remaining / length,
                ]);
                break;
            }
            remaining -= length;
        }
        let base = *points.last().unwrap();
        let length = ((tip[0] - base[0]).powi(2) + (tip[1] - base[1]).powi(2)).sqrt();
        let head_normal = [-(tip[1] - base[1]) / length, (tip[0] - base[0]) / length];
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (i, point) in points.iter().enumerate() {
            let previous = points[i.saturating_sub(1)];
            let next = points.get(i + 1).copied().unwrap_or(tip);
            let length = ((next[0] - previous[0]).powi(2) + (next[1] - previous[1]).powi(2)).sqrt();
            let normal = if i + 1 == points.len() {
                head_normal
            } else {
                [
                    -(next[1] - previous[1]) / length,
                    (next[0] - previous[0]) / length,
                ]
            };
            left.push([point[0] + 0.75 * normal[0], point[1] + 0.75 * normal[1]]);
            right.push([point[0] - 0.75 * normal[0], point[1] - 0.75 * normal[1]]);
        }
        left.extend([
            [
                base[0] + 1.7 * head_normal[0],
                base[1] + 1.7 * head_normal[1],
            ],
            tip,
            [
                base[0] - 1.7 * head_normal[0],
                base[1] - 1.7 * head_normal[1],
            ],
        ]);
        left.extend(right.into_iter().rev());
        self.path(&left, Some(ARROW_EDGE), ARROW_EDGE, 1.7, true);
        self.path(&left, Some(WHITE), WHITE, 0., true);
    }
    /// A bore's visible inner wall is the crescent between equal circles
    /// displaced along the block's projected depth axis (+X, -Y).
    fn bore(&mut self, c: [f32; 2], radius: f32, depth: [f32; 2]) {
        self.arc(
            c,
            [radius, radius],
            0.,
            std::f32::consts::TAU,
            INK,
            Some(CAVITY),
        );
        let d = (depth[0] * depth[0] + depth[1] * depth[1]).sqrt();
        let theta = depth[1].atan2(depth[0]);
        let alpha = (d / (2. * radius)).acos();
        let back = [c[0] + depth[0], c[1] + depth[1]];
        let circle = |origin: [f32; 2], start: f32, end: f32| -> Vec<[f32; 2]> {
            (0..=24)
                .map(|i| {
                    let t = start + (end - start) * i as f32 / 24.;
                    [origin[0] + radius * t.cos(), origin[1] + radius * t.sin()]
                })
                .collect()
        };
        let mut wall = circle(c, theta + alpha, theta - alpha + std::f32::consts::TAU);
        let inner = circle(
            back,
            theta + std::f32::consts::PI + alpha,
            theta + std::f32::consts::PI - alpha,
        );
        wall.extend(inner.iter().copied());
        self.path(&wall, Some(BLUE), Color32::TRANSPARENT, 0., true);
        self.line(&inner, INK);
        self.circle(c, radius, INK);
    }
    fn front_block(&mut self) {
        self.polygon(&[[3., 7.], [9., 3.], [22., 3.], [16., 7.]], LIGHT);
        self.polygon(&[[16., 7.], [22., 3.], [22., 17.], [16., 21.]], SIDE);
        self.polygon(&[[3., 7.], [16., 7.], [16., 21.], [3., 21.]], LIGHT);
    }
    fn cube(&mut self) {
        self.polygon(&[[4., 7.], [12., 3.], [20., 7.], [12., 11.]], LIGHT);
        self.polygon(&[[4., 7.], [12., 11.], [12., 21.], [4., 17.]], SIDE);
        self.polygon(&[[12., 11.], [20., 7.], [20., 17.], [12., 21.]], BLUE);
    }
}
fn art(icon: Icon) -> Art {
    use Icon::*;
    let mut a = Art(vec![]);
    let tau = std::f32::consts::TAU;
    match icon {
        Sketch => {
            a.polygon(&[[3., 8.], [14., 4.], [21., 14.], [10., 19.]], LIGHT);
            a.polygon(&[[6., 17.], [16., 5.], [19., 8.], [9., 20.]], GOLD);
            a.line(&[[6., 17.], [5., 21.], [9., 20.]], INK);
        }
        Extrude | PressPull => {
            a.cube();
            for path in &mut a.0 {
                for point in &mut path.points {
                    point[0] -= 2.;
                }
            }
            // The arrow is in its own column, clear of the solid's pale top face.
            a.arrow([21., 20.], [21., 3.]);
            if icon == PressPull {
                a.arrow([21., 4.], [21., 21.]);
            }
        }
        Hole => {
            a.front_block();
            a.bore([9.5, 14.], 4.7, [3., -2.]);
        }
        Revolve => {
            // A three-quarter cylinder: broad faces and one open radial cut.
            // Keep the rotation arrow clear of the solid at toolbar sizes.
            let arc = |start: f32, end: f32| -> Vec<[f32; 2]> {
                (0..=32)
                    .map(|i| {
                        let t = start + (end - start) * i as f32 / 32.;
                        [11. + 8. * t.cos(), 10. + 3.5 * t.sin()]
                    })
                    .collect()
            };
            let front = arc(1.27, std::f32::consts::PI);
            let mut wall = front.clone();
            wall.extend(front.iter().rev().map(|p| [p[0], p[1] + 8.]));
            a.polygon(&wall, BLUE);
            a.polygon(
                &[[11., 10.], [18.84, 10.7], [18.84, 18.7], [11., 18.]],
                SIDE,
            );
            a.polygon(
                &[[11., 10.], [13.37, 13.34], [13.37, 21.34], [11., 18.]],
                LIGHT,
            );
            let mut cap = arc(1.27, std::f32::consts::TAU + 0.2);
            cap.push([11., 10.]);
            a.polygon(&cap, LIGHT);
            a.curved_arrow(
                [11., 6.5],
                [8., 4.],
                std::f32::consts::PI + 0.15,
                std::f32::consts::TAU - 0.15,
            );
        }
        Fillet | Chamfer => {
            // An entire modified edge runs from the front profile to the back.
            // The large round/bevel silhouette remains distinct even at 16 px.
            let curve: Vec<[f32; 2]> = if icon == Fillet {
                (0..=20)
                    .map(|i| {
                        let t = -std::f32::consts::FRAC_PI_2
                            + i as f32 / 20. * std::f32::consts::FRAC_PI_2;
                        [9. + 6. * t.cos(), 14. + 6. * t.sin()]
                    })
                    .collect()
            } else {
                vec![[9., 8.], [15., 14.]]
            };
            a.polygon(&[[3., 8.], [9., 8.], [15., 4.], [9., 4.]], LIGHT);
            a.polygon(&[[15., 14.], [21., 10.], [21., 17.], [15., 21.]], SIDE);
            let mut front = vec![[3., 8.]];
            front.extend(curve.iter().copied());
            front.extend([[15., 21.], [3., 21.]]);
            a.polygon(&front, LIGHT);
            let mut surface = curve.clone();
            surface.extend(curve.iter().rev().map(|p| [p[0] + 6., p[1] - 4.]));
            a.polygon(&surface, BLUE);
            if icon == Fillet {
                let mut lit = curve[..=8].to_vec();
                lit.extend(curve[..=8].iter().rev().map(|p| [p[0] + 6., p[1] - 4.]));
                a.path(
                    &lit,
                    Some(Color32::from_rgb(142, 207, 249)),
                    Color32::TRANSPARENT,
                    0.,
                    true,
                );
                a.line(&curve, INK);
                let far: Vec<_> = curve.iter().map(|p| [p[0] + 6., p[1] - 4.]).collect();
                a.line(&far, INK);
            }
        }
        Shell => {
            a.front_block();
            // Both interior walls converge along the same (+6,-4) depth axis
            // as the exterior. The back rectangle is clipped by the front rim.
            a.polygon(&[[5., 9.], [14., 9.], [14., 19.], [5., 19.]], CAVITY);
            a.path(
                &[[5., 9.], [9.5, 9.], [9.5, 16.], [5., 19.]],
                Some(SIDE),
                Color32::TRANSPARENT,
                0.,
                true,
            );
            a.path(
                &[[5., 19.], [9.5, 16.], [14., 16.], [14., 19.]],
                Some(BLUE),
                Color32::TRANSPARENT,
                0.,
                true,
            );
            a.line(&[[9.5, 9.], [9.5, 16.], [14., 16.]], INK);
            a.path(
                &[[5., 9.], [14., 9.], [14., 19.], [5., 19.]],
                None,
                INK,
                1.,
                true,
            );
        }
        Mirror => {
            a.line(&[[12., 2.], [12., 22.]], GOLD);
            a.polygon(&[[3., 9.], [9., 5.], [9., 18.], [3., 15.]], BLUE);
            a.polygon(&[[15., 5.], [21., 9.], [21., 15.], [15., 18.]], LIGHT);
        }
        RectPattern => {
            for y in [4., 13.] {
                for x in [3., 12.] {
                    a.polygon(
                        &[[x, y + 2.], [x + 4., y], [x + 8., y + 2.], [x + 4., y + 4.]],
                        LIGHT,
                    );
                    a.polygon(
                        &[
                            [x, y + 2.],
                            [x + 4., y + 4.],
                            [x + 8., y + 2.],
                            [x + 8., y + 7.],
                            [x + 4., y + 9.],
                            [x, y + 7.],
                        ],
                        BLUE,
                    );
                }
            }
        }
        CircularPattern => {
            a.arc([12., 12.], [7., 7.], 0., tau, GOLD, None);
            for t in [0., 2.094, 4.189] {
                let c = [12. + 7. * f32::cos(t), 12. + 7. * f32::sin(t)];
                a.polygon(
                    &[
                        [c[0] - 3., c[1] - 2.],
                        [c[0] + 1., c[1] - 3.],
                        [c[0] + 3., c[1] + 1.],
                        [c[0] - 1., c[1] + 3.],
                    ],
                    BLUE,
                );
            }
        }
        OffsetPlane | AngledPlane => {
            a.polygon(&[[3., 16.], [13., 11.], [22., 15.], [12., 20.]], SIDE);
            let p = if icon == OffsetPlane {
                vec![[3., 7.], [13., 2.], [22., 6.], [12., 11.]]
            } else {
                vec![[4., 15.], [11., 2.], [21., 8.], [14., 20.]]
            };
            a.polygon(&p, GOLD);
            if icon == OffsetPlane {
                a.arrow([4., 14.], [4., 6.]);
            }
        }
        Measure => {
            a.polygon(&[[3., 16.], [17., 4.], [22., 10.], [8., 22.]], LIGHT);
            for i in 0..5 {
                let x = 6. + i as f32 * 2.7;
                let y = 15. - i as f32 * 2.3;
                a.line(&[[x, y], [x + 2., y + 2.]], INK);
            }
        }
        Insert => {
            a.cube();
            a.arrow([2., 2.], [10., 11.]);
        }
        Select | WindowSelect => {
            if icon == WindowSelect {
                a.line(&[[2., 10.], [2., 3.], [16., 3.], [16., 10.]], BLUE);
                a.line(&[[2., 14.], [2., 19.], [7., 19.]], BLUE);
            }
            a.path(
                &[
                    [9., 5.],
                    [9., 21.],
                    [13., 17.],
                    [16., 22.],
                    [19., 20.],
                    [16., 15.],
                    [22., 14.],
                ],
                Some(WHITE),
                ARROW_EDGE,
                1.,
                true,
            );
        }
        Params => {
            a.line(&[[4., 7.], [20., 7.]], INK);
            a.line(&[[4., 16.], [20., 16.]], INK);
            a.polygon(&[[8., 4.], [12., 4.], [12., 10.], [8., 10.]], GOLD);
            a.polygon(&[[15., 13.], [19., 13.], [19., 19.], [15., 19.]], BLUE);
        }
        Finish => {
            a.circle([12., 12.], 9., GREEN);
            a.line(&[[6., 12.], [10., 16.], [18., 7.]], GREEN);
        }
        Line => {
            a.line(&[[4., 19.], [20., 5.]], BLUE);
            a.dot([4., 19.], GOLD);
            a.dot([20., 5.], GOLD);
        }
        Rectangle | CenterRectangle => {
            a.line(
                &[[3., 6.], [21., 6.], [21., 19.], [3., 19.], [3., 6.]],
                BLUE,
            );
            a.dot([3., 6.], GOLD);
            a.dot([21., 19.], GOLD);
            if icon == CenterRectangle {
                a.line(&[[6., 8.], [18., 17.]], SIDE);
                a.line(&[[6., 17.], [18., 8.]], SIDE);
                a.dot([12., 12.5], GOLD);
            }
        }
        Circle => {
            a.circle([12., 12.], 8., BLUE);
            a.dot([12., 12.], GOLD);
            a.line(&[[12., 12.], [19., 8.]], SIDE);
        }
        Arc | ThreePointArc | TangentArc => {
            a.arc([12., 14.], [9., 9.], -2.9, -0.25, BLUE, None);
            a.dot([3.3, 11.8], GOLD);
            a.dot([20.7, 11.8], GOLD);
            if icon == Arc {
                a.dot([12., 14.], GOLD);
                a.line(&[[12., 14.], [19., 8.]], SIDE);
            } else if icon == ThreePointArc {
                a.dot([12., 5.], GOLD);
            } else {
                a.line(&[[2., 21.], [3.3, 11.8]], SIDE);
            }
        }
        Point => {
            a.line(&[[4., 12.], [20., 12.]], SIDE);
            a.line(&[[12., 4.], [12., 20.]], SIDE);
            a.dot([12., 12.], GOLD);
            a.circle([12., 12.], 3., BLUE);
        }
        Dimension => {
            a.line(&[[4., 4.], [4., 20.]], SIDE);
            a.line(&[[20., 4.], [20., 20.]], SIDE);
            a.arrow([12., 12.], [4., 12.]);
            a.arrow([12., 12.], [20., 12.]);
        }
        Project => {
            a.polygon(&[[3., 17.], [13., 12.], [21., 16.], [11., 21.]], LIGHT);
            a.line(&[[4., 3.], [19., 3.], [19., 9.], [4., 9.], [4., 3.]], BLUE);
            a.arrow([11., 8.], [11., 17.]);
        }
        Trim => {
            // Two balanced finger loops, crossed blades and a single hinge.
            a.line(&[[2., 7.], [7., 7.]], BLUE);
            a.line(&[[15., 7.], [22., 7.]], BLUE);
            a.polygon(&[[9., 14.], [3.3, 3.5], [6.5, 10.], [11., 13.]], LIGHT);
            a.polygon(&[[9., 13.], [17.8, 3.5], [15.9, 10.], [11., 14.]], LIGHT);
            a.line(&[[6.8, 15.2], [10., 13.], [13.2, 15.2]], INK);
            a.circle([6., 18.], 3., INK);
            a.circle([14., 18.], 3., INK);
            a.arc([10., 13.], [1., 1.], 0., tau, INK, Some(INK));
        }
        Extend => {
            a.line(&[[3., 17.], [12., 9.]], BLUE);
            a.line(&[[12., 9.], [21., 2.]], GOLD);
            a.line(&[[21., 2.], [21., 21.]], SIDE);
            a.arrow([10., 16.], [18., 9.]);
        }
        Offset => {
            a.arc([12., 16.], [9., 11.], -2.8, -0.3, BLUE, None);
            a.arc([12., 16.], [5., 6.], -2.8, -0.3, GOLD, None);
            a.arrow([12., 10.], [12., 5.]);
        }
        HorizontalVertical => {
            a.line(&[[3., 18.], [3., 5.], [21., 5.]], BLUE);
            a.line(&[[7., 15.], [7., 9.], [14., 9.]], RED);
        }
        Coincident => {
            a.line(&[[3., 19.], [18., 4.]], BLUE);
            a.circle([12., 10.], 4., RED);
            a.dot([12., 10.], RED);
        }
        Tangent => {
            a.circle([12., 9.], 6., BLUE);
            a.line(&[[3., 15.], [21., 15.]], RED);
            a.dot([12., 15.], RED);
        }
        Equal => {
            a.line(&[[3., 4.], [20., 4.]], BLUE);
            a.line(&[[3., 20.], [20., 20.]], BLUE);
            a.line(&[[7., 10.], [17., 10.]], RED);
            a.line(&[[7., 14.], [17., 14.]], RED);
        }
        Parallel => {
            a.line(&[[4., 19.], [13., 4.]], RED);
            a.line(&[[12., 20.], [21., 5.]], RED);
        }
        Perpendicular => {
            a.line(&[[4., 4.], [4., 20.], [21., 20.]], BLUE);
            a.line(&[[4., 15.], [9., 15.], [9., 20.]], RED);
        }
        Fix => {
            a.arc([12., 9.], [5., 6.], std::f32::consts::PI, tau, RED, None);
            a.polygon(&[[5., 9.], [19., 9.], [19., 21.], [5., 21.]], LIGHT);
            a.dot([12., 14.], RED);
            a.line(&[[12., 14.], [12., 18.]], RED);
        }
        Midpoint => {
            a.line(&[[2., 17.], [22., 17.]], BLUE);
            a.path(&[[7., 15.], [12., 6.], [17., 15.]], None, RED, 1.5, true);
            a.dot([12., 17.], GOLD);
        }
        Concentric => {
            a.circle([12., 12.], 9., BLUE);
            a.circle([12., 12.], 5., RED);
            a.dot([12., 12.], GOLD);
        }
        Collinear => {
            a.line(&[[2., 19.], [9., 12.]], BLUE);
            a.line(&[[15., 6.], [22., -1. + 4.]], BLUE);
            a.line(&[[10., 11.], [14., 7.]], RED);
        }
        Symmetry => {
            a.line(&[[12., 2.], [12., 22.]], RED);
            a.path(
                &[[3., 8.], [9., 5.], [9., 19.], [3., 16.]],
                None,
                BLUE,
                1.5,
                true,
            );
            a.path(
                &[[15., 5.], [21., 8.], [21., 16.], [15., 19.]],
                None,
                BLUE,
                1.5,
                true,
            );
        }
    }
    a
}
pub fn paint(painter: &egui::Painter, rect: Rect, icon: Icon) {
    paint_with_theme(painter, rect, icon, painter.ctx().style().visuals.dark_mode);
}

/// Triangulate concave artwork, optionally feathering an unoutlined white fill.
/// A centered one-pixel fringe smooths the arrow without widening its shaft.
fn filled_mesh(points: &[Pos2], fill: Color32, feather: f32) -> egui::Mesh {
    let mut mesh = egui::Mesh::default();
    let mut inner = points.to_vec();
    let mut outer = Vec::new();
    if feather > 0. {
        let area: f32 = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .take(points.len())
            .map(|(a, b)| a.x * b.y - b.x * a.y)
            .sum();
        let winding = area.signum();
        for (i, point) in points.iter().enumerate() {
            let before = (*point - points[(i + points.len() - 1) % points.len()]).normalized();
            let after = (points[(i + 1) % points.len()] - *point).normalized();
            let n0 = egui::vec2(before.y, -before.x) * winding;
            let n1 = egui::vec2(after.y, -after.x) * winding;
            let normal = (n0 + n1) / (1. + n0.dot(n1)).max(0.5);
            let offset = normal * (feather * 0.5);
            inner[i] = *point - offset;
            outer.push(*point + offset);
        }
    }
    let coords: Vec<f64> = inner
        .iter()
        .flat_map(|p| [p.x as f64, p.y as f64])
        .collect();
    let indices = earcutr::earcut(&coords, &[], 2).expect("valid tool icon polygon");
    for (i, &point) in inner.iter().enumerate() {
        mesh.colored_vertex(point, fill);
        if feather > 0. {
            mesh.colored_vertex(outer[i], Color32::TRANSPARENT);
        }
    }
    let stride = if feather > 0. { 2 } else { 1 };
    mesh.indices = indices.into_iter().map(|i| i as u32 * stride).collect();
    if feather > 0. {
        for i in 0..points.len() as u32 {
            let next = (i + 1) % points.len() as u32;
            mesh.add_triangle(i * 2, next * 2, i * 2 + 1);
            mesh.add_triangle(i * 2 + 1, next * 2, next * 2 + 1);
        }
    }
    mesh
}

pub fn paint_with_theme(painter: &egui::Painter, rect: Rect, icon: Icon, dark: bool) {
    let scale = rect.width().min(rect.height()) / 24.;
    let offset = rect.center() - egui::vec2(12. * scale, 12. * scale);
    for mut path in art(icon).0 {
        if dark && matches!(icon, Icon::Trim) && path.fill.is_none() && path.stroke == INK {
            path.stroke = WHITE;
        }

        let points: Vec<Pos2> = path
            .points
            .iter()
            .map(|p| offset + egui::vec2(p[0] * scale, p[1] * scale))
            .collect();
        let physical_min = 1. / painter.ctx().pixels_per_point();
        let stroke = Stroke::new(
            if path.width == 0. {
                0.
            } else {
                (path.width * scale).max(physical_min)
            },
            path.stroke,
        );
        if let Some(fill) = path.fill {
            // Stroked artwork already has a feathered outline. The joined
            // arrow's white foreground has no stroke to hide hard mesh edges.
            let feather = if path.width == 0. && fill == WHITE {
                physical_min
            } else {
                0.
            };
            painter.add(filled_mesh(&points, fill, feather));
        }
        painter.add(egui::epaint::PathShape {
            points,
            closed: path.closed,
            fill: Color32::TRANSPARENT,
            stroke: stroke.into(),
        });
    }
}
#[cfg(test)]
fn svg(icon: Icon) -> String {
    let mut svg =
        String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 24 24\">\n");
    let color = |c: Color32| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b());
    for p in art(icon).0 {
        let points = p
            .points
            .iter()
            .map(|p| format!("{},{}", p[0], p[1]))
            .collect::<Vec<_>>()
            .join(" ");
        svg.push_str(&format!(
            "<{} points=\"{}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{}\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>\n",
            if p.closed { "polygon" } else { "polyline" },
            points,
            p.fill.map(color).unwrap_or_else(|| "none".into()),
            color(p.stroke),
            p.width
        ));
    }
    svg.push_str("</svg>\n");
    svg
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tool_artwork_is_finite_distinct_and_exports_from_the_render_geometry() {
        let mut seen = std::collections::HashSet::new();
        for icon in Icon::ALL {
            let art = art(icon);
            assert!(!art.0.is_empty());
            assert!(art
                .0
                .iter()
                .all(|p| p.points.len() > 1 && p.points.iter().flatten().all(|v| v.is_finite())));
            for path in &art.0 {
                if path.fill.is_some() {
                    let coords: Vec<f64> =
                        path.points.iter().flatten().map(|&v| v as f64).collect();
                    assert!(
                        !earcutr::earcut(&coords, &[], 2).unwrap().is_empty(),
                        "empty fill {icon:?}"
                    );
                }
            }
            let svg = svg(icon);
            assert!(seen.insert(svg.clone()), "duplicate {icon:?}");
            if let Some(dir) = std::env::var_os("REFUSION_ICON_ARTIFACTS") {
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("{}.svg", icon.name())),
                    &svg,
                )
                .unwrap();
            }
            let saved = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets/tool-icons")
                .join(format!("{}.svg", icon.name()));
            assert_eq!(
                std::fs::read_to_string(saved).unwrap(),
                svg,
                "SVG drift for {icon:?}; regenerate assets"
            );
        }
    }
}
