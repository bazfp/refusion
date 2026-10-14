use glam::{DMat4, DQuat, DVec2, DVec3, Mat4};

#[derive(Clone, Debug)]
pub struct Camera {
    pub target: DVec3,
    pub distance: f64,
    /// Rotation taking camera space (−Z forward, +Y up) to world space.
    pub rot: DQuat,
    pub fov_y: f64,
    pub ortho: bool,
    /// Animation: (from, to, t ∈ 0..1)
    anim: Option<(Pose, Pose, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct Pose {
    target: DVec3,
    distance: f64,
    rot: DQuat,
}

/// A ray in world space.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: DVec3,
    pub dir: DVec3,
}

impl Ray {
    /// Intersection parameter with a plane (point, normal).
    pub fn hit_plane(&self, p: DVec3, n: DVec3) -> Option<DVec3> {
        let d = self.dir.dot(n);
        if d.abs() < 1e-12 {
            return None;
        }
        let t = (p - self.origin).dot(n) / d;
        Some(self.origin + self.dir * t)
    }
}

impl Default for Camera {
    fn default() -> Self {
        let mut c = Camera {
            target: DVec3::ZERO,
            distance: 250.0,
            rot: DQuat::IDENTITY,
            fov_y: 35f64.to_radians(),
            ortho: true,
            anim: None,
        };
        c.rot = Self::iso_rot();
        c
    }
}

impl Camera {
    pub fn iso_rot() -> DQuat {
        DQuat::from_rotation_y(45f64.to_radians()) * DQuat::from_rotation_x(-30f64.to_radians())
    }

    pub fn eye(&self) -> DVec3 {
        self.target + self.rot * DVec3::Z * self.distance
    }
    pub fn forward(&self) -> DVec3 {
        self.rot * -DVec3::Z
    }
    pub fn right(&self) -> DVec3 {
        self.rot * DVec3::X
    }
    pub fn up(&self) -> DVec3 {
        self.rot * DVec3::Y
    }

    pub fn view(&self) -> DMat4 {
        DMat4::look_at_rh(self.eye(), self.target, self.up())
    }

    pub fn proj(&self, aspect: f64) -> DMat4 {
        let near = (self.distance * 0.002).max(1e-3);
        let far = self.distance * 200.0 + 1e4;
        if self.ortho {
            let h = self.distance * (self.fov_y * 0.5).tan();
            DMat4::orthographic_rh(-h * aspect, h * aspect, -h, h, -far, far)
        } else {
            DMat4::perspective_rh(self.fov_y, aspect, near, far)
        }
    }

    /// Depth bias for the orthographic projection: about 0.2 % of the view
    /// distance, expressed in NDC depth units (0 in perspective).
    pub fn ortho_bias(&self) -> f32 {
        if !self.ortho {
            return 0.0;
        }
        let far = self.distance * 200.0 + 1e4;
        (0.002 * self.distance / (2.0 * far)) as f32
    }

    pub fn view_proj(&self, aspect: f64) -> Mat4 {
        (self.proj(aspect) * self.view()).as_mat4()
    }

    /// Ray through a viewport pixel (pixel coordinates, origin top-left).
    pub fn ray(&self, px: DVec2, size: DVec2) -> Ray {
        let ndc = DVec2::new(2.0 * px.x / size.x - 1.0, 1.0 - 2.0 * px.y / size.y);
        let inv = (self.proj(size.x / size.y) * self.view()).inverse();
        let p0 = inv.project_point3(DVec3::new(ndc.x, ndc.y, 0.0));
        let p1 = inv.project_point3(DVec3::new(ndc.x, ndc.y, 1.0));
        if self.ortho {
            Ray {
                origin: p0 - (p1 - p0).normalize() * self.distance * 100.0,
                dir: (p1 - p0).normalize(),
            }
        } else {
            Ray {
                origin: self.eye(),
                dir: (p1 - self.eye()).normalize(),
            }
        }
    }

    /// World → pixel coordinates (None if behind the camera).
    pub fn project(&self, p: DVec3, size: DVec2) -> Option<DVec2> {
        let clip = (self.proj(size.x / size.y) * self.view()) * p.extend(1.0);
        if clip.w <= 1e-9 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(DVec2::new(
            (ndc.x + 1.0) * 0.5 * size.x,
            (1.0 - ndc.y) * 0.5 * size.y,
        ))
    }

    /// World units per pixel at the target depth.
    pub fn units_per_pixel(&self, viewport_h: f64) -> f64 {
        2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_h.max(1.0)
    }

    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.anim = None;
        let yaw = DQuat::from_rotation_y(-dx * 0.008);
        let pitch = DQuat::from_axis_angle(self.right(), -dy * 0.008);
        self.rot = (yaw * pitch * self.rot).normalize();
    }

    pub fn pan(&mut self, dx: f64, dy: f64, viewport_h: f64) {
        self.anim = None;
        let k = self.units_per_pixel(viewport_h);
        self.target += (-self.right() * dx + self.up() * dy) * k;
    }

    /// Zoom towards the point under the cursor.
    pub fn zoom(&mut self, steps: f64, cursor: DVec2, size: DVec2) {
        self.anim = None;
        let f = (-steps * 0.0015).exp().clamp(0.2, 5.0);
        let before = self
            .ray(cursor, size)
            .hit_plane(self.target, self.forward());
        self.distance = (self.distance * f).clamp(0.05, 1e6);
        if let Some(b) = before {
            let after = self
                .ray(cursor, size)
                .hit_plane(self.target, self.forward());
            if let Some(a) = after {
                self.target += b - a;
            }
        }
    }

    /// Animate to look straight at a plane (sketch mode), keeping the target on it.
    pub fn look_at_plane(&mut self, origin: DVec3, x: DVec3, y: DVec3) {
        let z = x.cross(y).normalize();
        let rot = DQuat::from_mat3(&glam::DMat3::from_cols(x.normalize(), y.normalize(), z));
        let n = z;
        let t = self.target - n * (self.target - origin).dot(n);
        self.animate_to(t, self.distance, rot);
    }

    pub fn set_view(&mut self, rot: DQuat) {
        self.animate_to(self.target, self.distance, rot);
    }

    pub fn fit(&mut self, lo: DVec3, hi: DVec3) {
        let c = (lo + hi) * 0.5;
        let r = ((hi - lo).length() * 0.5).max(1.0);
        let d = r / (self.fov_y * 0.5).sin() * 1.1;
        self.animate_to(c, d, self.rot);
    }

    fn animate_to(&mut self, target: DVec3, distance: f64, rot: DQuat) {
        let from = Pose {
            target: self.target,
            distance: self.distance,
            rot: self.rot,
        };
        self.anim = Some((
            from,
            Pose {
                target,
                distance,
                rot,
            },
            0.0,
        ));
    }

    /// Advance animation; returns true while animating.
    pub fn tick(&mut self, dt: f64) -> bool {
        let Some((a, b, t)) = self.anim.as_mut() else {
            return false;
        };
        *t = (*t + dt / 0.35).min(1.0);
        let s = *t * *t * (3.0 - 2.0 * *t);
        self.target = a.target.lerp(b.target, s);
        self.distance = a.distance + (b.distance - a.distance) * s;
        self.rot = a.rot.slerp(b.rot, s);
        if *t >= 1.0 {
            self.anim = None;
        }
        true
    }
}

pub fn named_view(name: &str) -> DQuat {
    use std::f64::consts::FRAC_PI_2;
    match name {
        "Front" => DQuat::IDENTITY,
        "Back" => DQuat::from_rotation_y(std::f64::consts::PI),
        "Right" => DQuat::from_rotation_y(FRAC_PI_2),
        "Left" => DQuat::from_rotation_y(-FRAC_PI_2),
        "Top" => DQuat::from_rotation_x(-FRAC_PI_2),
        "Bottom" => DQuat::from_rotation_x(FRAC_PI_2),
        _ => Camera::iso_rot(),
    }
}
