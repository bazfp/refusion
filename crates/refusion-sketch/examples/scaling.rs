//! Repeatable local CPU scaling probe: cargo run -p refusion-sketch --example scaling.
use refusion_sketch::{
    find_profiles,
    solver::{Eq, Pt, System},
    Sketch,
};
use std::time::Instant;

fn elapsed(mut work: impl FnMut()) -> f64 {
    work(); // warm-up
    let mut samples = [0.; 3];
    for sample in &mut samples {
        let start = Instant::now();
        work();
        *sample = start.elapsed().as_secs_f64() * 1000.;
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}
fn main() {
    for count in [128, 256, 512] {
        let mut linear = System::default();
        for i in 0..count {
            let p = linear.add_param(i as f64);
            if i > 0 {
                linear.eqs.push(Eq::Lin {
                    a: p,
                    b: p - 1,
                    k: 1.,
                });
            }
        }
        let ms = elapsed(|| assert_eq!(linear.analyze().dof, 1));
        println!("analysis linear vars={count} median_ms={ms:.3}");
        let mut nonlinear = System::default();
        for i in 0..count / 2 {
            nonlinear.add_param(i as f64);
            nonlinear.add_param(0.);
            if i > 0 {
                nonlinear.eqs.push(Eq::Dist {
                    p: Pt(2 * i - 2, 2 * i - 1),
                    q: Pt(2 * i, 2 * i + 1),
                    d: 1.,
                });
            }
        }
        let ms = elapsed(|| assert_eq!(nonlinear.analyze().dof, count - (count / 2 - 1)));
        println!("analysis nonlinear vars={count} median_ms={ms:.3}");
    }
    for count in [250, 500, 1000, 2000] {
        let mut sk = Sketch::new();
        for i in 0..count {
            sk.add_line_pts([3. * i as f64, 0.], [3. * i as f64 + 1., 1.]);
        }
        let ms = elapsed(|| assert!(find_profiles(&sk).is_empty()));
        println!("profiles disjoint curves={count} median_ms={ms:.3}");
    }
    for count in [10, 20, 40] {
        let mut sk = Sketch::new();
        let end = (count - 1) as f64;
        for i in 0..count {
            sk.add_line_pts([i as f64, 0.], [i as f64, end]);
            sk.add_line_pts([0., i as f64], [end, i as f64]);
        }
        let ms = elapsed(|| assert_eq!(find_profiles(&sk).len(), (count - 1) * (count - 1)));
        println!("profiles crossing curves={} median_ms={ms:.3}", 2 * count);
    }
}
