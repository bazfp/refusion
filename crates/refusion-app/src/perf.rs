//! Opt-in viewport benchmark; ordinary sessions collect no samples.
use std::time::Instant;
pub struct ViewportProfiler {
    warmup: usize,
    samples: Vec<f64>,
    intervals: Vec<f64>,
    previous: Option<Instant>,
}
impl ViewportProfiler {
    pub fn from_env() -> Option<Self> {
        std::env::var_os("REFUSION_PERF").map(|_| Self {
            warmup: 30,
            samples: Vec::with_capacity(180),
            intervals: Vec::with_capacity(180),
            previous: None,
        })
    }
    pub fn record(&mut self, start: Instant, bodies: usize, picks: usize) {
        if bodies == 0 {
            return;
        }
        let interval = self
            .previous
            .map(|p| start.duration_since(p).as_secs_f64() * 1000.);
        self.previous = Some(start);
        if self.warmup > 0 {
            self.warmup -= 1;
            return;
        }
        self.intervals.push(interval.unwrap_or(0.));
        self.samples.push(start.elapsed().as_secs_f64() * 1000.);
        if self.samples.len() == 180 {
            self.samples.sort_by(f64::total_cmp);
            self.intervals.sort_by(f64::total_cmp);
            eprintln!(
                "refusion-perf: bodies={bodies} picks={picks} viewport CPU median={:.3}ms p95={:.3}ms frame interval median={:.3}ms p95={:.3}ms",
                self.samples[90], self.samples[171], self.intervals[90], self.intervals[171]
            );
            self.samples.clear();
            self.intervals.clear();
        }
    }
}
