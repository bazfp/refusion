//! A bounded mailbox for disposable derived work. Pending requests and results
//! each occupy one slot; domain callers reject stale results using tickets.
use std::sync::{Arc, Condvar, Mutex};
struct State<I, O> {
    input: Option<I>,
    output: Option<O>,
    closed: bool,
}
pub struct LatestWorker<I, O> {
    state: Arc<(Mutex<State<I, O>>, Condvar)>,
}
impl<I: Send + 'static, O: Send + 'static> LatestWorker<I, O> {
    pub fn new(
        name: &str,
        ctx: eframe::egui::Context,
        work: impl Fn(I) -> O + Send + 'static,
    ) -> Self {
        let state = Arc::new((
            Mutex::new(State {
                input: None,
                output: None,
                closed: false,
            }),
            Condvar::new(),
        ));
        let shared = state.clone();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || loop {
                let input = {
                    let (lock, wake) = &*shared;
                    let mut s = lock.lock().unwrap();
                    while s.input.is_none() && !s.closed {
                        s = wake.wait(s).unwrap();
                    }
                    if s.closed {
                        return;
                    }
                    s.input.take().unwrap()
                };
                let output = work(input);
                shared.0.lock().unwrap().output = Some(output);
                ctx.request_repaint();
            })
            .expect("derived data worker");
        Self { state }
    }
    pub fn submit(&self, input: I) {
        self.state.0.lock().unwrap().input = Some(input);
        self.state.1.notify_one();
    }
    pub fn poll(&self) -> Option<O> {
        self.state.0.lock().unwrap().output.take()
    }
}
impl<I, O> Drop for LatestWorker<I, O> {
    fn drop(&mut self) {
        let mut s = self.state.0.lock().unwrap();
        s.closed = true;
        s.input = None;
        self.state.1.notify_one();
    }
}
