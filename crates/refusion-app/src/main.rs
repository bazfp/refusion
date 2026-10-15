//! refusion — a native Linux parametric CAD modeller.
//!
//! UI: egui (immediate mode, GPU-drawn — no Qt). Viewport: wgpu (Vulkan on
//! Linux). Sketch solver, profiles, timeline: Rust. B-rep kernel: OpenCASCADE.

mod app;
mod camera;
mod compute;
mod dim_draw;
mod file_load;
mod gpu;
mod latest_worker;
mod measure;
mod mesh_lod;
mod perf;
mod scene;
mod shell_worker;
mod sketch_derived;
mod sketch_mode;
mod tool_icons;
mod view_cube;

use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
use eframe::wgpu;

fn main() -> eframe::Result {
    if std::env::args_os().nth(1).as_deref() == Some(std::ffi::OsStr::new(shell_worker::ARG)) {
        shell_worker::entry();
        return Ok(());
    }
    refusion_model::kernel::set_shell_executor(shell_worker::shell)
        .expect("install Shell process isolation before starting workers");
    let open = std::env::args().nth(1).map(std::path::PathBuf::from);
    let wgpu_setup = WgpuSetup::CreateNew(WgpuSetupCreateNew {
        // Prefer Vulkan and the discrete GPU.
        instance_descriptor: wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        },
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    });
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 920.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("refusion")
            .with_app_id("refusion"),
        wgpu_options: WgpuConfiguration {
            wgpu_setup,
            ..Default::default()
        },
        vsync: true,
        ..Default::default()
    };
    eframe::run_native(
        "refusion",
        options,
        Box::new(move |cc| Ok(Box::new(app::RefusionApp::new(cc, open)))),
    )
}
