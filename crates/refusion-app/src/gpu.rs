//! wgpu renderer for the 3D viewport.
//!
//! Renders into an offscreen 4×MSAA target that egui composites as an image,
//! plus an R32Uint id buffer used for picking (only a 15×15 px window around
//! the cursor is rendered and read back, so picking costs well under a ms).

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;

pub const MSAA: u32 = 4;
const COLOR_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const ID_FMT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
pub const PICK_WIN: u32 = 15;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub viewport: [f32; 4],
    pub light: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub face: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BodyInstance {
    pub base: u32,
    pub color: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInst {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub color: u32,
    pub width: f32,
    pub id: u32,
    pub style: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointInst {
    pub pos: [f32; 3],
    pub color: u32,
    pub size: f32,
    pub id: u32,
    pub shape: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlatVertex {
    pub pos: [f32; 3],
    pub color: u32,
    pub id: u32,
}

pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16 | (a as u32) << 24
}

/// One body to draw: cached GPU mesh key + per-frame instance data.
pub struct BodyDraw<'a> {
    pub key: u64,
    pub mesh: &'a refusion_model::kernel::Mesh,
    pub base: u32,
    pub edge_base: u32,
    pub color: u32,
}

#[derive(Default)]
pub struct FrameData<'a> {
    pub bodies: Vec<BodyDraw<'a>>,
    pub lines: Vec<LineInst>,
    pub overlay_lines: Vec<LineInst>,
    pub points: Vec<PointInst>,
    pub flat_v: Vec<FlatVertex>,
    pub flat_i: Vec<u32>,
    /// The first `under_count` flat indices are origin planes: in the pick
    /// pass they are drawn before bodies so a body face always wins.
    pub under_count: u32,
    pub flags: Vec<u32>,
    pub view_proj: glam::Mat4,
    pub eye: glam::Vec3,
    pub light: glam::Vec3,
    pub clear: [f64; 3],
    /// Orthographic depth bias (NDC per unit k), 0 for perspective.
    pub ortho_bias: f32,
}

struct BodyGpu {
    edges: wgpu::Buffer,
    edge_count: u32,
    edge_offset: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
    inst: wgpu::Buffer,
    used: bool,
}

struct Targets {
    size: [u32; 2],
    color_msaa: wgpu::TextureView,
    color: wgpu::TextureView,
    #[cfg(test)]
    color_texture: wgpu::Texture,
    depth: wgpu::TextureView,
    id: wgpu::Texture,
    id_view: wgpu::TextureView,
    id_depth: wgpu::TextureView,
    egui_id: egui::TextureId,
}

/// A growable GPU buffer.
struct DynBuf {
    buf: wgpu::Buffer,
    cap: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl DynBuf {
    fn new(device: &wgpu::Device, label: &'static str, usage: wgpu::BufferUsages) -> Self {
        let cap = 4096;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        DynBuf {
            buf,
            cap,
            usage,
            label,
        }
    }
    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let need = (data.len() as u64).max(16);
        if need > self.cap {
            self.cap = need.next_power_of_two();
            self.buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.cap,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buf, 0, data);
        }
    }
}

struct Pipes {
    mesh: wgpu::RenderPipeline,
    flat: wgpu::RenderPipeline,
    line: wgpu::RenderPipeline,
    point: wgpu::RenderPipeline,
}

pub struct Gpu {
    color: Pipes,
    id: Pipes,
    bgl: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    globals_pick: wgpu::Buffer,
    flags: DynBuf,
    bind: Option<wgpu::BindGroup>,
    bind_pick: Option<wgpu::BindGroup>,
    line_zero: wgpu::Buffer,
    lines: DynBuf,
    overlay: DynBuf,
    points: DynBuf,
    flat_v: DynBuf,
    flat_i: DynBuf,
    counts: [u32; 5], // lines, overlay, points, flat indices, of which "under" planes
    pick_pending: Option<PendingPick>,
    pick_ready: Option<(u64, Vec<u32>)>,
    pick_stamp: u64,
    readback: wgpu::Buffer,
    bodies: HashMap<u64, BodyGpu>,
    draw_list: Vec<u64>,
    targets: Option<Targets>,
}

impl Gpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("refusion shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mesh_buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<MeshVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<BodyInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![3 => Uint32, 4 => Uint32],
            },
        ];
        let flat_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FlatVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Uint32],
        }];
        let line_buffers = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LineInst>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32, 3 => Float32, 4 => Uint32, 5 => Uint32],
            },
            wgpu::VertexBufferLayout {
                array_stride: 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![6 => Uint32],
            },
        ];
        let point_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointInst>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32, 2 => Float32, 3 => Uint32, 4 => Uint32],
        }];

        let make = |vs: &str,
                    fs: &str,
                    buffers: &[wgpu::VertexBufferLayout],
                    id_pass: bool,
                    depth_write: bool,
                    blend: bool| {
            let format = if id_pass { ID_FMT } else { COLOR_FMT };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FMT,
                    depth_write_enabled: depth_write,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: if id_pass { 1 } else { MSAA },
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: if blend && !id_pass {
                            Some(wgpu::BlendState::ALPHA_BLENDING)
                        } else {
                            None
                        },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        let color = Pipes {
            mesh: make("vs_mesh", "fs_mesh", &mesh_buffers, false, true, false),
            flat: make("vs_flat", "fs_flat", &flat_buffers, false, false, true),
            line: make("vs_line", "fs_line", &line_buffers, false, false, true),
            point: make("vs_point", "fs_point", &point_buffers, false, false, true),
        };
        let id = Pipes {
            mesh: make("vs_mesh", "fs_mesh_id", &mesh_buffers, true, true, false),
            flat: make("vs_flat", "fs_flat_id", &flat_buffers, true, false, false),
            line: make("vs_line", "fs_line_id", &line_buffers, true, true, false),
            point: make("vs_point", "fs_point_id", &point_buffers, true, true, false),
        };
        let uni = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Globals>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let vb = wgpu::BufferUsages::VERTEX;
        Gpu {
            color,
            id,
            globals: uni("globals"),
            globals_pick: uni("globals pick"),
            flags: DynBuf::new(device, "flags", wgpu::BufferUsages::STORAGE),
            bgl,
            bind: None,
            bind_pick: None,
            line_zero: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dynamic line zero offset"),
                contents: bytemuck::cast_slice(&[0u32; 6]),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            lines: DynBuf::new(device, "lines", vb),
            overlay: DynBuf::new(device, "overlay", vb),
            points: DynBuf::new(device, "points", vb),
            flat_v: DynBuf::new(device, "flat v", vb),
            flat_i: DynBuf::new(device, "flat i", wgpu::BufferUsages::INDEX),
            counts: [0; 5],
            pick_pending: None,
            pick_ready: None,
            pick_stamp: 0,
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick readback"),
                size: 256 * PICK_WIN as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bodies: HashMap::new(),
            draw_list: Vec::new(),
            targets: None,
        }
    }

    fn ensure_targets(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
        size: [u32; 2],
    ) -> egui::TextureId {
        if let Some(t) = &self.targets {
            if t.size == size {
                return t.egui_id;
            }
        }
        let tex = |label, format, samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let color_msaa = tex("color msaa", COLOR_FMT, MSAA, ra).create_view(&Default::default());
        let readback_usage = if cfg!(test) {
            wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::empty()
        };
        let color_texture = tex(
            "color",
            COLOR_FMT,
            1,
            ra | wgpu::TextureUsages::TEXTURE_BINDING | readback_usage,
        );
        let color = color_texture.create_view(&Default::default());
        let depth = tex("depth", DEPTH_FMT, MSAA, ra).create_view(&Default::default());
        let id = tex("id", ID_FMT, 1, ra | wgpu::TextureUsages::COPY_SRC);
        let id_view = id.create_view(&Default::default());
        let id_depth = tex("id depth", DEPTH_FMT, 1, ra).create_view(&Default::default());
        let egui_id = match &self.targets {
            Some(old) => {
                renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    &color,
                    wgpu::FilterMode::Linear,
                    old.egui_id,
                );
                old.egui_id
            }
            None => renderer.register_native_texture(device, &color, wgpu::FilterMode::Linear),
        };
        self.targets = Some(Targets {
            size,
            color_msaa,
            color,
            depth,
            id,
            id_view,
            id_depth,
            egui_id,
            #[cfg(test)]
            color_texture,
        });
        egui_id
    }

    /// Start a frame: evict body meshes not used by the previous frame.
    pub fn begin_frame(&mut self) {
        self.bodies.retain(|_, b| b.used);
        for b in self.bodies.values_mut() {
            b.used = false;
        }
    }

    /// Upload frame data and render the colour pass. Returns the egui texture to show.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut egui_wgpu::Renderer,
        size: [u32; 2],
        f: &FrameData,
    ) -> egui::TextureId {
        let size = [size[0].max(1), size[1].max(1)];
        let tex_id = self.ensure_targets(device, renderer, size);
        self.upload(device, queue, size, f);
        self.draw_color(device, queue, f);
        tex_id
    }

    /// Upload a frame's geometry (a later `pick` uses the most recent upload).
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: [u32; 2],
        f: &FrameData,
    ) {
        self.pick_stamp = pick_stamp(f, size);
        self.draw_list.clear();
        for b in &f.bodies {
            let entry = self.bodies.entry(b.key).or_insert_with(|| {
                let verts: Vec<MeshVertex> = b
                    .mesh
                    .positions
                    .iter()
                    .zip(&b.mesh.normals)
                    .enumerate()
                    .map(|(i, (p, n))| MeshVertex {
                        pos: *p,
                        nrm: *n,
                        face: 0 * i as u32,
                    })
                    .collect();
                // Per-vertex face ids: OCCT meshes faces separately, so a vertex belongs to one face.
                let mut verts = verts;
                for (t, &face) in b.mesh.tri_face.iter().enumerate() {
                    for k in 0..3 {
                        verts[b.mesh.indices[3 * t + k] as usize].face = face;
                    }
                }
                let edges = body_edges(b.mesh);
                BodyGpu {
                    edges: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("cached body edges"),
                        contents: if edges.is_empty() {
                            &[0; 40]
                        } else {
                            bytemuck::cast_slice(&edges)
                        },
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    edge_count: edges.len() as u32,
                    edge_offset: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("body edge pick base"),
                        size: 24,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    vbuf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("body v"),
                        contents: bytemuck::cast_slice(&verts),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    ibuf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("body i"),
                        contents: bytemuck::cast_slice(&b.mesh.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    count: b.mesh.indices.len() as u32,
                    inst: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("body inst"),
                        size: std::mem::size_of::<BodyInstance>() as u64,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    used: false,
                }
            });
            entry.used = true;
            queue.write_buffer(
                &entry.edge_offset,
                0,
                bytemuck::cast_slice(&[b.edge_base; 6]),
            );
            queue.write_buffer(
                &entry.inst,
                0,
                bytemuck::bytes_of(&BodyInstance {
                    base: b.base,
                    color: b.color,
                }),
            );
            self.draw_list.push(b.key);
        }

        self.lines
            .write(device, queue, bytemuck::cast_slice(&f.lines));
        self.overlay
            .write(device, queue, bytemuck::cast_slice(&f.overlay_lines));
        self.points
            .write(device, queue, bytemuck::cast_slice(&f.points));
        self.flat_v
            .write(device, queue, bytemuck::cast_slice(&f.flat_v));
        self.flat_i
            .write(device, queue, bytemuck::cast_slice(&f.flat_i));
        self.counts = [
            f.lines.len() as u32,
            f.overlay_lines.len() as u32,
            f.points.len() as u32,
            f.flat_i.len() as u32,
            f.under_count.min(f.flat_i.len() as u32),
        ];
        let mut flags = f.flags.clone();
        if flags.is_empty() {
            flags.push(0);
        }
        let old_cap = self.flags.cap;
        self.flags
            .write(device, queue, bytemuck::cast_slice(&flags));

        let mut gl = Globals {
            view_proj: f.view_proj.to_cols_array_2d(),
            eye: f.eye.extend(1.0).into(),
            viewport: [size[0] as f32, size[1] as f32, 1.0, f.ortho_bias],
            light: f.light.extend(0.0).into(),
        };
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&gl));
        gl.viewport[2] = 3.0; // fatter lines and points in the pick pass
        queue.write_buffer(&self.globals_pick, 0, bytemuck::bytes_of(&gl));
        if self.bind.is_none() || self.flags.cap != old_cap {
            let mk = |u: &wgpu::Buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("globals"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: u.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.flags.buf.as_entire_binding(),
                        },
                    ],
                })
            };
            self.bind = Some(mk(&self.globals));
            self.bind_pick = Some(mk(&self.globals_pick));
        }
    }

    fn draw_color(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, f: &FrameData) {
        let t = self.targets.as_ref().unwrap();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("viewport"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("color"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.color_msaa,
                    resolve_target: Some(&t.color),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f.clear[0],
                            g: f.clear[1],
                            b: f.clear[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.draw(&mut pass, &self.color, self.bind.as_ref().unwrap());
        }
        queue.submit([enc.finish()]);
    }

    /// Size the offscreen targets without drawing (so a pick can run first).
    pub fn ensure_size(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut egui_wgpu::Renderer,
        size: [u32; 2],
    ) {
        self.ensure_targets(device, renderer, [size[0].max(1), size[1].max(1)]);
    }

    fn draw<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        p: &'p Pipes,
        bind: &'p wgpu::BindGroup,
    ) {
        pass.set_bind_group(0, bind, &[]);
        let id_pass = std::ptr::eq(p, &self.id);
        let under = if id_pass { self.counts[4] } else { 0 };
        if under > 0 {
            pass.set_pipeline(&p.flat);
            pass.set_vertex_buffer(0, self.flat_v.buf.slice(..));
            pass.set_index_buffer(self.flat_i.buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..under, 0, 0..1);
        }
        pass.set_pipeline(&p.mesh);
        for key in &self.draw_list {
            let b = &self.bodies[key];
            pass.set_vertex_buffer(0, b.vbuf.slice(..));
            pass.set_vertex_buffer(1, b.inst.slice(..));
            pass.set_index_buffer(b.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..b.count, 0, 0..1);
        }
        if self.counts[3] > under {
            pass.set_pipeline(&p.flat);
            pass.set_vertex_buffer(0, self.flat_v.buf.slice(..));
            pass.set_index_buffer(self.flat_i.buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(under..self.counts[3], 0, 0..1);
        }
        pass.set_pipeline(&p.line);
        for key in &self.draw_list {
            let b = &self.bodies[key];
            if b.edge_count > 0 {
                pass.set_vertex_buffer(0, b.edges.slice(..));
                pass.set_vertex_buffer(1, b.edge_offset.slice(..));
                pass.draw(0..6, 0..b.edge_count);
            }
        }
        pass.set_vertex_buffer(1, self.line_zero.slice(..));
        if self.counts[0] > 0 {
            pass.set_pipeline(&p.line);
            pass.set_vertex_buffer(0, self.lines.buf.slice(..));
            pass.draw(0..6, 0..self.counts[0]);
        }
        if self.counts[1] > 0 {
            pass.set_pipeline(&p.line);
            pass.set_vertex_buffer(0, self.overlay.buf.slice(..));
            pass.draw(0..6, 0..self.counts[1]);
        }
        if self.counts[2] > 0 {
            pass.set_pipeline(&p.point);
            pass.set_vertex_buffer(0, self.points.buf.slice(..));
            pass.draw(0..6, 0..self.counts[2]);
        }
    }

    /// Ids in the window around a pixel (row-major, PICK_WIN²; 0 = nothing).
    /// Uses the data uploaded by the last `render`.
    pub fn pick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        px: [u32; 2],
    ) -> Option<Vec<u32>> {
        let stamp = self.pick_stamp ^ ((px[0] as u64) << 32 | px[1] as u64);
        if self.pick_pending.is_some() {
            self.finish_pick(device);
        }
        if let Some((key, ids)) = &self.pick_ready {
            if *key == stamp {
                return Some(ids.clone());
            }
        }
        if self.pick_pending.is_some() {
            return None;
        }
        self.pick_pending = self.start_pick(device, queue, px, &self.readback);
        None
    }

    /// Geometry/camera context, independent of the mouse pixel and highlights.
    pub fn pick_scene_stamp(&self) -> u64 {
        self.pick_stamp
    }

    fn start_pick(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        px: [u32; 2],
        buffer: &wgpu::Buffer,
    ) -> Option<PendingPick> {
        let stamp = self.pick_stamp ^ ((px[0] as u64) << 32 | px[1] as u64);
        let Some(t) = &self.targets else { return None };
        let Some(bind) = &self.bind_pick else {
            return None;
        };
        let half = PICK_WIN / 2;
        let x0 = px[0]
            .saturating_sub(half)
            .min(t.size[0].saturating_sub(PICK_WIN));
        let y0 = px[1]
            .saturating_sub(half)
            .min(t.size[1].saturating_sub(PICK_WIN));
        let w = PICK_WIN.min(t.size[0]);
        let h = PICK_WIN.min(t.size[1]);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pick"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("id"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.id_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.id_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_scissor_rect(x0, y0, w, h);
            self.draw(&mut pass, &self.id, bind);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &t.id,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let (tx, rx) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        Some(PendingPick {
            stamp,
            px,
            x0,
            y0,
            w,
            h,
            rx,
        })
    }
    /// Capture this exact uploaded scene without waiting for the GPU.
    pub fn request_click(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        px: [u32; 2],
    ) -> Option<ClickPick> {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("click readback"),
            size: 256 * PICK_WIN as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pending = self.start_pick(device, queue, px, &buffer)?;
        Some(ClickPick { buffer, pending })
    }

    fn finish_pick(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::Maintain::Poll);
        let Some(pending) = self.pick_pending.as_ref() else {
            return;
        };
        let Ok(result) = pending.rx.try_recv() else {
            return;
        };
        let pending = self.pick_pending.take().unwrap();
        if result.is_err() {
            self.pick_ready = None;
            return;
        }
        let PendingPick {
            stamp,
            px,
            x0,
            y0,
            w,
            h,
            ..
        } = pending;
        let out = read_window(&self.readback, px, x0, y0, w, h);
        self.pick_ready = Some((stamp, out));
    }
}

/// Immutable edge geometry uses local zero-based edge IDs; a tiny per-body
/// buffer supplies the frame's pick-table base without reuploading segments.
fn body_edges(mesh: &refusion_model::kernel::Mesh) -> Vec<LineInst> {
    let mut lines = Vec::with_capacity(mesh.edge_points.len().saturating_sub(mesh.edge_count()));
    for e in 1..=mesh.edge_count() as u32 {
        for w in mesh.edge_polyline(e).windows(2) {
            lines.push(LineInst {
                p0: w[0],
                p1: w[1],
                color: crate::scene::colors::edge(),
                width: 1.6,
                id: e - 1,
                style: 0,
            });
        }
    }
    lines
}

pub struct ClickPick {
    buffer: wgpu::Buffer,
    pending: PendingPick,
}
impl ClickPick {
    pub fn poll(&mut self, device: &wgpu::Device) -> Option<Result<Vec<u32>, String>> {
        let _ = device.poll(wgpu::Maintain::Poll);
        match self.pending.rx.try_recv() {
            Ok(Ok(())) => {
                let p = &self.pending;
                Some(Ok(read_window(&self.buffer, p.px, p.x0, p.y0, p.w, p.h)))
            }
            Ok(Err(e)) => Some(Err(format!("Selection readback failed: {e}"))),
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err("Selection readback was interrupted".into()))
            }
            Err(mpsc::TryRecvError::Empty) => None,
        }
    }
}
fn read_window(buffer: &wgpu::Buffer, px: [u32; 2], x0: u32, y0: u32, w: u32, h: u32) -> Vec<u32> {
    let half = PICK_WIN / 2;
    let slice = buffer.slice(..);
    let mut out = vec![0u32; (PICK_WIN * PICK_WIN) as usize];
    {
        let data = slice.get_mapped_range();
        for row in 0..h {
            for col in 0..w {
                let o = (row * 256 + col * 4) as usize;
                // Place into the window so the cursor stays at the centre index.
                let gx = x0 + col;
                let gy = y0 + row;
                let wx = gx as i64 - (px[0] as i64 - half as i64);
                let wy = gy as i64 - (px[1] as i64 - half as i64);
                if (0..PICK_WIN as i64).contains(&wx) && (0..PICK_WIN as i64).contains(&wy) {
                    out[(wy as u32 * PICK_WIN + wx as u32) as usize] =
                        u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
                }
            }
        }
    }
    buffer.unmap();
    out
}

struct PendingPick {
    stamp: u64,
    px: [u32; 2],
    x0: u32,
    y0: u32,
    w: u32,
    h: u32,
    rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}
fn pick_stamp(f: &FrameData, size: [u32; 2]) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    size.hash(&mut hash);
    bytemuck::cast_slice::<f32, u8>(&f.view_proj.to_cols_array()).hash(&mut hash);
    f.ortho_bias.to_bits().hash(&mut hash);
    for b in &f.bodies {
        (b.key, b.base, b.edge_base).hash(&mut hash);
    }
    bytemuck::cast_slice::<LineInst, u8>(&f.lines).hash(&mut hash);
    bytemuck::cast_slice::<LineInst, u8>(&f.overlay_lines).hash(&mut hash);
    bytemuck::cast_slice::<PointInst, u8>(&f.points).hash(&mut hash);
    bytemuck::cast_slice::<FlatVertex, u8>(&f.flat_v).hash(&mut hash);
    f.flat_i.hash(&mut hash);
    hash.finish()
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    #[test]
    fn cached_segments_keep_local_edge_ids_and_exact_geometry() {
        let mesh = refusion_model::kernel::Mesh {
            edge_points: vec![
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [2., 1., 0.],
                [3., 1., 0.],
            ],
            edge_starts: vec![0, 3, 3, 5],
            ..Default::default()
        };
        let lines = body_edges(&mesh);
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines.iter().map(|l| l.id).collect::<Vec<_>>(),
            vec![0, 0, 2]
        );
        assert_eq!(lines[0].p0, [0., 0., 0.]);
        assert_eq!(lines[2].p1, [3., 1., 0.]);
    }
    #[test]
    fn hover_stamp_invalidates_for_geometry_camera_and_pick_base_but_not_highlights() {
        let mesh = refusion_model::kernel::Mesh::default();
        let mut frame = FrameData::default();
        frame.bodies.push(BodyDraw {
            key: 1,
            mesh: &mesh,
            base: 1,
            edge_base: 7,
            color: 0,
        });
        let a = pick_stamp(&frame, [800, 600]);
        frame.flags.push(3);
        assert_eq!(a, pick_stamp(&frame, [800, 600]));
        frame.bodies[0].edge_base = 8;
        assert_ne!(a, pick_stamp(&frame, [800, 600]));
        frame.bodies[0].edge_base = 7;
        frame.view_proj = glam::Mat4::from_translation(glam::Vec3::X);
        assert_ne!(a, pick_stamp(&frame, [800, 600]));
        frame.view_proj = glam::Mat4::IDENTITY;
        frame.bodies[0].key = 2;
        assert_ne!(a, pick_stamp(&frame, [800, 600]));
        assert_ne!(
            pick_stamp(&frame, [800, 600]),
            pick_stamp(&frame, [900, 600])
        );
    }
}
#[cfg(test)]
pub(crate) mod async_gpu_tests {
    use super::*;
    // Vulkan adapter/device lifecycle must be isolated on software drivers.
    pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    pub(crate) fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        struct Wake;
        impl std::task::Wake for Wake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = std::task::Waker::from(std::sync::Arc::new(Wake));
        let mut ctx = std::task::Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            if let std::task::Poll::Ready(v) = future.as_mut().poll(&mut ctx) {
                return v;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    #[test]
    #[ignore = "requires a Vulkan adapter; renders both RFZ fixtures and reads back pixels"]
    fn all_features_fixtures_render_on_gpu() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Vulkan adapter");
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let mut renderer = egui_wgpu::Renderer::new(&device, COLOR_FMT, None, 1, false);
        let mut gpu = Gpu::new(&device);
        for (name, fixture) in [
            (
                "all-features",
                include_str!("../../../examples/all-features.rfz"),
            ),
            (
                "all-features-combined",
                include_str!("../../../examples/all-features-combined.rfz"),
            ),
        ] {
            let doc = refusion_model::Document::from_json(fixture).unwrap();
            assert!(doc.errors.is_empty());
            let meshes = doc
                .state()
                .bodies
                .iter()
                .map(|b| {
                    (
                        b.shape.uid(),
                        std::sync::Arc::new(b.shape.mesh(0.15, 0.25).unwrap()),
                    )
                })
                .collect();
            let mut scene = crate::scene::fixture_scene(&doc, &meshes);
            let mut lo = glam::DVec3::splat(f64::INFINITY);
            let mut hi = glam::DVec3::splat(f64::NEG_INFINITY);
            for draw in &scene.frame.bodies {
                let (a, b) = draw.mesh.bounds.unwrap();
                lo = lo.min(glam::DVec3::from_array(a));
                hi = hi.max(glam::DVec3::from_array(b));
            }
            let mut camera = crate::camera::Camera::default();
            camera.fit(lo, hi);
            camera.tick(1.);
            let size = [768, 768];
            scene.frame.view_proj = camera.view_proj(1.);
            scene.frame.eye = camera.eye().as_vec3();
            scene.frame.light =
                crate::scene::light_dir(camera.right(), camera.up(), camera.forward());
            scene.frame.clear = [0.83, 0.86, 0.90];
            scene.frame.ortho_bias = camera.ortho_bias();
            // Exercise the complete scene, including sketch and plane pipelines.
            gpu.render(&device, &queue, &mut renderer, size, &scene.frame);
            // Isolate solid rendering from sketch/plane overlays for pixel coverage.
            scene.frame.lines.clear();
            scene.frame.points.clear();
            scene.frame.flat_i.clear();
            scene.frame.flat_v.clear();
            scene.frame.under_count = 0;
            gpu.render(&device, &queue, &mut renderer, size, &scene.frame);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fixture pixels"),
                size: 768 * 768 * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &gpu.targets.as_ref().unwrap().color_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(768 * 4),
                        rows_per_image: Some(768),
                    },
                },
                wgpu::Extent3d {
                    width: 768,
                    height: 768,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device.poll(wgpu::Maintain::Wait);
            rx.recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
                .unwrap();
            let pixels = buffer.slice(..).get_mapped_range();
            let background = &pixels[..4];
            let covered = pixels.chunks_exact(4).filter(|p| *p != background).count();
            assert!(
                covered > 2000 && covered < 768 * 768 * 9 / 10,
                "{name}: {covered} visible pixels"
            );
            if let Some(dir) = std::env::var_os("REFUSION_RENDER_ARTIFACTS") {
                let mut ppm = b"P6\n768 768\n255\n".to_vec();
                for p in pixels.chunks_exact(4) {
                    ppm.extend_from_slice(&p[..3]);
                }
                std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")), ppm)
                    .unwrap();
            }
            drop(pixels);
            buffer.unmap();
            // Isolate each body so gallery occlusion cannot hide its pick coverage.
            for draw in std::mem::take(&mut scene.frame.bodies) {
                let (a, b) = draw.mesh.bounds.unwrap();
                camera.fit(glam::DVec3::from_array(a), glam::DVec3::from_array(b));
                camera.tick(1.);
                scene.frame.bodies = vec![draw];
                scene.frame.view_proj = camera.view_proj(1.);
                scene.frame.eye = camera.eye().as_vec3();
                scene.frame.light =
                    crate::scene::light_dir(camera.right(), camera.up(), camera.forward());
                scene.frame.clear = [0.83, 0.86, 0.90];
                scene.frame.ortho_bias = camera.ortho_bias();
                gpu.render(&device, &queue, &mut renderer, size, &scene.frame);
                let draw = &scene.frame.bodies[0];
                let mut hit = false;
                for tri in draw
                    .mesh
                    .indices
                    .chunks_exact(3)
                    .step_by((draw.mesh.indices.len() / 3 / 24).max(1))
                {
                    let center = tri
                        .iter()
                        .map(|&i| glam::Vec3::from_array(draw.mesh.positions[i as usize]))
                        .sum::<glam::Vec3>()
                        / 3.;
                    let Some(px) = camera.project(center.as_dvec3(), glam::DVec2::splat(768.))
                    else {
                        continue;
                    };
                    if !(8. ..760.).contains(&px.x) || !(8. ..760.).contains(&px.y) {
                        continue;
                    }
                    let mut click = gpu
                        .request_click(&device, &queue, [px.x as u32, px.y as u32])
                        .unwrap();
                    let start = std::time::Instant::now();
                    let ids = loop {
                        if let Some(result) = click.poll(&device) {
                            break result.unwrap();
                        }
                        assert!(start.elapsed() < std::time::Duration::from_secs(10));
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    };
                    hit = ids.iter().any(|&id| id >= draw.base && id < draw.edge_base);
                    if hit {
                        break;
                    }
                }
                assert!(
                    hit,
                    "{name}: body at pick base {} has no visible GPU pick",
                    draw.base
                );
            }
        }
    }
    #[test]
    #[ignore = "requires a Vulkan adapter; run explicitly during GUI validation"]
    fn queued_clicks_keep_their_scene_when_geometry_changes_and_viewport_resizes() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Vulkan adapter");
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let mut renderer = egui_wgpu::Renderer::new(&device, COLOR_FMT, None, 1, false);
        let mut gpu = Gpu::new(&device);
        gpu.ensure_size(&device, &mut renderer, [32, 32]);
        let mut frame = FrameData::default();
        frame.flags = vec![0; 3];
        frame.points.push(PointInst {
            pos: [0., 0., 0.5],
            color: 0,
            size: 8.,
            id: 1,
            shape: 0,
        });
        gpu.upload(&device, &queue, [32, 32], &frame);
        let mut first = gpu.request_click(&device, &queue, [16, 16]).unwrap();
        frame.points[0].id = 2;
        gpu.upload(&device, &queue, [32, 32], &frame);
        let mut second = gpu.request_click(&device, &queue, [16, 16]).unwrap();
        gpu.ensure_size(&device, &mut renderer, [16, 16]);
        frame.points.clear();
        gpu.upload(&device, &queue, [16, 16], &frame);
        let mut empty = gpu.request_click(&device, &queue, [8, 8]).unwrap();
        let poll = |readback: &mut ClickPick| {
            let start = std::time::Instant::now();
            loop {
                if let Some(result) = readback.poll(&device) {
                    break result.unwrap();
                }
                assert!(start.elapsed() < std::time::Duration::from_secs(10));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        let a = poll(&mut first);
        let b = poll(&mut second);
        let c = poll(&mut empty);
        assert_eq!(a[(PICK_WIN * PICK_WIN / 2) as usize], 1);
        assert_eq!(b[(PICK_WIN * PICK_WIN / 2) as usize], 2);
        assert!(c.iter().all(|id| *id == 0));

        // Hover distinguishes pending work from a completed empty window, and
        // highlight-only uploads reuse the result without another readback.
        gpu.ensure_size(&device, &mut renderer, [32, 32]);
        frame.points.push(PointInst {
            pos: [0., 0., 0.5],
            color: 0,
            size: 8.,
            id: 1,
            shape: 0,
        });
        gpu.upload(&device, &queue, [32, 32], &frame);
        assert!(gpu.pick(&device, &queue, [16, 16]).is_none());
        let poll_hover = |gpu: &mut Gpu, px| {
            let start = std::time::Instant::now();
            loop {
                if let Some(ids) = gpu.pick(&device, &queue, px) {
                    break ids;
                }
                assert!(start.elapsed() < std::time::Duration::from_secs(10));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        let hit = poll_hover(&mut gpu, [16, 16]);
        assert_eq!(hit[(PICK_WIN * PICK_WIN / 2) as usize], 1);
        let stamp = gpu.pick_scene_stamp();
        frame.flags[1] = 1;
        gpu.upload(&device, &queue, [32, 32], &frame);
        assert_eq!(gpu.pick_scene_stamp(), stamp);
        assert_eq!(gpu.pick(&device, &queue, [16, 16]), Some(hit));
        assert!(gpu.pick(&device, &queue, [31, 31]).is_none());
        assert!(poll_hover(&mut gpu, [31, 31]).iter().all(|id| *id == 0));
        frame.points.clear();
        gpu.upload(&device, &queue, [32, 32], &frame);
        assert_ne!(gpu.pick_scene_stamp(), stamp);
        assert!(gpu.pick(&device, &queue, [16, 16]).is_none());
        assert!(poll_hover(&mut gpu, [16, 16]).iter().all(|id| *id == 0));
    }
}
