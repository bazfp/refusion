// refusion viewport shaders. Colour pipelines write gamma-encoded colour to an
// Rgba8Unorm target that egui composites; *_id pipelines write pick ids to an
// R32Uint target for GPU picking.

struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // x, y: viewport size in px; z: line/point size multiplier (larger in the pick pass);
    // w: orthographic depth bias per unit of k (0 for perspective)
    viewport: vec4<f32>,
    light: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> flags: array<u32>;

const HOVER: vec3<f32> = vec3<f32>(0.30, 0.62, 1.00);
const SELECT: vec3<f32> = vec3<f32>(0.10, 0.45, 0.98);
// Selected sketch curves/points: distinct from the blue of unconstrained geometry.
const SELECT_CURVE: vec3<f32> = vec3<f32>(0.0, 0.80, 0.95);

fn unpack(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u)) / 255.0;
}
fn to_srgb(c: vec3<f32>) -> vec3<f32> { return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)); }
fn to_lin(c: vec3<f32>) -> vec3<f32> { return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2)); }
fn flag(id: u32) -> u32 {
    if (id < arrayLength(&flags)) { return flags[id]; }
    return 0u;
}
// Move a clip-space position towards the camera by a fraction k of its depth.
fn bias(p: vec4<f32>, k: f32) -> vec4<f32> {
    var q = p;
    if (g.viewport.w > 0.0) {
        // Orthographic depth is linear over a large range: bias by a fixed
        // world distance (computed on the CPU) instead.
        q.z = q.z - g.viewport.w * (k / 0.0025) * q.w;
    } else {
        q.z = q.z - k * (q.w - q.z);
    }
    return q;
}

// ------------------------------------------------------------------ bodies

struct MeshOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) wp: vec3<f32>,
    @location(2) @interpolate(flat) id: u32,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_mesh(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>, @location(2) face: u32,
           @location(3) base: u32, @location(4) color: u32) -> MeshOut {
    var o: MeshOut;
    o.pos = g.view_proj * vec4<f32>(p, 1.0);
    o.n = n;
    o.wp = p;
    o.id = base + face - 1u;
    o.color = unpack(color);
    return o;
}

@fragment
fn fs_mesh(i: MeshOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
    var n = normalize(i.n);
    if (!ff) { n = -n; }
    let v = normalize(g.eye.xyz - i.wp);
    let l = normalize(g.light.xyz);
    let base = to_lin(i.color.rgb);
    let hemi = 0.18 + 0.10 * n.y;
    let diff = max(dot(n, l), 0.0) * 0.62 + max(dot(n, v), 0.0) * 0.28 + hemi;
    let h = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), 64.0) * 0.22;
    var c = base * diff + vec3<f32>(spec);
    let f = flag(i.id);
    if ((f & 2u) != 0u) {
        c = mix(c, to_lin(SELECT), 0.45);
    } else if ((f & 1u) != 0u) {
        c = mix(c, to_lin(HOVER), 0.30);
    }
    return vec4<f32>(to_srgb(c), i.color.a);
}

@fragment
fn fs_mesh_id(i: MeshOut) -> @location(0) u32 {
    return i.id;
}

// ------------------------------------------------------------------ flat (profiles, planes)

struct FlatOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) id: u32,
};

@vertex
fn vs_flat(@location(0) p: vec3<f32>, @location(1) color: u32, @location(2) id: u32) -> FlatOut {
    var o: FlatOut;
    o.pos = bias(g.view_proj * vec4<f32>(p, 1.0), 0.0008);
    o.color = unpack(color);
    o.id = id;
    return o;
}

@fragment
fn fs_flat(i: FlatOut) -> @location(0) vec4<f32> {
    var c = i.color;
    let f = flag(i.id);
    if ((f & 2u) != 0u) {
        c = vec4<f32>(mix(c.rgb, SELECT, 0.65), max(c.a, 0.55));
    } else if ((f & 1u) != 0u) {
        c = vec4<f32>(mix(c.rgb, HOVER, 0.45), max(c.a, 0.40));
    }
    return c;
}

@fragment
fn fs_flat_id(i: FlatOut) -> @location(0) u32 {
    if (i.id == 0u) { discard; }
    return i.id;
}

// ------------------------------------------------------------------ thick lines

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) id: u32,
    @location(2) @interpolate(linear) along: f32,
    @location(3) @interpolate(flat) style: u32,
};

// style bit 0: dashed; bit 1: always on top
@vertex
fn vs_line(@builtin(vertex_index) vi: u32,
           @location(0) p0: vec3<f32>, @location(1) p1: vec3<f32>, @location(2) color: u32,
           @location(3) width: f32, @location(4) id: u32, @location(5) style: u32, @location(6) offset: u32) -> LineOut {
    var o: LineOut;
    o.color = unpack(color);
    o.id = id + offset;
    o.style = style;
    var c0 = g.view_proj * vec4<f32>(p0, 1.0);
    var c1 = g.view_proj * vec4<f32>(p1, 1.0);
    let eps = 1e-5;
    if (c0.w < eps && c1.w < eps) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        o.along = 0.0;
        return o;
    }
    if (c0.w < eps) { c0 = mix(c0, c1, (eps - c0.w) / (c1.w - c0.w)); }
    if (c1.w < eps) { c1 = mix(c1, c0, (eps - c1.w) / (c0.w - c1.w)); }
    let half_vp = g.viewport.xy * 0.5;
    let s0 = c0.xy / c0.w * half_vp;
    let s1 = c1.xy / c1.w * half_vp;
    var d = s1 - s0;
    let len = length(d);
    if (len < 1e-6) { d = vec2<f32>(1.0, 0.0); } else { d = d / len; }
    let nrm = vec2<f32>(-d.y, d.x);
    var wscale = 1.0;
    if ((flag(id) & 3u) != 0u) { wscale = 1.7; }
    let w = max(width * g.viewport.z * wscale, 1.0) * 0.5;
    var ts = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
    var ss = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let t = ts[vi];
    let side = ss[vi];
    var c = mix(c0, c1, t);
    let off = nrm * side * w + d * (t * 2.0 - 1.0) * w;
    c.x = c.x + off.x / half_vp.x * c.w;
    c.y = c.y + off.y / half_vp.y * c.w;
    if ((style & 2u) != 0u) {
        c.z = 0.0;
    } else {
        c = bias(c, 0.0025);
    }
    o.pos = c;
    o.along = t * len;
    return o;
}

@fragment
fn fs_line(i: LineOut) -> @location(0) vec4<f32> {
    if ((i.style & 1u) != 0u && fract(i.along / 10.0) > 0.55) { discard; }
    var c = i.color;
    let f = flag(i.id);
    if ((f & 2u) != 0u) {
        c = vec4<f32>(SELECT_CURVE, 1.0);
    } else if ((f & 1u) != 0u) {
        c = vec4<f32>(HOVER, 1.0);
    }
    return c;
}

@fragment
fn fs_line_id(i: LineOut) -> @location(0) u32 {
    if (i.id == 0u) { discard; }
    return i.id;
}

// ------------------------------------------------------------------ points

struct PointOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(flat) id: u32,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) shape: u32,
};

@vertex
fn vs_point(@builtin(vertex_index) vi: u32,
            @location(0) p: vec3<f32>, @location(1) color: u32, @location(2) size: f32,
            @location(3) id: u32, @location(4) shape: u32) -> PointOut {
    var o: PointOut;
    var cx = array<f32, 6>(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
    var cy = array<f32, 6>(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0);
    let corner = vec2<f32>(cx[vi], cy[vi]);
    var c = g.view_proj * vec4<f32>(p, 1.0);
    let half_px = size * g.viewport.z * 0.5;
    let half_vp = g.viewport.xy * 0.5;
    c.x = c.x + corner.x * half_px / half_vp.x * c.w;
    c.y = c.y + corner.y * half_px / half_vp.y * c.w;
    if ((shape & 4u) != 0u) { c.z = 0.0; } else { c = bias(c, 0.004); }
    o.pos = c;
    o.color = unpack(color);
    o.id = id;
    o.uv = corner;
    o.shape = shape;
    return o;
}

@fragment
fn fs_point(i: PointOut) -> @location(0) vec4<f32> {
    let r = length(i.uv);
    let kind = i.shape & 3u;
    if (kind != 1u && r > 1.0) { discard; }
    var c = i.color;
    let f = flag(i.id);
    if ((f & 2u) != 0u) {
        c = vec4<f32>(SELECT_CURVE, 1.0);
    } else if ((f & 1u) != 0u) {
        c = vec4<f32>(HOVER, 1.0);
    }
    if (kind == 2u && r < 0.55) { c = vec4<f32>(1.0, 1.0, 1.0, 1.0); }
    return c;
}

@fragment
fn fs_point_id(i: PointOut) -> @location(0) u32 {
    if (i.id == 0u || length(i.uv) > 1.0) { discard; }
    return i.id;
}
