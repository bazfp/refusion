<div align="center">

# refusion

### Sketch an idea. Shape a solid. Keep it editable.

A native parametric CAD modeller for Linux, built with Rust and OpenCASCADE.

![Version](https://img.shields.io/badge/version-0.1.1-2563eb?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Linux-334155?style=flat-square)
![Rust](https://img.shields.io/badge/Rust-1.99-f97316?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-059669?style=flat-square)](LICENSE)

[Download v0.1.1](https://github.com/bazfp/refusion/releases/tag/v0.1.1) · [Features](#from-sketch-to-solid) · [Controls](#make-your-first-part) · [Build from source](#build-from-source)

</div>

refusion brings constrained sketches, solid modelling, and an editable feature timeline into one GPU-rendered desktop workspace. Give a dimension a name, use it in an expression, and change it later to recompute the model.

**v0.1.1 is an early release.** It is ready for exploration and feedback; save copies of important designs. The current target is Linux x86-64 with a working Vulkan or OpenGL driver, on X11 or Wayland.

## From sketch to solid

| Work with | What you can do |
| :--- | :--- |
| **Constrained sketches** | Draw lines, rectangles, circles, arcs, and points. Add geometric constraints, dimensions, construction geometry, and projected edges. Trim, extend, and offset curves. |
| **Solid features** | Extrude and revolve profiles; join, cut, or intersect bodies. Add fillets, chamfers, holes, shells, mirrors, and rectangular or circular patterns. |
| **Parameters & history** | Use expressions such as `width / 2 + 3 mm`. Edit, suppress, or delete features and move the timeline marker to revisit an earlier state. |
| **A responsive viewport** | Orbit, pan, zoom, switch standard views, measure geometry, and preview modelling operations before applying them. |
| **Files & exchange** | Save editable `.rfz` designs; import STEP, IGES, and `.f3d`; export STEP and binary or ASCII STL. |

F3D import reconstructs supported solid bodies and sketch geometry. It does not restore the source feature timeline, sketch constraints, dimensions, or construction flags, and some geometry may be approximated or omitted. Import notices report these limitations.

## Get started

The Docker build provides the compiler and native build dependencies. You need Docker with Buildx; the resulting application runs directly on your desktop.

```sh
scripts/docker-build.sh
./dist/refusion dist/examples/bracket.rfz
```

The first build compiles OpenCASCADE. Later builds reuse Docker layers and Cargo caches. The binary is built on Debian Bookworm and targets Linux x86-64 with glibc 2.36 or newer. Your desktop must provide graphics drivers and the X11 or Wayland runtime libraries.

If you downloaded a release archive, extract it and run `./refusion examples/bracket.rfz` from the extracted directory. Keep the included license notices with redistributed binaries.

## Make your first part

1. Press **S** and choose an origin plane to create a sketch.
2. Press **R** to draw a rectangle. Press **D** to add dimensions.
3. Choose **Finish Sketch**, press **E**, and select the enclosed profile.
4. Enter an extrusion distance and confirm the operation.
5. Save with **Ctrl+S**. Double-click a timeline feature to edit it later.

| Action | Control |
| :--- | :--- |
| Orbit | Right-drag, Shift + middle-drag, or Alt + left-drag |
| Pan / zoom | Middle-drag / mouse wheel |
| Fit / reset view | **F6** / ViewCube home button |
| Sketch geometry | **L** line · **R** rectangle · **C** circle · **A** centre arc |
| Dimension / project | **D** / **P** |
| Construction / grid snap | **X** / **G** |
| Extrude / fillet / hole | **E** / **F** / **H** |
| Measure | **I** |
| Select / cancel | **Space** / **Esc** |
| Undo / redo | **Ctrl+Z** / **Ctrl+Y** |

Sketch geometry changes from blue to black when fully constrained. The status bar reports the remaining degrees of freedom. Named parameters are available from the **ƒx Parameters** control.

## Explore the examples

Three editable designs are included:

- **[bracket.rfz](examples/bracket.rfz)** — a compact plate-and-boss workflow demonstrating parameter edits and feature replay.
- **[all-features.rfz](examples/all-features.rfz)** — a gallery covering supported saved feature types and sketch constraint variants.
- **[all-features-combined.rfz](examples/all-features-combined.rfz)** — the gallery plus an enclosure workflow on one evolving body.

```sh
./dist/refusion dist/examples/all-features-combined.rfz
```

## Build from source

Install Rust through rustup, plus a C++17 compiler, CMake, Make, curl, and the development dependencies required by your Linux graphics stack. The repository pins Rust 1.99.0.

```sh
scripts/build-occt.sh
cargo build --release --locked -p refusion-app
./target/release/refusion examples/bracket.rfz
```

OpenCASCADE 7.8.1 is built as static libraries under `third_party/occt`. To use an existing compatible installation, set `OCCT_DIR` to its prefix; it must contain `include/opencascade` and the static `libTK*.a` libraries under `lib`.

Validate a checkout with:

```sh
cargo fmt --all --check
cargo test --workspace --locked
```

Or run the tests in the build container with `scripts/docker-build.sh test`. Tests cover sketch solving and profiles, solid geometry, parameters and history, file round trips, and UI interactions. Optional GPU rendering tests are ignored by default and require a graphics adapter.

| Crate | Responsibility |
| :--- | :--- |
| `refusion-sketch` | Sketch geometry, constraints, solver, and closed profiles |
| `refusion-kernel` | Safe Rust API and a C++ shim over OpenCASCADE |
| `refusion-f3d` | F3D archive and geometry import |
| `refusion-model` | Parameters, feature timeline, recomputation, and persistence |
| `refusion-app` | Desktop interface, viewport rendering, picking, and tools |

## Current scope

The focus is sketch-based part modelling. Splines and ellipses, sweep and loft, components and assemblies, drawings, and CAM are not implemented. Source models with unsupported geometry may not import completely.

Bug reports are most useful with reproduction steps, your Linux distribution and GPU, and a small design that you can share publicly.

## License

The refusion source is available under the [MIT license](LICENSE). OpenCASCADE is a separate dependency under LGPL 2.1 with its additional exception; its notices are included in packaged binaries. Other dependencies retain their respective licenses.
