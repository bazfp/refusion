
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepBuilderAPI_Copy.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepPrimAPI_MakeHalfSpace.hxx>
#include <BRepLib.hxx>
#include <gp_Ax1.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepLib.hxx>
#include <BRepLib_ToolTriangulatedShape.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GCPnts_TangentialDeflection.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeSegment.hxx>
#include <GProp_GProps.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Writer.hxx>
#include <ShapeFix_Face.hxx>
#include <ShapeFix_Shape.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <Standard_Failure.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <BRep_Builder.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Pln.hxx>
#include <Geom_Plane.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_ToroidalSurface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_Line.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_SurfaceOfLinearExtrusion.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>
#include <BRepOffsetAPI_MakeFilling.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <ShapeFix_Wire.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColgp_Array2OfPnt.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_Array2OfReal.hxx>
#include <STEPControl_Reader.hxx>
#include <IGESControl_Reader.hxx>
#include <BRepBuilderAPI_Sewing.hxx>
#include <BRepBuilderAPI_MakeSolid.hxx>
#include <ShapeFix_Solid.hxx>
#include <ShapeFix_Shell.hxx>
#include <BRep_Builder.hxx>
#include <sstream>
#include <cstdlib>
#include <BRepTools.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <NCollection_DataMap.hxx>
#include <TopTools_ShapeMapHasher.hxx>
#include <BRepTools_History.hxx>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <limits>

struct RfShape {
    TopoDS_Shape s;
    TopTools_IndexedMapOfShape faces, edges, vertices;
    std::vector<uint64_t> ftags, etags;
    std::vector<std::array<int, 2>> adjacency;
    std::vector<int> adjacency_counts;
    bool indexed = false;
    void index() {
        if (indexed) return;
        TopExp::MapShapes(s, TopAbs_FACE, faces);
        TopExp::MapShapes(s, TopAbs_EDGE, edges);
        TopExp::MapShapes(s, TopAbs_VERTEX, vertices);
        // Immutable topology: build once during worker geometry preparation.
        // Rebuilding all ancestry for every queried edge made face selection
        // and persistent edge naming quadratic on dense imported solids.
        TopTools_IndexedDataMapOfShapeListOfShape e2f;
        TopExp::MapShapesAndAncestors(s, TopAbs_EDGE, TopAbs_FACE, e2f);
        adjacency.resize(edges.Extent());
        adjacency_counts.resize(edges.Extent(), 0);
        for (int i = 1; i <= edges.Extent(); ++i) {
            if (!e2f.Contains(edges(i))) continue;
            int n = 0;
            for (TopTools_ListIteratorOfListOfShape it(e2f.FindFromKey(edges(i))); it.More() && n < 2; it.Next()) {
                int face = faces.FindIndex(it.Value());
                if (n && adjacency[i-1][0] == face) continue;
                adjacency[i-1][n++] = face;
            }
            adjacency_counts[i-1] = n;
        }
        indexed = true;
    }
};

struct RfMesh {
    std::vector<float> pos, nrm;
    std::vector<uint32_t> idx, tri_face;
    std::vector<float> edge_pts;       // xyz triples
    std::vector<uint32_t> edge_starts; // per edge: start offset (in points); final sentinel
};

static thread_local std::string g_err;
// OCCT exchange controllers use process-global Interface_Static settings.
static std::mutex g_exchange_lock;
extern "C" const char* rf_last_error() { return g_err.c_str(); }

static RfShape* wrap(const TopoDS_Shape& s) {
    if (s.IsNull()) {
        g_err = "operation produced an empty shape";
        return nullptr;
    }
    auto* r = new RfShape();
    r->s = s;
    return r;
}

#define RF_TRY try {
#define RF_CATCH(ret)                                                            \
    }                                                                            \
    catch (const Standard_Failure& e) {                                          \
        g_err = std::string(e.DynamicType()->Name()) + ": " +                    \
                (e.GetMessageString() ? e.GetMessageString() : "");              \
        return ret;                                                              \
    }                                                                            \
    catch (const std::exception& e) {                                            \
        g_err = e.what();                                                        \
        return ret;                                                              \
    }                                                                            \
    catch (...) {                                                                \
        g_err = "unknown kernel error";                                          \
        return ret;                                                              \
    }


typedef NCollection_DataMap<TopoDS_Shape, uint64_t, TopTools_ShapeMapHasher> TagMap;

static uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static uint64_t mix(uint64_t a, uint64_t b) { return splitmix(a ^ splitmix(b)); }
static uint64_t mix3(uint64_t a, uint64_t b, uint64_t c) { return mix(mix(a, b), c); }

// Faces of `in` keyed to their tags.
static void bind_faces(RfShape* in, TagMap& m) {
    in->index();
    for (int i = 1; i <= in->faces.Extent(); ++i)
        if (i - 1 < (int)in->ftags.size()) m.Bind(in->faces(i), in->ftags[i - 1]);
}

// Assign tags to every face of r from map m (untracked faces get 0).
static void apply_tags(RfShape* r, const TagMap& m) {
    r->index();
    r->ftags.assign(r->faces.Extent(), 0);
    for (int i = 1; i <= r->faces.Extent(); ++i)
        if (m.IsBound(r->faces(i))) r->ftags[i - 1] = m.Find(r->faces(i));
}

// Carry tags of `in` through an operation's history onto its result.
template <class Op>
static TagMap carry(RfShape* in, Op& op) {
    TagMap m;
    in->index();
    for (int i = 1; i <= in->faces.Extent(); ++i) {
        const TopoDS_Shape& f = in->faces(i);
        uint64_t t = i - 1 < (int)in->ftags.size() ? in->ftags[i - 1] : 0;
        m.Bind(f, t);
        for (TopTools_ListIteratorOfListOfShape it(op.Modified(f)); it.More(); it.Next())
            if (!m.IsBound(it.Value())) m.Bind(it.Value(), t);
    }
    return m;
}

static TopoDS_Shape clean_tracked(const TopoDS_Shape& s, TagMap& m) {
    ShapeUpgrade_UnifySameDomain u(s, true, true, true);
    u.SetSafeInputMode(true);
    u.Build();
    Handle(BRepTools_History) h = u.History();
    TopTools_IndexedMapOfShape fs;
    TopExp::MapShapes(s, TopAbs_FACE, fs);
    for (int i = 1; i <= fs.Extent(); ++i) {
        if (!m.IsBound(fs(i))) continue;
        uint64_t t = m.Find(fs(i));
        if (!h.IsNull())
            for (TopTools_ListIteratorOfListOfShape it(h->Modified(fs(i))); it.More(); it.Next())
                if (!m.IsBound(it.Value())) m.Bind(it.Value(), t);
    }
    return u.Shape();
}

extern "C" void rf_face_tags(RfShape* s, uint64_t* out) {
    s->index();
    for (int i = 0; i < s->faces.Extent(); ++i) out[i] = i < (int)s->ftags.size() ? s->ftags[i] : 0;
}

// The (up to two) faces adjacent to an edge, as 1-based face ids. Returns count.
extern "C" int rf_edge_faces(RfShape* s, int eid, int* out) {
    s->index();
    if (eid < 1 || eid > s->edges.Extent()) return 0;
    int n = s->adjacency_counts[eid-1];
    for (int i = 0; i < n; ++i) out[i] = s->adjacency[eid-1][i];
    return n;
}

extern "C" void rf_shape_free(RfShape* s) { delete s; }


extern "C" int rf_count(RfShape* s, int kind) {
    RF_TRY
    s->index();
    return kind == 0 ? s->faces.Extent() : s->edges.Extent();
    RF_CATCH(-1)
}

// ------------------------------------------------------------------ profiles

// frame: origin(3) xdir(3) ydir(3)
// segs: per segment 8 doubles: kind(0=line,1=arc) ax ay bx by cx cy ccw
extern "C" RfShape* rf_make_face(const double* frame, const int* loop_lens, int nloops, const double* segs, const uint64_t* seg_tags) {
    RF_TRY
    gp_Pnt O(frame[0], frame[1], frame[2]);
    gp_Dir X(frame[3], frame[4], frame[5]);
    gp_Dir Y(frame[6], frame[7], frame[8]);
    gp_Dir N = X.Crossed(Y);
    auto to3 = [&](double u, double v) {
        return gp_Pnt(O.X() + u * X.X() + v * Y.X(), O.Y() + u * X.Y() + v * Y.Y(), O.Z() + u * X.Z() + v * Y.Z());
    };
    gp_Pln pln(gp_Ax3(O, N, X));
    std::vector<TopoDS_Wire> wires;
    const double* p = segs;
    for (int l = 0; l < nloops; ++l) {
        BRepBuilderAPI_MakeWire mw;
        for (int k = 0; k < loop_lens[l]; ++k, p += 8) {
            gp_Pnt a = to3(p[1], p[2]), b = to3(p[3], p[4]);
            TopoDS_Edge e;
            if (p[0] == 0.0) {
                e = BRepBuilderAPI_MakeEdge(GC_MakeSegment(a, b).Value());
            } else {
                gp_Pnt c = to3(p[5], p[6]);
                double r = c.Distance(a);
                bool ccw = p[7] != 0.0;
                gp_Circ circ(gp_Ax2(c, ccw ? N : N.Reversed(), gp_Vec(c, a)), r);
                e = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(circ, a, b, true).Value());
            }
            mw.Add(e);
            if (!mw.IsDone()) {
                g_err = "profile loop is not connected";
                return nullptr;
            }
        }
        wires.push_back(mw.Wire());
    }
    if (wires.empty()) {
        g_err = "empty profile";
        return nullptr;
    }
    BRepBuilderAPI_MakeFace mf(pln, wires[0], true);
    for (size_t i = 1; i < wires.size(); ++i) mf.Add(wires[i]);
    if (!mf.IsDone()) {
        g_err = "could not build a face from the profile";
        return nullptr;
    }
    ShapeFix_Face fix(mf.Face());
    fix.Perform();
    RfShape* r = wrap(fix.Face());
    if (!r) return nullptr;
    // Name each face edge after the profile segment it came from (nearest midpoint).
    std::vector<gp_Pnt> mids;
    int total = 0;
    for (int l = 0; l < nloops; ++l) total += loop_lens[l];
    for (int k = 0; k < total; ++k) {
        const double* q = segs + 8 * k;
        if (q[0] == 0.0) {
            mids.push_back(to3(0.5 * (q[1] + q[3]), 0.5 * (q[2] + q[4])));
        } else {
            double ta = std::atan2(q[2] - q[6], q[1] - q[5]), tb = std::atan2(q[4] - q[6], q[3] - q[5]);
            double sw = q[7] != 0.0 ? tb - ta : ta - tb;
            while (sw <= 1e-12) sw += 2 * M_PI;
            double tm = q[7] != 0.0 ? ta + 0.5 * sw : ta - 0.5 * sw;
            double rr = std::hypot(q[1] - q[5], q[2] - q[6]);
            mids.push_back(to3(q[5] + rr * std::cos(tm), q[6] + rr * std::sin(tm)));
        }
    }
    r->index();
    r->etags.assign(r->edges.Extent(), 0);
    for (int i = 1; i <= r->edges.Extent(); ++i) {
        BRepAdaptor_Curve c(TopoDS::Edge(r->edges(i)));
        gp_Pnt m = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()));
        int best = -1;
        double bd = 1e300;
        for (int k = 0; k < total; ++k) {
            double d = m.Distance(mids[k]);
            if (d < bd) { bd = d; best = k; }
        }
        if (best >= 0) r->etags[i - 1] = seg_tags[best];
    }
    return r;
    RF_CATCH(nullptr)
}

// Combine several faces (multi-profile selection) into one compound.
extern "C" RfShape* rf_compound(const RfShape* const* shapes, int n) {
    RF_TRY
    BRep_Builder b;
    TopoDS_Compound c;
    b.MakeCompound(c);
    TagMap m;
    for (int i = 0; i < n; ++i) {
        b.Add(c, shapes[i]->s);
        bind_faces(const_cast<RfShape*>(shapes[i]), m);
    }
    RfShape* r = wrap(c);
    if (r) apply_tags(r, m);
    return r;
    RF_CATCH(nullptr)
}

// ------------------------------------------------------------------ sweeps

// Side faces are named mix(feature, 1, profile-edge tag); caps mix(feature, 2|3).
// This is the analogue of ASM's sweep_annotation_data::get_profile_progenitor.
template <class Mk>
static void name_sweep(RfShape* r, Mk& mk, const TopoDS_Shape& base, RfShape* face, uint64_t feat, bool caps) {
    TagMap m;
    TopTools_IndexedMapOfShape be;
    TopExp::MapShapes(base, TopAbs_EDGE, be);
    face->index();
    for (int i = 1; i <= be.Extent(); ++i) {
        uint64_t et = i - 1 < (int)face->etags.size() ? face->etags[i - 1] : (uint64_t)i;
        for (TopTools_ListIteratorOfListOfShape it(mk.Generated(be(i))); it.More(); it.Next())
            if (it.Value().ShapeType() == TopAbs_FACE) m.Bind(it.Value(), mix3(feat, 1, et));
    }
    if (caps) {
        for (TopExp_Explorer ex(mk.FirstShape(), TopAbs_FACE); ex.More(); ex.Next()) m.Bind(ex.Current(), mix(feat, 2));
        for (TopExp_Explorer ex(mk.LastShape(), TopAbs_FACE); ex.More(); ex.Next()) m.Bind(ex.Current(), mix(feat, 3));
    }
    apply_tags(r, m);
}

extern "C" RfShape* rf_extrude(RfShape* face, const double* dir, double d0, double d1, uint64_t feat) {
    RF_TRY
    gp_Vec v(dir[0], dir[1], dir[2]);
    v.Normalize();
    TopoDS_Shape base = face->s;
    if (std::abs(d0) > 1e-12) {
        gp_Trsf t;
        t.SetTranslation(v * d0);
        base = base.Moved(TopLoc_Location(t));  // same topology, so edge order is preserved
    }
    BRepPrimAPI_MakePrism mk(base, v * (d1 - d0), true);
    mk.Build();
    if (!mk.IsDone()) {
        g_err = "extrude failed";
        return nullptr;
    }
    RfShape* r = wrap(mk.Shape());
    if (r) name_sweep(r, mk, base, face, feat, true);
    return r;
    RF_CATCH(nullptr)
}

// The profile plane is neutral; positive draft removes material away from it.
extern "C" RfShape* rf_taper(RfShape* solid, const double* dir, const double* origin, double angle) {
    RF_TRY
    BRepOffsetAPI_DraftAngle draft(solid->s);
    gp_Dir axis(dir[0],dir[1],dir[2]);
    gp_Pln neutral(gp_Pnt(origin[0],origin[1],origin[2]), axis);
    solid->index();
    for (int i=1; i<=solid->faces.Extent(); ++i) {
        TopoDS_Face face=TopoDS::Face(solid->faces(i));
        BRepAdaptor_Surface surf(face);
        if (surf.GetType()==GeomAbs_Plane && std::abs(surf.Plane().Axis().Direction().Dot(axis))>1.-1e-8) continue;
        draft.Add(face, axis, angle, neutral);
        if (!draft.AddDone()) { g_err="Taper cannot be applied to this profile or angle"; return nullptr; }
    }
    draft.Build();
    if (!draft.IsDone() || !BRepCheck_Analyzer(draft.Shape()).IsValid()) { g_err="Taper collapses the extrusion or creates invalid geometry"; return nullptr; }
    auto* result=wrap(draft.Shape());
    if (result) {
        TagMap names=carry(solid,draft);
        for (int i=1;i<=solid->faces.Extent();++i) {
            TopoDS_Shape modified=draft.ModifiedShape(solid->faces(i));
            for (TopExp_Explorer ex(modified,TopAbs_FACE);ex.More();ex.Next()) if (!names.IsBound(ex.Current())) names.Bind(ex.Current(),solid->ftags[i-1]);
        }
        apply_tags(result,names);
    }
    return result;
    RF_CATCH(nullptr)
}
extern "C" RfShape* rf_trim_plane(RfShape* solid, const double* origin, const double* normal, const double* inside, uint64_t feat) {
    RF_TRY
    TopoDS_Face plane=BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(origin[0],origin[1],origin[2]),gp_Dir(normal[0],normal[1],normal[2]))).Face();
    TopoDS_Solid half=BRepPrimAPI_MakeHalfSpace(plane,gp_Pnt(inside[0],inside[1],inside[2])).Solid();
    BRepAlgoAPI_Common cut;
    cut.SetNonDestructive(true);
    TopTools_ListOfShape args,tools; args.Append(solid->s); tools.Append(half);
    cut.SetArguments(args); cut.SetTools(tools); cut.Build();
    if (!cut.IsDone() || !BRepCheck_Analyzer(cut.Shape()).IsValid()) { g_err="Extrude could not terminate at the selected face"; return nullptr; }
    auto* result=wrap(cut.Shape());
    if (result) {
        TagMap names=carry(solid,cut);
        for (TopTools_ListIteratorOfListOfShape it(cut.Generated(plane));it.More();it.Next()) if (!names.IsBound(it.Value())) names.Bind(it.Value(),mix(feat,3));
        // Boolean history may report the plane as modified rather than generated.
        for (TopTools_ListIteratorOfListOfShape it(cut.Modified(plane));it.More();it.Next()) if (!names.IsBound(it.Value())) names.Bind(it.Value(),mix(feat,3));
        apply_tags(result,names);
        for (auto& tag:result->ftags) if (!tag) tag=mix(feat,3);
    }
    return result;
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_revolve(RfShape* face, const double* origin, const double* axis, double angle, uint64_t feat) {
    RF_TRY
    gp_Ax1 ax(gp_Pnt(origin[0], origin[1], origin[2]), gp_Dir(axis[0], axis[1], axis[2]));
    BRepPrimAPI_MakeRevol mk(face->s, ax, angle, true);
    mk.Build();
    if (!mk.IsDone()) {
        g_err = "revolve failed";
        return nullptr;
    }
    RfShape* r = wrap(mk.Shape());
    if (r) name_sweep(r, mk, face->s, face, feat, std::abs(angle) < 2 * M_PI - 1e-9);
    return r;
    RF_CATCH(nullptr)
}

// A face of a body as a standalone shape (for extruding body faces). Writes,
// for each edge of the new face (its own 1-based order), the id of the same
// edge in the body, so the caller can name the edges after the body's edges.
extern "C" RfShape* rf_face_copy(RfShape* s, int fid, int* body_edge_ids, int max) {
    RF_TRY
    s->index();
    if (fid < 1 || fid > s->faces.Extent()) { g_err = "bad face id"; return nullptr; }
    RfShape* r = wrap(s->faces(fid));
    if (!r) return nullptr;
    r->index();
    for (int i = 1; i <= r->edges.Extent() && i <= max; ++i) body_edge_ids[i - 1] = s->edges.FindIndex(r->edges(i));
    return r;
    RF_CATCH(nullptr)
}

extern "C" int rf_set_edge_tags(RfShape* s, const uint64_t* tags, int n) {
    RF_TRY
    s->index();
    s->etags.assign(tags, tags + n);
    return 1;
    RF_CATCH(0)
}

// ------------------------------------------------------------------ measure

extern "C" int rf_vertex_count(RfShape* s) {
    s->index();
    return s->vertices.Extent();
}

extern "C" int rf_vertex_pos(RfShape* s, int id, double* out) {
    s->index();
    if (id < 1 || id > s->vertices.Extent()) return 0;
    gp_Pnt p = BRep_Tool::Pnt(TopoDS::Vertex(s->vertices(id)));
    out[0] = p.X(); out[1] = p.Y(); out[2] = p.Z();
    return 1;
}

// kind: 0 whole shape, 1 face, 2 edge, 3 vertex (1-based ids)
static bool sub_shape(RfShape* s, int kind, int id, TopoDS_Shape& out) {
    s->index();
    const TopTools_IndexedMapOfShape* m = kind == 1 ? &s->faces : kind == 2 ? &s->edges : kind == 3 ? &s->vertices : nullptr;
    if (!m) { out = s->s; return true; }
    if (id < 1 || id > m->Extent()) return false;
    out = (*m)(id);
    return true;
}

// Minimum distance between two (sub-)shapes. out: distance, p1 xyz, p2 xyz.
extern "C" int rf_measure(RfShape* a, int ka, int ia, RfShape* b, int kb, int ib, double* out) {
    RF_TRY
    TopoDS_Shape sa, sb;
    if (!sub_shape(a, ka, ia, sa) || !sub_shape(b, kb, ib, sb)) { g_err = "bad measure target"; return 0; }
    BRepExtrema_DistShapeShape d(sa, sb);
    if (!d.IsDone() || d.NbSolution() < 1) { g_err = "distance could not be computed"; return 0; }
    gp_Pnt p1 = d.PointOnShape1(1), p2 = d.PointOnShape2(1);
    out[0] = d.Value();
    out[1] = p1.X(); out[2] = p1.Y(); out[3] = p1.Z();
    out[4] = p2.X(); out[5] = p2.Y(); out[6] = p2.Z();
    return 1;
    RF_CATCH(0)
}

extern "C" RfShape* rf_make_vertex(const double* p) {
    RF_TRY
    return wrap(BRepBuilderAPI_MakeVertex(gp_Pnt(p[0], p[1], p[2])).Vertex());
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_make_segment(const double* a, const double* b) {
    RF_TRY
    return wrap(BRepBuilderAPI_MakeEdge(gp_Pnt(a[0], a[1], a[2]), gp_Pnt(b[0], b[1], b[2])).Edge());
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_box(double x, double y, double z) {
    RF_TRY
    RfShape* r = wrap(BRepPrimAPI_MakeBox(x, y, z).Shape());
    if (r) {
        r->index();
        for (int i = 0; i < r->faces.Extent(); ++i) r->ftags.push_back(mix(0xB0, i));
    }
    return r;
    RF_CATCH(nullptr)
}

// ------------------------------------------------------------------ booleans

// Background modelling must not update topology shared with the display.
template <class Op>
static void build_readonly_boolean(Op& op, const TopoDS_Shape& a, const TopoDS_Shape& b) {
    TopTools_ListOfShape arguments, tools;
    arguments.Append(a); tools.Append(b);
    op.SetArguments(arguments); op.SetTools(tools);
    op.SetNonDestructive(true);
    op.Build();
}

// op: 0 = fuse (join), 1 = cut, 2 = common (intersect)
template <class Op>
static RfShape* finish_bool(RfShape* a, RfShape* b, Op& f, const char* what) {
    if (!f.IsDone() || f.HasErrors()) { g_err = std::string(what) + " failed"; return nullptr; }
    TagMap m = carry(a, f);
    TagMap mb = carry(b, f);
    for (TagMap::Iterator it(mb); it.More(); it.Next())
        if (!m.IsBound(it.Key())) m.Bind(it.Key(), it.Value());
    TopoDS_Shape out = clean_tracked(f.Shape(), m);
    RfShape* r = wrap(out);
    if (r) apply_tags(r, m);
    return r;
}

extern "C" RfShape* rf_boolean(RfShape* a, RfShape* b, int op) {
    RF_TRY
    if (op == 0) {
        BRepAlgoAPI_Fuse f; build_readonly_boolean(f,a->s,b->s);
        return finish_bool(a, b, f, "join");
    } else if (op == 1) {
        BRepAlgoAPI_Cut f; build_readonly_boolean(f,a->s,b->s);
        return finish_bool(a, b, f, "cut");
    } else {
        BRepAlgoAPI_Common f; build_readonly_boolean(f,a->s,b->s);
        return finish_bool(a, b, f, "intersect");
    }
    RF_CATCH(nullptr)
}

extern "C" int rf_touches(const RfShape* a, const RfShape* b) {
    RF_TRY
    Bnd_Box ba, bb;
    BRepBndLib::Add(a->s, ba);
    BRepBndLib::Add(b->s, bb);
    ba.Enlarge(1e-6);
    if (ba.IsOut(bb)) return 0;
    BRepExtrema_DistShapeShape d(a->s, b->s);
    if (d.IsDone() && d.Value() < 1e-6) return 1;
    // One inside the other: no boundary contact but a common volume.
    BRepAlgoAPI_Common c; build_readonly_boolean(c,a->s,b->s);
    if (!c.IsDone()) return 0;
    GProp_GProps g;
    BRepGProp::VolumeProperties(c.Shape(), g);
    return std::abs(g.Mass()) > 1e-9 ? 1 : 0;
    RF_CATCH(0)
}

extern "C" int rf_solid_count(const RfShape* s) {
    RF_TRY
    int n = 0;
    for (TopExp_Explorer ex(s->s, TopAbs_SOLID); ex.More(); ex.Next()) ++n;
    return n;
    RF_CATCH(0)
}

extern "C" RfShape* rf_solid_get(RfShape* s, int i) {
    RF_TRY
    int n = 0;
    for (TopExp_Explorer ex(s->s, TopAbs_SOLID); ex.More(); ex.Next(), ++n)
        if (n == i) {
            TagMap m;
            bind_faces(s, m);
            RfShape* r = wrap(ex.Current());
            if (r) apply_tags(r, m);
            return r;
        }
    g_err = "solid index out of range";
    return nullptr;
    RF_CATCH(nullptr)
}

extern "C" int rf_bbox(const RfShape* s, double* out) {
    RF_TRY
    Bnd_Box b;
    BRepBndLib::Add(s->s, b);
    if (b.IsVoid()) return 0;
    b.Get(out[0], out[1], out[2], out[3], out[4], out[5]);
    return 1;
    RF_CATCH(0)
}

extern "C" double rf_volume(const RfShape* s) {
    RF_TRY
    GProp_GProps g;
    BRepGProp::VolumeProperties(s->s, g);
    return g.Mass();
    RF_CATCH(0.0)
}

// ------------------------------------------------------------------ blends

// Copy transforms retain progenitor names while giving each instance its own
// namespace. Copy geometry so worker meshing never mutates the source body.
extern "C" RfShape* rf_transform(RfShape* s, int kind, const double* origin, const double* vector, double angle, uint64_t feat) {
    RF_TRY
    gp_Trsf tr;
    if (kind == 0) tr.SetTranslation(gp_Vec(vector[0], vector[1], vector[2]));
    else if (kind == 1) tr.SetRotation(gp_Ax1(gp_Pnt(origin[0], origin[1], origin[2]), gp_Dir(vector[0], vector[1], vector[2])), angle);
    else tr.SetMirror(gp_Ax2(gp_Pnt(origin[0], origin[1], origin[2]), gp_Dir(vector[0], vector[1], vector[2])));
    BRepBuilderAPI_Transform mk(s->s, tr, true);
    if (!mk.IsDone()) { g_err = "body transform failed"; return nullptr; }
    TagMap m = carry(s, mk);
    RfShape* r = wrap(mk.Shape());
    if (r) {
        apply_tags(r, m);
        if (feat) for (auto& tag : r->ftags) tag = mix3(feat, 7, tag);
    }
    return r;
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_shell(RfShape* s, const int* ids, int n, double thickness, uint64_t feat, int round_join) {
    RF_TRY
    s->index();
    TopTools_ListOfShape removed;
    for (int i=0; i<n; ++i) {
        if (ids[i] < 1 || ids[i] > s->faces.Extent()) { g_err = "bad shell face id"; return nullptr; }
        removed.Append(s->faces(ids[i]));
    }
    BRepOffsetAPI_MakeThickSolid mk;
    mk.MakeThickSolidByJoin(s->s, removed, thickness, 1e-6, BRepOffset_Skin, false, false, round_join ? GeomAbs_Arc : GeomAbs_Intersection);
    if (!mk.IsDone() || mk.Shape().IsNull() || !BRepCheck_Analyzer(mk.Shape()).IsValid()) { g_err = "shell failed (thickness may exceed available space)"; return nullptr; }
    TagMap m = carry(s, mk);
    for (int i=1; i<=s->faces.Extent(); ++i) {
        auto tag = i-1 < (int)s->ftags.size() ? s->ftags[i-1] : 0;
        for (TopTools_ListIteratorOfListOfShape it(mk.Generated(s->faces(i))); it.More(); it.Next())
            if (it.Value().ShapeType() == TopAbs_FACE && !m.IsBound(it.Value())) m.Bind(it.Value(), mix3(feat, 6, tag));
    }
    // With no removed faces OCCT returns the offset solid, rather than the
    // material between the original and offset shells. Build that material.
    if (n == 0) {
        TopoDS_Shape offsetShape = mk.Shape();
        if (offsetShape.ShapeType() == TopAbs_SOLID) {
            auto solid = TopoDS::Solid(offsetShape);
            if (!BRepLib::OrientClosedSolid(solid)) { g_err = "offset shell is not closed"; return nullptr; }
            offsetShape = solid;
        }
        GProp_GProps original, offset;
        BRepGProp::VolumeProperties(s->s, original);
        BRepGProp::VolumeProperties(offsetShape, offset);
        if (offset.Mass() <= 1e-9 || (thickness < 0 && offset.Mass() >= original.Mass()) || (thickness > 0 && offset.Mass() <= original.Mass())) { g_err = "shell thickness collapses the interior"; return nullptr; }
        auto* inner = wrap(offsetShape);
        if (!inner) return nullptr;
        inner->index();
        TopoDS_Compound sourceFaces, offsetFaces;
        BRep_Builder boundary;
        boundary.MakeCompound(sourceFaces); boundary.MakeCompound(offsetFaces);
        for (int i=1; i<=s->faces.Extent(); ++i) boundary.Add(sourceFaces,s->faces(i));
        for (int i=1; i<=inner->faces.Extent(); ++i) boundary.Add(offsetFaces,inner->faces(i));
        BRepExtrema_DistShapeShape wallDistance(sourceFaces,offsetFaces);
        if (!wallDistance.IsDone() || wallDistance.Value()+1e-5 < std::abs(thickness)) { delete inner; g_err = "shell thickness collapses or intersects the interior"; return nullptr; }
        apply_tags(inner,m);
        for (auto& t : inner->ftags) t = mix3(feat,6,t);
        BRepAlgoAPI_Cut cut;
        build_readonly_boolean(cut,thickness < 0 ? s->s : inner->s, thickness < 0 ? inner->s : s->s);
        auto* result = finish_bool(s,inner,cut,"closed shell");
        delete inner;
        return result;
    }
    auto* r = wrap(mk.Shape());
    if (r) apply_tags(r, m);
    return r;
    RF_CATCH(nullptr)
}

// Edge ids are 1-based indices into the shape's edge map (as returned by rf_mesh).
// Blend faces are named mix(feature, 4, edge tag) — ASM's blend_annotation_data.
template <class Mk>
static RfShape* finish_blend(RfShape* s, Mk& mk, const int* ids, int n, uint64_t feat, const uint64_t* etags) {
    TagMap m = carry(s, mk);
    for (int i = 0; i < n; ++i)
        for (TopTools_ListIteratorOfListOfShape it(mk.Generated(s->edges(ids[i]))); it.More(); it.Next())
            if (it.Value().ShapeType() == TopAbs_FACE && !m.IsBound(it.Value())) m.Bind(it.Value(), mix3(feat, 4, etags[i]));
    RfShape* r = wrap(mk.Shape());
    if (r) apply_tags(r, m);
    return r;
}

extern "C" RfShape* rf_fillet(RfShape* s, const int* ids, int n, double radius, uint64_t feat, const uint64_t* etags) {
    RF_TRY
    s->index();
    BRepFilletAPI_MakeFillet mk(s->s, ChFi3d_Rational);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 1 || ids[i] > s->edges.Extent()) { g_err = "bad edge id"; return nullptr; }
        mk.Add(radius, TopoDS::Edge(s->edges(ids[i])));
    }
    mk.Build();
    if (!mk.IsDone()) {
        g_err = "fillet failed (radius too large for the adjacent faces?)";
        return nullptr;
    }
    return finish_blend(s, mk, ids, n, feat, etags);
    RF_CATCH(nullptr)
}

// d2 < 0 means equal distance. Asymmetric chamfers measure d1 on the first adjacent face.
extern "C" RfShape* rf_chamfer(RfShape* s, const int* ids, int n, double d1, double d2, uint64_t feat, const uint64_t* etags) {
    RF_TRY
    s->index();
    TopTools_IndexedDataMapOfShapeListOfShape e2f;
    TopExp::MapShapesAndAncestors(s->s, TopAbs_EDGE, TopAbs_FACE, e2f);
    BRepFilletAPI_MakeChamfer mk(s->s);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 1 || ids[i] > s->edges.Extent()) { g_err = "bad edge id"; return nullptr; }
        const TopoDS_Edge& e = TopoDS::Edge(s->edges(ids[i]));
        if (d2 < 0) {
            mk.Add(d1, e);
        } else {
            const TopoDS_Face& f = TopoDS::Face(e2f.FindFromKey(e).First());
            mk.Add(d1, d2, e, f);
        }
    }
    mk.Build();
    if (!mk.IsDone()) {
        g_err = "chamfer failed (distance too large?)";
        return nullptr;
    }
    return finish_blend(s, mk, ids, n, feat, etags);
    RF_CATCH(nullptr)
}

// ------------------------------------------------------------------ queries

// out[0]=type (0 line, 1 circle, 2 other), out[1..4]=start, [4..7]=end, [7..10]=mid,
// [10..13]=tangent at mid, [13]=length, [14..17]=circle centre, [17..20]=axis,
// [20]=radius, [21]=closed
extern "C" int rf_edge_info(RfShape* s, int id, double* out) {
    RF_TRY
    s->index();
    if (id < 1 || id > s->edges.Extent()) return 0;
    const TopoDS_Edge& e = TopoDS::Edge(s->edges(id));
    if (BRep_Tool::Degenerated(e)) return 0;
    BRepAdaptor_Curve c(e);
    double u0 = c.FirstParameter(), u1 = c.LastParameter(), um = 0.5 * (u0 + u1);
    gp_Pnt p0 = c.Value(u0), p1 = c.Value(u1), pm;
    gp_Vec tm;
    c.D1(um, pm, tm);
    if (tm.Magnitude() > 1e-12) tm.Normalize();
    std::memset(out, 0, sizeof(double) * 22);
    out[0] = c.GetType() == GeomAbs_Line ? 0 : c.GetType() == GeomAbs_Circle ? 1 : 2;
    double v[] = {p0.X(), p0.Y(), p0.Z(), p1.X(), p1.Y(), p1.Z(), pm.X(), pm.Y(), pm.Z(), tm.X(), tm.Y(), tm.Z()};
    std::memcpy(out + 1, v, sizeof v);
    out[13] = GCPnts_AbscissaPoint::Length(c);
    if (c.GetType() == GeomAbs_Circle) {
        gp_Circ ci = c.Circle();
        out[14] = ci.Location().X(); out[15] = ci.Location().Y(); out[16] = ci.Location().Z();
        out[17] = ci.Axis().Direction().X(); out[18] = ci.Axis().Direction().Y(); out[19] = ci.Axis().Direction().Z();
        out[20] = ci.Radius();
    }
    out[21] = p0.Distance(p1) < 1e-9 ? 1 : 0;
    return 1;
    RF_CATCH(0)
}

// Sample an edge polyline; returns number of points written (<= max).
extern "C" int rf_edge_sample(RfShape* s, int id, double defl, double* out, int max) {
    RF_TRY
    s->index();
    if (id < 1 || id > s->edges.Extent()) return 0;
    BRepAdaptor_Curve c(TopoDS::Edge(s->edges(id)));
    GCPnts_TangentialDeflection td(c, 0.1, defl);
    int n = std::min(td.NbPoints(), max);
    for (int i = 0; i < n; ++i) {
        gp_Pnt p = td.Value(i + 1);
        out[3 * i] = p.X(); out[3 * i + 1] = p.Y(); out[3 * i + 2] = p.Z();
    }
    return n;
    RF_CATCH(0)
}

// out[0]=type (0 plane, 1 other), [1..4]=origin, [4..7]=outward normal, [7..10]=xdir,
// [10..13]=centroid, [13]=area
extern "C" int rf_face_info(RfShape* s, int id, double* out) {
    RF_TRY
    s->index();
    if (id < 1 || id > s->faces.Extent()) return 0;
    const TopoDS_Face& f = TopoDS::Face(s->faces(id));
    BRepAdaptor_Surface a(f);
    std::memset(out, 0, sizeof(double) * 14);
    GProp_GProps g;
    BRepGProp::SurfaceProperties(f, g);
    gp_Pnt cg = g.CentreOfMass();
    out[10] = cg.X(); out[11] = cg.Y(); out[12] = cg.Z();
    out[13] = g.Mass();
    if (a.GetType() == GeomAbs_Plane) {
        gp_Pln p = a.Plane();
        gp_Dir n = p.Axis().Direction();
        gp_Dir x = p.XAxis().Direction();
        if (f.Orientation() == TopAbs_REVERSED) { n.Reverse(); x.Reverse(); }
        out[0] = 0;
        out[1] = p.Location().X(); out[2] = p.Location().Y(); out[3] = p.Location().Z();
        out[4] = n.X(); out[5] = n.Y(); out[6] = n.Z();
        out[7] = x.X(); out[8] = x.Y(); out[9] = x.Z();
    } else {
        out[0] = 1;
    }
    return 1;
    RF_CATCH(0)
}

// ------------------------------------------------------------------ meshing

// clean != 0 discards existing triangulations first (so a coarser export
// setting is honoured even if the shape was meshed finer for display).
extern "C" RfMesh* rf_mesh(RfShape* s, double lin_defl, double ang_defl, int clean) {
    RF_TRY
    // Shape handles and their subshapes may alias across Rust Arc clones and
    // timeline states. Triangulation and normals belong to an owned deep copy.
    BRepBuilderAPI_Copy copy(s->s, true, false);
    RfShape isolated;
    isolated.s = copy.Shape();
    s = &isolated;
    s->index();
    if (clean) BRepTools::Clean(s->s);
    BRepMesh_IncrementalMesh(s->s, lin_defl, false, ang_defl, true);
    auto m = std::make_unique<RfMesh>();
    for (int fi = 1; fi <= s->faces.Extent(); ++fi) {
        const TopoDS_Face& f = TopoDS::Face(s->faces(fi));
        TopLoc_Location loc;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(f, loc);
        if (tri.IsNull()) continue;
        if (!tri->HasNormals()) BRepLib_ToolTriangulatedShape::ComputeNormals(f, tri);
        bool rev = f.Orientation() == TopAbs_REVERSED;
        gp_Trsf tr = loc.Transformation();
        uint32_t base = (uint32_t)(m->pos.size() / 3);
        for (int i = 1; i <= tri->NbNodes(); ++i) {
            gp_Pnt p = tri->Node(i).Transformed(tr);
            gp_Dir n = tri->Normal(i);
            gp_Vec nv(n);
            nv.Transform(tr);
            if (rev) nv.Reverse();
            m->pos.insert(m->pos.end(), {(float)p.X(), (float)p.Y(), (float)p.Z()});
            m->nrm.insert(m->nrm.end(), {(float)nv.X(), (float)nv.Y(), (float)nv.Z()});
        }
        for (int i = 1; i <= tri->NbTriangles(); ++i) {
            int a, b, c;
            tri->Triangle(i).Get(a, b, c);
            if (rev) std::swap(b, c);
            m->idx.insert(m->idx.end(), {base + a - 1, base + b - 1, base + c - 1});
            m->tri_face.push_back((uint32_t)fi);
        }
    }
    for (int ei = 1; ei <= s->edges.Extent(); ++ei) {
        m->edge_starts.push_back((uint32_t)(m->edge_pts.size() / 3));
        const TopoDS_Edge& e = TopoDS::Edge(s->edges(ei));
        if (BRep_Tool::Degenerated(e)) continue;
        BRepAdaptor_Curve c(e);
        GCPnts_TangentialDeflection td(c, ang_defl, lin_defl);
        for (int i = 1; i <= td.NbPoints(); ++i) {
            gp_Pnt p = td.Value(i);
            m->edge_pts.insert(m->edge_pts.end(), {(float)p.X(), (float)p.Y(), (float)p.Z()});
        }
    }
    m->edge_starts.push_back((uint32_t)(m->edge_pts.size() / 3));
    return m.release();
    RF_CATCH(nullptr)
}

extern "C" int rf_mesh_counts(const RfMesh* m, uint32_t* out) {
    RF_TRY
    const auto max = std::numeric_limits<uint32_t>::max();
    if (m->pos.size()/3 > max || m->idx.size()/3 > max || m->edge_pts.size()/3 > max || m->edge_starts.size() > max) {
        g_err = "Mesh exceeds the 32-bit buffer size limit";
        return 0;
    }
    out[0] = (uint32_t)(m->pos.size() / 3);
    out[1] = (uint32_t)(m->idx.size() / 3);
    out[2] = (uint32_t)(m->edge_pts.size() / 3);
    out[3] = (uint32_t)m->edge_starts.size();
    return 1;
    RF_CATCH(0)
}

extern "C" void rf_mesh_copy(const RfMesh* m, float* pos, float* nrm, uint32_t* idx, uint32_t* tri_face, float* edge_pts, uint32_t* edge_starts) {
    std::memcpy(pos, m->pos.data(), m->pos.size() * sizeof(float));
    std::memcpy(nrm, m->nrm.data(), m->nrm.size() * sizeof(float));
    std::memcpy(idx, m->idx.data(), m->idx.size() * sizeof(uint32_t));
    std::memcpy(tri_face, m->tri_face.data(), m->tri_face.size() * sizeof(uint32_t));
    std::memcpy(edge_pts, m->edge_pts.data(), m->edge_pts.size() * sizeof(float));
    std::memcpy(edge_starts, m->edge_starts.data(), m->edge_starts.size() * sizeof(uint32_t));
}

extern "C" void rf_mesh_free(RfMesh* m) { delete m; }

// ------------------------------------------------------------------ export

// ------------------------------------------------------------------ import

// Sew loose faces/shells into closed solids where possible.
static TopoDS_Shape solidify(const TopoDS_Shape& in) {
    int nsolids = 0;
    for (TopExp_Explorer ex(in, TopAbs_SOLID); ex.More(); ex.Next()) ++nsolids;
    if (nsolids > 0) return in;
    BRepBuilderAPI_Sewing sew(1e-4);
    sew.Add(in);
    sew.Perform();
    TopoDS_Shape sewn = sew.SewedShape();
    BRep_Builder b;
    TopoDS_Compound out;
    b.MakeCompound(out);
    int added = 0;
    for (TopExp_Explorer ex(sewn, TopAbs_SHELL); ex.More(); ex.Next()) {
        TopoDS_Shell sh = TopoDS::Shell(ex.Current());
        if (!BRep_Tool::IsClosed(sh)) { b.Add(out, sh); ++added; continue; }
        BRepBuilderAPI_MakeSolid ms(sh);
        if (ms.IsDone()) {
            ShapeFix_Solid fix(ms.Solid());
            fix.Perform();
            b.Add(out, fix.Solid());
        } else {
            b.Add(out, sh);
        }
        ++added;
    }
    return added ? TopoDS_Shape(out) : sewn;
}

extern "C" RfShape* rf_read_step(const char* path) {
    RF_TRY
    std::lock_guard<std::mutex> exchange(g_exchange_lock);
    STEPControl_Reader r;
    if (r.ReadFile(path) != IFSelect_RetDone) { g_err = "could not read STEP file"; return nullptr; }
    r.TransferRoots();
    TopoDS_Shape s = r.OneShape();  // STEP lengths are converted to millimetres
    if (s.IsNull()) { g_err = "STEP file contains no geometry"; return nullptr; }
    return wrap(solidify(s));
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_read_iges(const char* path) {
    RF_TRY
    std::lock_guard<std::mutex> exchange(g_exchange_lock);
    IGESControl_Reader r;
    if (r.ReadFile(path) != IFSelect_RetDone) { g_err = "could not read IGES file"; return nullptr; }
    r.TransferRoots();
    TopoDS_Shape s = r.OneShape();
    if (s.IsNull()) { g_err = "IGES file contains no geometry"; return nullptr; }
    return wrap(solidify(s));
    RF_CATCH(nullptr)
}

// Shape <-> OCCT BRep text (for embedding imported geometry in documents).
extern "C" char* rf_to_brep(const RfShape* s) {
    RF_TRY
    std::ostringstream os;
    BRepTools::Write(s->s, os);
    std::string str = os.str();
    char* out = (char*)std::malloc(str.size() + 1);
    std::memcpy(out, str.c_str(), str.size() + 1);
    return out;
    RF_CATCH(nullptr)
}

extern "C" void rf_free_str(char* p) { std::free(p); }

extern "C" RfShape* rf_from_brep(const char* text) {
    RF_TRY
    std::istringstream is(text);
    TopoDS_Shape s;
    BRep_Builder b;
    BRepTools::Read(s, is, b);
    if (s.IsNull()) { g_err = "invalid BRep data"; return nullptr; }
    return wrap(s);
    RF_CATCH(nullptr)
}

extern "C" RfShape* rf_with_face_tags(RfShape* s, const uint64_t* tags, int n) {
    RF_TRY
    // A fresh wrapper has independent naming/index data; immutable topology
    // remains shared. Never mutate a wrapper already published to Rust.
    auto r = std::make_unique<RfShape>();
    r->s = s->s;
    r->index();
    r->ftags.assign(tags, tags + n);
    r->etags = s->etags;
    return r.release();
    RF_CATCH(nullptr)
}


struct RfBuilder {
    std::vector<TopoDS_Vertex> verts;
    std::vector<Handle(Geom_Curve)> curves;
    std::vector<Handle(Geom_Surface)> surfs;
    std::vector<TopoDS_Edge> edges;
    std::vector<TopoDS_Face> faces;
    double tol = 1e-5;
};

extern "C" RfBuilder* rf_bld_new() {
    RF_TRY
    return new RfBuilder();
    RF_CATCH(nullptr)
}
extern "C" void rf_bld_free(RfBuilder* b) { delete b; }

extern "C" int rf_bld_vertex(RfBuilder* b, const double* p) {
    RF_TRY
    BRep_Builder bb;
    TopoDS_Vertex v;
    bb.MakeVertex(v, gp_Pnt(p[0], p[1], p[2]), b->tol);
    b->verts.push_back(v);
    return (int)b->verts.size() - 1;
    RF_CATCH(-1)
}

static int push_curve(RfBuilder* b, const Handle(Geom_Curve)& c) {
    b->curves.push_back(c);
    return (int)b->curves.size() - 1;
}
static int push_surf(RfBuilder* b, const Handle(Geom_Surface)& s) {
    b->surfs.push_back(s);
    return (int)b->surfs.size() - 1;
}

extern "C" int rf_bld_line(RfBuilder* b, const double* p, const double* d) {
    RF_TRY
    return push_curve(b, new Geom_Line(gp_Pnt(p[0], p[1], p[2]), gp_Dir(d[0], d[1], d[2])));
    RF_CATCH(-1)
}

// Circle (ratio == 1) or ellipse; `major` is the major-axis vector.
extern "C" int rf_bld_ellipse(RfBuilder* b, const double* c, const double* n, const double* major, double ratio) {
    RF_TRY
    gp_Vec mj(major[0], major[1], major[2]);
    double R = mj.Magnitude();
    gp_Ax2 ax(gp_Pnt(c[0], c[1], c[2]), gp_Dir(n[0], n[1], n[2]), gp_Dir(mj));
    if (std::abs(ratio - 1.0) < 1e-12) return push_curve(b, new Geom_Circle(ax, R));
    if (ratio > 1.0) {  // keep major >= minor for OCCT
        gp_Ax2 ax2(ax.Location(), ax.Direction(), ax.YDirection());
        return push_curve(b, new Geom_Ellipse(ax2, R * ratio, R));
    }
    return push_curve(b, new Geom_Ellipse(ax, R, R * ratio));
    RF_CATCH(-1)
}

// Knots with multiplicities already in OCCT form (clamped ends = degree+1).
// poles: xyz per pole (weights separate, may be null).
extern "C" int rf_bld_bspline_curve(RfBuilder* b, int degree, int nk, const double* knots, const int* mults,
                                    int np, const double* poles, const double* weights, int periodic) {
    RF_TRY
    TColgp_Array1OfPnt P(1, np);
    for (int i = 0; i < np; ++i) P(i + 1) = gp_Pnt(poles[3 * i], poles[3 * i + 1], poles[3 * i + 2]);
    TColStd_Array1OfReal K(1, nk);
    TColStd_Array1OfInteger M(1, nk);
    for (int i = 0; i < nk; ++i) { K(i + 1) = knots[i]; M(i + 1) = mults[i]; }
    if (weights) {
        TColStd_Array1OfReal W(1, np);
        for (int i = 0; i < np; ++i) W(i + 1) = weights[i];
        return push_curve(b, new Geom_BSplineCurve(P, W, K, M, degree, periodic != 0));
    }
    return push_curve(b, new Geom_BSplineCurve(P, K, M, degree, periodic != 0));
    RF_CATCH(-1)
}

extern "C" int rf_bld_plane(RfBuilder* b, const double* p, const double* n, const double* u) {
    RF_TRY
    return push_surf(b, new Geom_Plane(gp_Ax3(gp_Pnt(p[0], p[1], p[2]), gp_Dir(n[0], n[1], n[2]), gp_Dir(u[0], u[1], u[2]))));
    RF_CATCH(-1)
}

// Cone/cylinder: base circle (centre, axis, major vector), sin/cos of half angle.
// The radius grows along the axis by tan(half angle).
extern "C" int rf_bld_cone(RfBuilder* b, const double* c, const double* n, const double* major, double sina, double cosa) {
    RF_TRY
    gp_Vec mj(major[0], major[1], major[2]);
    double R = mj.Magnitude();
    gp_Ax3 ax(gp_Pnt(c[0], c[1], c[2]), gp_Dir(n[0], n[1], n[2]), gp_Dir(mj));
    if (std::abs(sina) < 1e-12) return push_surf(b, new Geom_CylindricalSurface(ax, R));
    double ang = std::atan2(sina, std::abs(cosa));
    return push_surf(b, new Geom_ConicalSurface(ax, ang, R));
    RF_CATCH(-1)
}

extern "C" int rf_bld_sphere(RfBuilder* b, const double* c, double r, const double* u, const double* pole) {
    RF_TRY
    return push_surf(b, new Geom_SphericalSurface(gp_Ax3(gp_Pnt(c[0], c[1], c[2]), gp_Dir(pole[0], pole[1], pole[2]), gp_Dir(u[0], u[1], u[2])), std::abs(r)));
    RF_CATCH(-1)
}

extern "C" int rf_bld_torus(RfBuilder* b, const double* c, const double* n, double R, double r, const double* u) {
    RF_TRY
    return push_surf(b, new Geom_ToroidalSurface(gp_Ax3(gp_Pnt(c[0], c[1], c[2]), gp_Dir(n[0], n[1], n[2]), gp_Dir(u[0], u[1], u[2])), std::abs(R), std::abs(r)));
    RF_CATCH(-1)
}

// poles row-major [iu][iv], xyz each; weights same layout or null.
extern "C" int rf_bld_bspline_surface(RfBuilder* b, int du, int dv, int nku, const double* ku, const int* mu, int nkv,
                                      const double* kv, const int* mv, int nu, int nv, const double* poles,
                                      const double* weights, int uper, int vper) {
    RF_TRY
    TColgp_Array2OfPnt P(1, nu, 1, nv);
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j) {
            const double* q = poles + 3 * (i * nv + j);
            P(i + 1, j + 1) = gp_Pnt(q[0], q[1], q[2]);
        }
    TColStd_Array1OfReal KU(1, nku), KV(1, nkv);
    TColStd_Array1OfInteger MU(1, nku), MV(1, nkv);
    for (int i = 0; i < nku; ++i) { KU(i + 1) = ku[i]; MU(i + 1) = mu[i]; }
    for (int i = 0; i < nkv; ++i) { KV(i + 1) = kv[i]; MV(i + 1) = mv[i]; }
    if (weights) {
        TColStd_Array2OfReal W(1, nu, 1, nv);
        for (int i = 0; i < nu; ++i)
            for (int j = 0; j < nv; ++j) W(i + 1, j + 1) = weights[i * nv + j];
        return push_surf(b, new Geom_BSplineSurface(P, W, KU, KV, MU, MV, du, dv, uper != 0, vper != 0));
    }
    return push_surf(b, new Geom_BSplineSurface(P, KU, KV, MU, MV, du, dv, uper != 0, vper != 0));
    RF_CATCH(-1)
}

extern "C" int rf_bld_extrusion(RfBuilder* b, int curve, const double* d) {
    RF_TRY
    if (curve < 0 || curve >= (int)b->curves.size()) return -1;
    return push_surf(b, new Geom_SurfaceOfLinearExtrusion(b->curves[curve], gp_Dir(d[0], d[1], d[2])));
    RF_CATCH(-1)
}

static double project_param(const Handle(Geom_Curve)& c, const gp_Pnt& p) {
    GeomAPI_ProjectPointOnCurve pr(p, c);
    if (pr.NbPoints() > 0) return pr.LowerDistanceParameter();
    // fall back to the nearer end
    return p.Distance(c->Value(c->FirstParameter())) < p.Distance(c->Value(c->LastParameter())) ? c->FirstParameter() : c->LastParameter();
}

// Edge from vertex vs to ve along curve (or a straight segment if curve < 0).
// reversed: the edge runs against the curve direction.
extern "C" int rf_bld_edge(RfBuilder* b, int curve, int vs, int ve, int reversed) {
    RF_TRY
    TopoDS_Vertex V1 = b->verts.at(vs), V2 = b->verts.at(ve);
    gp_Pnt P1 = BRep_Tool::Pnt(V1), P2 = BRep_Tool::Pnt(V2);
    TopoDS_Edge E;
    if (curve < 0) {
        E = BRepBuilderAPI_MakeEdge(V1, V2);
    } else {
        Handle(Geom_Curve) c = b->curves.at(curve);
        bool closed_edge = vs == ve || P1.Distance(P2) < b->tol * 10;
        double per = c->IsPeriodic() ? c->Period() : 0.0;
        double a, z;
        TopoDS_Vertex Va = V1, Vz = V2;
        if (closed_edge) {
            a = c->IsPeriodic() ? project_param(c, P1) : c->FirstParameter();
            z = c->IsPeriodic() ? a + per : c->LastParameter();
        } else {
            double t1 = project_param(c, P1), t2 = project_param(c, P2);
            if (!reversed) { a = t1; z = t2; }
            else { a = t2; z = t1; Va = V2; Vz = V1; }
            if (per > 0) { while (z <= a + 1e-12) z += per; while (z - a > per + 1e-9) z -= per; }
        }
        BRepBuilderAPI_MakeEdge me(c, Va, Vz, a, z);
        if (!me.IsDone()) {
            // tolerance mismatch: rebuild from parameters with fresh vertices
            BRepBuilderAPI_MakeEdge me2(c, a, z);
            if (!me2.IsDone()) { g_err = "edge construction failed"; return -1; }
            E = me2.Edge();
        } else {
            E = me.Edge();
        }
        if (reversed && !closed_edge) E.Reverse();
        else if (reversed && closed_edge) E.Reverse();
    }
    b->edges.push_back(E);
    return (int)b->edges.size() - 1;
    RF_CATCH(-1)
}

// A face: surface (or -1 to fill from the boundary), sense, loops of
// (edge id, coedge reversed) pairs following ACIS coedge order.
extern "C" int rf_bld_face(RfBuilder* b, int surf, int reversed, int nloops, const int* loop_lens, const int* coedges) {
    RF_TRY
    BRep_Builder bb;
    std::vector<TopoDS_Wire> wires;
    const int* q = coedges;
    for (int l = 0; l < nloops; ++l) {
        TopoDS_Wire w;
        bb.MakeWire(w);
        for (int k = 0; k < loop_lens[l]; ++k, q += 2) {
            TopoDS_Edge e = b->edges.at(q[0]);
            bb.Add(w, q[1] ? TopoDS::Edge(e.Reversed()) : e);
        }
        w.Closed(true);
        wires.push_back(w);
    }
    TopoDS_Face F;
    if (surf < 0) {
        // Unsupported surface type: fill a surface through the outer boundary.
        BRepOffsetAPI_MakeFilling fill;
        for (TopExp_Explorer ex(wires.at(0), TopAbs_EDGE); ex.More(); ex.Next()) fill.Add(TopoDS::Edge(ex.Current()), GeomAbs_C0);
        fill.Build();
        if (!fill.IsDone()) { g_err = "filling failed"; return -1; }
        F = TopoDS::Face(fill.Shape());
    } else {
        bb.MakeFace(F, b->surfs.at(surf), b->tol);
        for (auto& w : wires) bb.Add(F, reversed ? TopoDS::Wire(w.Reversed()) : w);
        if (reversed) F.Reverse();
        // Planes need no stored pcurves or seams; repairing 40k mesh-like
        // planar faces dominated import time, so only repair curved faces.
        if (!b->surfs.at(surf)->IsKind(STANDARD_TYPE(Geom_Plane))) {
            ShapeFix_Face fix(F);
            fix.SetPrecision(b->tol);
            fix.FixMissingSeamMode() = 1;
            fix.Perform();
            F = fix.Face();
        }
    }
    b->faces.push_back(F);
    return (int)b->faces.size() - 1;
    RF_CATCH(-1)
}

// Shell from faces → solid (when closed), transformed by trsf (3x4 row-major).
extern "C" RfShape* rf_bld_body(RfBuilder* b, int n, const int* face_ids, const double* trsf) {
    RF_TRY
    // Face repair (pcurves, seams) can replace edges, so faces no longer
    // share them: sew to restore shared topology, then close into a solid.
    // Fast path: faces that still share edges already form a closed shell.
    TopoDS_Shape sewn;
    {
        BRep_Builder bb;
        TopoDS_Shell sh;
        bb.MakeShell(sh);
        for (int i = 0; i < n; ++i) bb.Add(sh, b->faces.at(face_ids[i]));
        if (BRep_Tool::IsClosed(sh)) {
            sh.Closed(true);
            sewn = sh;
        }
    }
    if (sewn.IsNull()) {
        BRepBuilderAPI_Sewing sew(b->tol * 10);
        for (int i = 0; i < n; ++i) sew.Add(b->faces.at(face_ids[i]));
        sew.Perform();
        sewn = sew.SewedShape();
    }
    // Shared-topology planar imports (including dense faceted bodies) need no
    // sewing or global repair when the constructed solid is already valid.
    if (sewn.ShapeType() == TopAbs_SHELL && BRep_Tool::IsClosed(TopoDS::Shell(sewn))) {
        BRepBuilderAPI_MakeSolid solid(TopoDS::Shell(sewn));
        if (solid.IsDone()) {
            TopoDS_Solid candidate = solid.Solid();
            if (BRepLib::OrientClosedSolid(candidate) && BRepCheck_Analyzer(candidate, false).IsValid()) {
                TopoDS_Shape ready = candidate;
                if (trsf) {
                    gp_Trsf t;
                    t.SetValues(trsf[0], trsf[1], trsf[2], trsf[3], trsf[4], trsf[5], trsf[6], trsf[7], trsf[8], trsf[9], trsf[10], trsf[11]);
                    ready = BRepBuilderAPI_Transform(ready, t, true).Shape();
                }
                // The builder stays usable and owns mutable repair inputs.
                // Published shapes must never share those inputs with it.
                return wrap(trsf ? ready : BRepBuilderAPI_Copy(ready, true, false).Shape());
            }
        }
    }
    TopoDS_Shape result = sewn;
    ShapeFix_Shell fsh;
    TopoDS_Shape fixed = sewn;
    for (TopExp_Explorer ex(sewn, TopAbs_SHELL); ex.More(); ex.Next()) {
        fsh.Init(TopoDS::Shell(ex.Current()));
        fsh.Perform();
        fixed = fsh.Shape();
        break;
    }
    for (TopExp_Explorer ex(fixed, TopAbs_SHELL); ex.More(); ex.Next()) {
        TopoDS_Shell s1 = TopoDS::Shell(ex.Current());
        if (BRep_Tool::IsClosed(s1)) {
            BRepBuilderAPI_MakeSolid ms(s1);
            if (ms.IsDone()) {
                ShapeFix_Solid fs(ms.Solid());
                fs.Perform();
                result = fs.Solid();
                GProp_GProps g;
                BRepGProp::VolumeProperties(result, g);
                if (g.Mass() < 0) result.Reverse();
            }
        } else {
            result = fixed;
        }
        break;
    }
    if (n < 5000 || !BRepCheck_Analyzer(result, false).IsValid()) {
        ShapeFix_Shape fall(result);
        fall.Perform();
        result = fall.Shape();
    }
    if (trsf) {
        gp_Trsf t;
        t.SetValues(trsf[0], trsf[1], trsf[2], trsf[3], trsf[4], trsf[5], trsf[6], trsf[7], trsf[8], trsf[9], trsf[10], trsf[11]);
        result = BRepBuilderAPI_Transform(result, t, true).Shape();
    }
    return wrap(trsf ? result : BRepBuilderAPI_Copy(result, true, false).Shape());
    RF_CATCH(nullptr)
}

// 1 if the shape passes OCCT's validity check.
extern "C" int rf_is_valid(const RfShape* s) {
    RF_TRY
    BRepCheck_Analyzer a(s->s);
    return a.IsValid() ? 1 : 0;
    RF_CATCH(0)
}

extern "C" int rf_is_solid(const RfShape* s) {
    RF_TRY
    for (TopExp_Explorer ex(s->s, TopAbs_SOLID); ex.More(); ex.Next()) return 1;
    return 0;
    RF_CATCH(0)
}

extern "C" int rf_write_step(const RfShape* const* shapes, int n, const char* path) {
    RF_TRY
    std::lock_guard<std::mutex> exchange(g_exchange_lock);
    STEPControl_Writer w;
    Interface_Static::SetCVal("write.step.unit", "MM");
    for (int i = 0; i < n; ++i)
        if (w.Transfer(shapes[i]->s, STEPControl_AsIs) != IFSelect_RetDone) {
            g_err = "STEP transfer failed";
            return 0;
        }
    if (w.Write(path) != IFSelect_RetDone) {
        g_err = "could not write STEP file";
        return 0;
    }
    return 1;
    RF_CATCH(0)
}

