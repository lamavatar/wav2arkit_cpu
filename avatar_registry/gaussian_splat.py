"""3D Gaussian Splatting renderer for LAM per-person avatars (moderngl).

Renders the real LAM 3D Gaussian avatar from its asset pack:
``offset.ply`` (3DGS attributes) + ``skin.glb`` (neutral mesh + 51 ARKit morph
targets) + ``bsData.json`` (per-frame ARKit weights).

Authoritative recipe (from LAM_WebRender):
    p_i = neutral_i + offset_i + sum_k w_k * morphDelta_k_i      (one Gaussian per vertex)
    color_i   = f_dc_i            (degree-0 SH used directly as RGB)
    opacity_i = sigmoid(opacity)
    scale_i   = exp(scale)
    rot_i     = normalize(quat rot_0..3)   # rot_0 = w
Covariance is kept static in the canonical frame (matches the reference runtime).

Rendering is EWA splatting: project each Gaussian mean + 2D screen covariance,
sort back-to-front, draw as instanced quads with a Gaussian-weighted alpha and
premultiplied "over" blending.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from avatar_registry.glb_loader import _load_glb, _read_accessor, _indices_to_triangles  # noqa: E402

SH_C0 = 0.28209479177387814


# --------------------------------------------------------------------------- #
# Asset loading
# --------------------------------------------------------------------------- #
def _parse_ply(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    he = raw.index(marker) + len(marker)
    header = raw[:he].decode("ascii", errors="replace")
    count = 0
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])
    body = np.frombuffer(raw[he:], dtype="<f4", count=count * len(props)).reshape(count, len(props))
    return {p: body[:, i].astype(np.float32) for i, p in enumerate(props)}


def _load_skin_raw(glb_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load skin.glb POSITION + morph deltas in RAW mesh-local space (no node transform).

    The offset.ply offsets are expressed in this raw space, so we must NOT apply
    the flattening node/world transform used by the 2D pipeline.
    """
    gltf = _load_glb(glb_path)
    mesh = gltf.json["meshes"][0]
    primitive = mesh["primitives"][0]
    target_names = (mesh.get("extras") or {}).get("targetNames") or []

    neutral = _read_accessor(gltf, primitive["attributes"]["POSITION"]).astype(np.float32)
    indices = _read_accessor(gltf, primitive["indices"], fill_sparse=False)
    triangles = _indices_to_triangles(indices)

    morph_deltas: dict[str, np.ndarray] = {}
    for target, name in zip(primitive.get("targets", []), target_names):
        if "POSITION" not in target:
            continue
        delta = _read_accessor(gltf, target["POSITION"])
        if delta.ndim == 1:
            delta = delta.reshape(-1, 3)
        morph_deltas[name] = delta.astype(np.float32)
    return neutral, triangles, morph_deltas


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """(N,4) quaternion (w,x,y,z) -> (N,3,3) rotation matrices."""
    q = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    r = np.empty((n, 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """Single glTF quaternion [x,y,z,w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    nrm = np.sqrt(x * x + y * y + z * z + w * w)
    if nrm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / nrm, y / nrm, z / nrm, w / nrm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _trs_matrix(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = _quat_xyzw_to_mat(q) * np.asarray(s, np.float64)[None, :]
    m[:3, 3] = t
    return m


class Skeleton:
    """glTF skeleton + animation evaluator. Computes the head-bone skin matrix.

    Only the head joint drives the Gaussian centers (per LAM_WebRender), so we
    evaluate just the node chain from the root down to ``head`` each frame.
    """

    def __init__(self, glb_path: Path, head_name: str = "head"):
        gltf = _load_glb(glb_path)
        js = gltf.json
        self.nodes = js["nodes"]
        n = len(self.nodes)

        self.parent = [-1] * n
        for i, nd in enumerate(self.nodes):
            for c in nd.get("children", []) or []:
                self.parent[c] = i

        self.rest_T = np.zeros((n, 3))
        self.rest_R = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        self.rest_S = np.ones((n, 3))
        self.rest_M = [np.eye(4) for _ in range(n)]
        for i, nd in enumerate(self.nodes):
            if nd.get("matrix"):
                self.rest_M[i] = np.asarray(nd["matrix"], np.float64).reshape(4, 4).T
            else:
                t = np.asarray(nd.get("translation", [0, 0, 0]), np.float64)
                r = np.asarray(nd.get("rotation", [0, 0, 0, 1]), np.float64)
                s = np.asarray(nd.get("scale", [1, 1, 1]), np.float64)
                self.rest_T[i] = t
                self.rest_R[i] = r
                self.rest_S[i] = s
                self.rest_M[i] = _trs_matrix(t, r, s)

        skin = js["skins"][0]
        self.joints = skin["joints"]
        self.joint_index = {node: k for k, node in enumerate(self.joints)}
        ibm = _read_accessor(gltf, skin["inverseBindMatrices"]).reshape(-1, 4, 4)
        self.inv_bind = np.transpose(ibm, (0, 2, 1)).astype(np.float64)  # column-major -> row

        self.head_node = next(
            (i for i, nd in enumerate(self.nodes) if nd.get("name") == head_name), None
        )
        self.head_chain: list[int] = []
        if self.head_node is not None:
            i = self.head_node
            while i != -1:
                self.head_chain.append(i)
                i = self.parent[i]

        self.animations = []
        for a in js.get("animations", []):
            channels: dict[int, dict[str, tuple]] = {}
            duration = 0.0
            for ch in a["channels"]:
                node = ch["target"]["node"]
                path = ch["target"]["path"]
                samp = a["samplers"][ch["sampler"]]
                tin = _read_accessor(gltf, samp["input"]).reshape(-1).astype(np.float64)
                vout = _read_accessor(gltf, samp["output"]).astype(np.float64)
                channels.setdefault(node, {})[path] = (tin, vout, samp.get("interpolation", "LINEAR"))
                if tin.size:
                    duration = max(duration, float(tin[-1]))
            self.animations.append({"name": a.get("name", "?"), "channels": channels, "duration": duration})
        self.anim_names = [a["name"] for a in self.animations]

    @staticmethod
    def _sample(channel: tuple, tau: float, is_quat: bool) -> np.ndarray:
        tin, vout, interp = channel
        vals = vout[1::3] if interp == "CUBICSPLINE" else vout  # cubic: keep value, drop tangents
        if tau <= tin[0]:
            return vals[0]
        if tau >= tin[-1]:
            return vals[-1]
        i1 = int(np.searchsorted(tin, tau))
        i0 = i1 - 1
        if interp == "STEP":
            return vals[i0]
        span = tin[i1] - tin[i0]
        f = 0.0 if span <= 0 else (tau - tin[i0]) / span
        a, b = vals[i0], vals[i1]
        if is_quat:
            if np.dot(a, b) < 0:
                b = -b
            q = a * (1 - f) + b * f
            return q / max(np.linalg.norm(q), 1e-12)
        return a * (1 - f) + b * f

    def _local_matrix(self, node: int, channels: dict, tau: float) -> np.ndarray:
        ch = channels.get(node)
        if not ch:
            return self.rest_M[node]
        t = self.rest_T[node].copy()
        r = self.rest_R[node].copy()
        s = self.rest_S[node].copy()
        if "translation" in ch:
            t = self._sample(ch["translation"], tau, False)
        if "rotation" in ch:
            r = self._sample(ch["rotation"], tau, True)
        if "scale" in ch:
            s = self._sample(ch["scale"], tau, False)
        return _trs_matrix(t, r, s)

    def head_skin_matrix(self, anim_index: int, tau: float) -> np.ndarray:
        """4x4 matrix mapping bind-pose mesh points through the animated head bone."""
        if self.head_node is None or not self.animations:
            return np.eye(4, dtype=np.float32)
        channels = self.animations[anim_index]["channels"]
        gmat = np.eye(4)
        for node in reversed(self.head_chain):
            gmat = gmat @ self._local_matrix(node, channels, tau)
        inv = self.inv_bind[self.joint_index[self.head_node]]
        return (gmat @ inv).astype(np.float32)

    def anim_index(self, name: str) -> int:
        if name in self.anim_names:
            return self.anim_names.index(name)
        raise ValueError(f"animation '{name}' not found; available: {self.anim_names}")


class GaussianAvatar:
    """Parsed, render-ready Gaussian avatar (canonical-frame attributes)."""

    def __init__(self, avatar_dir: str | Path, *, sh_color: bool = False, min_opacity: float = 0.0):
        avatar_dir = Path(avatar_dir)
        if (avatar_dir / "arkitWithBSData").is_dir():
            avatar_dir = avatar_dir / "arkitWithBSData"
        self.dir = avatar_dir

        ply = _parse_ply(avatar_dir / "offset.ply")
        self.offset = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
        fdc = np.stack([ply["f_dc_0"], ply["f_dc_1"], ply["f_dc_2"]], axis=1).astype(np.float32)
        self.color = np.clip(0.5 + SH_C0 * fdc if sh_color else fdc, 0.0, 1.0)
        self.opacity = (1.0 / (1.0 + np.exp(-ply["opacity"]))).astype(np.float32)
        scale = np.exp(np.stack([ply["scale_0"], ply["scale_1"], ply["scale_2"]], axis=1)).astype(np.float32)
        quat = np.stack([ply["rot_0"], ply["rot_1"], ply["rot_2"], ply["rot_3"]], axis=1).astype(np.float32)

        self.neutral, self.triangles, self.morph_deltas = _load_skin_raw(avatar_dir / "skin.glb")

        rot = _quat_to_rotmat(quat)
        s2 = scale[:, :, None] * np.eye(3, dtype=np.float32)[None] * scale[:, None, :]
        self.cov3d = (rot @ s2 @ rot.transpose(0, 2, 1)).astype(np.float32)  # (N,3,3) world-frame

        if min_opacity > 0.0:
            keep = self.opacity >= min_opacity
            self.offset = self.offset[keep]
            self.color = self.color[keep]
            self.opacity = self.opacity[keep]
            self.cov3d = self.cov3d[keep]
            self.neutral = self.neutral[keep]
            self.morph_deltas = {k: v[keep] for k, v in self.morph_deltas.items()}

        self.num = self.offset.shape[0]
        self.skeleton: Skeleton | None = None

    def load_skeleton(self) -> Skeleton:
        """Lazily load the skeleton/animations from animation.glb."""
        if self.skeleton is None:
            self.skeleton = Skeleton(self.dir / "animation.glb")
        return self.skeleton

    def positions(
        self,
        weights: np.ndarray | None = None,
        morph_names: list[str] | None = None,
        *,
        head_matrix: np.ndarray | None = None,
        idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gaussian world positions for given ARKit weights (None -> neutral).

        If ``head_matrix`` (4x4) is given, the rigid head-bone transform is applied
        after morph deformation (matches LAM_WebRender splat-center skinning).
        If ``idx`` is given, only that subset of Gaussians is deformed/returned.
        """
        base = self.neutral + self.offset
        pos = base if idx is None else base[idx]
        if weights is not None and morph_names is not None:
            w = np.asarray(weights, dtype=np.float32)
            for k, name in enumerate(morph_names):
                if k >= w.shape[0] or w[k] == 0.0:
                    continue
                delta = self.morph_deltas.get(name)
                if delta is not None:
                    pos = pos + w[k] * (delta if idx is None else delta[idx])
        if head_matrix is not None:
            pos = pos @ head_matrix[:3, :3].T + head_matrix[:3, 3]
        return pos.astype(np.float32)

    def dynamic_mask(
        self,
        weights_seq: np.ndarray,
        morph_names: list[str],
        *,
        rel_threshold: float = 0.02,
    ) -> np.ndarray:
        """Boolean mask of Gaussians that move across an ARKit weight sequence.

        A Gaussian is "dynamic" if its peak per-frame displacement (relative to the
        sequence-mean pose) exceeds ``rel_threshold`` of the global peak. For a
        lip-sync clip this isolates the mouth/jaw region (plus any active blinks).
        """
        weights_seq = np.asarray(weights_seq, np.float32)
        if weights_seq.ndim == 1:
            weights_seq = weights_seq[None]
        present = []
        for k, name in enumerate(morph_names):
            if k >= weights_seq.shape[1]:
                break
            delta = self.morph_deltas.get(name)
            if delta is not None and float(np.ptp(weights_seq[:, k])) > 1e-6:
                present.append((k, delta))

        disp_max = np.zeros(self.num, np.float32)
        if not present:
            return disp_max > 1.0  # nothing moves
        wmean = weights_seq.mean(0)
        for f in range(weights_seq.shape[0]):
            dvec = np.zeros((self.num, 3), np.float32)
            for k, delta in present:
                wc = float(weights_seq[f, k] - wmean[k])
                if wc != 0.0:
                    dvec += wc * delta
            disp = np.sqrt((dvec * dvec).sum(1))
            np.maximum(disp_max, disp, out=disp_max)
        thr = rel_threshold * float(disp_max.max())
        return disp_max > max(thr, 1e-9)


# --------------------------------------------------------------------------- #
# Camera math
# --------------------------------------------------------------------------- #
def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenGL-style view matrix (camera looks down -Z)."""
    eye = np.asarray(eye, np.float64)
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -s @ eye
    m[1, 3] = -u @ eye
    m[2, 3] = f @ eye
    return m


def auto_camera(positions: np.ndarray, *, fovy_deg: float = 30.0, margin: float = 1.35,
                yaw_deg: float = 0.0, pitch_deg: float = 0.0):
    """Frame the avatar from the front (+Z). Returns (eye, center, up, fovy_rad)."""
    center = positions.mean(axis=0).astype(np.float64)
    extent = (positions.max(axis=0) - positions.min(axis=0)).astype(np.float64)
    half = float(max(extent[0], extent[1])) * 0.5 * margin
    fovy = np.radians(fovy_deg)
    dist = half / np.tan(fovy * 0.5)

    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    dir_cam = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
    ])
    eye = center + dir_cam * dist
    up = np.array([0.0, 1.0, 0.0])
    return eye, center, up, fovy


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
VERTEX_SHADER = """#version 330
in vec2 corner;
in vec2 i_center;
in float i_radius;
in vec3 i_conic;
in vec3 i_color;
in float i_alpha;
uniform vec2 viewport;
out vec2 v_d;
out vec3 v_conic;
out vec3 v_color;
out float v_alpha;
void main() {
    vec2 off = corner * i_radius;
    vec2 pix = i_center + off;
    vec2 ndc = vec2(pix.x / viewport.x * 2.0 - 1.0, 1.0 - pix.y / viewport.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_d = off;
    v_conic = i_conic;
    v_color = i_color;
    v_alpha = i_alpha;
}
"""

FRAGMENT_SHADER = """#version 330
in vec2 v_d;
in vec3 v_conic;
in vec3 v_color;
in float v_alpha;
out vec4 f_color;
void main() {
    float power = -0.5 * (v_conic.x * v_d.x * v_d.x
                        + 2.0 * v_conic.y * v_d.x * v_d.y
                        + v_conic.z * v_d.y * v_d.y);
    if (power > 0.0) discard;
    float a = v_alpha * exp(power);
    if (a < 0.00392) discard;
    f_color = vec4(v_color * a, a);  // premultiplied
}
"""


class GaussianSplatRenderer:
    backend = "gl"

    def __init__(self, avatar: GaussianAvatar, *, width: int = 1024, height: int = 1024):
        import os
        import moderngl

        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")

        self.avatar = avatar
        self.width = int(width)
        self.height = int(height)

        try:
            self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context(require=330)

        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.prog["viewport"] = (float(self.width), float(self.height))

        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype="f4")
        self.corner_vbo = self.ctx.buffer(corners.tobytes())

        # instance buffer: center(2) radius(1) conic(3) color(3) alpha(1) = 10 floats
        self.inst_vbo = self.ctx.buffer(reserve=avatar.num * 10 * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.corner_vbo, "2f", "corner"),
                (self.inst_vbo, "2f 1f 3f 3f 1f/i", "i_center", "i_radius", "i_conic", "i_color", "i_alpha"),
            ],
        )

        self.color_tex = self.ctx.texture((self.width, self.height), 4, dtype="f1")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex])
        self._fbo_cache: dict[tuple[int, int], tuple] = {}

    def _project_instances(self, means, cov3d, color, opacity, fovy, cx, cy, vw, vh):
        """Project a Gaussian set into instance attributes for a (vw x vh) viewport.

        ``cx, cy`` are the principal point in viewport pixels (top-left origin).
        The focal length is derived from the FULL image height so that the world
        scale is identical regardless of crop/viewport size.
        """
        Rv = self._view_R          # (3,3) float32, set by caller
        tv = self._view_t          # (3,) float32
        t = means @ Rv.T + tv      # view-space (N,3)
        zc = t[:, 2]
        front = zc < -1e-4
        zp = np.where(front, -zc, 1.0).astype(np.float32)
        inv = (1.0 / zp).astype(np.float32)

        fy = np.float32((self.height * 0.5) / np.tan(fovy * 0.5))
        fx = fy
        sx = np.float32(cx) + fx * t[:, 0] * inv
        sy = np.float32(cy) - fy * t[:, 1] * inv

        # M = J @ Rv  (N,2,3) without materializing the zero-filled J tensor
        tx_inv2 = t[:, 0] * inv * inv
        ty_inv2 = t[:, 1] * inv * inv
        n = means.shape[0]
        M = np.empty((n, 2, 3), np.float32)
        M[:, 0, :] = (fx * inv)[:, None] * Rv[0][None, :] + (fx * tx_inv2)[:, None] * Rv[2][None, :]
        M[:, 1, :] = (-fy * inv)[:, None] * Rv[1][None, :] + (-fy * ty_inv2)[:, None] * Rv[2][None, :]

        # cov2d = M Σ M^T  (N,2,2)
        MS = M @ cov3d
        a = np.einsum("nk,nk->n", MS[:, 0, :], M[:, 0, :]) + 0.3
        b = np.einsum("nk,nk->n", MS[:, 0, :], M[:, 1, :])
        c = np.einsum("nk,nk->n", MS[:, 1, :], M[:, 1, :]) + 0.3

        det = a * c - b * b
        valid = front & (det > 1e-9)
        det_safe = np.where(valid, det, 1.0).astype(np.float32)
        inv_det = (1.0 / det_safe).astype(np.float32)

        mid = 0.5 * (a + c)
        lam = mid + np.sqrt(np.maximum(mid * mid - det, 0.0))
        radius = (3.0 * np.sqrt(np.maximum(lam, 1e-8))).astype(np.float32)
        valid &= (sx + radius >= 0) & (sx - radius <= vw) & (sy + radius >= 0) & (sy - radius <= vh)

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return None
        order = idx[np.argsort(zc[idx], kind="stable")]  # far first -> back-to-front

        inst = np.empty((order.size, 10), np.float32)
        inst[:, 0] = sx[order]
        inst[:, 1] = sy[order]
        inst[:, 2] = radius[order]
        inst[:, 3] = c[order] * inv_det[order]
        inst[:, 4] = -b[order] * inv_det[order]
        inst[:, 5] = a[order] * inv_det[order]
        inst[:, 6:9] = color[order]
        inst[:, 9] = opacity[order]
        return inst

    def _project_points(self, pts, fovy):
        """Screen (x, y) in full-image pixels for the current view (no covariance)."""
        Rv = self._view_R
        tv = self._view_t
        t = pts @ Rv.T + tv
        zc = t[:, 2]
        zp = np.where(zc < -1e-4, -zc, 1.0)
        inv = 1.0 / zp
        fy = (self.height * 0.5) / np.tan(fovy * 0.5)
        sx = self.width * 0.5 + fy * t[:, 0] * inv
        sy = self.height * 0.5 - fy * t[:, 1] * inv
        return sx, sy

    def _get_fbo(self, w: int, h: int):
        key = (int(w), int(h))
        if key not in self._fbo_cache:
            tex = self.ctx.texture(key, 4, dtype="f1")
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._fbo_cache[key] = (tex, fbo)
        return self._fbo_cache[key][1]

    def _draw_premult(self, inst, fbo, vw, vh):
        """Rasterize instances into ``fbo`` and read back premultiplied float RGBA."""
        import moderngl

        self.prog["viewport"] = (float(vw), float(vh))
        fbo.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        out = np.zeros((int(vh), int(vw), 4), np.float32)
        if inst is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self.inst_vbo.orphan(inst.nbytes)
            self.inst_vbo.write(inst.tobytes())
            self.vao.render(moderngl.TRIANGLES, instances=inst.shape[0])
            data = fbo.read(components=4, dtype="f1")
            out = np.frombuffer(data, np.uint8).reshape(int(vh), int(vw), 4).astype(np.float32) / 255.0
            out = np.flipud(out)
        return np.ascontiguousarray(out)

    def fixed_camera(self, camera: dict | None = None) -> tuple[np.ndarray, float]:
        """Compute a stable (view, fovy) from the neutral pose, for sequence rendering."""
        cam = camera or {}
        eye, center, up, fovy = auto_camera(self.avatar.positions(), **cam)
        return look_at(eye, center, up), fovy

    def render(
        self,
        weights: np.ndarray | None = None,
        morph_names: list[str] | None = None,
        *,
        background: tuple[int, int, int] = (255, 255, 255),
        camera: dict | None = None,
        head_matrix: np.ndarray | None = None,
        view: np.ndarray | None = None,
        fovy: float | None = None,
    ) -> np.ndarray:
        self._means = self.avatar.positions(weights, morph_names, head_matrix=head_matrix)
        if view is None or fovy is None:
            eye, center, up, fovy = auto_camera(self._means, **(camera or {}))
            view = look_at(eye, center, up)

        self._view_R = view[:3, :3].astype(np.float32)
        self._view_t = view[:3, 3].astype(np.float32)

        av = self.avatar
        W, H = self.width, self.height
        inst = self._project_instances(
            self._means, av.cov3d, av.color, av.opacity, fovy, W * 0.5, H * 0.5, W, H
        )
        rendered = self._draw_premult(inst, self.fbo, W, H)

        # composite premultiplied over background
        bg = np.array(background, np.float32) / 255.0
        alpha = rendered[:, :, 3:4]
        rgb = rendered[:, :, :3] + (1.0 - alpha) * bg[None, None, :]
        return np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

    def release(self):
        for tex, fbo in self._fbo_cache.values():
            for obj in (fbo, tex):
                try:
                    obj.release()
                except Exception:
                    pass
        for obj in (self.vao, self.inst_vbo, self.corner_vbo, self.color_tex, self.fbo, self.prog, self.ctx):
            try:
                obj.release()
            except Exception:
                pass


def _load_bsdata(path: Path):
    from export_utils import ARKIT_BLENDSHAPE_NAMES

    bs = json.loads(path.read_text())
    names = bs.get("names") or bs.get("metadata", {}).get("blendshape_names") or ARKIT_BLENDSHAPE_NAMES
    fps = float(bs.get("metadata", {}).get("fps", 30.0))
    weights = np.asarray([f["weights"] for f in bs["frames"]], np.float32)
    return names, fps, weights


def _encode_video(frames_dir: Path, video_path: Path, fps: float) -> bool:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-framerate", f"{fps:g}",
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _render_sequence(renderer, avatar, args, bg) -> int:
    import time

    from PIL import Image

    names, fps, weights = _load_bsdata(avatar.dir / "bsData.json")
    fps = args.fps if args.fps > 0 else fps
    if args.max_frames > 0:
        weights = weights[: args.max_frames]
    weights = weights[:: max(1, args.frame_step)]
    n_frames = weights.shape[0]

    # Fixed camera from neutral pose -> stable video.
    view, fovy = renderer.fixed_camera({"yaw_deg": args.yaw, "pitch_deg": args.pitch})

    skel = None
    anim_idx = 0
    clip_dur = 0.0
    if args.animation:
        skel = avatar.load_skeleton()
        anim_idx = skel.anim_index(args.animation)
        clip_dur = skel.animations[anim_idx]["duration"]

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    times: list[float] = []
    print(
        f"backend={renderer.backend} num_gaussians={avatar.num} size={args.size} "
        f"frames={n_frames} fps={fps:g} head_bone={args.animation or 'off'}"
    )
    for i in range(n_frames):
        head_mat = None
        if skel is not None:
            tau = (i / fps) % clip_dur if clip_dur > 0 else 0.0
            head_mat = skel.head_skin_matrix(anim_idx, tau)

        t0 = time.perf_counter()
        img = renderer.render(
            weights[i], names, background=bg, head_matrix=head_mat, view=view, fovy=fovy
        )
        Image.fromarray(img).save(frames_dir / f"frame_{i:05d}.png")
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)

        avg = sum(times) / len(times)
        print(
            f"  frame {i + 1:04d}/{n_frames}  render={ms:7.1f} ms  "
            f"(avg {avg:6.1f} ms, {1000.0 / avg:5.1f} fps)",
            flush=True,
        )

    arr = np.array(times)
    print(
        f"\nsummary: frames={n_frames}  mean={arr.mean():.1f} ms  median={np.median(arr):.1f} ms  "
        f"min={arr.min():.1f}  max={arr.max():.1f}  ~{1000.0 / arr.mean():.1f} fps  (render only)"
    )

    if args.video:
        ok = _encode_video(frames_dir, Path(args.video), fps)
        print(f"video: {'saved ' + args.video if ok else 'ffmpeg unavailable; PNG frames only'}")
    print(f"frames: {frames_dir}")
    renderer.release()
    return 0


def _project_screen(pts, Rv, tv, fovy, W, H):
    """Full-image pixel coords (x, y) for points under a view (GL-free)."""
    t = pts @ Rv.T + tv
    zc = t[:, 2]
    zp = np.where(zc < -1e-4, -zc, 1.0)
    inv = 1.0 / zp
    fy = (H * 0.5) / np.tan(fovy * 0.5)
    sx = W * 0.5 + fy * t[:, 0] * inv
    sy = H * 0.5 - fy * t[:, 1] * inv
    return sx, sy


def _crop_bbox_np(avatar, weights, names, dyn_idx, Rv, tv, fovy, W, H, *, margin=0.18):
    """GL-free mouth crop bbox over the full motion range of the dynamic set."""
    n = weights.shape[0]
    step = max(1, n // 240)
    sample = sorted({*range(0, n, step), n - 1})
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for f in sample:
        pts = avatar.positions(weights[f], names, idx=dyn_idx)
        sx, sy = _project_screen(pts, Rv, tv, fovy, W, H)
        x_min = min(x_min, float(sx.min())); x_max = max(x_max, float(sx.max()))
        y_min = min(y_min, float(sy.min())); y_max = max(y_max, float(sy.max()))
    mx = margin * (x_max - x_min) + 6.0
    my = margin * (y_max - y_min) + 6.0
    x0 = int(max(0, np.floor(x_min - mx)))
    y0 = int(max(0, np.floor(y_min - my)))
    x1 = int(min(W, np.ceil(x_max + mx)))
    y1 = int(min(H, np.ceil(y_max + my)))
    cw = max(2, x1 - x0); cw += cw % 2; cw = min(cw, W - x0)
    ch = max(2, y1 - y0); ch += ch % 2; ch = min(ch, H - y0)
    return x0, y0, cw, ch


def _mouth_crop_bbox(renderer, avatar, weights, names, dyn_idx, fovy, *, margin=0.18):
    """Screen-space bbox (x0, y0, w, h) covering the dynamic Gaussians over the clip."""
    W, H = renderer.width, renderer.height
    n = weights.shape[0]
    # Sample every frame (cheap, one-time) so the crop encloses the full motion
    # range of all dynamic Gaussians -> no holes after they are pulled from base.
    step = max(1, n // 240)
    sample = sorted({*range(0, n, step), n - 1})
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for f in sample:
        pts = avatar.positions(weights[f], names, idx=dyn_idx)
        sx, sy = renderer._project_points(pts, fovy)
        x_min = min(x_min, float(sx.min())); x_max = max(x_max, float(sx.max()))
        y_min = min(y_min, float(sy.min())); y_max = max(y_max, float(sy.max()))
    mx = margin * (x_max - x_min) + 6.0
    my = margin * (y_max - y_min) + 6.0
    x0 = int(max(0, np.floor(x_min - mx)))
    y0 = int(max(0, np.floor(y_min - my)))
    x1 = int(min(W, np.ceil(x_max + mx)))
    y1 = int(min(H, np.ceil(y_max + my)))
    cw = max(2, x1 - x0); cw += cw % 2; cw = min(cw, W - x0)
    ch = max(2, y1 - y0); ch += ch % 2; ch = min(ch, H - y0)
    return x0, y0, cw, ch


def _render_sequence_hybrid(renderer, avatar, args, bg) -> int:
    """Static full-face base rendered once + per-frame mouth-only patch composite."""
    import time

    from PIL import Image

    names, fps, weights = _load_bsdata(avatar.dir / "bsData.json")
    fps = args.fps if args.fps > 0 else fps
    if args.max_frames > 0:
        weights = weights[: args.max_frames]
    weights = weights[:: max(1, args.frame_step)]
    n_frames = weights.shape[0]

    if args.animation:
        print("note: --mouth-only assumes a static head; ignoring --animation (head-bone).")

    view, fovy = renderer.fixed_camera({"yaw_deg": args.yaw, "pitch_deg": args.pitch})
    renderer._view_R = view[:3, :3].astype(np.float32)
    renderer._view_t = view[:3, 3].astype(np.float32)

    # 1) dynamic (moving) Gaussian set for this clip.
    dyn_mask = avatar.dynamic_mask(weights, names, rel_threshold=args.dyn_threshold)
    dyn_idx = np.flatnonzero(dyn_mask)
    static_idx = np.flatnonzero(~dyn_mask)
    if dyn_idx.size == 0:
        print("no dynamic Gaussians detected; falling back to full sequence render.")
        return _render_sequence(renderer, avatar, args, bg)
    print(f"dynamic gaussians: {dyn_idx.size}/{avatar.num} ({100.0 * dyn_idx.size / avatar.num:.1f}%)")

    # 2) mouth crop bbox from the moving Gaussians.
    x0, y0, cw, ch = _mouth_crop_bbox(renderer, avatar, weights, names, dyn_idx, fovy,
                                      margin=args.crop_margin)
    print(f"mouth crop: origin=({x0},{y0}) size={cw}x{ch} "
          f"({100.0 * cw * ch / (renderer.width * renderer.height):.1f}% of frame)")

    # 3) static base: render the non-moving Gaussians once, composite over bg.
    W, H = renderer.width, renderer.height
    t_base = time.perf_counter()
    base_means = avatar.positions(idx=static_idx)
    base_inst = renderer._project_instances(
        base_means, avatar.cov3d[static_idx], avatar.color[static_idx],
        avatar.opacity[static_idx], fovy, W * 0.5, H * 0.5, W, H,
    )
    base_premult = renderer._draw_premult(base_inst, renderer.fbo, W, H)
    bgf = np.array(bg, np.float32) / 255.0
    base_a = base_premult[:, :, 3:4]
    base_rgb = np.clip(base_premult[:, :, :3] + (1.0 - base_a) * bgf[None, None, :], 0.0, 1.0)
    print(f"static base rendered once in {(time.perf_counter() - t_base) * 1000.0:.1f} ms")

    # Pre-slice dynamic attributes (constant across frames).
    dyn_cov = np.ascontiguousarray(avatar.cov3d[dyn_idx])
    dyn_col = np.ascontiguousarray(avatar.color[dyn_idx])
    dyn_op = np.ascontiguousarray(avatar.opacity[dyn_idx])
    cx = W * 0.5 - x0
    cy = H * 0.5 - y0
    patch_fbo = renderer._get_fbo(cw, ch)

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    times: list[float] = []
    print(
        f"backend={renderer.backend} mode=mouth-only num_gaussians={avatar.num} "
        f"patch={cw}x{ch} frames={n_frames} fps={fps:g}"
    )
    for i in range(n_frames):
        t0 = time.perf_counter()
        means = avatar.positions(weights[i], names, idx=dyn_idx)
        inst = renderer._project_instances(means, dyn_cov, dyn_col, dyn_op, fovy, cx, cy, cw, ch)
        patch = renderer._draw_premult(inst, patch_fbo, cw, ch)
        out = base_rgb.copy()
        region = out[y0:y0 + ch, x0:x0 + cw]
        pa = patch[:, :, 3:4]
        out[y0:y0 + ch, x0:x0 + cw] = patch[:, :, :3] + (1.0 - pa) * region
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)

        img = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(img)).save(frames_dir / f"frame_{i:05d}.png")

        avg = sum(times) / len(times)
        print(
            f"  frame {i + 1:04d}/{n_frames}  render={ms:7.1f} ms  "
            f"(avg {avg:6.1f} ms, {1000.0 / avg:5.1f} fps)",
            flush=True,
        )

    arr = np.array(times)
    print(
        f"\nsummary[mouth-only]: frames={n_frames}  mean={arr.mean():.1f} ms  "
        f"median={np.median(arr):.1f} ms  min={arr.min():.1f}  max={arr.max():.1f}  "
        f"~{1000.0 / arr.mean():.1f} fps  (patch+composite, save excluded)"
    )
    if args.video:
        ok = _encode_video(frames_dir, Path(args.video), fps)
        print(f"video: {'saved ' + args.video if ok else 'ffmpeg unavailable; PNG frames only'}")
    print(f"frames: {frames_dir}")
    renderer.release()
    return 0


# --------------------------------------------------------------------------- #
# Parallel (multi-process) sequence rendering
# --------------------------------------------------------------------------- #
# Frames are independent under a fixed camera, so they are split across worker
# processes. Each worker owns its own EGL/moderngl context; llvmpipe (software
# GL) already multi-threads a single draw, so we cap LP_NUM_THREADS per worker
# to keep total threads ~= core count and avoid oversubscription.
_CTX: dict = {}      # set in parent before Pool; inherited by fork children
_WORKER: dict = {}   # built once per worker process by the initializer


def _seq_worker_init():
    import os

    ctx = _CTX
    lp = ctx["lp_threads"]
    for var in ("LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(lp)

    avatar = ctx["avatar"]
    args = ctx["args"]
    view = ctx["view"]
    fovy = ctx["fovy"]
    renderer = GaussianSplatRenderer(avatar, width=args.size, height=args.size)
    renderer._view_R = view[:3, :3].astype(np.float32)
    renderer._view_t = view[:3, 3].astype(np.float32)

    w = {
        "r": renderer, "avatar": avatar, "args": args, "bg": ctx["bg"],
        "view": view, "fovy": fovy, "names": ctx["names"],
        "weights": ctx["weights"], "fps": ctx["fps"], "mode": ctx["mode"],
    }
    if ctx["mode"] == "hybrid":
        h = ctx["hybrid"]
        dyn_idx = h["dyn_idx"]
        static_idx = h["static_idx"]
        x0, y0, cw, ch = h["crop"]
        W, H = renderer.width, renderer.height
        # Static base rendered once per worker (GL only ever runs in the child).
        base_means = avatar.positions(idx=static_idx)
        binst = renderer._project_instances(base_means, avatar.cov3d[static_idx],
                                            avatar.color[static_idx], avatar.opacity[static_idx],
                                            fovy, W * 0.5, H * 0.5, W, H)
        bp = renderer._draw_premult(binst, renderer.fbo, W, H)
        bgf = np.array(ctx["bg"], np.float32) / 255.0
        ba = bp[:, :, 3:4]
        base_rgb = np.clip(bp[:, :, :3] + (1.0 - ba) * bgf[None, None, :], 0.0, 1.0)
        w.update(
            base_rgb=base_rgb, dyn_idx=dyn_idx, crop=h["crop"],
            dyn_cov=np.ascontiguousarray(avatar.cov3d[dyn_idx]),
            dyn_col=np.ascontiguousarray(avatar.color[dyn_idx]),
            dyn_op=np.ascontiguousarray(avatar.opacity[dyn_idx]),
            cx=renderer.width * 0.5 - x0, cy=renderer.height * 0.5 - y0,
            patch_fbo=renderer._get_fbo(cw, ch),
        )
    if args.animation and ctx["mode"] == "full":
        skel = avatar.load_skeleton()
        ai = skel.anim_index(args.animation)
        w.update(skel=skel, anim_idx=ai, clip_dur=skel.animations[ai]["duration"])

    global _WORKER
    _WORKER = w


def _seq_worker_render(i: int):
    import time

    from PIL import Image

    w = _WORKER
    r = w["r"]
    t0 = time.perf_counter()
    if w["mode"] == "hybrid":
        x0, y0, cw, ch = w["crop"]
        means = w["avatar"].positions(w["weights"][i], w["names"], idx=w["dyn_idx"])
        inst = r._project_instances(means, w["dyn_cov"], w["dyn_col"], w["dyn_op"],
                                    w["fovy"], w["cx"], w["cy"], cw, ch)
        patch = r._draw_premult(inst, w["patch_fbo"], cw, ch)
        out = w["base_rgb"].copy()
        region = out[y0:y0 + ch, x0:x0 + cw]
        pa = patch[:, :, 3:4]
        out[y0:y0 + ch, x0:x0 + cw] = patch[:, :, :3] + (1.0 - pa) * region
        img = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        head_mat = None
        if "skel" in w:
            cd = w["clip_dur"]
            tau = (i / w["fps"]) % cd if cd > 0 else 0.0
            head_mat = w["skel"].head_skin_matrix(w["anim_idx"], tau)
        img = r.render(w["weights"][i], w["names"], background=w["bg"],
                       head_matrix=head_mat, view=w["view"], fovy=w["fovy"])
    ms = (time.perf_counter() - t0) * 1000.0

    Image.fromarray(np.ascontiguousarray(img)).save(Path(w["args"].frames_dir) / f"frame_{i:05d}.png")
    return i, ms


def _render_sequence_parallel(avatar, args, bg) -> int:
    import multiprocessing as mp
    import os
    import time

    names, fps, weights = _load_bsdata(avatar.dir / "bsData.json")
    fps = args.fps if args.fps > 0 else fps
    if args.max_frames > 0:
        weights = weights[: args.max_frames]
    weights = weights[:: max(1, args.frame_step)]
    n_frames = weights.shape[0]

    # IMPORTANT: never touch GL/EGL in the parent process. EGL + llvmpipe driver
    # state is not fork-safe, so initializing a context here and then forking
    # deadlocks the children. Camera/crop are computed with pure NumPy; every GL
    # context (and the static base) is created inside the worker processes.
    W = H = int(args.size)
    eye, center, up, fovy = auto_camera(
        avatar.positions(), yaw_deg=args.yaw, pitch_deg=args.pitch
    )
    view = look_at(eye, center, up)
    Rv = view[:3, :3].astype(np.float32)
    tv = view[:3, 3].astype(np.float32)

    mode = "full"
    hybrid = None
    if args.mouth_only:
        if args.animation:
            print("note: --mouth-only assumes a static head; ignoring --animation.")
        dyn_mask = avatar.dynamic_mask(weights, names, rel_threshold=args.dyn_threshold)
        dyn_idx = np.flatnonzero(dyn_mask)
        static_idx = np.flatnonzero(~dyn_mask)
        if dyn_idx.size == 0:
            print("no dynamic Gaussians; using full-frame parallel render.")
        else:
            x0, y0, cw, ch = _crop_bbox_np(avatar, weights, names, dyn_idx, Rv, tv, fovy,
                                           W, H, margin=args.crop_margin)
            mode = "hybrid"
            hybrid = {"dyn_idx": dyn_idx, "static_idx": static_idx, "crop": (x0, y0, cw, ch)}
            print(f"dynamic gaussians: {dyn_idx.size}/{avatar.num} "
                  f"({100.0 * dyn_idx.size / avatar.num:.1f}%)  crop={cw}x{ch}@({x0},{y0})")

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, args.workers)
    lp_threads = max(1, (os.cpu_count() or 4) // workers)

    global _CTX
    _CTX = {
        "avatar": avatar, "args": args, "bg": bg, "view": view, "fovy": fovy,
        "names": names, "weights": weights, "fps": fps, "mode": mode,
        "hybrid": hybrid, "lp_threads": lp_threads,
    }

    print(
        f"backend=gl mode={mode} workers={workers} (lp_threads/worker={lp_threads}) "
        f"num_gaussians={avatar.num} size={args.size} frames={n_frames} fps={fps:g}"
    )

    times: dict[int, float] = {}
    ctxmp = mp.get_context("fork")
    t_all = time.perf_counter()
    with ctxmp.Pool(workers, initializer=_seq_worker_init) as pool:
        for i, ms in pool.imap_unordered(_seq_worker_render, range(n_frames), chunksize=1):
            times[i] = ms
            done = len(times)
            wall = time.perf_counter() - t_all
            print(
                f"  done {done:04d}/{n_frames}  frame={i:04d}  compute={ms:7.1f} ms  "
                f"(wall {wall:5.1f}s, {done / wall:5.1f} fps eff)",
                flush=True,
            )
    wall = time.perf_counter() - t_all

    arr = np.array([times[k] for k in sorted(times)])
    print(
        f"\nsummary[parallel x{workers} {mode}]: frames={n_frames}  "
        f"per-frame compute mean={arr.mean():.1f} ms median={np.median(arr):.1f} ms  | "
        f"wall={wall:.1f}s  throughput={n_frames / wall:.1f} fps  "
        f"(speedup vs 1 worker measured separately)"
    )
    if args.video:
        ok = _encode_video(frames_dir, Path(args.video), fps)
        print(f"video: {'saved ' + args.video if ok else 'ffmpeg unavailable; PNG frames only'}")
    print(f"frames: {frames_dir}")
    return 0


def _main() -> int:
    import argparse
    import time

    from PIL import Image

    from export_utils import ARKIT_BLENDSHAPE_NAMES

    parser = argparse.ArgumentParser(description="Render LAM 3D Gaussian avatar")
    parser.add_argument("--avatar", default=str(ROOT / "avatar_registry" / "lam3d_avatar" / "vfhq_case1"))
    parser.add_argument("--morph", default="", help="single ARKit morph to apply (e.g. jawOpen)")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--bsdata-frame", type=int, default=-1, help="render a specific bsData frame index")
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sh-color", action="store_true")
    parser.add_argument("--min-opacity", type=float, default=0.0, help="prune Gaussians below this opacity (speed)")
    parser.add_argument("--bg", default="255,255,255")
    parser.add_argument("--output", default=str(ROOT / "avatar_registry" / "output" / "gs_neutral.png"))
    # sequence mode
    parser.add_argument("--sequence", action="store_true", help="render the full bsData.json sequence")
    parser.add_argument("--fps", type=float, default=0.0, help="override playback fps (0 = use bsData metadata)")
    parser.add_argument("--max-frames", type=int, default=0, help="limit sequence frames (0 = all)")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--animation", default="", help="head-bone animation clip name (e.g. yumi_h5_a3_speak)")
    parser.add_argument("--frames-dir", default=str(ROOT / "avatar_registry" / "output" / "gs_seq_frames"))
    parser.add_argument("--video", default="", help="optional mp4 output path (needs ffmpeg)")
    parser.add_argument("--list-animations", action="store_true")
    # mouth-only hybrid mode (static base + dynamic mouth patch)
    parser.add_argument("--mouth-only", action="store_true",
                        help="hybrid render: static full-face base + per-frame mouth patch composite")
    parser.add_argument("--dyn-threshold", type=float, default=0.02,
                        help="dynamic-Gaussian displacement threshold (fraction of peak motion)")
    parser.add_argument("--crop-margin", type=float, default=0.18,
                        help="extra margin around the mouth crop bbox (fraction of bbox size)")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes for sequence rendering (1 = serial)")
    args = parser.parse_args()

    bg = tuple(int(c) for c in args.bg.split(","))
    avatar = GaussianAvatar(args.avatar, sh_color=args.sh_color, min_opacity=args.min_opacity)

    if args.list_animations:
        print("animations:", avatar.load_skeleton().anim_names)
        return 0

    if args.sequence and args.workers > 1:
        return _render_sequence_parallel(avatar, args, bg)

    renderer = GaussianSplatRenderer(avatar, width=args.size, height=args.size)

    if args.sequence:
        if args.mouth_only:
            return _render_sequence_hybrid(renderer, avatar, args, bg)
        return _render_sequence(renderer, avatar, args, bg)

    weights = None
    morph_names = ARKIT_BLENDSHAPE_NAMES
    head_mat = None
    if args.bsdata_frame >= 0:
        names, _, all_w = _load_bsdata(avatar.dir / "bsData.json")
        weights = all_w[args.bsdata_frame]
        morph_names = names
    elif args.morph:
        weights = np.zeros(len(ARKIT_BLENDSHAPE_NAMES), np.float32)
        weights[ARKIT_BLENDSHAPE_NAMES.index(args.morph)] = args.weight
    if args.animation:
        skel = avatar.load_skeleton()
        head_mat = skel.head_skin_matrix(skel.anim_index(args.animation), 0.0)

    cam = {"yaw_deg": args.yaw, "pitch_deg": args.pitch}
    t0 = time.perf_counter()
    img = renderer.render(weights, morph_names, background=bg, camera=cam, head_matrix=head_mat)
    ms = (time.perf_counter() - t0) * 1000

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    renderer.release()
    print(f"backend={renderer.backend} num_gaussians={avatar.num}\n  render_ms={ms:.1f}  output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
