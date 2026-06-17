"""Single draw-call mesh renderer for 2D avatar packs.

Replaces the per-triangle ``cv2.getAffineTransform`` + ``cv2.remap`` warp
(39,904 calls/frame) in :mod:`avatar_registry.warp` with one indexed draw of the
whole textured mesh. Shared vertices map to identical positions, so triangle
seams disappear and interpolation is continuous.

Primary backend is OpenGL via ``moderngl`` (standalone EGL context; works
headless on Mesa/llvmpipe). If a GL context cannot be created, a seam-free
NumPy barycentric rasterizer is used instead.

Data contract (matches :func:`avatar_registry.warp.render_mesh_frame`):
- ``neutral_2d``  : (N, 2) source/UV sample coords in texture pixels.
- ``deformed_2d`` : (N, 2) destination vertex positions in image pixels.
- ``triangles``   : (T, 3) int vertex indices.
- ``texture``     : (H, W, 3) uint8 RGB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VERTEX_SHADER = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

FRAGMENT_SHADER = """#version 330
uniform sampler2D tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(tex, v_uv).rgb, 1.0);
}
"""


def _normalized_uv(neutral_2d: np.ndarray, width: int, height: int, v_flip: bool) -> np.ndarray:
    """neutral pixel coords -> [0,1] UV (optionally flipping v)."""
    uv = np.empty((neutral_2d.shape[0], 2), dtype=np.float32)
    uv[:, 0] = neutral_2d[:, 0] / float(width)
    uv[:, 1] = neutral_2d[:, 1] / float(height)
    if v_flip:
        uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def _to_ndc(positions: np.ndarray, width: int, height: int) -> np.ndarray:
    """image pixel coords (origin top-left) -> OpenGL NDC."""
    ndc = np.empty((positions.shape[0], 2), dtype=np.float32)
    ndc[:, 0] = positions[:, 0] / float(width) * 2.0 - 1.0
    ndc[:, 1] = 1.0 - positions[:, 1] / float(height) * 2.0
    return ndc


def _composite(warped: np.ndarray, photo: np.ndarray, face_mask: np.ndarray | None) -> np.ndarray:
    """Feathered blend of warped face over the original photo (reuses warp logic)."""
    from avatar_registry.warp import _composite_with_mask

    return _composite_with_mask(warped, np.asarray(photo), face_mask)


class GLMeshRenderer:
    """OpenGL single-draw-call renderer. Context/texture/buffers set up once."""

    backend = "gl"

    def __init__(
        self,
        texture: np.ndarray,
        neutral_2d: np.ndarray,
        triangles: np.ndarray,
        *,
        v_flip: bool = False,
        flip_output: bool = True,
    ) -> None:
        import moderngl

        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")

        self.height = int(texture.shape[0])
        self.width = int(texture.shape[1])
        self.num_vertices = int(neutral_2d.shape[0])
        self.flip_output = flip_output

        try:
            self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
        except Exception:
            self.ctx = moderngl.create_standalone_context(require=330)

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )

        tex_rgb = np.ascontiguousarray(texture[..., :3].astype(np.uint8))
        self.tex = self.ctx.texture((self.width, self.height), 3, tex_rgb.tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = False
        self.tex.repeat_y = False
        self.prog["tex"] = 0

        uv = _normalized_uv(neutral_2d.astype(np.float32), self.width, self.height, v_flip)
        self.uv_vbo = self.ctx.buffer(np.ascontiguousarray(uv).tobytes())

        self.pos_vbo = self.ctx.buffer(reserve=self.num_vertices * 2 * 4, dynamic=True)

        idx = np.ascontiguousarray(triangles.astype("u4").ravel())
        self.ibo = self.ctx.buffer(idx.tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.pos_vbo, "2f", "in_pos"), (self.uv_vbo, "2f", "in_uv")],
            index_buffer=self.ibo,
            index_element_size=4,
        )

        self.color = self.ctx.texture((self.width, self.height), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color])

    def render(self, deformed_2d: np.ndarray, *, background: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        ndc = _to_ndc(np.asarray(deformed_2d, dtype=np.float32), self.width, self.height)
        self.pos_vbo.write(np.ascontiguousarray(ndc).tobytes())

        self.fbo.use()
        import moderngl

        self.ctx.disable(moderngl.DEPTH_TEST)
        bg = tuple(float(c) / 255.0 for c in background)
        self.ctx.clear(bg[0], bg[1], bg[2], 0.0)
        self.tex.use(0)
        self.vao.render(moderngl.TRIANGLES)

        data = self.fbo.read(components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        if self.flip_output:
            img = np.flipud(img)
        return np.ascontiguousarray(img)

    def render_composite(
        self,
        deformed_2d: np.ndarray,
        photo: np.ndarray,
        face_mask: np.ndarray | None,
        *,
        background: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        warped = self.render(deformed_2d, background=background)
        return _composite(warped, photo, face_mask)

    def release(self) -> None:
        for obj in (
            self.vao,
            self.ibo,
            self.uv_vbo,
            self.pos_vbo,
            self.tex,
            self.color,
            self.fbo,
            self.prog,
            self.ctx,
        ):
            try:
                obj.release()
            except Exception:
                pass


class NumpyMeshRenderer:
    """Seam-free CPU fallback: single-pass barycentric mesh rasterizer."""

    backend = "numpy"

    def __init__(
        self,
        texture: np.ndarray,
        neutral_2d: np.ndarray,
        triangles: np.ndarray,
        *,
        v_flip: bool = False,
    ) -> None:
        self.texture = np.ascontiguousarray(texture[..., :3].astype(np.uint8))
        self.height = int(self.texture.shape[0])
        self.width = int(self.texture.shape[1])
        self.src = neutral_2d.astype(np.float32)  # source sample coords (pixels)
        self.triangles = triangles.astype(np.int64)

    def render(self, deformed_2d: np.ndarray, *, background: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        out = np.full((self.height, self.width, 3), background, dtype=np.uint8)
        dst = np.asarray(deformed_2d, dtype=np.float32)
        for tri in self.triangles:
            self._raster_triangle(out, dst[tri], self.src[tri])
        return out

    def _raster_triangle(self, out: np.ndarray, dst: np.ndarray, src: np.ndarray) -> None:
        x_min = max(0, int(np.floor(dst[:, 0].min())))
        x_max = min(self.width - 1, int(np.ceil(dst[:, 0].max())))
        y_min = max(0, int(np.floor(dst[:, 1].min())))
        y_max = min(self.height - 1, int(np.ceil(dst[:, 1].max())))
        if x_min > x_max or y_min > y_max:
            return

        (x0, y0), (x1, y1), (x2, y2) = dst
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-8:
            return

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        gx, gy = np.meshgrid(xs, ys)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            return

        su = l0 * src[0, 0] + l1 * src[1, 0] + l2 * src[2, 0]
        sv = l0 * src[0, 1] + l1 * src[1, 1] + l2 * src[2, 1]
        sx = np.clip(np.round(su).astype(np.int64), 0, self.width - 1)
        sy = np.clip(np.round(sv).astype(np.int64), 0, self.height - 1)
        out[gy[inside], gx[inside]] = self.texture[sy[inside], sx[inside]]

    def render_composite(
        self,
        deformed_2d: np.ndarray,
        photo: np.ndarray,
        face_mask: np.ndarray | None,
        *,
        background: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        warped = self.render(deformed_2d, background=background)
        return _composite(warped, photo, face_mask)

    def release(self) -> None:  # symmetry with GLMeshRenderer
        pass


def make_mesh_renderer(
    texture: np.ndarray,
    neutral_2d: np.ndarray,
    triangles: np.ndarray,
    *,
    prefer: str = "gl",
    v_flip: bool = False,
):
    """Build a renderer. ``prefer`` in {"gl", "numpy", "auto"}.

    "gl"/"auto" try the OpenGL backend first and fall back to NumPy on failure.
    """
    if prefer in ("gl", "auto"):
        try:
            return GLMeshRenderer(texture, neutral_2d, triangles, v_flip=v_flip)
        except Exception as exc:  # pragma: no cover - depends on runtime GL
            print(
                f"[gl_render] GL backend unavailable ({exc!r}); using numpy fallback",
                file=sys.stderr,
            )
    return NumpyMeshRenderer(texture, neutral_2d, triangles, v_flip=v_flip)


def _main() -> int:
    import argparse
    import time

    from PIL import Image

    from avatar_registry.avatar_pack import load_avatar_pack
    from export_utils import ARKIT_BLENDSHAPE_NAMES

    parser = argparse.ArgumentParser(description="Render a single morph via mesh renderer")
    parser.add_argument("--pack", default=str(ROOT / "avatar_registry" / "packs" / "vfhq_case1"))
    parser.add_argument("--morph", default="jawOpen")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--composite", action="store_true")
    parser.add_argument(
        "--backend", choices=("gl", "numpy", "auto"), default="gl"
    )
    parser.add_argument("--output", default=str(ROOT / "avatar_registry" / "output" / "gl_jawOpen.png"))
    args = parser.parse_args()

    pack = load_avatar_pack(Path(args.pack))

    weights = np.zeros(52, dtype=np.float32)
    if args.morph not in ARKIT_BLENDSHAPE_NAMES:
        raise ValueError(f"Unknown morph: {args.morph}")
    missing = pack.meta.get("missing_shapes", [])
    if args.morph not in missing:
        weights[ARKIT_BLENDSHAPE_NAMES.index(args.morph)] = args.weight
    deformed = pack.deform(weights)

    renderer = make_mesh_renderer(
        pack.texture, pack.neutral_2d, pack.triangles, prefer=args.backend
    )

    t0 = time.perf_counter()
    if args.composite:
        image = renderer.render_composite(deformed, pack.texture, pack.face_mask)
    else:
        image = renderer.render(deformed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)
    renderer.release()

    print(
        f"backend={renderer.backend}  morph={args.morph} weight={args.weight}\n"
        f"  render_ms={elapsed_ms:.1f}  output={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
