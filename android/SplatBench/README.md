# SplatBench — on-device 3D Gaussian-splat render + parallel CPU benchmark

Renders the same LAM 3D Gaussian avatar as the desktop `gaussian_splat.py`
(EWA splatting: per-Gaussian morph deform → view projection → 2D covariance →
back-to-front sort → instanced "over" blend), but on Android via **OpenGL ES
3.x** on the device GPU (e.g. Adreno 640).

The per-frame **CPU instance build** stage (deform + projection + covariance +
cull + sort) is the parallelizable hot path. The app sweeps worker-thread counts
(1, 2, 4, …, #cores) and reports median build time, GPU time, and throughput so
you can see how the CPU stage scales on the device.

Target device used for design: Samsung SM-G975U1 (Snapdragon 855 / SM8150,
Adreno 640, GLES 3.2, Android 12, arm64-v8a).

## 1. Bake the avatar asset (already done once)

The app reads a flat binary `vfhq_case1.splat` from `app/src/main/assets/`.
It is produced from the LAM avatar pack by the Python baker:

```bash
# from the wav2arkit_cpu repo root
python -m avatar_registry.bake_android \
  --avatar avatar_registry/lam3d_avatar/vfhq_case1 \
  --output android/SplatBench/app/src/main/assets/vfhq_case1.splat
```

The committed asset already contains 20,018 Gaussians, 51 morphs, 302 frames
(~13 MB). Re-run the baker to use a different avatar (point `--avatar` at any
pack with `offset.ply` + `skin.glb` + `bsData.json`).

## 2. Build & run

Open `android/SplatBench/` in **Android Studio** (Giraffe+), let it sync (it
will fetch Gradle 8.7 + AGP 8.5), then Run on the device. Or from CLI once a
wrapper is generated:

```bash
cd android/SplatBench
gradle wrapper          # first time only (generates gradlew)
./gradlew installDebug  # build + install on a connected device
```

## 3. Using the app

- The avatar animates live at the top (vsync-limited preview).
- Toggle **Mouth-only** to render only the dynamic (mouth/jaw) Gaussian subset
  over a pre-rendered static base — the on-device version of the desktop hybrid
  mode. Fewer Gaussians ⇒ less CPU work per frame.
- Tap **Run Benchmark** to sweep thread counts. Results table:

```
threads  build(ms)    gpu(ms)      fps
1        ...          ...          ...   x1.00
2        ...          ...          ...   x...
4        ...          ...          ...   x...
6        ...          ...          ...   x...
```

`build(ms)` is the parallel CPU stage (median), `gpu(ms)` is total-minus-build
(draw + `glFinish`), `fps` is 1000/total, `x…` is speedup vs the 1-thread run.

## Notes / parity with desktop

- Identical EWA recipe and shaders as `gaussian_splat.py` (`_project_instances`
  + the splat vertex/fragment shaders), rendered into a centered square viewport
  so the projection matches the square desktop frame.
- Unlike the desktop CPU path (software llvmpipe), here the GPU does the
  rasterization, so the benchmark isolates **CPU instance-build scaling** across
  threads while the GPU draw cost stays roughly fixed.
- Mouth-only on device does not crop the viewport (the GPU draw is cheap); the
  speedup comes from building only the dynamic subset on the CPU.
```
