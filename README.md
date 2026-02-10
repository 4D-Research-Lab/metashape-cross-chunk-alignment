  # metashape-cross-chunk-alignment

  Aligns a non-georeferenced Metashape chunk (with cameras/photos) to a georeferenced point cloud in another chunk using point cloud ICP
   registration. The entire chunk moves — cameras and point cloud together — so photos end up georeferenced.

  Developed by Ariicia Kuiters, based on [agisoft-llc/metashape-scripts](https://github.com/agisoft-llc/metashape-scripts) `align_model_to_model.py`, adapted for cross-chunk alignment. The original script is in `reference/align_chunk_to_model.py`.

  ## Requirements

  - Metashape Pro 2.3
  - Two chunks: one with photos (non-georeferenced), one with a georeferenced point cloud
  - Both chunks must have a CRS set (e.g. EPSG:28992)

  Dependencies (installed automatically via `pip_auto_install`):
  - open3d 0.19.0
  - scipy 1.12.0
  - numpy 1.26.4

  ## Usage

  1. In Metashape: **Tools > Run Script** → select `src/metashape-cross-chunk-alignment.py`
  2. Go to **Scripts > Align chunk to model (cross-chunk) V2**
  3. Select **From** (non-georeferenced chunk) and **To** (georeferenced reference)
  4. Click **Ok**

  ### Parameters

  | Parameter | Description |
  |---|---|
  | Scale ratio | Size ratio between target and source. Leave empty to auto-estimate from convex hulls |
  | Target resolution | Point spacing in target coords. Leave empty to auto-estimate |
  | Use initial alignment | Skip global RANSAC, start ICP from current position. Use for refinement after a first pass |
  | Preview intermediate alignment | Show Open3D windows at each stage (may crash on integrated GPUs due to OpenGL conflicts) |

  ## How it works

  1. Exports both point clouds to temporary PLY files
  2. Centers both clouds and estimates scale/resolution
  3. Runs RANSAC global registration (feature matching) for rough alignment
  4. Refines with coarse then fine ICP (Iterative Closest Point)
  5. Converts the ICP result from CRS projected space back to Metashape's internal coordinate system using local Cartesian frames
  6. Applies the transform to `chunk.transform.matrix`, moving cameras and point cloud together
