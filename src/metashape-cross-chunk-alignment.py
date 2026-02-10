# Align one model/point cloud to another, allowing FROM and TO objects in different chunks
#
# This is useful for alignment of two independently photogrammetry-reconstructed models/dense point clouds, or two LIDAR clouds with good overlap.
#
# Also this should work in case of registration between LIDAR points clouds and photogrammetry model,
# but please note that they should cover similar part of surface and you should specify accurate scale ratio.
#
# Alternatively - you can use selection+crop to remove from both point clouds/models everything except object part that presented in both entities.
# And then align with automatic scale+resolution+registration.
# Just don't forget to revert cropping after alignment execution to recover deleted parts.
# After such common-parts-based alignment it should be a good idea to finally optimize alignment for full entities without selection+cropping via option 'Use initial alignment'.
#
# This is a python script for Metashape Pro (2.x). Based on the script align_model_to_model in the scripts repository: https://github.com/agisoft-llc/metashape-scripts
#
# V2: Fixed coordinate transformation issues for cross-chunk alignment

#-----------------------------
# Basis
#-----------------------------

import Metashape
from PySide2 import QtGui, QtCore, QtWidgets

import os, sys, copy, time, itertools, tempfile
from pathlib import Path

import urllib.request, tempfile
from modules.pip_auto_install import pip_install

# Required packages
requirements_txt = """open3d==0.19.0
scipy==1.12.0
numpy==1.26.4"""
pip_install(requirements_txt)

import open3d as o3d
from scipy.spatial import ConvexHull
import numpy as np

try:
    o3d_registration = o3d.registration
except AttributeError:
    o3d_registration = o3d.pipelines.registration

# --------------------------------
# Functions: aligning point clouds
# --------------------------------

def align_two_point_clouds(from_points, to_points, scale_ratio=None,
                                 target_resolution=None,
                                 no_global_alignment=False,
                                 preview_intermediate_alignment=True):
    """
    Align FROM point cloud to TO point cloud (TO is assumed to be in world coordinates).
    FROM can be arbitrary. Only FROM is transformed; TO stays fixed.

    Returns the full transformation matrix that maps original FROM coords to original TO coords,
    including centering, scaling, and ICP result.
    """
    assert(isinstance(from_points, np.ndarray) and isinstance(to_points, np.ndarray))
    assert(from_points.shape[1] == to_points.shape[1] == 3)

    v_from, v_to = from_points.copy(), to_points.copy()

    # V2 FIX: Center BOTH clouds (v1 only centered v_to conditionally).
    # Without this, large coordinate differences (0 vs 155000) break RANSAC/ICP.
    c_from = np.mean(v_from, axis=0)
    c_to = np.mean(v_to, axis=0)
    v_from = v_from - c_from
    v_to = v_to - c_to

    print(f"FROM center: {c_from}")
    print(f"TO center: {c_to}")

    # --- Estimate scale ratio if not provided ---
    if scale_ratio is None:
        if no_global_alignment:
            scale_ratio = 1.0
        else:
            print("Estimating scale ratio using convex hulls...")
            v_from_sub = subsample_points(v_from, 100000)
            v_to_sub = subsample_points(v_to, 100000)
            scale_ratio = estimate_convex_hull_size(v_to_sub) / estimate_convex_hull_size(v_from_sub)
            print(f"Estimated scale_ratio={scale_ratio:.4f}")

    # --- Estimate target resolution if not provided ---
    if target_resolution is None:
        v_from_sub = subsample_points(v_from, 1000)
        v_to_sub = subsample_points(v_to, 1000)
        # V2 FIX: Multiply by scale_ratio (v1 missed this, so voxel sizes were wrong)
        source_res = 1.5 * estimate_resolution(v_from_sub) / np.sqrt(len(v_from) / len(v_from_sub)) * scale_ratio
        target_res = 1.5 * estimate_resolution(v_to_sub) / np.sqrt(len(v_to) / len(v_to_sub))
        target_resolution = max(source_res, target_res)
    print(f"Estimated target_resolution={target_resolution:.4f}")
    Metashape.app.update()

    # Apply scale to FROM cloud
    v_from = v_from * scale_ratio

    stage = 0
    total_stages = 2 if no_global_alignment else 3

    # --- Global registration ---
    if not no_global_alignment:
        stage += 1
        print("{}/{}: Global registration...".format(stage, total_stages))
        start = time.time()
        source_down, target_down, global_result = global_registration(
            v_from, v_to, global_voxel_size=64.0 * target_resolution
        )
        print("    estimated in {} s".format(time.time() - start))
        Metashape.app.update()
        transformation = global_result.transformation
        if preview_intermediate_alignment:
            draw_registration_result(source_down, target_down, transformation, title="Initial global alignment")
    else:
        transformation = np.eye(4)
        if preview_intermediate_alignment:
            print("Initial objects shown!")
            draw_registration_result(v_from, v_to, title="Initial alignment")

    # --- Coarse ICP ---
    stage += 1
    print("{}/{}: Coarse ICP registration...".format(stage, total_stages))
    start = time.time()
    icp_voxel_size_1 = 8.0 * target_resolution
    source_down_1 = downscale_point_cloud(to_point_cloud(v_from), icp_voxel_size_1)
    target_down_1 = downscale_point_cloud(to_point_cloud(v_to), icp_voxel_size_1)
    icp_result_1 = icp_registration(source_down_1, target_down_1, voxel_size=icp_voxel_size_1,
                                  transform_init=transformation, max_iterations=100)
    print("    estimated in {} s".format(time.time() - start))
    Metashape.app.update()
    transformation = icp_result_1.transformation
    if preview_intermediate_alignment:
        draw_registration_result(source_down_1, target_down_1, transformation, title="Coarse ICP alignment")

    # --- Fine ICP ---
    stage += 1
    print("{}/{}: Fine ICP registration...".format(stage, total_stages))
    start = time.time()
    icp_voxel_size_2 = 0.5 * target_resolution
    icp_result_2 = icp_registration(to_point_cloud(v_from), to_point_cloud(v_to),
                                  voxel_size=icp_voxel_size_2, transform_init=transformation, max_iterations=100)
    print("    estimated in {} s".format(time.time() - start))
    Metashape.app.update()
    transformation = icp_result_2.transformation
    if preview_intermediate_alignment:
        draw_registration_result(to_point_cloud(v_from), to_point_cloud(v_to),
                                 transformation, title="Fine ICP alignment")

    # V2 FIX: Compose full transformation including centering (v1 returned raw ICP matrix
    # without accounting for the center subtraction, so translation was wrong).
    # M = T_to @ ICP @ Scale @ T_from maps original FROM coords → original TO coords.

    T_from = np.eye(4)
    T_from[:3, 3] = -c_from

    S = np.diag([scale_ratio, scale_ratio, scale_ratio, 1.0])

    T_to = np.eye(4)
    T_to[:3, 3] = c_to

    M_full = T_to @ transformation @ S @ T_from

    M = Metashape.Matrix(M_full.tolist())

    print("Estimated ICP transformation (centered space):")
    print(transformation)
    print("Full transformation matrix (original FROM → original TO):")
    print(M)
    Metashape.app.update()

    return M, scale_ratio

def subsample_points(vs, n):
    if len(vs) <= n:
        return vs.copy()
    np.random.seed(len(vs))
    vs = vs.copy()
    np.random.shuffle(vs)
    return vs[:n]


def estimate_convex_hull_size(vs):
    try:
        hull = ConvexHull(vs)
    except Exception:
        return np.linalg.norm(vs.max(axis=0) - vs.min(axis=0))
    indices = np.unique(hull.vertices)
    hull_vs = vs[indices]
    dists = hull_vs[:, None, :] - hull_vs[None, :, :]
    dists = dists.reshape(-1, 3)
    dists = np.sum(dists * dists, axis=-1)
    size = np.sqrt(np.max(dists))
    return size


def estimate_resolution(vs):
    dists = vs[:, None, :] - vs[None, :, :]
    dists = np.sum(dists * dists, axis=-1)
    dists[dists == 0] = np.max(dists)
    min_dists = np.min(dists, axis=-1)
    resolution = np.sqrt(np.median(min_dists))
    return resolution


def to_point_cloud(vs):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(vs.copy())
    return pc


def downscale_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def estimate_points_features(pcd_down, voxel_size):
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d_registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh


def global_registration(v1, v2, global_voxel_size):
    # See http://www.open3d.org/docs/release/tutorial/Advanced/global_registration.html#global-registration
    source = to_point_cloud(v1)
    target = to_point_cloud(v2)
    source_down = downscale_point_cloud(source, global_voxel_size)
    target_down = downscale_point_cloud(target, global_voxel_size)
    source_fpfh = estimate_points_features(source_down, global_voxel_size)
    target_fpfh = estimate_points_features(target_down, global_voxel_size)

    distance_threshold = global_voxel_size * 2.0
    max_validation = np.min([len(source_down.points), len(target_down.points)]) // 2
    kwargs = {
        "source": source_down,
        "target": target_down,
        "source_feature": source_fpfh,
        "target_feature": target_fpfh,
        "max_correspondence_distance": distance_threshold,
        "estimation_method": o3d_registration.TransformationEstimationPointToPoint(False),
        "ransac_n": 4,
        "checkers": [
            o3d_registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d_registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        "criteria": o3d_registration.RANSACConvergenceCriteria(4000000, max_validation),
    }
    if o3d.__version__ not in ["0.{}.0".format(v) for v in range(12)]:
        # Introduced in 0.12.0 release
        kwargs["mutual_filter"] = True
    global_registration_result = o3d_registration.registration_ransac_based_on_feature_matching(**kwargs)
    return source_down, target_down, global_registration_result


def icp_registration(source, target, voxel_size, transform_init, max_iterations):
    # See http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html#icp-registration
    threshold = 8.0 * voxel_size
    reg_p2p = o3d_registration.registration_icp(
        source, target, threshold, transform_init,
        o3d_registration.TransformationEstimationPointToPoint(),
        o3d_registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    return reg_p2p


def draw_registration_result(source, target, transformation=None, title="Visualization"):
    Metashape.app.update()
    if isinstance(source, np.ndarray):
        source = to_point_cloud(source)
    if isinstance(target, np.ndarray):
        target = to_point_cloud(target)
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    vis.run()
    vis.destroy_window()


def read_ply(filename):
    # Simplified read PLY (vertices only)
    return np.asarray(o3d.io.read_point_cloud(filename).points)


def get_chunk_local_transform(chunk):
    """
    V2 NEW: Adapted from get_asset_T() in align_model_to_model.py for chunk-level transforms.
    Converts chunk.transform.matrix (internal→ECEF) into a local Cartesian frame
    so ICP results (in CRS space) can be mapped back correctly.

    Returns:
        T: Transform from chunk internal to local Cartesian
        Tlocal: Local frame matrix (for converting between local frames)
    """
    if chunk.crs is None:
        raise RuntimeError("Chunk '{}' must have a CRS defined for cross-chunk alignment".format(chunk.label or "unnamed"))

    # Get a representative point in the chunk (translation from chunk transform)
    world_transform = chunk.transform.matrix
    pt0 = world_transform.translation()

    # Project to get ground-level point in geocentric coords
    pt0 = chunk.crs.project(pt0)
    pt0.z = 0  # Flatten to ground level
    pt0 = chunk.crs.unproject(pt0)  # Back to geocentric

    # Get local Cartesian frame at this point
    Tlocal = chunk.crs.localframe(pt0)

    # Compose: chunk coords -> ECEF -> local Cartesian
    T = Tlocal * world_transform

    return T, Tlocal


# -------------------------
# DIALOGUE AND GUI SETTINGS
# -------------------------

class AlignModelDlg(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Align model/dense cloud (cross-chunk) V2")

        # List all objects across all chunks
        self.objects = []
        for chunk in Metashape.app.document.chunks:
            for pc in chunk.point_clouds:
                label = pc.label or "Dense Cloud"
                label += " ({} points, chunk: {})".format(pc.point_count, chunk.label or "unnamed")
                self.objects.append((chunk, pc.key, False, label))

        if len(self.objects) < 2:
            Metashape.app.messageBox("Need at least 2 point clouds in the project (can be in different chunks).")
            self.reject()
            return

        self.fromCombo = QtWidgets.QComboBox()
        self.toCombo = QtWidgets.QComboBox()
        for _, _, _, label in self.objects:
            self.fromCombo.addItem(label)
            self.toCombo.addItem(label)

        self.edtScaleRatio = QtWidgets.QLineEdit()
        self.edtScaleRatio.setPlaceholderText("Auto (from convex hull)")
        self.edtTargetResolution = QtWidgets.QLineEdit()
        self.edtTargetResolution.setPlaceholderText("Auto (from point spacing)")
        self.chkUseInitialAlignment = QtWidgets.QCheckBox("Use initial alignment")
        self.chkUseInitialAlignment.setToolTip("Skip global RANSAC registration. Use if chunks are already coarsely aligned.")
        self.chkPreview = QtWidgets.QCheckBox("Preview intermediate alignment")
        self.chkPreview.setToolTip("Show Open3D visualization windows at each ICP stage.")
        self.btnOk = QtWidgets.QPushButton("Ok")
        self.btnQuit = QtWidgets.QPushButton("Close")

        layout = QtWidgets.QGridLayout()
        layout.addWidget(QtWidgets.QLabel("From (will be moved):"), 0,0)
        layout.addWidget(self.fromCombo, 0,1)
        layout.addWidget(QtWidgets.QLabel("To (reference):"), 0,2)
        layout.addWidget(self.toCombo, 0,3)
        layout.addWidget(QtWidgets.QLabel("Scale ratio:"),1,0)
        layout.addWidget(self.edtScaleRatio,1,1)
        layout.addWidget(QtWidgets.QLabel("Target resolution:"),1,2)
        layout.addWidget(self.edtTargetResolution,1,3)
        layout.addWidget(self.chkUseInitialAlignment,2,1)
        layout.addWidget(self.chkPreview,2,3)
        layout.addWidget(self.btnOk,3,1)
        layout.addWidget(self.btnQuit,3,3)
        self.setLayout(layout)

        self.btnOk.clicked.connect(self.align)
        self.btnQuit.clicked.connect(self.reject)

        self.exec_()

    def align(self):
        # Get selected objects
        from_chunk, from_key, _, label1 = self.objects[self.fromCombo.currentIndex()]
        to_chunk, to_key, _, label2 = self.objects[self.toCombo.currentIndex()]

        if from_chunk == to_chunk and from_key == to_key:
            Metashape.app.messageBox("FROM and TO must be different point clouds.")
            return

        print("=" * 80)
        print("Aligning {} to {}...".format(label1, label2))
        print("=" * 80)

        # Export point clouds (ignore models — only point clouds supported)
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".ply"); tmp1.close()
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".ply"); tmp2.close()

        try:
            # Export FROM cloud
            print("Exporting FROM point cloud...")
            Metashape.app.document.chunk = from_chunk
            pc1 = next(p for p in from_chunk.point_clouds if p.key == from_key)
            from_chunk.exportPointCloud(
                tmp1.name, point_clouds=[pc1.key], binary=True,  # V2 FIX: pass key not object
                save_point_normal=False, save_point_color=False,
                save_point_confidence=False, save_comment=False,
                clip_to_region=False,  # V2 FIX: v1 used default (True), clipping points
                format=Metashape.PointCloudFormatPLY
            )

            # Export TO cloud
            print("Exporting TO point cloud...")
            Metashape.app.document.chunk = to_chunk
            pc2 = next(p for p in to_chunk.point_clouds if p.key == to_key)
            to_chunk.exportPointCloud(
                tmp2.name, point_clouds=[pc2.key], binary=True,  # V2 FIX: pass key not object
                save_point_normal=False, save_point_color=False,
                save_point_classification=False,
                save_point_confidence=False, save_comment=False,
                clip_to_region=False,  # V2 FIX: v1 used default (True), clipping points
                format=Metashape.PointCloudFormatPLY
            )

            # Read numpy arrays
            print("Reading PLY files...")
            v1 = read_ply(tmp1.name)
            v2 = read_ply(tmp2.name)
        finally:
            # Clean up temp files
            if os.path.exists(tmp1.name):
                os.remove(tmp1.name)
            if os.path.exists(tmp2.name):
                os.remove(tmp2.name)

        print("FROM points:", len(v1))
        print("TO points:", len(v2))

        scale_ratio = None if self.edtScaleRatio.text() == '' else float(self.edtScaleRatio.text())
        target_resolution = None if self.edtTargetResolution.text() == '' else float(self.edtTargetResolution.text())
        no_global_alignment = self.chkUseInitialAlignment.isChecked()
        preview = self.chkPreview.isChecked()

        # Calculate transformation (in CRS projected coordinate space)
        print("\nComputing ICP alignment...")
        M12, scale_ratio = align_two_point_clouds(v1, v2, scale_ratio, target_resolution,
                                     no_global_alignment, preview)

        # Verify the rotation matrix is valid
        R = np.array([[M12[r,c] for c in range(3)] for r in range(3)])
        det_R = np.linalg.det(R)
        print(f"det(R) = {det_R:.6f} (should be ~1.0 for proper rotation)")

        if not np.isfinite(np.array(M12)).all():
            Metashape.app.messageBox("ICP returned NaN matrix – alignment failed. Try different parameters.")
            return

        # Get local Cartesian transforms for coordinate conversion
        print("\nApplying transformation to chunk...")
        try:
            T_from, Tlocal_from = get_chunk_local_transform(from_chunk)
            _, Tlocal_to = get_chunk_local_transform(to_chunk)
        except RuntimeError as e:
            Metashape.app.messageBox(str(e))
            return

        # Initialize chunk transform if needed
        if from_chunk.transform is None:
            from_chunk.transform = Metashape.ChunkTransform()

        # Compute shift between the two local frames
        shift_to_from = Tlocal_to * Tlocal_from.inv()

        # V2 FIX: Convert ICP result (CRS space) to chunk transform space using local frames.
        # v1 used: to_chunk.transform.matrix * M12 * S (wrong: mixed CRS and ECEF spaces)
        # v1 also called updateTransform() which overwrote the alignment from reference data.
        from_chunk.transform.matrix = from_chunk.transform.matrix * T_from.inv() * shift_to_from.inv() * M12 * T_from

        print("FROM chunk transform (new):", from_chunk.transform.matrix)
        print("TO chunk transform:", to_chunk.transform.matrix)

        Metashape.app.update()
        print("\n" + "=" * 80)
        print("Alignment complete! Cameras and point cloud moved together.")
        print("=" * 80)
        self.reject()

def show_alignment_dialog():
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    AlignModelDlg()

# Add menu item
Metashape.app.addMenuItem("Scripts/Align chunk to model (cross-chunk) V2", show_alignment_dialog)
print("To execute this script press Scripts/Align chunk to model (cross-chunk) V2")
