# mesh.py

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import triangle  # For CDT
import refine    # <-- Import for circle-based refinement
from matplotlib.path import Path
from scipy.spatial import cKDTree

# ---------- IMAGE & OUTLINE EXTRACTION ----------
def extract_polygon_from_png(path, simplify=1.0):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")
    cnt = max(contours, key=cv2.contourArea)
    epsilon = simplify * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    poly = approx[:, 0, :]
    return poly

# ---------- PLOTTING HELPERS ----------
def plot_image_with_outline(img_path, polygon, show=True):
    img = Image.open(img_path)
    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.plot(polygon[:,0], polygon[:,1], color='red', lw=2)
    for i, (x, y) in enumerate(polygon):
        plt.text(x, y, str(i), fontsize=8, color='blue')
    plt.title("Image with Extracted Polygon Outline")
    plt.axis('off')
    if show:
        plt.show()

def plot_polygon(polygon, show=True):
    plt.figure(figsize=(6,6))
    plt.fill(polygon[:,0], polygon[:,1], color='orange', alpha=0.2)
    plt.plot(polygon[:,0], polygon[:,1], color='orange')
    for i, (x, y) in enumerate(polygon):
        plt.text(x, y, str(i), fontsize=8, color='blue')
    plt.title("Polygon Points (with indices)")
    plt.axis('equal')
    if show:
        plt.show()

def plot_triangulation(polygon, tri_vertices, tri_faces, show=True):
    plt.figure(figsize=(8,8))
    plt.triplot(tri_vertices[:,0], tri_vertices[:,1], tri_faces, color='purple', lw=1)
    plt.plot(polygon[:,0], polygon[:,1], color='orange')
    for i, (x, y) in enumerate(polygon):
        plt.text(x, y, str(i), fontsize=8, color='blue')
    plt.title("Constrained Delaunay Triangulation (CDT)")
    plt.axis('equal')
    if show:
        plt.show()

# ----------- NEW: Adaptive Interior Point Generation -----------
def local_spacing_per_vertex(polygon):
    """
    Compute local spacing for each polygon vertex as average length
    of its adjacent edges.
    """
    pts = np.array(polygon)
    N = len(pts)
    spacing = np.zeros(N)
    for i in range(N):
        prev_i = (i - 1) % N
        next_i = (i + 1) % N
        dist_prev = np.linalg.norm(pts[i] - pts[prev_i])
        dist_next = np.linalg.norm(pts[i] - pts[next_i])
        spacing[i] = (dist_prev + dist_next) / 2
    return pts, spacing

def generate_adaptive_interior_points(polygon, base_spacing=10, max_spacing=40, factor=1.5):
    poly = np.array(polygon)
    path = Path(poly)

    verts, local_spacings = local_spacing_per_vertex(poly)
    kdtree = cKDTree(verts)

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)

    fine_spacing = base_spacing / 2
    x_vals = np.arange(xmin, xmax, fine_spacing)
    y_vals = np.arange(ymin, ymax, fine_spacing)
    grid = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)

    inside_mask = path.contains_points(grid)
    candidates = grid[inside_mask]

    keep_mask = []
    for point in candidates:
        dist, idx = kdtree.query(point)
        local_space = local_spacings[idx]
        thresh = np.clip(local_space * factor, base_spacing, max_spacing)
        keep_mask.append(dist < thresh)

    adaptive_points = candidates[np.array(keep_mask)]
    return adaptive_points

# ----------- MODIFIED triangulate_polygon -----------
def triangulate_polygon(polygon, spacing=10):
    pts = np.array(polygon)
    N = len(pts)
    segments = np.array([[i, (i+1)%N] for i in range(N)], dtype=np.int32)

    # --- Use adaptive interior points ---
    interior_pts = generate_adaptive_interior_points(pts, base_spacing=spacing)

    all_vertices = np.vstack([pts, interior_pts]) if len(interior_pts) > 0 else pts
    A = dict(vertices=all_vertices, segments=segments)
    result = triangle.triangulate(A, 'p')
    tri_vertices = result['vertices']
    tri_faces = result['triangles']
    return tri_vertices, tri_faces

# --------- (OPTIONAL) EXPORT HELPERS -----------
def export_polygon_as_txt(polygon, path):
    np.savetxt(path, polygon, fmt='%.4f')

def export_mesh_as_txt(tri_vertices, tri_faces, path):
    np.savetxt(path+'_vertices.txt', tri_vertices, fmt='%.4f')
    np.savetxt(path+'_faces.txt', tri_faces, fmt='%d')

# ---------- API for UI and scripting ----------
def load_and_extract_polygon(png_path, simplify=0.003):
    poly = extract_polygon_from_png(png_path, simplify=simplify)
    return poly

def get_triangulation(poly, spacing=10):
    return triangulate_polygon(poly, spacing=spacing)

# ---------- REFINEMENT TOOLS: call refine.py ----------
def refine_polygon_with_circle(polygon, center, radius, points_per_segment=2):
    return refine.insert_circle_intersections_in_polygon(
        polygon, center, radius, points_per_segment=points_per_segment
    )

def preview_circle_polygon_intersections(polygon, center, radius, points_per_segment=2):
    return refine.find_circle_polygon_intersections(
        polygon, center, radius, points_per_segment=points_per_segment
    )
