# refine.py

import numpy as np

def line_circle_intersections(p1, p2, center, radius, tol=1e-8):
    """
    Returns a list of (t_on_edge, intersection_point) tuples for intersection(s)
    between the segment p1->p2 and a circle (center, radius).
    Only returns points that are ON the segment (not on the infinite line).
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    center = np.array(center, dtype=float)

    d = p2 - p1
    f = p1 - center

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    discriminant = b * b - 4 * a * c
    points = []
    if discriminant < -tol:
        return points  # No intersection

    sqrt_disc = np.sqrt(max(discriminant, 0))
    if discriminant > tol:
        ts = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
    else:  # tangent
        ts = [(-b) / (2 * a)]

    for t in ts:
        if 0 - tol <= t <= 1 + tol:
            pt = p1 + t * d
            points.append((t, pt))
    return points

def insert_circle_intersections_in_polygon(polygon, center, radius, points_per_segment=2, tol=1e-8):
    """
    For each polygon edge, if the edge crosses the circle, insert points_per_segment
    equally spaced points (including the endpoints) between the two intersection points.
    If only a tangent, inserts the single point.
    Returns a new polygon array with points inserted in correct order.
    """
    poly = np.array(polygon, dtype=float)
    N = len(poly)
    insertions = []

    for i in range(N):
        p1 = poly[i]
        p2 = poly[(i + 1) % N]
        d = p2 - p1
        ints = line_circle_intersections(p1, p2, center, radius, tol=tol)

        if len(ints) == 2:
            (t1, pt1), (t2, pt2) = sorted(ints, key=lambda x: x[0])
            t_vals = np.linspace(t1, t2, points_per_segment)
            pts = [p1 + t * d for t in t_vals]
            for t, pt in zip(t_vals, pts):
                insertions.append((i + 1, t, pt))
        elif len(ints) == 1:
            t, pt = ints[0]
            insertions.append((i + 1, t, pt))

    # Sort by (insert-after-index, t) so all points on an edge are inserted in the right order
    insertions.sort(key=lambda tup: (tup[0], tup[1]))
    new_poly = np.array(poly)
    for idx, _, pt in reversed(insertions):
        new_poly = np.insert(new_poly, idx, pt, axis=0)
    return new_poly

def find_circle_polygon_intersections(polygon, center, radius, points_per_segment=2, tol=1e-8):
    """
    Returns all points that would be inserted by a circle refine with the given
    number of points per segment. Useful for UI preview (as red dots).
    """
    poly = np.array(polygon, dtype=float)
    N = len(poly)
    preview_points = []

    for i in range(N):
        p1 = poly[i]
        p2 = poly[(i + 1) % N]
        d = p2 - p1
        ints = line_circle_intersections(p1, p2, center, radius, tol)
        if len(ints) == 2:
            (t1, pt1), (t2, pt2) = sorted(ints, key=lambda x: x[0])
            t_vals = np.linspace(t1, t2, points_per_segment)
            pts = [p1 + t * d for t in t_vals]
            preview_points.extend(pts)
        elif len(ints) == 1:
            t, pt = ints[0]
            preview_points.append(pt)
    if len(preview_points) == 0:
        return None
    else:
        return np.array(preview_points)
