"""Normalized image polygons shared by truth validation and patch supervision."""

import math


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (
        abs(_cross(a, b, p)) <= 1e-12
        and min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
        and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12
    )


def validate_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return an open vertex ring; reject degenerate or self-intersecting outlines."""
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if not 3 <= len(points) <= 256 or len(set(points)) != len(points):
        raise ValueError("輪廓需要 3–256 個不同頂點")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p):
        raise ValueError("輪廓頂點必須位於影像內（0–1）")
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1], strict=True))
    if abs(area) <= 1e-10:
        raise ValueError("輪廓必須形成有面積的區域")
    count = len(points)
    for i in range(count):
        a, b = points[i], points[(i + 1) % count]
        c = points[(i + 2) % count]
        if abs(_cross(a, b, c)) <= 1e-12 and (
            (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) < 0
        ):
            raise ValueError("輪廓邊線不能重疊或折返")
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            c, d = points[j], points[(j + 1) % count]
            crosses = _cross(a, b, c) * _cross(a, b, d) < 0 and _cross(c, d, a) * _cross(c, d, b) < 0
            if crosses or any((_on_segment(a, b, c), _on_segment(a, b, d),
                               _on_segment(c, d, a), _on_segment(c, d, b))):
                raise ValueError("輪廓邊線不能交叉，請沿部位外緣依序標記")
    return points


def points_in_regions(centers, regions: list[dict], *, flip: bool = False):
    """Select patch centers inside real outlines, including edges; support legacy boxes."""
    import numpy as np

    centers = np.asarray(centers)
    x = 1 - centers[:, 0] if flip else centers[:, 0]
    y = centers[:, 1]
    hit = np.zeros(len(centers), dtype=bool)
    for region in regions:
        if region.get("shape", "rectangle") != "polygon":
            hit |= ((x >= region["x"]) & (x <= region["x"] + region["width"])
                    & (y >= region["y"]) & (y <= region["y"] + region["height"]))
            continue
        vertices = validate_polygon([(p["x"], p["y"]) for p in region["points"]])
        inside = np.zeros(len(centers), dtype=bool)
        boundary = np.zeros(len(centers), dtype=bool)
        for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True):
            cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
            boundary |= ((np.abs(cross) <= 1e-7) & (x >= min(a[0], b[0]) - 1e-7)
                         & (x <= max(a[0], b[0]) + 1e-7) & (y >= min(a[1], b[1]) - 1e-7)
                         & (y <= max(a[1], b[1]) + 1e-7))
            if a[1] != b[1]:
                intersects = (a[1] > y) != (b[1] > y)
                inside ^= intersects & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
        hit |= inside | boundary
    return hit


def region_area(region: dict) -> float:
    if region.get("shape") != "polygon":
        return region["width"] * region["height"]
    points = [(p["x"], p["y"]) for p in region["points"]]
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1], strict=True))) / 2


def region_rectangle_intersection(region: dict, bounds: tuple[float, float, float, float]) -> float:
    """Area within a crop, preserving concave polygon geometry for local supervision."""
    x0, y0, x1, y1 = bounds
    if region.get("shape") != "polygon":
        return (max(0, min(x1, region["x"] + region["width"]) - max(x0, region["x"]))
                * max(0, min(y1, region["y"] + region["height"]) - max(y0, region["y"])))
    points = [(p["x"], p["y"]) for p in region["points"]]
    for axis, limit, direction in ((0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)):
        if not points:
            return 0.0
        clipped = []
        previous = points[-1]
        for current in points:
            was_inside = (previous[axis] - limit) * direction >= 0
            is_inside = (current[axis] - limit) * direction >= 0
            if was_inside != is_inside:
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(tuple(previous[k] + t * (current[k] - previous[k]) for k in (0, 1)))
            if is_inside:
                clipped.append(current)
            previous = current
        points = clipped
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1], strict=True))) / 2 if points else 0.0
