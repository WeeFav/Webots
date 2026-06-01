import svgwrite
import numpy as np
from scipy.spatial.transform import Rotation as R
from extract_proto import extract_proto
import argparse

POSE_AXIS = np.array([0.707107, -0.707107, 0.0])
POSE_ANGLE = 3.14157
POSE_ROT = R.from_rotvec(POSE_AXIS * POSE_ANGLE)

def apply_transform(points, translation, rotation, extra_rot=None):
    """
    Apply Webots translation + axis-angle rotation.
    """
    axis = rotation[:3]
    angle = rotation[3]

    rot = R.from_rotvec(np.array(axis) * angle)
    if extra_rot is not None:
        rot = rot * extra_rot   # IMPORTANT: Webots order
        
    transformed = []

    for p in points:
        p_rot = rot.apply(p)
        p_world = p_rot + np.array(translation)
        transformed.append(p_world)

    return np.array(transformed)


def compute_bounds(roads, crossroads, forests, buildings, parkings, waters):
    """
    Compute world bounds for SVG sizing.
    """

    all_pts = []

    for road in roads:
        pts = apply_transform(road.wayPoints, road.translation, road.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])

    for cross in crossroads:
        pts = apply_transform(cross.shape, cross.translation, cross.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])
            
    for forest in forests:
        shape_3d = np.array([[p[0], -p[1], 0.0] for p in forest.shape])
        pts = apply_transform(shape_3d, forest.translation, forest.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])
            
    for building in buildings:
        shape_3d = np.array([[p[0], p[1], 0.0] for p in building.corners])
        pts = apply_transform(shape_3d, building.translation, building.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])

    for parking in parkings:
        pts = apply_transform(parking.point, parking.translation, parking.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])

    for water in waters:
        pts = apply_transform(water.point, water.translation, water.rotation)
        if pts.size > 0:
            all_pts.extend(pts[:, :2])
            
    all_pts = np.array(all_pts)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    return min_xy, max_xy

def create_svg(roads, crossroads, forests, buildings, parkings, waters, output_file="map.svg", scale=5.0, margin=50):
    min_xy, max_xy = compute_bounds(roads, crossroads, forests, buildings, parkings, waters)

    width = (max_xy[0] - min_xy[0]) * scale + margin * 2
    height = (max_xy[1] - min_xy[1]) * scale + margin * 2

    dwg = svgwrite.Drawing(output_file, size=(f"{width}px", f"{height}px"))

    def to_svg_coords(p):
        """
        Convert world coords to SVG coords.
        SVG y-axis points downward.
        """
        x = (p[0] - min_xy[0]) * scale + margin
        y = height - ((p[1] - min_xy[1]) * scale + margin)
        return (x, y)

    # =====================================================
    # Draw crossroads first
    # =====================================================
    for cross in crossroads:
        pts = apply_transform(cross.shape, cross.translation, cross.rotation, extra_rot=POSE_ROT)
        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        if len(svg_pts) >= 3:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_cmd.append("Z")

            path_data = " ".join(path_cmd)

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="gray",
                    stroke="gray",
                    stroke_width=1,
                    id=cross.id,
                )
            )
        else:
            print(f"Skipping invalid crossroad: {cross.name}")
            
    # =====================================================
    # Draw forests
    # =====================================================
    for forest in forests:
        # Convert 2D shape -> 3D points
        shape_3d = np.array([[p[0], -p[1], 0.0] for p in forest.shape])
        pts = apply_transform(shape_3d, forest.translation, forest.rotation)
        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        if len(svg_pts) >= 3:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_cmd.append("Z")

            path_data = " ".join(path_cmd)

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="#7fbf7f",      # light green
                    stroke="#7fbf7f",
                    stroke_width=1,
                    id=forest.id
                )
            )
        else:
            print(f"Skipping invalid forest: {forest.name}")
            
    # =====================================================
    # Draw roads
    # =====================================================
    for road in roads:
        pts = apply_transform(road.wayPoints, road.translation, road.rotation)

        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        # polyline needs at least 2 points
        if len(svg_pts) >= 2:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_data = " ".join(path_cmd)
            
            color = "yellow" if road.appearance == "CementTiles" else "black"

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="none",
                    stroke=color,
                    stroke_width=road.width * scale,
                    stroke_linecap="butt",
                    stroke_linejoin="round",
                    id=road.id
                )
            )
        else:
            print(f"Skipping invalid road: {road.name}")

    # =====================================================
    # Draw buildings
    # =====================================================
    for building in buildings:
        # Convert 2D shape -> 3D points
        shape_3d = np.array([[p[0], p[1], 0.0] for p in building.corners])
        pts = apply_transform(shape_3d, building.translation, building.rotation)
        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        if len(svg_pts) >= 3:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_cmd.append("Z")

            path_data = " ".join(path_cmd)

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="red",
                    stroke="red",
                    stroke_width=1,
                    id=building.id
                )
            )
        else:
            print(f"Skipping invalid building: {building.name}")

    # =====================================================
    # Draw parkings
    # =====================================================
    for parking in parkings:
        pts = apply_transform(parking.point, parking.translation, parking.rotation)
        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        if len(svg_pts) >= 3:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_cmd.append("Z")

            path_data = " ".join(path_cmd)

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="lightgray",
                    stroke="lightgray",
                    stroke_width=1,
                    id=parking.id,
                )
            )
        else:
            print(f"Skipping invalid parking: {parking.name}")
    
    # =====================================================
    # Draw waters
    # =====================================================
    for water in waters:
        pts = apply_transform(water.point, water.translation, water.rotation)
        svg_pts = [to_svg_coords(p[:2]) for p in pts]

        if len(svg_pts) >= 3:
            path_cmd = []

            for i, p in enumerate(svg_pts):
                if i == 0:
                    path_cmd.append(f"M {p[0]} {p[1]}")
                else:
                    path_cmd.append(f"L {p[0]} {p[1]}")

            path_cmd.append("Z")

            path_data = " ".join(path_cmd)

            dwg.add(
                dwg.path(
                    d=path_data,
                    fill="blue",
                    stroke="blue",
                    stroke_width=1,
                    id=water.id,
                )
            )
        else:
            print(f"Skipping invalid water: {water.name}")

    dwg.save()
    print(f"SVG saved to: {output_file}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    
    roads, crossroads, forests, buildings, parkings, waters = extract_proto(args.input)
    create_svg(roads, crossroads, forests, buildings, parkings, waters, output_file=args.output, scale=5)