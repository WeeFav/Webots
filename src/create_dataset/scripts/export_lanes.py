"""
export_lanes.py

Run this once (offline, before launching the C++ driver) to bake the lane
center-lines from a Webots world file into a plain-text cache that the C++
robot driver can read at startup with no Python dependency.

Usage:
    python export_lanes.py \
        --world /home/marvin/Webots/src/webots_launch/worlds/city.wbt \
        --out   /home/marvin/Webots/src/webots_launch/worlds/lanes.txt

File format (lanes.txt)
-----------------------
Line 1:   <num_lanes>
For each lane:
  Line N:   <num_points>
  Lines …:  <x> <y> <z>        (one point per line, space-separated doubles)

Example:
  3
  100
  1.234567 0.000000 -2.345678
  1.256789 0.000000 -2.367890
  ...
  100
  ...
"""

import argparse
import numpy as np
from lane_segmentation import extract_lanes

# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------

def save_lanes_txt(all_lanes_center: list, out_path: str) -> None:
    """
    Write lane data to a plain-text file parseable by C++.

    Format:
        <num_lanes>
        <num_points_lane_0>
        x0 y0 z0
        x1 y1 z1
        ...
        <num_points_lane_1>
        ...
    """
    with open(out_path, "w") as f:
        f.write(f"{len(all_lanes_center)}\n")
        for lane in all_lanes_center:
            pts = np.asarray(lane)          # (N, 3)
            f.write(f"{len(pts)}\n")
            for x, y, z in pts:
                # 9 decimal places — more than enough for sub-millimetre precision
                f.write(f"{x:.9f} {y:.9f} {z:.9f}\n")

    print(f"Saved {len(all_lanes_center)} lanes → {out_path}")


# ---------------------------------------------------------------------------
# Loader (for round-trip verification)
# ---------------------------------------------------------------------------

def load_lanes_txt(path: str) -> list[np.ndarray]:
    """
    Load a lanes.txt file back into Python.
    Returns a list of (N, 3) numpy arrays.
    """
    lanes = []
    with open(path, "r") as f:
        num_lanes = int(f.readline())
        for _ in range(num_lanes):
            num_points = int(f.readline())
            pts = np.empty((num_points, 3), dtype=np.float64)
            for i in range(num_points):
                pts[i] = list(map(float, f.readline().split()))
            lanes.append(pts)
    return lanes


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export Webots lane centres to a C++-parseable text file.")
    parser.add_argument("--world", required=True, help="Path to the .wbt world file.")
    parser.add_argument("--out",   required=True, help="Output .txt path.")
    parser.add_argument("--verify", action="store_true",
                        help="Reload the file after writing and print a summary.")
    args = parser.parse_args()

    print(f"Extracting lanes from: {args.world}")
    lanes = extract_lanes(args.world)
    save_lanes_txt(lanes, args.out)

    if args.verify:
        reloaded = load_lanes_txt(args.out)
        print(f"Verification: reloaded {len(reloaded)} lanes")
        for i, lane in enumerate(reloaded[:3]):          # show first 3
            print(f"  Lane {i}: {len(lane)} points, first={lane[0]}, last={lane[-1]}")


if __name__ == "__main__":
    main()