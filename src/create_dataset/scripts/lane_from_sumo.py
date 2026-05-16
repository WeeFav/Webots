"""
SUMO Lane Boundary Extractor
-----------------------------
Connects to a running SUMO simulation via TraCI and extracts left/right
lane boundaries for every lane in the network.

Usage:
    # Option 1 – start SUMO yourself, then run this script
    sumo --net-file my.net.xml --step-length 0.1 --remote-port 8813 &
    python sumo_lane_boundaries.py

    # Option 2 – let the script launch SUMO (set SUMO_CMD below)
    python sumo_lane_boundaries.py --launch
"""

import argparse
import sys

import numpy as np
import traci
import traci.constants as tc
import sumolib


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class SumoLaneBoundaryExtractor:
    """Extract left/right boundary polylines for every lane in a SUMO network."""

    def __init__(self, net, traci_connection=None):
        """
        Parameters
        ----------
        traci_connection : module or object, optional
            Pass a custom TraCI connection object if you are using the
            multi-client API (traci.connect / traci.StepListener).
            Defaults to the global `traci` module.
        """
        self.traci = traci_connection if traci_connection is not None else traci

        self.net = sumolib.net.readNet(net)
        self.xOffset = -self.net.getLocationOffset()[0]
        self.yOffset = -self.net.getLocationOffset()[1]

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def get_lane_boundary(self, lane_id: str) -> dict:
        shape = self.traci.lane.getShape(lane_id)
        width = self.traci.lane.getWidth(lane_id)

        pts = np.array(shape, dtype=float)
        pts[:, 0] += self.xOffset
        pts[:, 1] += self.yOffset

        left_boundary = []
        right_boundary = []

        for i in range(len(pts)):
            if i == 0:
                tangent = pts[i + 1] - pts[i]
            elif i == len(pts) - 1:
                tangent = pts[i] - pts[i - 1]
            else:
                tangent = pts[i + 1] - pts[i - 1]

            norm = np.linalg.norm(tangent)
            if norm == 0:
                if left_boundary:
                    left_boundary.append(left_boundary[-1])
                    right_boundary.append(right_boundary[-1])
                continue

            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])

            left_boundary.append(pts[i] + normal * width * 0.5)
            right_boundary.append(pts[i] - normal * width * 0.5)

        return {
            "left_line":  self.interpolate_to_100(left_boundary),
            "right_line": self.interpolate_to_100(right_boundary),
        }
            
    def interpolate_to_100(self, boundary: list) -> np.ndarray:
        arr = np.array(boundary, dtype=float)          # (N, 2)

        # Cumulative chord-length parameter t in [0, 1]
        deltas = np.diff(arr, axis=0)                  # (N-1, 2)
        seg_lengths = np.linalg.norm(deltas, axis=1)   # (N-1,)
        t = np.concatenate([[0], np.cumsum(seg_lengths)])
        t /= t[-1]                                     # normalise to [0, 1]

        t_uniform = np.linspace(0, 1, 100)

        x_interp = np.interp(t_uniform, t, arr[:, 0])
        y_interp = np.interp(t_uniform, t, arr[:, 1])

        return np.column_stack([x_interp, y_interp])  # (100, 2)

    # ------------------------------------------------------------------
    # Network-level helpers
    # ------------------------------------------------------------------

    def get_all_lane_ids(self) -> list[str]:
        """Return every lane ID present in the loaded network."""
        return list(self.traci.lane.getIDList())

    def get_lane_ids_for_edge(self, edge_id: str) -> list[str]:
        """
        Return the lane IDs that belong to *edge_id*.

        SUMO lane IDs follow the convention ``<edge_id>_<lane_index>``.
        We use :func:`traci.edge.getLaneNumber` to discover the count.
        """
        n_lanes = self.traci.edge.getLaneNumber(edge_id)
        return [f"{edge_id}_{i}" for i in range(n_lanes)]

    # ------------------------------------------------------------------
    # Main extraction method
    # ------------------------------------------------------------------

    def extract_all_lane_boundaries(
        self,
        edge_ids: list[str] | None = None,
        skip_internal: bool = True,
    ) -> dict[str, dict]:
        """
        Extract boundaries for all lanes (or a subset of edges).

        Parameters
        ----------
        edge_ids : list[str] or None
            If provided, only lanes belonging to these edges are processed.
            If ``None``, all lanes in the network are processed.
        skip_internal : bool
            When ``True`` (default) junction-internal lanes (whose IDs start
            with ``':'``) are skipped.  Set to ``False`` to include them.

        Returns
        -------
        dict
            ``{ lane_id: {'left_line': np.ndarray, 'right_line': np.ndarray} }``
        """
        if edge_ids is not None:
            lane_ids = []
            for eid in edge_ids:
                lane_ids.extend(self.get_lane_ids_for_edge(eid))
        else:
            lane_ids = self.get_all_lane_ids()

        if skip_internal:
            lane_ids = [lid for lid in lane_ids if not lid.startswith(":")]

        result: dict[str, dict] = {}
        errors: list[str] = []

        for lane_id in lane_ids:
            try:
                result[lane_id] = self.get_lane_boundary(lane_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"  {lane_id}: {exc}")

        if errors:
            print(
                f"[SumoLaneBoundaryExtractor] Could not process "
                f"{len(errors)} lane(s):\n" + "\n".join(errors),
                file=sys.stderr,
            )

        return result


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def get_all_lane_boundaries(
    host: str = "localhost",
    port: int = 8813,
    skip_internal: bool = True,
    net: str = ""
) -> dict[str, dict]:
    """
    Connect to a running SUMO/TraCI server and return lane boundaries.

    Parameters
    ----------
    host : str
        Hostname of the TraCI server (default ``"localhost"``).
    port : int
        Port of the TraCI server (default ``8813``).
    skip_internal : bool
        Skip junction-internal lanes when ``True`` (default).

    Returns
    -------
    dict
        ``{ lane_id: {'left_line': np.ndarray, 'right_line': np.ndarray} }``
    """
    traci.init(port=port, host=host)
    try:
        extractor = SumoLaneBoundaryExtractor(net)
        boundaries = extractor.extract_all_lane_boundaries(
            skip_internal=skip_internal
        )
    finally:
        traci.close()

    return boundaries


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lane boundaries from a SUMO network via TraCI."
    )
    parser.add_argument(
        "--host", default="localhost", help="TraCI server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8813, help="TraCI server port (default: 8813)"
    )
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Also extract junction-internal lanes (id starts with ':')",
    )
    parser.add_argument(
        "--net", default="", help="net file"
    )
    parser.add_argument(
        "--launch",
        metavar="NET_FILE",
        help="Path to a .net.xml file.  The script will launch SUMO itself.",
    )
    parser.add_argument(
        "--output",
        metavar="OUT_FILE",
        help="Save boundaries as a txt file.",
    )
    return parser.parse_args()


def _launch_sumo(net_file: str, port: int) -> None:
    """Start SUMO as a subprocess (requires SUMO_HOME to be set)."""
    import os
    import subprocess

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home is None:
        raise EnvironmentError(
            "SUMO_HOME environment variable is not set. "
            "Please set it to your SUMO installation directory."
        )

    sumo_bin = os.path.join(sumo_home, "bin", "sumo")
    cmd = [sumo_bin, "--net-file", net_file, "--remote-port", str(port), "--no-step-log"]
    print(f"[launcher] Starting SUMO: {' '.join(cmd)}")
    subprocess.Popen(cmd)  # non-blocking; TraCI client will connect shortly


def main() -> None:
    args = _parse_args()

    if args.launch:
        _launch_sumo(args.launch, args.port)
        import time; time.sleep(1)   # give SUMO a moment to start

    print(f"Connecting to TraCI at {args.host}:{args.port} …")
    boundaries = get_all_lane_boundaries(
        host=args.host,
        port=args.port,
        skip_internal=not args.include_internal,
        net=args.net
    )

    print(f"Extracted boundaries for {len(boundaries)} lane(s).")

    # Quick sanity print
    for lane_id, bnd in list(boundaries.items())[:3]:
        print(
            f"  {lane_id}: "
            f"left_line={bnd['left_line'].shape}, "
            f"right_line={bnd['right_line'].shape}"
        )
    if len(boundaries) > 3:
        print("  …")

    if args.output:
        with open(args.output, 'w') as f:
            for lane_id, bnd in boundaries.items():
                for side in ('left_line', 'right_line'):
                    pts = bnd[side]  # shape (100, 2)
                    # format: lane_id|side|n_points
                    f.write(f"{lane_id}|{side}|{len(pts)}\n")
                    for x, y in pts:
                        f.write(f"{x:.6f} {y:.6f}\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()