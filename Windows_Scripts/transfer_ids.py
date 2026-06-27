import os
import re
import math
import shutil

# Regular expression to extract node blocks: matches NodeName { ... } correctly nested
def extract_blocks_with_indices(text, node_name):
    blocks = []
    pattern = re.compile(rf"^\s*{node_name}\s*\{{", re.MULTILINE)

    for match in pattern.finditer(text):
        start = match.end()
        brace_count = 1
        i = start

        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1

        block_content = text[start:i-1].strip()
        # Return the content, the start index of the node, and the end index of the node
        blocks.append({
            "content": block_content,
            "start_idx": match.start(),
            "end_idx": i
        })

    return blocks

def parse_vector(val_str):
    return [float(x) for x in val_str.strip().split()]

def parse_vector_block(block):
    block = block.replace(",", "\n")
    vectors = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vector = [float(x) for x in line.split()]
            vectors.append(vector)
        except ValueError:
            pass
    return vectors

def parse_node(block_text, node_type):
    # Extract translation
    translation = [0.0, 0.0, 0.0]
    t_match = re.search(r"translation\s+([^\n]+)", block_text)
    if t_match:
        try:
            translation = parse_vector(t_match.group(1))
        except:
            pass
            
    # Extract id
    node_id = ""
    id_match = re.search(r"id\s+\"([^\"]+)\"", block_text)
    if id_match:
        node_id = id_match.group(1)
        
    # Extract name
    name = ""
    name_match = re.search(r"name\s+\"([^\"]+)\"", block_text)
    if name_match:
        name = name_match.group(1)

    # Extract geometry points (corners for SimpleBuilding, shape for Forest, point for Transform/Pose)
    points = []
    if node_type == "SimpleBuilding":
        pts_match = re.search(r"corners\s*\[(.*?)\]", block_text, re.DOTALL)
        if pts_match:
            points = parse_vector_block(pts_match.group(1))
    elif node_type == "Forest":
        pts_match = re.search(r"shape\s*\[(.*?)\]", block_text, re.DOTALL)
        if pts_match:
            points = parse_vector_block(pts_match.group(1))
    elif node_type == "Pose":
        # In Pose, the coordinates are inside geometry IndexedFaceSet Coordinate point [ ... ]
        pts_match = re.search(r"point\s*\[(.*?)\]", block_text, re.DOTALL)
        if pts_match:
            points = parse_vector_block(pts_match.group(1))
            
    return {
        "translation": translation,
        "id": node_id,
        "name": name,
        "points": points
    }

def get_node_geometry(node_data):
    t = node_data["translation"]
    pts = node_data["points"]
    if not pts:
        return {
            "num_vertices": 0,
            "width": 0.0,
            "height": 0.0,
            "centroid": t
        }
    
    # Exclude last point if it duplicates the first point
    pts_to_use = list(pts)
    if len(pts_to_use) > 1 and all(abs(x - y) < 1e-4 for x, y in zip(pts_to_use[0], pts_to_use[-1])):
        pts_to_use = pts_to_use[:-1]
        
    global_pts = [[t[0] + p[0], t[1] + p[1]] for p in pts_to_use]
    xs = [p[0] for p in global_pts]
    ys = [p[1] for p in global_pts]
    
    width = max(xs) - min(xs) if xs else 0.0
    height = max(ys) - min(ys) if ys else 0.0
    centroid = [sum(xs)/len(xs), sum(ys)/len(ys), t[2] if len(t) > 2 else 0.0] if xs else t
    
    return {
        "num_vertices": len(pts_to_use),
        "width": width,
        "height": height,
        "centroid": centroid
    }

def load_world_nodes(wbt_path):
    with open(wbt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    nodes_by_type = {}
    for nt in ["SimpleBuilding", "Forest", "Pose"]:
        blocks = extract_blocks_with_indices(content, nt)
        nodes_by_type[nt] = []
        for b in blocks:
            node_data = parse_node(b["content"], nt)
            node_data["start_idx"] = b["start_idx"]
            node_data["end_idx"] = b["end_idx"]
            geom = get_node_geometry(node_data)
            node_data.update(geom)
            nodes_by_type[nt].append(node_data)
            
    return nodes_by_type

def match_buildings(wo_nodes, w_nodes):
    matches = {}
    w_by_name = {n["name"]: n for n in w_nodes if n["name"]}
    
    for wo in wo_nodes:
        if wo["name"] and wo["name"] in w_by_name:
            matches[wo["start_idx"]] = w_by_name[wo["name"]]["id"]
        else:
            # Fallback to closest centroid
            min_dist = float('inf')
            best_id = None
            for w in w_nodes:
                d = math.sqrt((wo["centroid"][0] - w["centroid"][0])**2 + (wo["centroid"][1] - w["centroid"][1])**2)
                if d < min_dist:
                    min_dist = d
                    best_id = w["id"]
            if best_id and min_dist < 20.0:
                matches[wo["start_idx"]] = best_id
                
    return matches

def match_poses(wo_nodes, w_nodes):
    matches = {}
    # Pose nodes are identical in structure/translation, so match by closest centroid
    for wo in wo_nodes:
        min_dist = float('inf')
        best_id = None
        for w in w_nodes:
            d = math.sqrt((wo["centroid"][0] - w["centroid"][0])**2 + (wo["centroid"][1] - w["centroid"][1])**2)
            if d < min_dist:
                min_dist = d
                best_id = w["id"]
        if best_id and min_dist < 20.0:
            matches[wo["start_idx"]] = best_id
            
    return matches

def match_forests(wo_nodes, w_nodes):
    # Forests might have shifted coordinates. Compute optimal assignment based on:
    # 1. Bounding box size similarity
    # 2. Vertex count similarity
    # 3. Global X centroid distance (which remains extremely close)
    pairs = []
    for idx_wo, f_wo in enumerate(wo_nodes):
        for f_w in w_nodes:
            w_diff = abs(f_wo["width"] - f_w["width"])
            h_diff = abs(f_wo["height"] - f_w["height"])
            v_diff = abs(f_wo["num_vertices"] - f_w["num_vertices"])
            # X centroid distance
            dx = abs(f_wo["centroid"][0] - f_w["centroid"][0])
            score = dx + 10.0 * (w_diff + h_diff) + 5.0 * v_diff
            pairs.append((idx_wo, f_w["id"], score, f_wo["start_idx"]))
            
    pairs.sort(key=lambda x: x[2])
    
    matched_wo = set()
    matched_w = set()
    matches = {}
    
    for idx_wo, source_id, score, start_idx in pairs:
        if idx_wo not in matched_wo and source_id not in matched_w:
            matched_wo.add(idx_wo)
            matched_w.add(source_id)
            matches[start_idx] = source_id
            
    return matches

def main():
    target_path = r"d:\Webots\map4.wbt"
    source_path = r"d:\Webots\map4_2.wbt"
    backup_path = r"d:\Webots\map4_backup.wbt"
    
    print("Loading nodes from source and target worlds...")
    source_nodes = load_world_nodes(source_path)
    target_nodes = load_world_nodes(target_path)
    
    print("\nMatching SimpleBuildings...")
    building_matches = match_buildings(target_nodes["SimpleBuilding"], source_nodes["SimpleBuilding"])
    print(f"Matched {len(building_matches)} / {len(target_nodes['SimpleBuilding'])} SimpleBuildings.")
    
    print("\nMatching Poses...")
    pose_matches = match_poses(target_nodes["Pose"], source_nodes["Pose"])
    print(f"Matched {len(pose_matches)} / {len(target_nodes['Pose'])} Poses.")
    
    print("\nMatching Forests...")
    forest_matches = match_forests(target_nodes["Forest"], source_nodes["Forest"])
    print(f"Matched {len(forest_matches)} / {len(target_nodes['Forest'])} Forests.")
    
    # Combine matches: maps start_idx of node in target file to ID to insert
    all_matches = {}
    all_matches.update(building_matches)
    all_matches.update(pose_matches)
    all_matches.update(forest_matches)
    
    # Sort target indices in reverse order to modify text from back to front
    sorted_indices = sorted(all_matches.keys(), reverse=True)
    
    print(f"\nCreating backup of target file: {backup_path}")
    shutil.copyfile(target_path, backup_path)
    
    with open(target_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("Inserting ID fields in-place...")
    for start_idx in sorted_indices:
        matched_id = all_matches[start_idx]
        # Find the first brace '{' after start_idx
        brace_pos = text.find("{", start_idx)
        if brace_pos != -1:
            insert_pos = brace_pos + 1
            # Check if there is already an ID field there (should not be, but safety check)
            check_slice = text[insert_pos:insert_pos+100]
            if "id \"" not in check_slice:
                id_str = f'\n  id "{matched_id}"'
                text = text[:insert_pos] + id_str + text[insert_pos:]
                
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print("\nValidation check of modified file...")
    # Load nodes from modified file to verify they now have IDs
    final_nodes = load_world_nodes(target_path)
    for nt in ["SimpleBuilding", "Forest", "Pose"]:
        total = len(final_nodes[nt])
        with_id = sum(1 for n in final_nodes[nt] if n["id"])
        print(f"  {nt}: total={total}, successfully updated with ID={with_id}")

if __name__ == "__main__":
    main()
