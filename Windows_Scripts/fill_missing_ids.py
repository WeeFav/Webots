import os
import re
import shutil

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
        blocks.append({
            "node_name": node_name,
            "content": block_content,
            "start_idx": match.start(),
            "end_idx": i
        })

    return blocks

def parse_node(block_text):
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

    return {
        "id": node_id,
        "name": name
    }

def main():
    target_path = r"d:\Webots\map4_backup.wbt"
    backup_path = r"d:\Webots\map4_backup_fill.wbt"
    
    if not os.path.exists(target_path):
        print(f"Target file {target_path} not found.")
        return
        
    print(f"Reading target file: {target_path}...")
    with open(target_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # 1. Dynamically scan all EXTERNPROTOs and common node types
    supported_types = {"SimpleBuilding", "Forest", "Pose", "Transform", "Road", "Crossroad"}
    extern_matches = re.findall(r'EXTERNPROTO\s+"(?:[^"]+/)?([^"/]+)\.proto"', text)
    for name in extern_matches:
        # Ignore backgrounds, floors, and basic textures/appearances
        if not any(x in name for x in ["Background", "Floor", "Soil", "CementTiles", "RoadLine"]):
            supported_types.add(name)
            
    print(f"Scanning for node types: {sorted(list(supported_types))}")
    
    # 2. Extract all blocks and collect seen IDs
    all_blocks = []
    seen_ids = set()
    
    for nt in supported_types:
        blocks = extract_blocks_with_indices(text, nt)
        for b in blocks:
            node_data = parse_node(b["content"])
            b.update(node_data)
            all_blocks.append(b)
            if b["id"]:
                seen_ids.add(b["id"])
                
    print(f"Found {len(all_blocks)} total target nodes.")
    print(f"Collected {len(seen_ids)} existing unique IDs.")
    
    # Helper to generate unique IDs
    unique_counter = 1
    def generate_unique_id(seen):
        nonlocal unique_counter
        while True:
            candidate = f"gen_{unique_counter}"
            if candidate not in seen:
                seen.add(candidate)
                return candidate
            unique_counter += 1
            
    def make_unique_id(candidate, seen):
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        suffix = 1
        while True:
            new_candidate = f"{candidate}_{suffix}"
            if new_candidate not in seen:
                seen.add(new_candidate)
                return new_candidate
            suffix += 1

    # 3. Plan the ID insertions
    insertions = [] # list of (start_idx, new_id)
    skipped_count = 0
    updated_count = 0
    
    for b in all_blocks:
        if b["id"]:
            skipped_count += 1
            continue
            
        # Node has no ID. Determine what to use.
        if b["name"]:
            new_id = make_unique_id(b["name"], seen_ids)
        else:
            new_id = generate_unique_id(seen_ids)
            
        insertions.append((b["start_idx"], new_id))
        updated_count += 1
        
    print(f"Skipped {skipped_count} nodes that already have IDs.")
    print(f"Need to assign IDs to {updated_count} nodes.")
    
    if updated_count == 0:
        print("All nodes already have IDs. No changes needed.")
        return
        
    # 4. Create backup and apply insertions from back to front
    print(f"Creating backup file at: {backup_path}")
    shutil.copyfile(target_path, backup_path)
    
    # Sort insertions in reverse order of start_idx to prevent shift in indices
    insertions.sort(key=lambda x: x[0], reverse=True)
    
    print("Modifying file in-place...")
    for start_idx, new_id in insertions:
        # Find the first brace '{' after start_idx
        brace_pos = text.find("{", start_idx)
        if brace_pos != -1:
            insert_pos = brace_pos + 1
            id_str = f'\n  id "{new_id}"'
            text = text[:insert_pos] + id_str + text[insert_pos:]
            
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    print("IDs successfully filled. Verification results:")
    # Verification parse
    verify_text = text
    verify_blocks = []
    verify_seen = set()
    missing_id_count = 0
    
    for nt in supported_types:
        blocks = extract_blocks_with_indices(verify_text, nt)
        for b in blocks:
            node_data = parse_node(b["content"])
            if node_data["id"]:
                verify_seen.add(node_data["id"])
            else:
                missing_id_count += 1
                
    print(f"  Total unique IDs in updated file: {len(verify_seen)}")
    print(f"  Nodes still missing IDs: {missing_id_count}")

if __name__ == "__main__":
    main()
