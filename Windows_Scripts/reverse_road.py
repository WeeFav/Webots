import sys
import re
import argparse

def get_field_val(block, field_name, is_list=False):
    if is_list:
        pattern = r'\b' + field_name + r'\s*\[([^\]]*)\]'
    else:
        pattern = r'\b' + field_name + r'\s*"([^"]*)"'
    match = re.search(pattern, block)
    return match.group(1) if match else None

def set_field_val(block, field_name, new_val, is_list=False):
    if is_list:
        pattern = r'(\b' + field_name + r'\s*\[)[^\]]*(\])'
        replacement = rf'\g<1>{new_val}\g<2>'
    else:
        pattern = r'(\b' + field_name + r'\s*")[^"]*(")'
        replacement = rf'\g<1>{new_val}\g<2>'
    
    if re.search(pattern, block):
        return re.sub(pattern, replacement, block)
    else:
        # Field doesn't exist, insert it before the closing brace '}'
        idx = block.rfind('}')
        if idx != -1:
            if is_list:
                field_str = f'\n  {field_name} [{new_val}]\n'
            else:
                field_str = f'\n  {field_name} "{new_val}"\n'
            return block[:idx] + field_str + block[idx:]
        return block

def find_road_blocks(wbt_content):
    blocks = []
    idx = 0
    n = len(wbt_content)
    while True:
        match = re.search(r'\bRoad\s*\{', wbt_content[idx:])
        if not match:
            break
        
        start_pos = idx + match.start()
        brace_count = 1
        curr = start_pos + match.end()
        while curr < n and brace_count > 0:
            char = wbt_content[curr]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            curr += 1
        
        if brace_count == 0:
            blocks.append((start_pos, curr))
            idx = curr
        else:
            break
    return blocks

def reverse_roads_in_wbt(file_path, target_ids):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = find_road_blocks(content)
    print(f"Found {len(blocks)} Road nodes in file.")
    
    modified = False
    # Process blocks in reverse order of indices to avoid index shifting problems
    for start_idx, end_idx in reversed(blocks):
        block_text = content[start_idx:end_idx]
        road_id = get_field_val(block_text, 'id', is_list=False)
        if road_id in target_ids:
            print(f"Reversing road ID: {road_id}")
            
            # 1. Reverse wayPoints
            waypoints_content = get_field_val(block_text, 'wayPoints', is_list=True)
            if waypoints_content is not None:
                lines = [l.strip() for l in waypoints_content.split('\n') if l.strip()]
                reversed_lines = lines[::-1]
                new_waypoints_content = "\n" + "\n".join(f"    {l}" for l in reversed_lines) + "\n  "
                block_text = set_field_val(block_text, 'wayPoints', new_waypoints_content, is_list=True)
            
            # 2. Swap startJunction and endJunction
            start_junc_val = get_field_val(block_text, 'startJunction', is_list=False)
            end_junc_val = get_field_val(block_text, 'endJunction', is_list=False)
            if start_junc_val is not None or end_junc_val is not None:
                new_start = end_junc_val if end_junc_val is not None else ""
                new_end = start_junc_val if start_junc_val is not None else ""
                block_text = set_field_val(block_text, 'startJunction', new_start, is_list=False)
                block_text = set_field_val(block_text, 'endJunction', new_end, is_list=False)
            
            # 3. Swap startLine and endLine textures
            start_line_val = get_field_val(block_text, 'startLine', is_list=True)
            end_line_val = get_field_val(block_text, 'endLine', is_list=True)
            if start_line_val is not None or end_line_val is not None:
                new_start_line = end_line_val if end_line_val is not None else ""
                new_end_line = start_line_val if start_line_val is not None else ""
                block_text = set_field_val(block_text, 'startLine', new_start_line, is_list=True)
                block_text = set_field_val(block_text, 'endLine', new_end_line, is_list=True)
            
            # Replace the old block with modified block
            content = content[:start_idx] + block_text + content[end_idx:]
            modified = True
            
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("File updated successfully.")
    else:
        print("No matching road IDs found or no modifications made.")

def main():
    parser = argparse.ArgumentParser(description="Reverse direction of specific Road nodes in Webots .wbt file.")
    parser.add_argument("file", help="Path to the Webots .wbt file")
    parser.add_argument("ids", nargs="+", help="List of road IDs to flip")
    args = parser.parse_args()
    
    reverse_roads_in_wbt(args.file, args.ids)

if __name__ == "__main__":
    main()
