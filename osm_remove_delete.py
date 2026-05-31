import xml.etree.ElementTree as ET
import argparse

def remove_deleted_objects(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Remove top-level elements marked for deletion
    for elem in list(root):
        if elem.get("action") == "delete":
            root.remove(elem)

    tree.write(output_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()  
    remove_deleted_objects(args.input, args.output)