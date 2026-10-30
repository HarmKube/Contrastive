import os
import xml.etree.ElementTree as ET

# Paths
input_dir = "annotations"
output_dir = "labels"
classes_file = "classes.txt"

# Load class names
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

os.makedirs(output_dir, exist_ok=True)

def convert_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    yolo_annotations = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            print(f"Warning: '{class_name}' not in classes.txt")
            continue
        class_id = classes.index(class_name)

        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        yolo_annotations.append(yolo_line)

    return yolo_annotations

# Process XML files
for file in os.listdir(input_dir):
    if file.endswith(".xml"):
        xml_path = os.path.join(input_dir, file)
        yolo_lines = convert_annotation(xml_path)

        output_file = os.path.join(output_dir, file.replace(".xml", ".txt"))
        with open(output_file, "w") as f:
            f.write("\n".join(yolo_lines))
