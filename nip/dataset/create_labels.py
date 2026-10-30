import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
import yaml

def create_xml_annotation(img_path, img_width, img_height, boxes, class_names, confidence_scores):
    """
    Create XML annotation in Pascal VOC format.
    
    Args:
        img_path (str): Path to the image file
        img_width (int): Width of the image
        img_height (int): Height of the image
        boxes (list): List of bounding boxes in format [x1, y1, x2, y2]
        class_names (list): List of class names corresponding to each box
        confidence_scores (list): List of confidence scores for each box
        
    Returns:
        str: XML content as a string
    """
    # Create the root element
    root = ET.Element("annotation")
    
    # Add basic information
    folder = ET.SubElement(root, "folder")
    folder.text = os.path.basename(os.path.dirname(img_path))
    
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(img_path)
    
    path = ET.SubElement(root, "path")
    path.text = str(img_path)
    
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    # Add image size information
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"  # Assuming RGB images
    
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    # Add object information
    for i, (box, class_name, score) in enumerate(zip(boxes, class_names, confidence_scores)):
        obj = ET.SubElement(root, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = class_name
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        confidence = ET.SubElement(obj, "confidence")
        confidence.text = str(round(float(score), 6))
        
        bndbox = ET.SubElement(obj, "bndbox")
        
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box[0]))
        
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[1]))
        
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[2]))
        
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[3]))
    
    # Convert to string with proper formatting
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    return xmlstr


def create_labels_from_yolo(
    data,
    model,
    output=None,
    confidence_threshold=0.25,
    device="cuda",
    batch_size=16
):
    """
    Process images using a YOLO model and generate XML annotations.
    
    Args:
        data (str): Path to the YOLO data YAML file
        model (str): Path to the trained YOLO model weights
        output (str, optional): Path where to save XML annotations. If None, creates 'annotations' folder.
        confidence_threshold (float, optional): Confidence threshold for detections. Default is 0.25.
        device (str, optional): Device to run inference on ('cpu' or 'cuda'). Default is 'cuda'.
        batch_size (int, optional): Batch size for processing images. Default is 16.
        
    Returns:
        str: Path to the folder containing generated XML annotations
    """
    # convert batch_size to int
    batch_size = int(batch_size)

    # Load YOLO model
    print(f"Loading YOLO model from {model}")
    yolo_model = YOLO(model)
    
    # Handle data parameter - only support YAML files
    data_path = Path(data)
    if not data_path.suffix.lower() in ['.yaml', '.yml']:
        raise ValueError(f"Data parameter must be a YAML file, got {data_path}")
    
    # Load YAML configuration
    print(f"Loading dataset configuration from {data_path}")
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get base directory
    base_dir = Path(data_config.get('path', '.'))
    
    # Find the images directory
    images_dir = base_dir / "images"
    if not images_dir.exists():
        raise ValueError(f"Images directory not found at {images_dir}")
    
    # Setup output folder
    if output is None:
        # Create annotations directory at the same level as the image directory
        output = base_dir / "annotations"
    else:
        output = Path(output)
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Find all images in the images directory
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"Processing {len(image_files)} images...")
    
    # Process images in batches
    batch_count = (len(image_files) + batch_size - 1) // batch_size  # Ceiling division
    for batch_idx in tqdm(range(batch_count), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_paths = [str(img_path) for img_path in image_files[start_idx:end_idx]]
        
        # Run inference on batch
        results = yolo_model(batch_paths, conf=confidence_threshold, device=device, verbose=False)
        
        # Process each result
        for img_path, result in zip(image_files[start_idx:end_idx], results):
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Extract detection results
            boxes = []
            class_names = []
            confidence_scores = []
            
            for box in result.boxes:
                # Get box coordinates
                xyxy = box.xyxy[0].cpu().numpy()  # Convert tensor to numpy array
                boxes.append(xyxy)
                
                # Get class name
                cls_id = int(box.cls.item())
                class_names.append(result.names[cls_id])
                
                # Get confidence score
                confidence_scores.append(box.conf.item())
            
            # Create XML annotation
            xml_content = create_xml_annotation(
                img_path=str(img_path),
                img_width=img_width,
                img_height=img_height,
                boxes=boxes,
                class_names=class_names,
                confidence_scores=confidence_scores
            )
            
            # Save XML file
            xml_filename = img_path.stem + ".xml"
            xml_path = output / xml_filename
            
            with open(xml_path, "w") as f:
                f.write(xml_content)
    
    print(f"Completed! Generated XML annotation files in {output}")
    return str(output)

