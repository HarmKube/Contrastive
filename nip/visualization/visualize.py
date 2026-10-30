import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


def visualize_annotation(xml_path, output=None):
    """
    Visualize an image with bounding boxes and labels from an XML annotation file.
    
    Args:
        xml_path (str): Path to the XML annotation file in Pascal VOC format
        output (str, optional): Path where to save the visualization. 
                               If None, the visualization is displayed directly.
    
    Returns:
        None
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image path from the XML
    image_path = root.find('path').text
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Define colors for different classes (will cycle through these)
    colors = plt.cm.hsv(np.linspace(0, 1, 20))
    class_colors = {}
    legend_patches = []
    
    # Find and draw all objects
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # Get confidence score if available
        confidence = obj.find('confidence')
        conf_text = f" ({float(confidence.text):.2f})" if confidence is not None else ""
        
        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        # Calculate width and height of the box
        width = xmax - xmin
        height = ymax - ymin
        
        # Assign a color to this class if not already assigned
        if class_name not in class_colors:
            class_colors[class_name] = colors[len(class_colors) % len(colors)]
            legend_patches.append(Patch(facecolor='none', edgecolor=class_colors[class_name], label=class_name))
        
        color = class_colors[class_name]
        
        # Create rectangle patch
        rect = Rectangle((xmin, ymin), width, height, 
                         linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, f"{class_name}{conf_text}", 
                color=color, fontsize=5, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Specify the graphics layout
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_title(f"Annotation: {os.path.basename(xml_path)}")
    plt.tight_layout()
    
    # Save or display
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        plt.savefig(output, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Visualization saved to {output}")
    else:
        plt.show()
