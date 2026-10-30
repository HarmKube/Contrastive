import json
import re
from matplotlib import pyplot as plt
import yaml
import os
import xml.etree.ElementTree as ET
from collections import Counter
import numpy as np


def analyse_dataset(data: str, split: str = ''):
    """
    Analyze a dataset based on its YAML definition and return statistics.
    
    Args:
        dataset_definition: Path to the YAML file defining the dataset
        split: Dataset split to analyze (train, val, test). If empty, uses annotations folder
        
    Returns:
        Dictionary containing dataset statistics
    """
    
    # Load dataset definition
    with open(data, 'r') as f:
        definition = yaml.safe_load(f)
    
    # Extract class information from 'names' field
    names = definition.get('names', {})
    classes = [names.get(i, f"class_{i}") for i in range(max(names.keys()) + 1)] if names else []
    
    # Determine annotations directory path
    dataset_root = definition.get('path', os.path.dirname(data))
    
    if split:
        # Use specified split (train, val, test)
        split_dir = definition.get(split, '')
        annotations_dir = os.path.join(dataset_root,split_dir, 'annotations')
    else:
        # Default to annotations in the same folder as definition file
        annotations_dir = os.path.join(os.path.dirname(data), 'annotations')
    
    # Initialize statistics
    total_images = 0
    images_with_objects = 0
    objects_per_image = []
    objects_per_class = Counter()
    resolutions = set()

    # Process XML annotations
    if os.path.isdir(annotations_dir):
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        total_images = len(annotation_files)
        
        for annotation_file in annotation_files:
            try:
                tree = ET.parse(os.path.join(annotations_dir, annotation_file))
                root = tree.getroot()

                # Get image size
                height = int(root.find('size/height').text)
                resolutions.add(height)
                
                objects = root.findall('./object')
                num_objects = len(objects)

                objects_per_image.append(num_objects)
                
                if num_objects > 0:
                    images_with_objects += 1
                    
                # Count objects by class
                for obj in objects:
                    class_name = obj.find('name').text if obj.find('name') is not None else None
                    if class_name in names.values():
                        # Get class index from name
                        class_idx = [k for k, v in names.items() if v == class_name][0]
                        objects_per_class[classes[class_idx]] += 1
                    else:
                        # If class name not in names dict, just count it separately
                        objects_per_class[class_name] += 1
            except ET.ParseError:
                print(f"Error parsing XML file: {annotation_file}")
                continue
    
    # Calculate statistics
    total_objects = sum(objects_per_image)
    avg_objects_per_image = total_objects / total_images if total_images > 0 else 0
    max_objects_per_image_idx = max(range(len(objects_per_image)), key=objects_per_image.__getitem__) if objects_per_image else 0    
    
    # Prepare results
    stats = {
        'total_images': total_images,
        'images_with_objects': images_with_objects,
        'total_objects': total_objects,
        'avg_objects_per_image': avg_objects_per_image,
        'max_objects_per_image': objects_per_image[max_objects_per_image_idx],
        'max_objects_per_image_file': annotation_files[max_objects_per_image_idx],
        'resolutions': [f'{res}p' for res in resolutions],
        'objects_per_class': dict(objects_per_class)
    }

    with open(os.path.join(dataset_root, 'dataset_analysis.json'), 'w') as f:
        json.dump(stats, f)

    visualize_analysis(stats, dataset_root)
    
    return stats
    
def visualize_analysis(analysis: dict, output_dir: str = None):
    """
    Visualize the analysis results from a JSON file.
    
    Args:
        analysis (dict): Dictionary containing the analysis results
        output_dir (str, optional): Directory to save the output image. If None, the image will be displayed.
    """

    # Order objects
    analysis['objects_per_class'] = dict(sorted(analysis['objects_per_class'].items(), key=lambda x: x[1], reverse=True))

    # Create a bar chart for objects per class
    plt.figure(figsize=(12, 8))
    plt.bar(analysis['objects_per_class'].keys(), analysis['objects_per_class'].values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Objects')
    plt.title('Objects per Class')
    plt.xticks(rotation=90, ha='right')  # Rotate labels vertically and align them to the right
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'objects_per_class.png'))
    else:
        plt.show()

def compare_analysis(file_1: str, file_2: str):
    """
    Compare two dataset analyses by creating a side-by-side bar plot of object occurrences.
    
    Args:
        analysis_file_1 (str): Path to the first analysis JSON file
        analysis_file_2 (str): Path to the second analysis JSON file
    """
    # Load both analysis files
    with open(file_1, 'r') as f:
        analysis1 = json.load(f)
    with open(file_2, 'r') as f:
        analysis2 = json.load(f)

    dataset_name1 = file_1.split('/')[-2]
    dataset_name2 = file_2.split('/')[-2]
    
    total_objects1 = analysis1['total_objects']
    total_objects2 = analysis2['total_objects']

    # Get objects per class from both analyses
    objects1 = analysis1['objects_per_class']
    objects2 = analysis2['objects_per_class']
    
    # Sort classes based on first analysis counts
    sorted_classes = sorted(objects1.keys(), key=lambda x: objects1[x], reverse=True)
    
    # Get counts for both analyses in the same order
    counts1 = [float(objects1[cls])/total_objects1 for cls in sorted_classes]
    counts2 = [float(objects2.get(cls, 0))/total_objects2 for cls in sorted_classes]  # Use 0 if class not in second analysis
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(sorted_classes))
    r2 = [x + barWidth for x in r1]
    
    # Create the bars
    plt.bar(r1, counts1, width=barWidth, label=dataset_name1, color='skyblue')
    plt.bar(r2, counts2, width=barWidth, label=dataset_name2, color='lightcoral')
    
    # Add labels and title
    plt.xlabel('Classes')
    plt.ylabel('Relative Number of Objects')
    plt.title('Comparison of Object Occurrences')
    
    # Add class names as x-axis labels
    plt.xticks([r + barWidth/2 for r in range(len(sorted_classes))], sorted_classes, rotation=45, ha='right')
    
    # Add legend
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.dirname(file_1)
    plt.savefig(os.path.join(output_dir, f'{dataset_name1}_vs_{dataset_name2}.png'))
    plt.close()

def track_analysis(data: str, split: str = '', output_dir:str = ""):
    """
    Analyze tracking data from XML annotations and generate visualizations.
    Plots number of objects over track time
    
    Args:
        data (str): Path to the dataset definition YAML file
        split (str, optional): Dataset split to analyze (train, val, test). If empty, uses annotations folder
        output_dir (str, optional): Directory to save output visualizations. Defaults to "".
    """

    # Load dataset definition
    with open(data, 'r') as f:
        definition = yaml.safe_load(f)
    
    # Determine annotations directory path
    dataset_root = definition.get('path', os.path.dirname(data))
    
    if split:
        # Use specified split (train, val, test)
        split_dir = definition.get(split, '')
        annotations_dir = os.path.join(dataset_root,split_dir, 'annotations')
    else:
        # Default to annotations in the same folder as definition file
        annotations_dir = os.path.join(os.path.dirname(data), 'annotations')

    tracks = dict()

    # Process XML annotations
    if os.path.isdir(annotations_dir):
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        for annotation_file in annotation_files:
            r'^(\w+)#t=(\d+.\d+).xml'
            match = re.match(r'^(\w+)#t=(\d+.\d+).xml',annotation_file)
            track_name = match.groups()[0]
            track_time = float(match.groups()[1])

            if track_name not in tracks.keys():
                tracks[track_name] = []

            try:
                tree = ET.parse(os.path.join(annotations_dir, annotation_file))
                root = tree.getroot()
                
                objects = root.findall('./object')
                num_objects = len(objects)

                tracks[track_name].append([track_time, num_objects])

            except ET.ParseError:
                print(f"Error parsing XML file: {annotation_file}")
                continue
        
        # Plot each track with a different color
        for track_name, data in tracks.items():
            plt.figure(figsize=(12, 6))
            np_array = np.array(data)
            np_array = np_array[np_array[:, 0].argsort()]

            plt.plot(np_array[:,0], np_array[:,1], label=track_name, marker='o')        
            plt.xlabel('Time (seconds)')
            plt.ylabel('Number of Objects')
            plt.title('Object Count Over Time for Each Track')
            plt.legend()
            plt.grid(True)            

            plt.savefig(os.path.join(dataset_root, f'track_analysis_{track_name}.png'))