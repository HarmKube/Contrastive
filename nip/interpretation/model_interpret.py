import os
import traceback
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from captum.attr import LayerGradCam

import config.global_config as cfg


def get_grad_cam(yolo: YOLO, input: torch.Tensor, target, model_wrapper = None):
    """
    Generate Grad-CAM attention map using Captum's LayerGradCam for Yolov8 model.

    Args:
        model: YOLO model instance.
        target_layer_index: Index of the target layer for Grad-CAM.
        input: Input image as a NumPy array (H, W, C).
        target: Target class index for which Grad-CAM is computed.
        model_wrapper: Optional wrapper function for the model (if needed).

    Returns:
        Grad-CAM attention map as a NumPy array.
    """
    yolo_backbone_out_layer = [4,6,cfg.config.sslcon_config.yolo_backbone_end - 1 ]


    # Get the target layer
    target_layers = [yolo.model.model[x] for x in yolo_backbone_out_layer]

    # Wrap the model if a wrapper function is provided
    if model_wrapper:
        yolo = model_wrapper(yolo)

    # Initialize LayerGradCam
    grad_cams = [LayerGradCam(yolo, target_layer) for target_layer in target_layers]

    # Compute Grad-CAM
    attributions = [grad_cam.attribute(input, target=target) for grad_cam in grad_cams]

    # Convert attribution to NumPy array
    attention_maps = [attribution.squeeze().cpu().detach().numpy() for attribution in attributions]

    return attention_maps


def overlay_attention_map(image: np.ndarray, attention_maps: list[np.ndarray], 
                        alpha: float = 0.4) -> np.ndarray:
    """
    Overlay attention map on original image.
    
    Args:
        image: Original image (H, W, 3)
        attention_map: Attention map (H, W)
        alpha: Transparency of overlay
        
    Returns:
        Image with overlaid attention map
    """
    # Resize attention map to match image size
    attentions_resized = [cv2.resize(attention_map, (image.shape[1], image.shape[0])) for attention_map in attention_maps]
    
    attention_resized = sum(attentions_resized)

    # Normalize values to [-1, 1] range
    attention_min = attention_resized.min()
    attention_max = attention_resized.max()
    if attention_max - attention_min > 0:
        attention_normalized = 2 * (attention_resized - attention_min) / (attention_max - attention_min) - 1
    else:
        attention_normalized = np.zeros_like(attention_resized)
    
    # Convert to colormap
    colormap = cm.get_cmap('jet')
    attention_colored = colormap(attention_normalized)[:, :, :3]  # Remove alpha channel
    attention_colored = (attention_colored * 255).astype(np.uint8)

    # Overlay
    overlaid = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)
    
    return overlaid, attention_resized

def load_images(data:str, image_indices: List[int]):
    # Load dataset configuration
    with open(data, 'r') as file:
        dataset_config = yaml.safe_load(file)
    
    # Get dataset root path and validation images path
    dataset_root = dataset_config['path']
    val_images_path = os.path.join(dataset_root, dataset_config.get('val', 'images/val'))
    
    print(f"Loading images from: {val_images_path}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if os.path.exists(val_images_path):
        for file in sorted(os.listdir(val_images_path)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(val_images_path, file))
    else:
        raise FileNotFoundError(f"Validation images path not found: {val_images_path}")
    
    # Process specified image indices
    for idx in image_indices:
        if idx >= len(image_files):
            print(f"Warning: Index {idx} is out of range (max: {len(image_files) - 1})")
            continue
            
        image_path = image_files[idx]
        print(f"Processing image {idx}: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
            
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model (resize to model input size)
        input_size = 1920  # Standard YOLO input size
        resized_image = cv2.resize(original_image_rgb, (input_size, input_size))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor.requires_grad_(True)  # Enable gradient computation

        yield image_tensor, original_image_rgb, image_path, idx


def generate_attention_maps(model_path:str, output: str, data: str, 
                          image_indices: List[int], model_wrapper, visualize_bbox = False) -> None:
    """
    Generate attention maps for specified images using a trained YOLO model.
    
    Args:
        output: Path to training output folder containing best.pt
        data: Path to dataset YAML configuration
        image_indices: List of image indices to process
    """

    # Load the trained model
    inference_model = YOLO(model_path)

    output_folder = os.path.join(output, "interpretation")
    if not os.path.exists(output_folder):
        os.mkdir(os.path.join(output, "interpretation"))    
    
    # Process specified image indices
    for image_tensor, original_image_rgb, image_path, idx in load_images(data, image_indices):
        try:
            target = 0
            scaled_bbox = None
            # Get Bounding box if available and set to draw
            if visualize_bbox:
                # Use a clone for inference to avoid affecting the original tensor
                inference_tensor = image_tensor.clone().detach()
                predictions = inference_model(inference_tensor)
                if predictions[0].boxes is not None and len(predictions[0].boxes) > 0:
                    # Get the bounding box of the maximum confidence prediction
                    max_conf_idx = torch.argmax(predictions[0].boxes.conf)
                    max_conf_bbox_model = predictions[0].boxes.xyxy[max_conf_idx].cpu().numpy()
                    max_conf_class = predictions[0].boxes.cls[max_conf_idx].cpu().numpy()
                    target = int(max_conf_class)  # Use the class index as target
                    
                    # Scale bounding box from model input size (640x640) to original image size
                    model_input_size = 1920
                    orig_h, orig_w = original_image_rgb.shape[:2]
                    
                    # Scale coordinates
                    scale_x = orig_w / model_input_size
                    scale_y = orig_h / model_input_size
                    
                    scaled_bbox = [
                        max_conf_bbox_model[0] * scale_x,  # x1
                        max_conf_bbox_model[1] * scale_y,  # y1
                        max_conf_bbox_model[2] * scale_x,  # x2
                        max_conf_bbox_model[3] * scale_y   # y2
                    ]

            model = YOLO(model_path)
            # Generate attention map using original tensor with gradients
            attention_maps = get_grad_cam(model, image_tensor, target=target, model_wrapper=model_wrapper)
            
            # Overlay attention map on original image
            overlaid_image, attention_map = overlay_attention_map(original_image_rgb, attention_maps)

            # Draw bounding box if available
            if visualize_bbox and scaled_bbox is not None:
                overlaid_image = draw_bounding_box(overlaid_image, scaled_bbox)
                original_with_box = draw_bounding_box(original_image_rgb, scaled_bbox)
            else:
                original_with_box = original_image_rgb
            
            # Create visualization with original, attention map, and overlay
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image (with bounding box if available)
            axes[0].imshow(original_with_box)
            axes[0].set_title('Original Image' + (' + BBox' if visualize_bbox else ''))
            axes[0].axis('off')
            
            # Attention map
            im = axes[1].imshow(attention_map, cmap='jet')
            axes[1].set_title('Attention Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Overlaid image (with bounding box if available)
            axes[2].imshow(overlaid_image)
            axes[2].set_title('Attention Overlay' + (' + BBox' if visualize_bbox else ''))
            axes[2].axis('off')
            
            # Save the visualization
            output_filename = f"attention_map_idx_{idx}_{Path(image_path).stem}.png"
            output_path = os.path.join(output_folder, output_filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved attention map visualization: {output_path}")
            
            # Also save individual components
            individual_dir = os.path.join(output_folder, "individual_maps")
            os.makedirs(individual_dir, exist_ok=True)
            
            # Save attention map as separate image
            attention_colored = cm.jet(attention_map)[:, :, :3]
            attention_image = (attention_colored * 255).astype(np.uint8)
            attention_path = os.path.join(individual_dir, f"attention_idx_{idx}_{Path(image_path).stem}.png")
            cv2.imwrite(attention_path, cv2.cvtColor(attention_image, cv2.COLOR_RGB2BGR))
            
            # Save overlay
            overlay_path = os.path.join(individual_dir, f"overlay_idx_{idx}_{Path(image_path).stem}.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            traceback.print_exc()
            continue
    
    print(f"Attention map generation completed. Results saved in: {output_folder}")


def interpret_global(model:str, output: str, data: str, 
                          indices: str):
    """Generate global attention maps for specified images using a trained YOLO model."""

    indices_list = [int(idx.strip()) for idx in indices.split(',') if idx.strip()]

    def model_wrapper(yolo:YOLO):
        """
        Wrap Yolo model to only use Backbone. 
        Also sum the output of the last layer to get a single output for Grad-CAM. 
        """
        backbone = yolo.model.model[:cfg.config.sslcon_config.yolo_backbone_end]
        def forward(x):
            x = backbone(x)
            return torch.sum(x, dim=(1,2,3)).unsqueeze(1)  # Sum over spatial dimensions to get a single scalar output per image in batch
        
        return forward
    
    generate_attention_maps(model, output, data, indices_list, 
                            model_wrapper=model_wrapper)
    
def interpret_local(model:str, output: str, data: str, 
                          indices: str):
    """Generate local attention maps for specified images using a trained YOLO model."""

    indices_list = [int(idx.strip()) for idx in indices.split(',') if idx.strip()]

    def model_wrapper(yolo: YOLO):
        """
        Wrap YOLO model to work with raw model outputs.
        This wrapper ensures compatibility with LayerGradCam by returning
        a single scalar value from the raw model predictions.
        """
        
        def forward(x):
            # Get raw predictions from the model (before post-processing)
            raw_predictions = yolo.model(x)
            # raw_predictions is typically a list of tensors
            # For YOLOv8, this contains [batch_size, num_predictions, 4+num_classes]
            # where 4 is bbox coords and the rest are class probabilities
            
            if isinstance(raw_predictions, (list, tuple)):
                # Take the first prediction tensor
                pred_tensor = raw_predictions[0]
            else:
                pred_tensor = raw_predictions
            
            # Extract confidence scores (assuming last columns are class probabilities)
            # For YOLOv8: shape is [batch_size, BBox and classes, predictions]
            if len(pred_tensor.shape) == 3:
                # Get the maximum confidence across all predictions and classes
                # Skip the first 4 columns (bbox coordinates)
                class_scores = pred_tensor[:, 4:, :]

                class_scores_sum = torch.sum(class_scores, dim=2)
            else:
                raise ValueError("Unexpected prediction tensor shape: {}".format(pred_tensor.shape))
            
            # Return the maximum confidence, ensuring it's properly shaped
            print(class_scores_sum.shape)
            return class_scores_sum
        
        return forward
    
    generate_attention_maps(model, output, data, indices_list,
                            model_wrapper=model_wrapper, visualize_bbox=True)

def draw_bounding_box(image: np.ndarray, bbox: List[float], color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box on image.
    
    Args:
        image: Image array (H, W, 3)
        bbox: Bounding box coordinates [x1, y1, x2, y2] in image coordinates
        color: RGB color tuple for the box
        thickness: Line thickness
        
    Returns:
        Image with bounding box drawn
    """
    if bbox is None:
        return image
    
    image_with_box = image.copy()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    
    # Draw rectangle
    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), color, thickness)
    
    # Add label
    label = f"Max Conf Box"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(image_with_box, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
    cv2.putText(image_with_box, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_with_box