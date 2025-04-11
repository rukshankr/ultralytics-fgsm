import cv2
from ultralytics import YOLO
import numpy as np
import os

yolo_model_path = r"E:\PhD\PyCharmProjects\plant-phenotyping-codes\25Mar2025\yolo11n.pt"
output_folder = r".\\"
dataset_yaml_path = r"E:\PhD\Sorgham Datasets\Sorghum_UAV_U_Tokyo\Sorghum_UAV_U_YOLO\data.yaml"

photos = []
bboxes = []


model = YOLO(yolo_model_path)

def save_yolo_labels(bboxes, save_path, filename):
    """
    Save YOLO bounding boxes to a .txt file.

    Args:
        bboxes (np.ndarray): (N, 4) array with [x_center, y_center, width, height] normalized to [0, 1]
        save_path (str): Directory to save the .txt file
        filename (str): Filename (without extension), e.g., "image_001"
    """
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, f"{filename}.txt")

    with open(txt_path, 'w') as f:
        for bbox in bboxes:
            x, y, w, h = bbox
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")



def draw_yolo_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw YOLO-format bounding boxes on an image.

    Args:
        image (np.ndarray): HWC image in uint8 format (e.g., (640, 640, 3))
        bboxes (np.ndarray): (N, 4) array of normalized [x_center, y_center, width, height]
        color (tuple): BGR color for the boxes
        thickness (int): Line thickness

    Returns:
        np.ndarray: image with boxes drawn
    """
    img_h, img_w = image.shape[:2]

    for bbox in bboxes:
        x_center, y_center, box_w, box_h = bbox

        # Convert from normalized to absolute pixel values
        x_center *= img_w
        y_center *= img_h
        box_w *= img_w
        box_h *= img_h

        # Calculate top-left and bottom-right corners
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image


def put_in_eval_mode(trainer):
    photos.append(trainer.adversarial_images)
    bboxes.append(trainer.out_boxes)
    print(trainer.adversarial_images.shape)


def chw_float_tensor_to_hwc_uint8(image_tensor):
    # Ensure it's detached from computation graph and converted to numpy
    image_np = image_tensor.detach().cpu().numpy()  # shape: (3, H, W)

    # If it's normalized to [0, 1], scale to [0, 255]
    # if image_np.max() <= 1.1: # 1.0 + 0.1 for augmentations done on the image
    image_np = image_np * 255.0

    # Convert to uint8 and transpose to HWC format
    image_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    image_hwc = np.transpose(image_uint8, (1, 2, 0))  # (H, W, C)

    return image_hwc

if __name__ == "__main__":
    model.add_callback("on_train_batch_end", put_in_eval_mode)
    results = model.train(data=dataset_yaml_path,
                          epochs=1,
                          imgsz=1024,
                          augment=False,
                          mosaic=0.0,
                          rect=True,
                          advstyle=True,
                          batch=1)

    for b, batch in enumerate(photos):
        batch_boxes = bboxes[b]
        for i, image in enumerate(batch):
            img = chw_float_tensor_to_hwc_uint8(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = batch_boxes.detach().cpu().numpy()
            labeled_img = draw_yolo_bboxes(img.copy(), label)
            print(img.shape)

            cv2.imwrite(output_folder +f"FGMS_{b}_{i}.jpg", img)
            save_yolo_labels(label, output_folder + "labels",f"FGMS_{b}_{i}")
