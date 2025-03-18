import cv2
import torch
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = './checkpoint_ssd.pth.tar'
checkpoint = torch.load(checkpoint, map_location=device)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200):
    """
    Detect objects in an image with a trained SSD300 and return annotated image.

    :param original_image: image, a PIL Image
    :param min_score: minimum score threshold
    :param max_overlap: maximum IoU for non-max suppression
    :param top_k: top K results to keep
    :return: annotated image, a PIL Image
    """
    image = normalize(to_tensor(resize(original_image))).to(device)

    # Forward propagation
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores,
                                                             min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')

    # Scale boxes back to original image size
    original_dims = torch.FloatTensor([original_image.width, original_image.height,
                                       original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return original_image

    # Annotate image
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("./SSD/calibril.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i in range(det_boxes.size(0)):
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline="red", width=2)

        text_location = [box_location[0] + 2., box_location[1] - 10]
        draw.text(xy=text_location, text=det_labels[i].upper(), fill="red", font=font)

    return annotated_image

def detect_video(video_path, output_path=None, min_score=0.2, max_overlap=0.5, top_k=200):
    """
    Perform object detection on a video file using SSD.

    :param video_path: Path to input video
    :param output_path: Path to save output video (optional)
    :param min_score: Minimum score for detection
    :param max_overlap: Max overlap for non-max suppression
    :param top_k: Top K detections per frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect objects
        annotated_image = detect(pil_image, min_score, max_overlap, top_k)

        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

        # Display frame
        cv2.imshow("Object Detection", frame)

        # Save frame if output path is given
        if output_path:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'input_video.mp4'  # 변경 가능
    output_path = 'output_video.mp4'  # 변경 가능
    detect_video(video_path, output_path)
