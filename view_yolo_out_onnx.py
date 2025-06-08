import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
import torch
import torchvision.ops as ops  # For NMS

# === Load YOLOv10 ONNX model with OpenVINO (Intel GPU) ===
session = ort.InferenceSession(
    "PIPE.onnx",
    providers=["OpenVINOExecutionProvider"]
)

# === Helper: preprocess frame ===
def preprocess(frame, input_shape):
    img = cv2.resize(frame, input_shape)
    img = img[..., ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img

# === Helper: postprocess for YOLOv10 raw output ===
def postprocess(output, orig_shape, input_shape=(640, 640), conf_thres=0.3, iou_thres=0.45):
    pred = output[0]  # (1, num_dets, 84)
    pred = torch.tensor(pred[0])  # Remove batch dimension

    boxes = []
    scores = []
    class_ids = []

    for row in pred:
        cx, cy, w, h, obj_conf, *class_probs = row.tolist()
        class_probs = torch.tensor(class_probs)
        class_conf, cls = torch.max(class_probs, dim=0)
        score = obj_conf * class_conf

        if score < conf_thres:
            continue

        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Rescale to original image size
        scale_x = orig_shape[1] / input_shape[0]
        scale_y = orig_shape[0] / input_shape[1]
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        class_ids.append(cls.item())

    if not boxes:
        return []

    # Apply NMS
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    keep = ops.nms(boxes, scores, iou_thres)

    result = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i].int().tolist()
        score = scores[i].item()
        cls = class_ids[i]
        result.append((x1, y1, x2, y2, score, cls))

    return result

# === Webcam setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Determine input shape from model ===
input_name = session.get_inputs()[0].name
input_shape = tuple(session.get_inputs()[0].shape[2:])  # (640, 640)

# === Setup matplotlib display ===
plt.ion()
fig, ax = plt.subplots()
im = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_input = preprocess(frame, input_shape)
    outputs = session.run(None, {input_name: img_input})

    # Postprocess to get boxes
    boxes = postprocess(outputs, frame.shape[:2], input_shape)

    # Draw boxes
    for i, (x1, y1, x2, y2, score, cls) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{i} ({score:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Convert to RGB and display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if im is None:
        im = ax.imshow(frame_rgb)
    else:
        im.set_data(frame_rgb)
    plt.pause(0.001)

    if plt.get_fignums() == []:
        break

cap.release()
plt.ioff()
plt.close()
