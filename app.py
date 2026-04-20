import gradio as gr
from gradio_client import Client, handle_file
import torch
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from dotenv import load_dotenv

# Nạp mã bảo mật từ file .env (nếu có)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# 1. KẾT NỐI API ĐẾN XƯỞNG TRELLIS GỐC
# ==========================================
# Kết nối đến Space chính thức của Microsoft/JeffreyXiang
# Nếu có HF_TOKEN, tốc độ sẽ ưu tiên và không bị giới hạn
print("🔗 Đang kết nối API với hệ thống TRELLIS...")
client = Client("JeffreyXiang/TRELLIS", hf_token=HF_TOKEN)

# ==========================================
# 2. NẠP MÔ HÌNH XỬ LÝ ẢNH TẠI CHỖ
# ==========================================
from ultralytics import YOLO
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔍 Đang nạp YOLOv11 & BiRefNet...")
yolo_model = YOLO("yolo11n.pt") 
birefnet_model = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
birefnet_model.to(device)

birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. LOGIC XỬ LÝ (GIỮ NGUYÊN CÔNG THỨC CỦA BÁC)
# ==========================================
def process_image_locally(input_image):
    # YOLO soi vật thể
    results = yolo_model(input_image, conf=0.5)
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = map(int, boxes[np.argmax(areas)])
        input_image = input_image.crop((x1, y1, x2, y2))

    # BiRefNet tỉa nền & Căn giữa 80%
    img_rgb = input_image.convert("RGB")
    input_tensor = birefnet_transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet_model(input_tensor)[-1].sigmoid().cpu()
    
    mask = transforms.ToPILImage()(preds[0].squeeze()).resize(img_rgb.size)
    img_rgba = img_rgb.copy()
    img_rgba.putalpha(mask)
    
    # Logic 80% của bác
    mask_np = np.array(mask)
    coords = cv2.findNonZero(mask_np)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_rgba.crop((x, y, x+w, y+h))
        max_dim = max(w, h)
        new_size = int(max_dim / 0.8)
        final_img = Image.new("RGBA", (new_size, new_size), (0, 0, 0, 0))
        final_img.paste(cropped, ((new_size - w) // 2, (new_size - h) // 2))
        return final_img
    return img_rgba

# ==========================================
# 4. HÀM CHÍNH GỌI API
# ==========================================
def run_3d_engine(input_image):
    if input_image is None: return "❌ Cần ảnh!", None, None
    
    yield "⚙️ Đang tiền xử lý ảnh (YOLO + BiRefNet)...", None, None
    processed_image = process_image_locally(input_image)
    
    temp_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    processed_image.save(temp_path)
    
    yield "🌐 Đang gửi sang Microsoft TRELLIS qua API...", processed_image, None
    
    try:
        # Gọi API nặn 3D
        result = client.predict(
            image=handle_file(temp_path),
            seed=0,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3,
            slat_sampling_steps=12,
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/image_to_3d"
        )
        yield "🎉 Thành công! Đã nhận file 3D.", processed_image, result[0]
    except Exception as e:
        yield f"❌ Lỗi kết nối API: {str(e)}", processed_image, None

# ==========================================
# 5. GIAO DIỆN
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 SaaS 3D - GitHub Professional Edition")
    with gr.Row():
        with gr.Column():
            in_img = gr.Image(type="pil")
            run_btn = gr.Button("NẶN 3D NGAY", variant="primary")
        with gr.Column():
            txt_status = gr.Textbox(label="Trạng thái")
            out_pre = gr.Image(label="Ảnh phôi")
            out_3d = gr.Model3D(label="Kết quả 3D")

    run_btn.click(run_3d_engine, in_img, [txt_status, out_pre, out_3d])

demo.launch()
