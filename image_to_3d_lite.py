import numpy as np
import pyvista as pv
import cv2
from PIL import Image
from transformers import pipeline
import time
import os

pv.OFF_SCREEN = True 

def image_to_lightweight_3d(image_path, output_file="proxy_model.ply"):
    print(f"Bắt đầu xử lý ảnh: {image_path}")
    start_time = time.time()
    
    # Mở ảnh và giữ nguyên kênh Alpha (trong suốt) nếu có
    img = Image.open(image_path).convert("RGBA")
    
    # 1. Chạy AI ước lượng độ sâu (AI chỉ cần ảnh RGB)
    print("1. Đang tải AI phân tích chiều sâu...")
    img_rgb = img.convert("RGB")
    pipe = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    depth_result = pipe(img_rgb)["depth"]
    
    depth_np = np.array(depth_result).astype(np.float32)
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    
    # 2. Tạo lưới (Low-Poly)
    print("2. Tạo lưới Không gian 3D...")
    target_res = 64
    h, w, _ = np.array(img).shape
    aspect_ratio = w / h
    
    grid_w = target_res
    grid_h = int(target_res / aspect_ratio)
    
    img_cv = np.array(img) # Ảnh RGBA
    tex_rgb = cv2.resize(img_cv[:, :, :3], (grid_w, grid_h)) # Lấy màu
    tex_alpha = cv2.resize(img_cv[:, :, 3], (grid_w, grid_h)) # Lấy độ trong suốt
    depth_small = cv2.resize(depth_norm, (grid_w, grid_h))
    
    # 3. Đùn khối 3D trục Z
    print("3. Đang đùn khối 3D trục Z...")
    x = np.linspace(-aspect_ratio, aspect_ratio, grid_w)
    y = np.linspace(-1, 1, grid_h)
    x, y = np.meshgrid(x, y)
    
    z = depth_small * 0.3 
    
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["colors"] = tex_rgb.reshape(-1, 3)
    grid.point_data["alpha"] = tex_alpha.flatten() # Nạp dữ liệu trong suốt
    
    # CẮT BỎ PHẦN NỀN TRONG SUỐT (Chỉ giữ lại điểm ảnh có Alpha > 128)
    mesh = grid.threshold(128, scalars="alpha")
    mesh = mesh.extract_surface()
    
    # 4. Ép dung lượng
    print("4. Ép dung lượng (Decimation) & Lưu file...")
    mesh = mesh.decimate(0.6)
    mesh.save(output_file)
    
    print(f"✅ Hoàn tất! Đã lưu '{output_file}' trong {round(time.time() - start_time, 2)} giây.")

if __name__ == "__main__":
    # Đổi thành tìm file sofa.png
    if os.path.exists("sofa.png"):
        image_to_lightweight_3d("sofa.png", "sofa_proxy_sieu_nhe.ply")
    else:
        print("❌ Không tìm thấy file sofa.png. Vui lòng tải ảnh PNG đã tách nền lên!")
