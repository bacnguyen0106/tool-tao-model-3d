import numpy as np
import pyvista as pv
import cv2
from PIL import Image
from transformers import pipeline
import time
import os

# Tắt chế độ hiển thị cửa sổ (cực kỳ quan trọng khi chạy trên server)
pv.OFF_SCREEN = True 

def image_to_lightweight_3d(image_path, output_file="proxy_model.ply"):
    print(f"Bắt đầu xử lý ảnh: {image_path}")
    start_time = time.time()
    
    # 1. Tải ảnh và phân tích chiều sâu
    print("1. Đang tải AI phân tích chiều sâu (Depth Map)...")
    img = Image.open(image_path).convert("RGB")
    pipe = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
    depth_result = pipe(img)["depth"]
    
    depth_np = np.array(depth_result).astype(np.float32)
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    
    # 2. Tạo lưới tối giản (Low-Poly)
    print("2. Tạo lưới Không gian 3D...")
    target_res = 64
    h, w, _ = np.array(img).shape
    aspect_ratio = w / h
    
    grid_w = target_res
    grid_h = int(target_res / aspect_ratio)
    
    img_cv = np.array(img)
    tex_small = cv2.resize(img_cv, (grid_w, grid_h))
    depth_small = cv2.resize(depth_norm, (grid_w, grid_h))
    
    # 3. Đùn khối 3D
    print("3. Đang đùn khối 3D trục Z...")
    x = np.linspace(-aspect_ratio, aspect_ratio, grid_w)
    y = np.linspace(-1, 1, grid_h)
    x, y = np.meshgrid(x, y)
    
    z = depth_small * 0.3 
    
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["colors"] = tex_small.reshape(-1, 3)
    mesh = grid.extract_surface()
    
    # 4. Tối ưu hóa và Lưu file
    print("4. Ép dung lượng (Decimation) & Lưu file...")
    mesh = mesh.decimate(0.6)
    mesh.save(output_file)
    
    print(f"✅ Hoàn tất! Đã lưu '{output_file}' trong {round(time.time() - start_time, 2)} giây.")

if __name__ == "__main__":
    if os.path.exists("sofa.jpg"):
        image_to_lightweight_3d("sofa.jpg", "sofa_proxy_sieu_nhe.ply")
    else:
        print("❌ Không tìm thấy file sofa.jpg. Vui lòng kiểm tra lại!")
