import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="糙米不完善粒检测系统",
    page_icon="🌾",
    layout="wide"
)
def process_image(image):
    # 中值滤波降噪
    median = cv2.medianBlur(image, 5)
    
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    
    # 背景分离（白色背景）
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    foreground = cv2.bitwise_and(median, median, mask=~mask)
    
    # 提取V通道进行增强
    v_channel = hsv[:, :, 2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_v = clahe.apply(v_channel)
    
    # 合并增强后的通道
    hsv[:, :, 2] = enhanced_v
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 边缘检测（Canny算法）
    edges = cv2.Canny(enhanced_img, 50, 150)
    
    # 形态学处理连接边缘
    kernel = np.ones((3,3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 特征分析
    defect_contours = []
    color_features = []
    shape_features = []
    
    avg_hue = np.mean(hsv[:,:,0][closed_edges == 255])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 过滤小面积噪声
            continue
        
        # 形状特征（基于最小外接圆）
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circularity = area / (np.pi * radius**2) if radius != 0 else 0
        
        # 颜色特征（基于边缘区域）
        mask = np.zeros_like(closed_edges)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_hue = cv2.mean(hsv[:,:,0], mask=mask)[0]
        hue_diff = abs(mean_hue - avg_hue)
        
        # 分类逻辑
        if circularity < 0.65 or hue_diff > 20:
            defect_contours.append(cnt)
        
        color_features.append(hue_diff)
        shape_features.append(circularity)
    
    return {
        'median': median,
        'hsv': hsv,
        'enhanced': enhanced_img,
        'edges': edges,
        'closed_edges': closed_edges,
        'contours': contours,
        'defects': defect_contours,
        'avg_hue': avg_hue,
        'color_features': color_features,
        'shape_features': shape_features
    }

# Streamlit界面
st.title("🌾糙米不完善粒检测系统")
uploaded_file = st.file_uploader("上传糙米图像", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = process_image(bgr_img)
    
    # 显示处理流程
    st.subheader("图像处理流程")
    cols = st.columns(3)
    cols[0].image(cv2.cvtColor(results['median'], cv2.COLOR_BGR2RGB), 
                caption="中值滤波降噪")
    cols[1].image(results['hsv'][:,:,0], 
                caption="HSV-H通道", clamp=True)
    cols[2].image(cv2.cvtColor(results['enhanced'], cv2.COLOR_BGR2RGB), 
                caption="CLAHE增强")
    
    cols = st.columns(2)
    cols[0].image(results['edges'], 
                caption="Canny边缘检测", clamp=True)
    cols[1].image(results['closed_edges'], 
                caption="闭合边缘", clamp=True)
    
    # 显示检测结果
    result_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(result_img, results['defects'], -1, (255,0,0), 2)
    
    st.subheader(f"检测结果：发现{len(results['defects'])}个不完善粒")

      # 下载报告
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    st.download_button(
        label="下载检测报告",
        data=result_pil.tobytes(),
        file_name="检测报告.png",
        mime="image/png"
    )
else:
    st.info("请上传包含糙米的图像文件")

st.markdown("---")
st.caption("检测标准：圆形度<0.65 或 色调差异>20")
st.caption("Developed by LSQ| 检测算法版本3.0")