import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 页面配置
st.set_page_config(
    page_title="颗粒缺陷检测系统",
    page_icon="🌾",
    layout="wide"
)

def detect_defective_grains(image):
    """核心检测逻辑"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(
        cleaned, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 筛选有效轮廓（面积大于50像素）
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    
    # 绘制检测结果
    result_img = image.copy()
    cv2.drawContours(result_img, valid_contours, -1, (0,255,0), 2)
    
    return result_img, len(valid_contours), gray, thresh

def advanced_processing(image):
    # HSV颜色空间分析
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 设置颜色阈值（需实际调整）
    lower = np.array([15, 50, 50])
    upper = np.array([35, 255, 255])
    
    # 创建颜色掩模
    mask = cv2.inRange(hsv, lower, upper)
    
    # 结合形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return refined_mask
def advanced_preprocessing(img):
    # 同态滤波消除光照影响
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    Y = img_YUV[:,:,0].astype(float)/255
    
    # 高斯滤波核
    rows, cols = Y.shape
    sigma = 30
    X, Y = np.meshgrid(np.linspace(0,cols-1,cols), np.linspace(0,rows-1,rows))
    gaussian_kernel = np.exp(-((X - cols/2)**2 + (Y - rows/2)**2)/(2*sigma**2))
    
    # 频域处理
    Y_fft = np.fft.fft2(Y)
    Y_fft_shift = np.fft.fftshift(Y_fft)
    Y_filtered = Y_fft_shift * (1 + 0.5*gaussian_kernel)  # 高频增强
    
    # 逆变换
    Y_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(Y_filtered)))
    img_YUV[:,:,0] = np.uint8(np.clip(Y_reconstructed*255, 0, 255))
    return cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

def multi_scale_segmentation(img):
    # 构建高斯金字塔
    pyramid = [cv2.pyrDown(img)]
    for _ in range(2):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    
    # 多尺度检测
    all_contours = []
    for level, layer in enumerate(pyramid):
        gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50*level, 150*level)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 坐标映射回原图尺寸
        scale_factor = 2 ** (level + 1)
        scaled_contours = [cnt * scale_factor for cnt in contours]
        all_contours.extend(scaled_contours)
    
    return all_contours
import torch
from torchvision import transforms

class DefectDetector:
    def __init__(self):
        self.classifier = torch.load('defect_classifier.pth')
        self.segmentor = torch.load('segmentation_unet.pth')
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect(self, img):
        # 第一阶段：粗检测
        tensor_img = self.transform(Image.fromarray(img))
        with torch.no_grad():
            pred = self.classifier(tensor_img.unsqueeze(0))
        
        if pred[0][1] < 0.5:  # 置信度阈值
            return []
        
        # 第二阶段：精细分割
        mask = self.segmentor(tensor_img.unsqueeze(0))
        return self.postprocess_mask(mask)

# 网页界面
st.title("🌾 糙米不完善粒检测系统")
st.markdown("---")

# 侧边栏参数设置
with st.sidebar:
    st.header("检测参数设置")
    min_area = st.slider("最小颗粒面积", 10, 200, 50)
    blur_size = st.slider("降噪强度", 3, 15, 9, step=2)
    threshold_type = st.selectbox("阈值方法", ["自适应阈值", "全局阈值"])

# 文件上传组件
uploaded_file = st.file_uploader(
    "上传待检测图片（支持JPG/PNG）",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # 转换上传文件为OpenCV格式
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    with st.spinner("正在分析中..."):
        # 执行检测
        result_img, defect_count, gray, thresh = detect_defective_grains(image)
        
        # 显示结果
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="原始图像", use_container_width=True, channels="BGR")
        with col2:
            st.image(result_img, caption="检测结果", use_container_width=True, channels="BGR")
    with st.expander("查看中间处理过程"):
        tab1, tab2, tab3 = st.tabs(["灰度图", "二值化", "颜色分析"])
        
        with tab1:
            st.image(gray, caption="灰度处理", use_column_width=True)
        with tab2:
            st.image(thresh, caption="二值化结果", use_column_width=True)
        with tab3:
            refined_mask = advanced_processing(image)
            st.image(refined_mask, caption="颜色分析", use_column_width=True)
       
        # 显示统计信息
        st.success(f"检测完成！发现疑似缺陷颗粒：{defect_count}处")
        st.metric("缺陷颗粒数量", defect_count)
        
        # 添加下载按钮
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.download_button(
            label="下载检测报告",
            data=result_pil.tobytes(),
            file_name="检测结果.png",
            mime="image/png"
        )

else:
    st.info("请上传需要检测的图片文件")

st.markdown("---")
st.caption("Developed by LSQ| 检测算法版本2.0")
