import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()
model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

def preprocess_image(image_array):
    """
    统一预处理：将任意来源的图像转换为 MNIST 格式张量。
    MNIST 格式：0.0 = 黑色背景, 1.0 = 白色数字
    """
    img = np.array(image_array)
    
    if img.ndim == 3:
        if img.shape[2] == 4:
            # RGBA（来自 Sketchpad）：直接用 alpha 通道作为数字掩码
            # alpha: 0=透明(无笔迹), 255=不透明(有笔迹)
            img = img[:, :, 3].astype(np.float32)
        elif img.shape[2] == 3:
            # RGB（来自上传图片）：转灰度，然后反色
            # 原图：白纸(255) + 黑字(0) → 反色后：黑底(0) + 白字(255)
            img = (255.0 - np.mean(img, axis=2)).astype(np.float32)
    else:
        img = img.astype(np.float32)
    
    # 此时 img：0=背景, 较大值=数字（与 MNIST 格式一致）
    
    # 裁剪数字区域
    foreground = np.where(img > 30)
    if len(foreground[0]) > 20:
        y_min = max(0, foreground[0].min() - 4)
        y_max = min(img.shape[0] - 1, foreground[0].max() + 4)
        x_min = max(0, foreground[1].min() - 4)
        x_max = min(img.shape[1] - 1, foreground[1].max() + 4)
        img = img[y_min:y_max+1, x_min:x_max+1]
    
    # 正方形填充（黑色背景）
    h, w = img.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off+h, x_off:x_off+w] = img
    
    # 缩放到 28x28
    pil = Image.fromarray(padded.astype(np.uint8))
    pil = pil.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 归一化到 [0, 1]
    img_np = np.array(pil).astype(np.float32) / 255.0
    return torch.tensor(img_np).unsqueeze(0).unsqueeze(0)

def predict(image_tensor):
    if image_tensor is None:
        return "处理失败"
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item() * 100
    
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    result = f"预测结果: 数字 {predicted}\n置信度: {confidence:.2f}%\n\nTop-3:\n"
    for i in range(3):
        result += f"\u2022 数字 {int(top3_idx[0][i].item())}: {float(top3_prob[0][i].item())*100:.2f}%\n"
    
    return result

def predict_image(image):
    if image is None:
        return "请上传图片"
    return predict(preprocess_image(image))

def predict_sketch(sketch_data):
    if sketch_data is None:
        return "请在画板上书写数字"
    
    try:
        image_data = None
        
        if isinstance(sketch_data, dict):
            # Sketchpad 返回 dict: {'background', 'layers', 'composite'}
            if 'layers' in sketch_data and sketch_data['layers']:
                layer = sketch_data['layers'][0]
                if isinstance(layer, np.ndarray):
                    image_data = layer
        elif isinstance(sketch_data, np.ndarray):
            image_data = sketch_data
        
        if image_data is None:
            return "无法解析画板数据"
        
        return predict(preprocess_image(image_data))
    
    except Exception as e:
        return f"处理失败: {str(e)}"

with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# \U0001f3af 手写数字识别系统")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### \U0001f4e4 上传图片识别")
            image_input = gr.Image(type="pil", label="选择图片", height=200)
            image_btn = gr.Button("\U0001f680 识别图片")
            
            gr.Markdown("### \u270f\ufe0f 手写画板识别")
            sketch_input = gr.Sketchpad(label="在此书写数字", height=200)
            sketch_btn = gr.Button("\U0001f680 识别手写")
        
        with gr.Column(scale=1):
            gr.Markdown("### \U0001f4ca 图片识别结果")
            image_result = gr.Textbox(label="结果", lines=8)
            
            gr.Markdown("### \U0001f4ca 手写识别结果")
            sketch_result = gr.Textbox(label="结果", lines=8)
    
    image_btn.click(predict_image, inputs=[image_input], outputs=[image_result])
    sketch_btn.click(predict_sketch, inputs=[sketch_input], outputs=[sketch_result])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
