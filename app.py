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

def preprocess_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_np = np.array(image)
    image_np = 255 - image_np
    image_np = image_np / 255.0
    image_np = image_np.astype(np.float32)
    image_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
    return image_tensor

def predict_digit(image):
    if image is None:
        return "请上传或绘制一张数字图片", None
    
    processed = preprocess_image(image)
    
    with torch.no_grad():
        output = model(processed)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item() * 100
    
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3_results = [(int(top3_idx[0][i].item()), float(top3_prob[0][i].item()) * 100) for i in range(3)]
    
    result_text = f"预测结果: {predicted} (置信度: {confidence:.2f}%)\n\nTop-3 预测:\n"
    for digit, prob in top3_results:
        result_text += f"  • 数字 {digit}: {prob:.2f}%\n"
    
    return result_text, top3_results

with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# 🎯 手写数字识别系统")
    gr.Markdown("上传一张手写数字图片（0-9），或在画板上直接书写，系统将自动识别数字。")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传图片")
            image_input = gr.Image(type="pil", label="选择图片", height=200)
            
            gr.Markdown("### ✏️ 或在画板上书写")
            sketchpad = gr.Sketchpad(label="在此书写数字", height=200)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 预测结果")
            result_output = gr.Textbox(label="识别结果", lines=6, interactive=False)
            
            gr.Markdown("### 🔍 置信度分布")
            bar_plot = gr.BarPlot(label="Top-3 概率分布", x="数字", y="置信度", height=200)
    
    with gr.Row():
        submit_btn = gr.Button("🚀 开始识别", size="lg")
        clear_btn = gr.Button("🗑️ 清除", size="sm")
    
    def on_submit(image, sketch):
        input_image = sketch if sketch is not None else image
        if input_image is None:
            return "请上传或绘制一张数字图片", None
        
        result_text, top3 = predict_digit(input_image)
        
        plot_data = {"数字": [str(d[0]) for d in top3], "置信度": [d[1] for d in top3]}
        return result_text, plot_data
    
    submit_btn.click(
        fn=on_submit,
        inputs=[image_input, sketchpad],
        outputs=[result_output, bar_plot]
    )
    
    def on_clear():
        return None, None, "请上传或绘制一张数字图片", None
    
    clear_btn.click(
        fn=on_clear,
        outputs=[image_input, sketchpad, result_output, bar_plot]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 💡 使用说明")
    gr.Markdown("1. **上传图片**: 点击左侧上传区域，选择一张手写数字图片\n2. **手写输入**: 在画板上用鼠标或触摸书写数字\n3. **开始识别**: 点击\"开始识别\"按钮获取结果\n4. **清除**: 点击\"清除\"按钮重置输入")

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())