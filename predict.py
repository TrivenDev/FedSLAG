from unet.unet_model_feat import UNet

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# 加载模型
model = UNet(n_channels=1, n_classes=1,bilinear=True)
model.load_state_dict(torch.load('log/FedNEW_GMM_bbbl_2024-06-26_08-34-49/BUSIS/epoch294_best_2024-06-26.pth'))
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

def infer_and_show(image_path):

    input_image = Image.open(image_path).convert('L')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        output = torch.sigmoid(output)

    # 转换模型输出为图像
    output_image = output.squeeze(0).cpu().numpy()
    # Ensure output_image is 2D; this part depends on your specific case
    if len(output_image.shape) > 2:
        output_image = output_image.reshape(output_image.shape[1], output_image.shape[2])
    output_image = (output_image * 255).astype('uint8')
    
    try:
        output_pil = Image.fromarray(output_image)
    except Exception as e:
        print(f"Error converting output to image: {e}")
        return
    
    
   
    # 在窗口中显示输入和分割后的图像
    root = tk.Tk()
    root.title("Image Viewer")
    input_img = ImageTk.PhotoImage(input_image.resize((224, 224)))
    output_img = ImageTk.PhotoImage(output_pil.resize((224, 224)))
    label_image_path = image_path.replace("original", "GT")
    try:
        label_image = Image.open(label_image_path).convert('L')
        label_image.thumbnail((160, 160))
        label_img = ImageTk.PhotoImage(label_image.resize((224, 224)))
    except Exception as e:
        print(f"Error loading label image: {e}")
        label_img = None

    input_panel = tk.Label(root, image=input_img) # 
    input_panel.image = input_img
    input_panel.pack(side="left", fill="both", expand="yes")

    output_frame = tk.Frame(root)
    output_frame.pack(side="right", fill="both", expand="yes")

    output_panel = tk.Label(output_frame, image=output_img)
    output_panel.image = output_img  # Keep a reference to the image
    output_panel.pack(side="top", fill="both", expand="yes", anchor="w")

    if label_img:
        label_panel = tk.Label(output_frame, image=label_img)
        label_panel.image = label_img  # Keep a reference to the image
        label_panel.pack(side="top", fill="both", expand="yes", anchor="w")

    root.mainloop()

# 从文件对话框中选择图像
file_path = "dataset/breast/BUS/original/malignant (115).png"

infer_and_show(file_path)