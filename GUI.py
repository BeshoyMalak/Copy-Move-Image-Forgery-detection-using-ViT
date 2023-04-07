import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torch
from torchvision.utils import save_image
import cv2 
import PIL 
from transformers import ViTFeatureExtractor
from transformers import ViTMAEForPreTraining

import os

# used to supress display of warnings
import warnings
warnings.filterwarnings("ignore")


from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=3):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None


class forgged_image():
    def __init__(self, image):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base")
        self.pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        torch.manual_seed(2)
        self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
        self.visualize()
        
    def show_image(self, image, title=''):
        
        imagenet_mean = np.array(self.feature_extractor.image_mean)
        imagenet_std = np.array(self.feature_extractor.image_std)
        # image is [H, W, 3]
        assert image.shape[2] == 3
        #plt.figure(figsize = (12, 9))
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
        return


    def apply_postprocessing_filter(self):
        # Load heatmap
        heatmap = cv2.imread('heatmap.jpg', cv2.IMREAD_GRAYSCALE)

        # Threshold heatmap to create binary image
        _, binary_heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # Apply morphological filters
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(binary_heatmap, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        plt.imshow(closing)
        plt.title("The forged object", fontsize=16)
        plt.axis('off')

        cv2.imwrite(r'D:\Free lancing\Sara, ViT, Copy cut\Results\forged_object.jpg', closing)
        return

    def visualize(self):
        # forward pass
        outputs = self.model(self.pixel_values)
        y = self.model.unpatchify(outputs.logits)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()

        # visualize the mask
        mask = outputs.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
        mask = self.model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', self.pixel_values)

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [18, 18]

        plt.subplot(2, 3, 1)
        self.show_image(x[0], "original")

        plt.subplot(2, 3, 2)
        self.show_image(im_masked[0], "masked")

        plt.subplot(2, 3, 3)
        self.show_image(y[0], "reconstruction")

        plt.subplot(2, 3, 4)
        self.show_image(im_paste[0], "reconstruction + visible")
        
        difference_image = im_paste[0] - x[0]
        # Save difference image to disk
        save_image(difference_image.permute(2,0,1), r'D:\Free lancing\Sara, ViT, Copy cut\Results\heatmap.jpg')
        
        plt.subplot(2, 3, 5)
        self.show_image(difference_image , "Difference")
        plt.show()
        
        self.apply_postprocessing_filter()
        
    
    

class ImageForm(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        master.geometry("500x500")
        self.pack()
        self.create_widgets()
        

    def create_widgets(self):
        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack(side="top")

        self.show_button = tk.Button(self, text="Apply the model", command=self.apply_model, state="disabled")
        self.show_button.pack(side="top")

        self.quit_button = tk.Button(self, text="Quit", command=self.master.destroy)
        self.quit_button.pack(side="bottom")



    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.show_button.config(state="normal")

    def apply_model(self):#here where we apply the model and display the output

        MODEL_PATH = r'model.pt'
        model2 = torch.load(MODEL_PATH)

        EVAL_BATCH = 1

        # Load the image
        #image_path = '/content/forged/001_F_JC3.jpg'
        #image = Image.open(image_path).convert('RGB')
        image = self.image
        # Transform the image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # Send to appropriate computing device
        device = torch.device('cpu')
        #torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        input_batch = input_batch.to(device)

        # Disable grad
        with torch.no_grad():
            # Generate prediction
            #tensor([1](or) /[0](for)) <class 'torch.Tensor'>
            prediction, loss = model2(input_batch, torch.tensor([0]))

        # Predicted class value using argmax
        predicted_class = np.argmax(prediction.cpu())
        
        enc_dict = {'forged': 0, 'original': 1}
        value_predicted = list(enc_dict.keys())[list(enc_dict.values()).index(predicted_class)]

        # Show result
        plt.imshow(image)
        plt.title(f'Prediction: {value_predicted}')
        plt.show()
        if value_predicted == "forged":
            # make random mask reproducible (comment out to make it change)
            forged = forgged_image(image= image)
            #forged.visualize()

        

root = tk.Tk()
app = ImageForm(master=root)
app.mainloop()
