import tkinter as tk;
import customtkinter as ctk;

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#create app
app=tk.Tk()
app.geometry("532x622")
app.title("Stable bud")
ctk.set_appearance_mode("dark")
promt=ctk.CTKEntry(height=40, width=512, text_font=("Arial",20),text_color="balck", fg_color="white")
promt.place(x=10,y=10)

lmain=ctk.CTkLabel(height=512,width=512)
lmain.place(x=10, y=110)

modelid="Compvis/stable-diffusion-v1-4"
pipe=StableDiffusionPipeline.from_pretrained(modelid,revision="fp16",torch_dtype=torch.float16,use_auth_token=auth_token)
pipe.to(device)


def generate():
    with autocast(device):
        image=pipe(promt.get(),guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img=ImageTk.photoImage(image)
    lmain.configure(image=img)
    
        
    



trigger=ctk.CTKButton(height=40,width=512, text_font=("Arial",20),text_color="balck", fg_color="white")
trigger.configure(text="Generate")
trigger.place(x=206,y=60)



app.mainloop()
