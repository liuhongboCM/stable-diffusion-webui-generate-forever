from __future__ import annotations
import logging
import gradio as gr
import modules.scripts as scripts
from modules.ui import *
from modules.processing import process_images
import _thread
import threading
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
queue_lock = threading.Lock()
isGeneratingForever = False
numOfGeneratingForever = 0

def forever(p):
        global isGeneratingForever
        while isGeneratingForever:
            time.sleep(0.01)
            with queue_lock:
               p.seed = -1
               p.n_iter = 1
               p.sd_model = shared.sd_model
               process_images(p)

class Script(scripts.Script):
    def handleInterrupt(self):
        print("interrupt")
        global isGeneratingForever 
        global numOfGeneratingForever
        isGeneratingForever = False
        numOfGeneratingForever = 0

    def returnOneimg(p):
        return process_images(p)

    def title(self):
        return "GenerateForever"
    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Generate Forever", open=True):
                interruptButton = gr.Button(value = "Interrupt",elem_id = "interruptButton")
                interruptButton.click(self.handleInterrupt,[],[])
                   
        return [
            interruptButton
        ]
    
    def run(self, p, interruptButton):
        global isGeneratingForever
        if isGeneratingForever == False:
            print("change isGeneratingForever True!")
            isGeneratingForever = True
        global numOfGeneratingForever
        if numOfGeneratingForever == 0:
           numOfGeneratingForever += 1
           _thread.start_new_thread(forever,(p,))
        print("return !")
        with queue_lock:
            res = process_images(p)
        return res


        
        







