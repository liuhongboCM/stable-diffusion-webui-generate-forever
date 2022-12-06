from __future__ import annotations
import logging
import gradio as gr
import modules.scripts as scripts
from modules.ui import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class Script(scripts.Script):
    def handleInterrupt(self):
        print("interrupt")
        shared.state.interrupt()

    def title(self):
        return f"GenerateForever"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        with gr.Group():
            with gr.Accordion("Generate Forever", open=True):
                generateForeverBtn = gr.Button(value="Generate Forever",elem_id = "generateForeverButton")
                interruptButton = gr.Button(value = "Interrupt",elem_id = "interruptButton")
                interruptButton.click(self.handleInterrupt,[],[])
                   
        return [
            generateForeverBtn,
            interruptButton
        ]
        







