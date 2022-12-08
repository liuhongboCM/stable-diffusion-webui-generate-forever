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
            time.sleep(0.1)
            
            with queue_lock:
               shared.state.begin()
               p.seed = -1
               p.n_iter = 1
               p.sd_model = shared.sd_model
               print("sampler",p.sampler)
               process_images(p)
               shared.state.end()
            



class Script(scripts.Script):
    prompt = None
    negativePrompt = None
    samplingSteps = None
    samplerName = None
    width = None
    height = None
    cfgScale = None
    modelHash = None
    def handleInfoButtonClick(self):
        return self.modelHash,self.prompt,self.negativePrompt,self.samplingSteps,self.samplerName,self.width,self.height,self.cfgScale
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
                modelHashText = gr.Text(label = "model hash",interactive = False,value = self.modelHash)
                promptText = gr.Textbox(label = "prompt",interactive = False,value = self.prompt)
                negativePromptText = gr.Textbox(label = "negative prompt",interactive = False,value = self.negativePrompt)
                samplingStepsText = gr.Text(label = "sampling steps",interactive = False,value = self.samplingSteps)
                samplerNameText = gr.Text(label = "sampler name",interactive = False,value = self.samplerName)
                widthText = gr.Text(label = "width",interactive = False,value = self.width)
                heightText = gr.Text(label = "height",interactive = False,value = self.height)
                cfgScaleText = gr.Text(label = "cfgScale",interactive = False,value = self.cfgScale)
                infoButton = gr.Button(value = "Refresh forever info",variant = 'primary')
                infoButton.click(self.handleInfoButtonClick,inputs = [],outputs = [modelHashText,promptText,negativePromptText,samplingStepsText,samplerNameText,widthText,heightText,cfgScaleText])


                   
        return [
            interruptButton,
            modelHashText,
            promptText,
            negativePromptText,
            samplingStepsText,
            samplerNameText,
            widthText,
            heightText,
            cfgScaleText,
            infoButton
        ]
    
    def run(self, p, interruptButton,modelHashText,
            promptText,
            negativePromptText,
            samplingStepsText,
            samplerNameText,
            widthText,
            heightText,
            cfgScaleText,
            infoButton):
        print("p.prompt_first",p.prompt)
        print("p.negative_prompt",p.negative_prompt)
        print("p.steps",p.steps)
        print("p.sampler_name",p.sampler_name)
        print("p.width",p.width)
        print("p.height",p.height)
        print("p.cfg_scale",p.cfg_scale)
        print("p.model_hash",shared.sd_model.sd_model_hash)
        self.prompt = p.prompt
        self.negativePrompt = p.negative_prompt
        self.samplingSteps = p.steps
        self.samplerName = p.sampler_name
        self.width = p.width
        self.height = p.height
        self.cfgScale = p.cfg_scale
        self.modelHash = shared.sd_model.sd_model_hash

        global isGeneratingForever
        if isGeneratingForever == False:
            print("change isGeneratingForever True!")
            isGeneratingForever = True
        global numOfGeneratingForever
        with queue_lock:
            res = process_images(p)
        if numOfGeneratingForever == 0:
           numOfGeneratingForever += 1
           _thread.start_new_thread(forever,(p,))
        return res