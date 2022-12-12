from __future__ import annotations
import logging
import gradio as gr
import modules.scripts as scripts
from modules.ui import *
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
import _thread
import threading
from modules import script_callbacks
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
queue_lock = threading.Lock()
isGeneratingForever = False
numOfGeneratingForever = 0
firstRes =None
sd_model_hash = None
foreverState = "no forever task"

def forever(p):
        global isGeneratingForever
        while isGeneratingForever:
            time.sleep(0.1)
            with queue_lock:
               print("isGeneratingForever:",isGeneratingForever)
               print("begin torch_gc processing(shared.state.begin())")
               shared.state.begin()
               print("torch_gc processing(shared.state.begin()) done")
               p.seed = -1
               p.n_iter = 1
               p.sd_model = shared.sd_model
               print("begin process_images")
               process_images(p)
               print("process_images done")
               print("begin torch_gc processing(shared.state.end())")
               shared.state.end()
               print("torch_gc processing(shared.state.end()) done")
class Script(scripts.Script):
    def handleInfoButtonClick(self):
        global firstRes
        global sd_model_hash
        if firstRes is not None:
            print("processing:",type(firstRes))
            if isinstance(firstRes,StableDiffusionProcessingTxt2Img):#txt2img型
                if firstRes.enable_hr == False:
                    info = firstRes.prompt + '\n' + "Negative prompt: "+firstRes.negative_prompt + '\n' + "Steps: " + str(firstRes.steps) + ', ' + "Sampler: " + firstRes.sampler_name + ', ' + "CFG scale: " + str(firstRes.cfg_scale) + ', ' + "Seed: " + "-1" + ', ' + "Size: " + str(firstRes.width) + "x" + str(firstRes.height) + ", " + "Model hash: " + sd_model_hash
                    return info
                else:
                    info = firstRes.prompt + '\n' + "Negative prompt: "+firstRes.negative_prompt + '\n' + "Steps: " + str(firstRes.steps) + ', ' + "Sampler: " + firstRes.sampler_name + ', ' + "CFG scale: " + str(firstRes.cfg_scale) + ', ' + "Seed: " + "-1" + ', ' + "Size: " + str(firstRes.width) + "x" + str(firstRes.height) + ", " + "Model hash: " + sd_model_hash + ", " + "Denoising strength: " + str(firstRes.denoising_strength) + ', ' + "First pass size: " + str(firstRes.firstphase_width) + 'x' + str(firstRes.firstphase_height)
                    return info 
            else:#img2img型
                info = firstRes.prompt + '\n' + "Negative prompt: "+firstRes.negative_prompt + '\n' + "Steps: " + str(firstRes.steps) + ', ' + "Sampler: " + firstRes.sampler_name + ', ' + "CFG scale: " + str(firstRes.cfg_scale) + ', ' + "Seed: " + "-1" + ', ' + "Size: " + str(firstRes.width) + "x" + str(firstRes.height) + ", " + "Model hash: " + sd_model_hash + ", " + "Denoising strength: " + str(firstRes.denoising_strength) + ', ' + "Mask blur: " + str(firstRes.mask_blur)
                return info
        else:
            return "No info"
    def handleInterrupt(self):
        print("interrupt")
        global isGeneratingForever 
        global numOfGeneratingForever
        isGeneratingForever = False
        print("isGeneratingForever",isGeneratingForever)
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
                generateInfoText = gr.Text(label = "Generate Info",interactive = False)
                infoButton = gr.Button(value = "Refresh forever info",variant = 'primary')
                infoButton.click(self.handleInfoButtonClick,inputs = [],outputs = [generateInfoText])


                   
        return [
            interruptButton,
            generateInfoText,
            infoButton
        ]
    
    def run(self, p, 
            interruptButton,
            generateInfoText,
            infoButton):
        # print("p.prompt_first",p.prompt)
        # print("p.negative_prompt",p.negative_prompt)
        # print("p.steps",p.steps)
        # print("p.sampler_name",p.sampler_name)
        # print("p.width",p.width)
        # print("p.height",p.height)
        # print("p.cfg_scale",p.cfg_scale)
        # print("p.model_hash",shared.sd_model.sd_model_hash)
        global firstRes
        firstRes = p

        global isGeneratingForever
        if isGeneratingForever == False:
            print("change isGeneratingForever True!")
            isGeneratingForever = True
        global numOfGeneratingForever
        with queue_lock:
            print("process first image")
            res = process_images(p)
            print("first image process done")
            global sd_model_hash
            sd_model_hash = res.sd_model_hash
        print("isGeneratingForever:",isGeneratingForever)
        print("numOfGeneratingForever:",numOfGeneratingForever)
        if numOfGeneratingForever == 0:
           numOfGeneratingForever += 1
           print("numOfGeneratingForever:",numOfGeneratingForever)
           print("start new thread")
           _thread.start_new_thread(forever,(p,))
        return res
def refreshForeverState():
    global foreverState
    if isGeneratingForever:
        print("numOfGeneratingForever:",numOfGeneratingForever)
        foreverState = "forever running"
       
    else:
        print("numOfGeneratingForever:",numOfGeneratingForever)
        foreverState = "no forever task"
    return foreverState
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as images_history:
        gr.Textbox(value = "state",elem_id="forever_state") 
    return (images_history , "Forever", "forever")
def on_ui_settings():
    global foreverState
    section = ('forever', "Generate Forever")
    shared.opts.add_option("forever-state", shared.OptionInfo(foreverState, "Forever State", gr.Textbox,{"interactive":False},refresh=refreshForeverState,section=section))
script_callbacks.on_ui_settings(on_ui_settings)
# script_callbacks.on_ui_tabs(on_ui_tabs)
