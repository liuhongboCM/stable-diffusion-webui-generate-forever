from __future__ import annotations
import logging
import json
import gradio as gr
import modules.scripts as scripts
from modules.ui import *
from modules.processing import StableDiffusionProcessingTxt2Img, process_images
import _thread
import threading
import time
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
queue_lock = threading.Lock()
isGeneratingForever = False
numOfGeneratingForever = 0
process_info = None
sd_model_hash = None
taskId = ""
foreverState = ""


class Script(scripts.Script):
    foreverPath = './outputs/tasks/'
    def generateTaskId(self):
        global taskId
        fConfig = open('config.json','r')
        configDic = json.load(fConfig)
        print("machine-id:",configDic['machine-id'])
        # taskId = configDic['machine-id'] + '-' + str((lambda:int(round(time.time() * 1000)))())
        #暂时写死
        taskId = "sd05-1670982305082"
        print("taskId:",taskId)
        f = open('task_status.log','w')
        f.write(taskId)
        f.close()
    def forever(self,p):
        global isGeneratingForever
        global foreverState
        while isGeneratingForever:
            foreverState = "forever task running"
            time.sleep(0.1)
            with queue_lock:
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
        foreverState = "no forever task"

    def handleInfoButtonClick(self):
        global process_info
        global sd_model_hash
        global taskId
        global foreverState
        if process_info is not None:
            print("processing:", type(process_info))
            if isinstance(process_info, StableDiffusionProcessingTxt2Img):  # txt2img型
                if process_info.enable_hr == False:
                    info = "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model hash: {}".format(
                        process_info.prompt,
                        process_info.negative_prompt,
                        process_info.steps,
                        process_info.sampler_name,
                        process_info.cfg_scale,
                        -1,
                        process_info.width,
                        process_info.height,
                        sd_model_hash
                    )
                    return info,taskId,foreverState,self.foreverPath + taskId
                else:
                    info = "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model hash: {}, Denoising strength: {}, First pass size: {}x{}".format(
                        process_info.prompt,
                        process_info.negative_prompt,
                        process_info.steps,
                        process_info.sampler_name,
                        process_info.cfg_scale,
                        -1,
                        process_info.width,
                        process_info.height,
                        sd_model_hash,
                        process_info.denoising_strength,
                        process_info.firstphase_width,
                        process_info.firstphase_height
                    )
                    return info,taskId,foreverState,self.foreverPath + taskId
            else:  # img2img型
                info = "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model hash: {}, Denoising strength: {}, Mask blur: {}".format(
                        process_info.prompt,
                        process_info.negative_prompt,
                        process_info.steps,
                        process_info.sampler_name,
                        process_info.cfg_scale,
                        -1,
                        process_info.width,
                        process_info.height,
                        sd_model_hash,
                        process_info.denoising_strength,
                        process_info.mask_blur
                    )
                return info,taskId,foreverState,self.foreverPath + taskId
        else:
            return "No info",taskId,foreverState,self.foreverPath + taskId

    def handleInterrupt(self):
        print("interrupt")
        global isGeneratingForever
        global numOfGeneratingForever
        isGeneratingForever = False
        numOfGeneratingForever = 0
        f = open('task_status.log','w')
        f.close()

    def returnOneimg(p):
        return process_images(p)

    def title(self):
        return "GenerateForever"

    def ui(self, is_img2img):
        global foreverState
        global taskId
        with gr.Group():
            with gr.Accordion("Generate Forever", open=True):
                interruptButton = gr.Button(
                    value="Interrupt", elem_id="interruptButton")
                interruptButton.click(self.handleInterrupt, [], [])
                foreverStateText = gr.Textbox(label = "Forever State",value = foreverState,elem_id = "forever_state_text",interactive = False)
                taskIdText = gr.Textbox(label = "Task ID",interactive=False)
                generateInfoText = gr.Text(
                    label="Generate Info", interactive=False)
                savePathText = gr.Textbox(label = "Images Save Path",value = self.foreverPath + taskId,interactive = False)
                infoButton = gr.Button(
                    value="Refresh forever info", variant='primary')
                infoButton.click(self.handleInfoButtonClick,
                                 inputs=[], outputs=[generateInfoText,taskIdText,foreverStateText,savePathText])

        return [
            interruptButton,
            generateInfoText,
            infoButton,
            taskIdText,
            foreverStateText,
            savePathText
        ]

    def run(self, p,
            interruptButton,
            generateInfoText,
            infoButton,
            taskIdText,
            foreverStateText,
            savePathText):
        global process_info
        process_info = p
        print(process_info.outpath_samples)
        print(process_info.outpath_grids)
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
        print("isGeneratingForever:", isGeneratingForever)
        print("numOfGeneratingForever:", numOfGeneratingForever)
        if numOfGeneratingForever == 0:
            numOfGeneratingForever += 1
            print("numOfGeneratingForever:", numOfGeneratingForever)
            print("start new thread")
            self.generateTaskId()

            dirs = self.foreverPath + taskId
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            p.outpath_samples = dirs
            #存储prompts信息
            txtInfoPath = os.path.join(dirs,taskId + ".txt")
            info,a,b,c = self.handleInfoButtonClick()
            infoFile = open(txtInfoPath,'w')
            infoFile.write(info)
            infoFile.close()
            
            _thread.start_new_thread(self.forever, (p,))

            

        return res



processListId=[]
processIndex = 0
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as generate_forever:
        with gradio.Column(elem_id = "forever_batch"):
            processDropDown = gradio.Dropdown(label = "Tasks")
            with gr.Blocks(title = "Task info"):
                taskIdText = gr.Textbox(label = "Task Id",interactive = False)
                taskInfoText = gr.Textbox(label = "Generate Info",lines = 5,interactive = False)
                imgsSavePath = gr.Textbox(label = "Images Save Path",interactive = False)
                


            deleteButton = gr.Button(value = "Delete Batch",elem_id = "forever_batch_delete_btn")
            with gradio.Column(elem_id = "forever_batch_item"):
                with gradio.Row():
                    promptText = gradio.Textbox(label = "Prompt",placeholder = "Prompt",lines = 2)
                    with gradio.Column(scale = 1):
                        getGenerateInfoButton = gradio.Button(value = "↙️",elem_id = "get_generate_info_btn")
                        processNumber = gradio.Slider(maximum = 10000,value = 3000,label = "Batch Count",interactive = True)
                negativePromptText = gradio.Textbox(label = "Negative Prompt",placeholder = "Negative Prompt",lines = 2)
                with gradio.Row():
                    samplingStepsSlider = gradio.Slider(maximum = 150,label = "Sampling Steps",value = 20,interactive = True)
                    samplingMethodDropdown = gradio.Dropdown(choices = ["Euler a","Euler", "LMS", "Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM fast","DPM adaptive","LMS Karras","DPM2 Karras","DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DDIM","PLMS"],label = "Sampling Method",value = "Euler a",interactive = True)
                with gradio.Row():
                    widthSlider = gradio.Slider(maximum = 2048,value = 512,label = "Width",interactive = True)
                    heightSlider = gradio.Slider(maximum = 2048,value = 512,label = "Height",interactive = True)
                    cfgScaleSlider = gradio.Slider(maximum = 30,value = 7,label = "CFG Scale",interactive = True)
        addButton = gr.Button(value = "Add Batch",variant = "primary",elem_id = "forever_batch_add_btn")
        
       

      
    return (generate_forever , "Txt2Img Forever Batch", "forever batch"),




def on_ui_settings():
    section = ('generate-forever', "Generate Forever")
    shared.opts.add_option("machine-id", shared.OptionInfo("", "set an machine id", section=section))

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)


