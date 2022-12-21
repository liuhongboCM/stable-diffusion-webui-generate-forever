
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
from modules import script_callbacks
from modules.txt2img import txt2img
from modules.call_queue import queue_lock
foreverTasksLock = threading.Lock()
foreverTasks = []
foreverTasksIndex = []
foreverPath = './outputs/tasks/'
status = "free"#free,busy
statusLock = threading.Lock()
taskStatusFLock = threading.Lock()
class ForeverTask():
    taskId = ""
    prompt = ""
    negativePrompt = ""
    samplingSteps = 20,
    samplingMethod = "Euler a"
    width = 512
    height = 512
    cfgScale = 7
    batchCount = 10000
    status = "waiting"# waiting,processing,paused,completed
    generateInfo=""
    seed = -1
    modelHash = ""
    imgsSavePath = ""
    oringinalBatchCount = 10000
    def __init__(self,taskId,prompt,negativePrompt,samplingSteps,samplingMethod,width,height,cfgScale,batchCount,generateInfo,modelHash,imgsSavePath):
        self.taskId = taskId
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.samplingSteps = int(samplingSteps)
        self.samplingMethod = samplingMethod
        self.width = int(width)
        self.height = int(height)
        self.cfgScale = float(cfgScale)
        self.batchCount = int(batchCount)
        self.generateInfo = generateInfo
        self.modelHash = modelHash
        self.imgsSavePath = imgsSavePath
        self.oringinalBatchCount = self.batchCount

class ForeverMonitor(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def monitor(self):
        while True:
            time.sleep(1)
            global status
            if status == "free":
                global foreverTasks
                for task in foreverTasks:
                    if task.status == "waiting":
                        task.status = "processing"
                        threading.Thread(processBatch(task), daemon = True).start
                        break
            else:
                pass
            
            


    def run(self):
        self.monitor()
            


def processBatch(task):
    fConfig = open('config.json','r')
    configDic = json.load(fConfig)
    print("machine-id:",configDic['machine-id'])
    task_id = task.taskId
    saveDir = task.imgsSavePath
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        infoTxt = open(saveDir + "/" + task.taskId + ".txt",'w')
        infoTxt.write(task.generateInfo)
        infoTxt.close()
    with taskStatusFLock:
        taskIdF = open("task_status.log",'w')
        taskIdF.write(task_id)
        taskIdF.close()
    global status
    while task.batchCount > 0 and task.status == "processing":
        time.sleep(0.1)
        with queue_lock:
            with statusLock:
                status = "busy"
            task.batchCount -= 1
            # txt2img(prompt = task.prompt,negative_prompt = task.negativePrompt,prompt_style = None,prompt_style2 = None,steps = task.samplingSteps,sampler_index = 1,restore_faces = False,tiling = False,n_iter = 1,batch_size = 1,cfg_scale = task.cfgScale,seed = -1,subseed= -1, subseed_strength= 0, seed_resize_from_h= 0, seed_resize_from_w= 0, seed_enable_extras= False, height= task.height, width= task.width, enable_hr= False, denoising_strength= 0.7, firstphase_width= 0, firstphase_height= 0)
            p = StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            outpath_samples=saveDir,
            outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
            prompt=task.prompt,
            styles=[None, None],
            negative_prompt=task.negativePrompt,
            seed=task.seed,
            subseed=-1,
            subseed_strength=0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            seed_enable_extras=False,
            sampler_name=task.samplingMethod,
            batch_size=1,
            n_iter=1,
            steps=task.samplingSteps,
            cfg_scale=task.cfgScale,
            width=task.width,
            height=task.height,
            restore_faces=False,
            tiling=False,
            enable_hr=False,
            denoising_strength= None,
            firstphase_width=None,
            firstphase_height=None,
            )
            p.scripts = modules.scripts.scripts_txt2img
            args =   (0, '<div class="dynamic-prompting">\n    <h3><strong>Combinations</strong></h3>\n\n    Choose a number of terms from a list, in this case we choose two artists: \n    <code class="codeblock">{2$$artist1|artist2|artist3}</code><br/>\n\n    If $$ is not provided, then 1$$ is assumed.<br/><br/>\n\n    If the chosen number of terms is greater than the available terms, then some terms will be duplicated, otherwise chosen terms will be unique. This is useful in the case of wildcards, e.g.\n    <code class="codeblock">{2$$__artist__}</code> is equivalent to <code class="codeblock">{2$$__artist__|__artist__}</code><br/><br/>\n\n    A range can be provided:\n    <code class="codeblock">{1-3$$artist1|artist2|artist3}</code><br/>\n    In this case, a random number of artists between 1 and 3 is chosen.<br/><br/>\n\n    Wildcards can be used and the joiner can also be specified:\n    <code class="codeblock">{{1-$$and$$__adjective__}}</code><br/>\n\n    Here, a random number between 1 and 3 words from adjective.txt will be chosen and joined together with the word \'and\' instead of the default comma.\n\n    <br/><br/>\n\n    <h3><strong>Wildcards</strong></h3>\n    \n\n    <br/>\n    If the groups wont drop down click <strong onclick="check_collapsibles()" style="cursor: pointer">here</strong> to fix the issue.\n\n    <br/><br/>\n\n    <code class="codeblock">WILDCARD_DIR: /home/lunaon/stable_diffusion/stable-diffusion-webui/extensions/dynamic-prompts/wildcards</code><br/>\n    <small onload="check_collapsibles()">You can add more wildcards by creating a text file with one term per line and name is mywildcards.txt. Place it in /home/lunaon/stable_diffusion/stable-diffusion-webui/extensions/dynamic-prompts/wildcards. <code class="codeblock">__&#60;folder&#62;/mywildcards__</code> will then become available.</small>\n</div>\n\n', True, False, 1, False, False, False, 100, 0.7, False, False, False, False, False, False, False, False, False, '', 1, '', 0, '', True, False, False, 'Interrupt', '', 'Refresh forever info', '', '', './outputs/tasks/')
            p.script_args = args
            if cmd_opts.enable_console_prompts:
                print(f"\ntxt2img: {prompt}", file=shared.progress_print_out)
            processed = modules.scripts.scripts_txt2img.run(p, *args)
            if processed is None:
                processed = process_images(p)
            p.close()
            shared.total_tqdm.clear()
    if task.batchCount == 0:
        task.status = "completed"
    with statusLock:
        status = "free"
    with taskStatusFLock:
        taskIdF = open("task_status.log",'w')
        taskIdF.write('')
        taskIdF.close()


def addForeverBatch(prompt,negativePrompt,samplingSteps,samplingMethod,width,height,cfgScale,batchCount,modelHash):
    global foreverTasks
    global foreverTasksIndex
    if 'Negative prompt' in prompt:#用户误操作，没有转化prompt，不做处理维持原样
        return prompt,negativePrompt,samplingSteps,samplingMethod,width,height,cfgScale,batchCount,gr.update(choices = foreverTasksIndex,interactive = True),modelHash
    fConfig = open('config.json','r')
    configDic = json.load(fConfig)
    print("machine-id:",configDic['machine-id'])
    taskId = configDic['machine-id'] + '-' + str((lambda:int(round(time.time() * 1000)))())
    print("taskId:",taskId)
    generateInfo = getGenerateInfo(prompt,negativePrompt,samplingSteps,samplingMethod,cfgScale,width,height,modelHash)
    global foreverPath
    imgsSavePath = foreverPath + taskId
    newForeverTask = ForeverTask(taskId,prompt,negativePrompt,samplingSteps,samplingMethod,width,height,cfgScale,batchCount,generateInfo,modelHash,imgsSavePath)
    with foreverTasksLock:
        foreverTasks.append(newForeverTask)
        foreverTasksIndex.append(taskId)
    return "","",20,"Euler a",512,512,7,10000,gr.update(choices = foreverTasksIndex,interactive = True),""
def getGenerateInfo(prompt,negativePrompt,samplingSteps,samplingMethod,cfgScale,width,height,modelHash):
    if modelHash != "":
        info = "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}, Model hash: {}".format(
                            prompt,
                            negativePrompt,
                            samplingSteps,
                            samplingMethod,
                            cfgScale,
                            -1,
                            width,
                            height,
                            modelHash
                        )
    else:
        info = "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, CFG scale: {}, Seed: {}, Size: {}x{}".format(
                            prompt,
                            negativePrompt,
                            samplingSteps,
                            samplingMethod,
                            cfgScale,
                            -1,
                            width,
                            height
                        )
    return info
def findTask(taskId):
    with foreverTasksLock:
        global foreverTasks
        for task in foreverTasks:
            if task.taskId == taskId:
                return task
    print("error:can't find taskId:",taskId)
    return None
def showTaskInfo(taskId):
    tempTask = findTask(taskId)
    if tempTask.status == "paused":
        return tempTask.taskId,tempTask.status,tempTask.generateInfo,tempTask.imgsSavePath,tempTask.batchCount,gr.update(visible = True),gr.update(visible = False),gr.update(visible = True),gr.update(visible = False),gr.update(visible = True)
    elif tempTask.status == "processing":
        return tempTask.taskId,tempTask.status,tempTask.generateInfo,tempTask.imgsSavePath,tempTask.batchCount,gr.update(visible = True),gr.update(visible = True),gr.update(visible = False),gr.update(visible = False),gr.update(visible = True)
    elif tempTask.status == "completed":
        return tempTask.taskId,tempTask.status,tempTask.generateInfo,tempTask.imgsSavePath,tempTask.batchCount,gr.update(visible = True),gr.update(visible = False),gr.update(visible = False),gr.update(visible = False),gr.update(visible = True)
    else:
        return tempTask.taskId,tempTask.status,tempTask.generateInfo,tempTask.imgsSavePath,tempTask.batchCount,gr.update(visible = True),gr.update(visible = True),gr.update(visible = False),gr.update(visible = True),gr.update(visible = True)
def delete(taskId):
    task = findTask(taskId)
    if task is not None:
        if task.status == "processing":
            task.status = "paused"
            global status
            while status == "busy":
                time.sleep(0.5)
        global foreverTasks
        global foreverTasksIndex
        with foreverTasksLock:
            foreverTasks.remove(task)
            foreverTasksIndex.remove(taskId)
    return gr.update(choices = foreverTasksIndex,interactive = True), "","","","","",gr.update(visible = False),gr.update(visible = False) ,gr.update(visible = False) ,gr.update(visible = False),gr.update(visible = False)         
def insert(taskId):
    task = findTask(taskId)
    if task is not None:
        global foreverTasks
        global status
        global foreverTasksIndex
        findProcessing = False
        with foreverTasksLock:
            for item in foreverTasks:
                if item.status == "processing":
                    findProcessing = True
                    foreverTasks.remove(task)
                    foreverTasks.insert(0,task)
                    foreverTasksIndex.remove(taskId)
                    foreverTasksIndex.insert(0,taskId)
                    item.status = "waiting"
                    while status == "busy":
                        time.sleep(0.5)
                    break
            if findProcessing == False:
                foreverTasks.remove(task)
                foreverTasks.insert(0,task)
                foreverTasksIndex.remove(taskId)
                foreverTasksIndex.insert(0,taskId)
    return gr.update(choices = foreverTasksIndex,value = taskId,interactive = True),taskId,task.status,task.batchCount,gr.update(visible = False)
def pause(taskId):
    task = findTask(taskId)
    if task is not None:
        task.status = "paused"
        return task.status,task.batchCount,gr.update(visible = False),gr.update(visible = True),gr.update(visible = False)
    else:
        raise Error("can't find a task of taskId:",taskId)
def start(taskId):
    task = findTask(taskId)
    if task is not None:
        task.status = "waiting"
        return task.status,gr.update(visible = True),gr.update(visible = False),gr.update(visible = True)
    else:
        raise Error("can't find a task of taskId:",taskId)
def syncTasks():
    global foreverTasksIndex
    return gr.update(choices = foreverTasksIndex)
def refreshLeftBatchCountAndStatus(taskId):
    task = findTask(taskId)
    if task is not None:
        return task.batchCount,task.status
    else:
        raise Error("can't find a task of taskId:",taskId)                    

            

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as generate_forever:
        with gr.Column(elem_id = "forever_batch"):
            syncButton = gr.Button(value = "Synchronize Tasks",variant = "primary")
            with gr.Blocks(title = "Task info"):
                processDropDown = gr.Dropdown(label = "Tasks")
                taskIdText = gr.Textbox(label = "Task Id",interactive = False)
                taskStatusText = gr.Textbox(label = "Task Status",interactive = False)
                with gr.Row():
                    batchCountLeftText = gr.Textbox(label = "Counts Left",interactive = False)
                    refreshBatchCountAndStatusButton = gr.Button(value = "Refresh Left BatchCount And Status",visible = False)
                taskInfoText = gr.Textbox(label = "Generate Info",lines = 5,interactive = False)
                imgsSavePathText = gr.Textbox(label = "Images Save Path",interactive = False)
                with gr.Column():
                    insertButton = gr.Button(value = "Insert",visible = False)
                    pauseButtoon = gr.Button(value = "Pause",visible = False)
                    startButton = gr.Button(value = "Start",visible = False)
                    deleteButton = gr.Button(value = "Delete Batch",elem_id = "forever_batch_delete_btn",visible = False)

            with gr.Column(elem_id = "forever_batch_item"):
                with gr.Row():
                    promptText = gr.Textbox(label = "Prompt",placeholder = "Prompt",lines = 2)
                    with gr.Column(scale = 1):
                        getGenerateInfoButton = gr.Button(value = "↙️",elem_id = "convert_generate_info_btn")
                        batchCountSlider = gr.Slider(maximum = 10000,value = 10000,label = "Batch Count",interactive = True)
                negativePromptText = gr.Textbox(label = "Negative Prompt",placeholder = "Negative Prompt",lines = 2,interactive = True)
                with gr.Row():
                    samplingStepsSlider = gr.Slider(maximum = 150,label = "Sampling Steps",value = 20,interactive = True)
                    samplingMethodDropdown = gr.Dropdown(choices = ["Euler a","Euler", "LMS", "Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM fast","DPM adaptive","LMS Karras","DPM2 Karras","DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DDIM","PLMS"],label = "Sampling Method",value = "Euler a",interactive = True)
                with gr.Row():
                    widthSlider = gr.Slider(maximum = 2048,value = 512,label = "Width",interactive = True)
                    heightSlider = gr.Slider(maximum = 2048,value = 512,label = "Height",interactive = True)
                    cfgScaleSlider = gr.Slider(maximum = 30,value = 7,label = "CFG Scale",interactive = True)
                with gr.Row(visible = False):
                    #hidden items
                    modelHashText = gr.Textbox(interactive = False,value = "")
        addButton = gr.Button(value = "Add Batch",variant = "primary",elem_id = "forever_batch_add_btn")
        getGenerateInfoButton.click(fn=None,_js = "convertPromptsInfo",inputs = [promptText],outputs = [promptText,negativePromptText,samplingStepsSlider,samplingMethodDropdown,cfgScaleSlider,widthSlider,heightSlider,modelHashText])
        addButton.click(fn = addForeverBatch,inputs=[promptText,negativePromptText,samplingStepsSlider,samplingMethodDropdown,widthSlider,heightSlider,cfgScaleSlider,batchCountSlider,modelHashText],outputs=[promptText,negativePromptText,samplingStepsSlider,samplingMethodDropdown,widthSlider,heightSlider,cfgScaleSlider,batchCountSlider,processDropDown,modelHashText])
        processDropDown.change(fn = showTaskInfo,inputs=[processDropDown],outputs=[taskIdText,taskStatusText,taskInfoText,imgsSavePathText,batchCountLeftText,deleteButton,pauseButtoon,startButton,insertButton,refreshBatchCountAndStatusButton])
        deleteButton.click(fn = delete,inputs = [taskIdText],outputs = [processDropDown,taskIdText,taskStatusText,batchCountLeftText,taskInfoText,imgsSavePathText,startButton,pauseButtoon,insertButton,deleteButton,refreshBatchCountAndStatusButton])
        insertButton.click(fn = insert,inputs = [taskIdText],outputs = [processDropDown,taskIdText,taskStatusText,batchCountLeftText,insertButton])
        pauseButtoon.click(fn = pause,inputs = [taskIdText],outputs = [taskStatusText,batchCountLeftText,pauseButtoon,startButton,insertButton])
        startButton.click(fn = start,inputs = [taskIdText],outputs = [taskStatusText,pauseButtoon,startButton,insertButton])
        syncButton.click(fn = syncTasks,inputs = [],outputs = [processDropDown])
        refreshBatchCountAndStatusButton.click(fn = refreshLeftBatchCountAndStatus,inputs = [taskIdText],outputs = [batchCountLeftText,taskStatusText])
    return (generate_forever , "txt2img Forever", "forever batch"),

script_callbacks.on_ui_tabs(on_ui_tabs)
def on_ui_settings():
    section = ('generate-forever', "Generate Forever")
    shared.opts.add_option("machine-id", shared.OptionInfo("", "set an machine id", section=section))

script_callbacks.on_ui_settings(on_ui_settings)
monitor = ForeverMonitor().start()
