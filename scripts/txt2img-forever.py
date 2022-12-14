
import gradio as gr
from modules import script_callbacks
class foreverTask():
    taskId = ""
    prompt = ""
    negativePrompt = ""
    samplingSteps = 20,
    samplingMethod = "Euler a"
    width = 512
    height = 512
    cfgScale = 7
    batchCount = 3000
    
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as generate_forever:
        with gr.Column(elem_id = "forever_batch"):
            processDropDown = gr.Dropdown(label = "Tasks")
            with gr.Blocks(title = "Task info"):
                taskIdText = gr.Textbox(label = "Task Id",interactive = False)
                taskInfoText = gr.Textbox(label = "Generate Info",lines = 5,interactive = False)
                imgsSavePath = gr.Textbox(label = "Images Save Path",interactive = False)
                
            deleteButton = gr.Button(value = "Delete Batch",elem_id = "forever_batch_delete_btn")
            with gr.Column(elem_id = "forever_batch_item"):
                with gr.Row():
                    promptText = gr.Textbox(label = "Prompt",placeholder = "Prompt",lines = 2)
                    with gr.Column(scale = 1):
                        getGenerateInfoButton = gr.Button(value = "↙️",elem_id = "convert_generate_info_btn")
                        processNumber = gr.Slider(maximum = 10000,value = 3000,label = "Batch Count",interactive = True)
                negativePromptText = gr.Textbox(label = "Negative Prompt",placeholder = "Negative Prompt",lines = 2)
                with gr.Row():
                    samplingStepsSlider = gr.Slider(maximum = 150,label = "Sampling Steps",value = 20,interactive = True)
                    samplingMethodDropdown = gr.Dropdown(choices = ["Euler a","Euler", "LMS", "Heun","DPM2","DPM2 a","DPM++ 2S a","DPM++ 2M","DPM++ SDE","DPM fast","DPM adaptive","LMS Karras","DPM2 Karras","DPM2 a Karras","DPM++ 2S a Karras","DPM++ 2M Karras","DPM++ SDE Karras","DDIM","PLMS"],label = "Sampling Method",value = "Euler a",interactive = True)
                with gr.Row():
                    widthSlider = gr.Slider(maximum = 2048,value = 512,label = "Width",interactive = True)
                    heightSlider = gr.Slider(maximum = 2048,value = 512,label = "Height",interactive = True)
                    cfgScaleSlider = gr.Slider(maximum = 30,value = 7,label = "CFG Scale",interactive = True)
        addButton = gr.Button(value = "Add Batch",variant = "primary",elem_id = "forever_batch_add_btn")
        getGenerateInfoButton.click(fn=None,_js = "convertPromptsInfo",inputs = [promptText],outputs = [promptText,negativePromptText,samplingStepsSlider,samplingMethodDropdown,cfgScaleSlider,widthSlider,heightSlider])
    return (generate_forever , "txt2img Forever", "forever batch"),

script_callbacks.on_ui_tabs(on_ui_tabs)