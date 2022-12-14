function convertPromptsInfo(input){
    var splitInput = input.split('\n')
    dynamicPrompt = splitInput[0]
    dynamicNegativePrompt = splitInput[1].split(': ')[1]
    others = splitInput[2]
    othersSplit = others.split(', ')
    steps = othersSplit[0].split(': ')[1]
    sampler = othersSplit[1].split(': ')[1]
    cfgScale = othersSplit[2].split(': ')[1]
    seed = othersSplit[3].split(': ')[1]
    width = othersSplit[4].split(': ')[1].split('x')[0]
    height = othersSplit[4].split(': ')[1].split('x')[1]
    return [dynamicPrompt,dynamicNegativePrompt,steps,sampler,cfgScale,width,height]



}