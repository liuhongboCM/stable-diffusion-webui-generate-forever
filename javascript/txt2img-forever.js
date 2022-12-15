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
    modelHash = ""
    if(othersSplit.length>5){
        if(othersSplit[5].split(': ').length>1){
            modelHash = othersSplit[5].split(': ')[1]

        }
        
    }
    console.log(modelHash)
    
    return [dynamicPrompt,dynamicNegativePrompt,steps,sampler,cfgScale,width,height,modelHash]



}