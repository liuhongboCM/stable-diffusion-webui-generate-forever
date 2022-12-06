setTimeout(()=>{
  var GenerateForeverBtn = gradioApp().getElementById("generateForeverButton")
  GenerateForeverBtn.addEventListener("click",()=>{
    var slider = gradioApp().getElementById("batch_count")
    console.log("slider",slider)
    var input = slider.getElementsByTagName("input")
    console.log(input)
    input[0].max = "1000000"
    input[1].max = "1000000"

  })

},500) 

