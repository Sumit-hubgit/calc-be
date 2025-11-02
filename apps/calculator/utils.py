# import torch
# from transformers import pipeline, BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration
# from PIL import Image

# # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )


# model_id = "llava-hf/llava-1.5-7b-hf"
# processor = AutoProcessor.from_pretrained(model_id)
# model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
# # pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# def analyze_image(image: Image):
#     prompt = "USER: <image>\nAnalyze the equation or expression in this image, and return answer in format: {expr: given equation in LaTeX format, result: calculated answer}"

#     inputs = processor(prompt, images=[image], padding=True, return_tensors="pt").to("cuda")
#     for k, v in inputs.items():
#         print(k,v.shape)

#     output = model.generate(**inputs, max_new_tokens=20)
#     generated_text = processor.batch_decode(output, skip_special_tokens=True)
#     for text in generated_text:
#         print(text.split("ASSISTANT:")[-1])

import google.generativeai as genai
import json
from PIL import Image
from constants import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

def analyze_image(img: Image, dict_of_vars: dict):
    """
    Analyzes a mathematical image and returns structured results.
    Forces Gemini to return valid JSON for easier parsing.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    dict_of_vars_str = json.dumps(dict_of_vars, ensure_ascii=False)
    
    prompt = (
        f"You have been given an image with some mathematical expressions, equations, or graphical problems, "
    f"and you need to solve them. "
    f"Use PEMDAS for mathematical expressions. "
    f"For example: "
    f"Q. 2 + 3 * 4 -> (3*4)=12, 2+12=14. "
    f"Q. 2 + 3 + 5 * 4 - 8 / 2 -> 21. "
    f"Only one type of problem will appear: "
    f"1. Simple math expressions: return a LIST of ONE DICT like "
    f"[{{'expr': '2 + 2', 'result': 4}}]. "
    f"2. Equations with variables: return COMMA SEPARATED DICTS, each with 'expr', 'result', 'assign': True. "
    f"3. Variable assignments like x=4: return LIST of DICTS with 'assign': True. "
    f"4. Graphical math problems: return LIST of ONE DICT like [{{'expr': '<description>', 'result': '<answer>'}}]. "
    f"5. Abstract concepts: return same format as above. "
    f"Include any user variables from: {dict_of_vars_str}. "
    f"Return ONLY valid JSON. DO NOT USE BACKTICKS, MARKDOWN, OR PYTHON DICT SYNTAX. "
    f"Keys and strings must be properly quoted."
    )

    # Generate response from Gemini
    response = model.generate_content([prompt, img])
    print("Raw response from Gemini:", response.text)

    # Parse response as JSON
    answers = []
    try:
        answers = json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini API: {e}")
        print("Raw response was:", response.text)

    # Ensure every answer has 'assign' key
    for answer in answers:
        if 'assign' in answer:
            answer['assign'] = True
        else:
            answer['assign'] = False

    print("Processed answers:", answers)
    return answers
