from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import time
#pay attention to this line, not import from transformers, import from our GitHub repo's eval folder qwen2_5_vl
from eval.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from datetime import datetime
import torch

curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DROP_METHOD = 'feature'
DROP_THRESHOLD = 0.5
DROP_ABSOLUTE = True
DR_SAVE_PATH = "drop_{curr_time}.jsonl"

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "wyccccc/TimeChatOnline-7B", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("wyccccc/TimeChatOnline-7B")
test_questions = [
    "Is there any clothes hanger next to the towel? What is their relative relation?",
]
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/mnt/bn/xyxdata/home/codes/my_projs/open-eqa/assets/scene0246_00.mp4",
                # "min_pixels": 336*336,
                # "max_pixels": 336*336,
                # "max_frames": 1016,
                # "min_frames": 4,
                # "fps": 1.0
            },
            {
                "type": "text", 
                "text": test_questions[0]
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    drop_method=DROP_METHOD,
    drop_threshold=DROP_THRESHOLD,
    drop_absolute=DROP_ABSOLUTE,
    dr_save_path=DR_SAVE_PATH,
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)