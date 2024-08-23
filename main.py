# import re
# import argparse
# from string import punctuation

# import torch
# import yaml
# import numpy as np
# from torch.utils.data import DataLoader
# from g2p_en import G2p
# from pypinyin import pinyin, Style

# from utils.model import get_model, get_vocoder
# from utils.tools import to_device, synth_samples
# from dataset import TextDataset
# from text import text_to_sequence
# import nltk



# from fastapi import FastAPI, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.requests import Request

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# nltk.download('averaged_perceptron_tagger_eng')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def read_lexicon(lex_path):
#     lexicon = {}
#     with open(lex_path) as f:
#         for line in f:
#             temp = re.split(r"\s+", line.strip("\n"))
#             word = temp[0]
#             phones = temp[1:]
#             if word.lower() not in lexicon:
#                 lexicon[word.lower()] = phones
#     return lexicon


# def preprocess_english(text, preprocess_config):
#     text = text.rstrip(punctuation)
#     lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

#     g2p = G2p()
#     phones = []
#     words = re.split(r"([,;.\-\?\!\s+])", text)
#     for w in words:
#         if w.lower() in lexicon:
#             phones += lexicon[w.lower()]
#         else:
#             phones += list(filter(lambda p: p != " ", g2p(w)))
#     phones = "{" + "}{".join(phones) + "}"
#     phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
#     phones = phones.replace("}{", " ")

#     print("Raw Text Sequence: {}".format(text))
#     print("Phoneme Sequence: {}".format(phones))
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )

#     return np.array(sequence)


# def synthesize(model, step, configs, vocoder, batchs, control_values):
#     preprocess_config, model_config, train_config = configs
#     pitch_control, energy_control, duration_control = control_values

#     for batch in batchs:
#         batch = to_device(batch, device)
#         with torch.no_grad():
#             # Forward
#             output = model(
#                 *(batch[2:]),
#                 p_control=pitch_control,
#                 e_control=energy_control,
#                 d_control=duration_control
#             )
#             synth_samples(
#                 batch,
#                 output,
#                 vocoder,
#                 model_config,
#                 preprocess_config,
#                 train_config["path"]["result_path"],
#             )

# # if __name__ == "__main__":

# #     restore_step=90000
# #     mode="single"
# #     text=""
# #     speaker_id=0
# #     preprocess_config=r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\preprocess.yaml"
# #     model_config=r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\model.yaml"
# #     train_config=r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\train.yaml"
# #     pitch_control=1.0
# #     energy_control=1.0
# #     duration_control=1.0
    
# #     if mode == "single":
# #         assert args.source is None and args.text is not None

# #     # Read Config
# #     preprocess_config = yaml.load(
# #         open(args.preprocess_config, "r"), Loader=yaml.FullLoader
# #     )
# #     model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
# #     train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
# #     configs = (preprocess_config, model_config, train_config)

# #     # Get model
# #     model = get_model(args, configs, device, train=False)

# #     # Load vocoder
# #     vocoder = get_vocoder(model_config, device)

# #     # # Preprocess texts
# #     # if args.mode == "batch":
# #     #     # Get dataset
# #     #     dataset = TextDataset(args.source, preprocess_config)
# #     #     batchs = DataLoader(
# #     #         dataset,
# #     #         batch_size=8,
# #     #         collate_fn=dataset.collate_fn,
# #     #     )
# #     # if args.mode == "single":
# #     #     ids = raw_texts = [args.text[:100]]
# #     #     speakers = np.array([args.speaker_id])
        
# #     #     if preprocess_config["preprocessing"]["text"]["language"] == "en":
# #     #         texts = np.array([preprocess_english(args.text, preprocess_config)])
# #     #     elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
# #     #         texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
# #     #     text_lens = np.array([len(texts[0])])
# #     #     batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

# #     # control_values = args.pitch_control, args.energy_control, args.duration_control

# #     # synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

# class Args(dict):
#     def __getattr__(self, item):
#         if item in self:
#             return self[item]
#         raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

#     def __setattr__(self, key, value):
#         self[key] = value


# @app.on_event("startup")
# async def load_model():
    
#     global args, all_configs,configs,models
#     args=Args({
#     "restore_step":900000
#     ,"mode":"single"
#     ,"text":""
#     ,"speaker_id":0
#     ,"preprocess_config":r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\preprocess.yaml"
#     ,"model_config":r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\model.yaml"
#     ,"train_config":r"C:\Users\07032\github_projects\texmin\FastSpeech2\config\LJSpeech\train.yaml"
#     ,"pitch_control":1.0
#     ,"energy_control":1.0
#     ,"duration_control":1.0
#     ,"source":None
#     })
#     if args.mode == "single":
#         assert args.source is None and args.text is not None

#     # Read Config
#     all_configs=Args({
#     "preprocess_config":yaml.load(
#         open(args.preprocess_config, "r"), Loader=yaml.FullLoader
#     )
#     ,"model_config":yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
#     ,"train_config":yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
#     })

#     configs = (all_configs.preprocess_config, all_configs.model_config, all_configs.train_config)

#     # Get model
#     model = get_model(args, configs, device, train=False)

#     # Load vocoder
#     vocoder = get_vocoder(all_configs.model_config, device)
#     models=Args({"model":model,"vocoder":vocoder})


#     print("model loaded")

# @app.get("/", response_class=HTMLResponse)
# async def get_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/submit", response_class=HTMLResponse)
# async def handle_form(request: Request, data: str = Form(...)):
#     # Process the received data
#     print(type(data),type(request))
#     print()
#     args["text"]=data
#     if args.mode == "single":
#         ids = raw_texts = [args.text]
#         speakers = np.array([args.speaker_id])
        
#         if all_configs.preprocess_config["preprocessing"]["text"]["language"] == "en":
#             texts = np.array([preprocess_english(args.text, all_configs.preprocess_config)])
        
#         text_lens = np.array([len(texts[0])])
#         batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

#     control_values = args.pitch_control, args.energy_control, args.duration_control

#     synthesize(models.model, args.restore_step, configs, models.vocoder, batchs, control_values)

#     audio_url=r"C:\Users\07032\github_projects\texmin\FastSpeech2\output\result\LJSpeech\sound.wav"







import os

def get_file_sizes(directory):
    file_sizes = {}
    
    # Iterate over all files in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_sizes[file_path] = file_size//(1024*1024)
    
    return file_sizes

# Example usage
folder_path = r"C:\Users\07032\github_projects\text_to_speech"
print(folder_path)
sizes = get_file_sizes(folder_path)

for file_path, size in sizes.items():
    if size>1:
        print(f"File: {file_path} Size: {size} mb")
