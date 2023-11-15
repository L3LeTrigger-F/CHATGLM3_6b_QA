#!/usr/bin/env python
# encoding: utf-8  

"""
@file: main.py
@description: 执行入口函数
@create: 2023/11/01 10:33
"""
import os
import logging
from utils.pdf_loader import load_pdf,load_h5
import transformers
import torch
from torch.utils.data import DataLoader
from utils.step_runner import StepRunner,save_ckpt,load_ckpt
from torchkeras import KerasModel 
from transformers import AutoModel
from utils.data_preprocess import QADataset
from peft import get_peft_model, LoraConfig, TaskType
data_path="/home/lhl/nlp/LoRA/data/data"
trained=True

if __name__ == "__main__":
    # logger.info("============start=============")
    model_name="chatglm3-6b"
    max_source_length = 128
    max_target_length = 512
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda('cuda:0')
    filename="/home/lhl/nlp/LoRA/data/process_data"
    data_path="/home/lhl/nlp/LoRA/data/data"
    h5_path="/home/lhl/nlp/LoRA/data/process_data/QADataset_2"
    if not os.path.exists(filename):
        load_pdf(data_path)
    question_list, answer_list=load_h5(h5_path)
    train_data=QADataset(question_list, answer_list,tokenizer, max_source_length, max_target_length)
    my_loader = DataLoader(train_data,batch_size=64,shuffle=True,num_workers = 0,drop_last=True)
    model.supports_gradient_checkpointing = True  #节约cuda
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    peft_model_id = "/home/lhl/nlp/LoRA/chatglm3-6b" 
    
    peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1,
)
    # peft_config= peft_config.from_pretrained(peft_model_id)
    
    model = get_peft_model(model, peft_config)
    # model.load_adapter("22h/cabrita-lora-v0-1", adapter_name="portuguese_alpaca")
    # model.load_adapter()
    # model.is_parallelizable = True
    # model.model_parallel = True
    model.print_trainable_parameters()
    
    KerasModel.StepRunner = StepRunner
    KerasModel.save_ckpt = save_ckpt 
    KerasModel.load_ckpt = load_ckpt 
    keras_model = KerasModel(model,loss_fn = None,
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5))
    ckpt_path = 'models/chatglm3-6b-QA_2'

    keras_model.fit(train_data = train_data,
                epochs=100,patience=3,
                monitor='train_loss',mode='min',
                ckpt_path = ckpt_path,
                mixed_precision='fp16'
               )
    model.save_pretrained("models/chatglm3-6b-QA_2", max_shard_size='1GB')
    tokenizer.save_pretrained("models/chatglm3-6b-QA_2")







