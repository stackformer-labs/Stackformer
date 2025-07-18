import os
import torch
import random
import numpy as np
from torch.utils.tensorboard import summary # check some other
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR
from transformers import get_scheduler

from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_optimizer(optim_name, model, lr, weight_decay):
    if optim_name.lower() == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name.lower() == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer {optim_name}")

def train_model(model,
                train_dataload,
                test_dataload,
                eval_dataload=None,
                output_dir = "output",
                lr = 5e-5,
                lr_scheduler_type = "linear",
                num_epoch=3,
                max_steps=None,
                optimizer_type="adamw",
                save_steps=500,
                report_to_tfboard=True,
                gead_accumulation_step=1,
                weight_decay=0.01,
                warmup_steps=100,
                logging_steps=100,
                seed=42,
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ):
    set_seed(seed)
    model.to(device)
    
    total_training_steps = max_steps if max_steps is not None else len(train_dataload) // gead_accumulation_step * num_epoch
    
    optimizer = get_optimizer(optimizer_type,model,lr,weight_decay)
    
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    os.makedirs(output_dir, exist_ok=True)
    write = summary(log_dir=output_dir) if report_to_tfboard else None
    
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    
    global_step = 0
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        pdar = tqdm(train_dataload, desc=f"epoch {epoch+1}/{num_epoch}",leave=False)
        
        for step, batch in enumerate(pdar):
            input, target = batch
            
            input = input.to(device)
            target = target.to(device)
            
            with torch.cuda.amp.autocast():
                output = model(input)
                loss = torch.nn.functional.cross_entropy(output,target)
                
            scaler.scale(loss).backward()
            
            if(step+1) % gead_accumulation_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step+=1
                
                if global_step % logging_steps == 0 and write:
                    write.add_scalar("loss/train",loss.item(),global_step)
                    write.add_scalar("lr",lr_scheduler.get_last_lr()[0],global_step)
                    
                    if global_step % save_steps == 0:
                        save_path = os.path.join(output_dir, f"step_{global_step}")
                        os.makedirs(save_path,exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(save_path,"model.pt"))
                        
            
            epoch_loss += loss.item()
            
            if max_steps and global_step >= max_steps:
                break
            
            print(f"Epoch {epoch+1} finished Loss: {epoch_loss:.4}")
            
            if eval_dataload:
                eval_loss = 0
                model.eval()
                with torch.no_grad():
                    for eval_batch in eval_dataload:
                        input,target = eval_batch
                        input = input.to(device)
                        target = target.to(device)
                        
                        with torch.cuda.amp.autocast():
                            output = model(input)
                            loss = torch.nn.functional.cross_entropy(output,target)
                            
                        eval_loss += loss.item()
                        model.train()
                    
                    avg_eval_loss = eval_loss / len(eval_dataload)
                    print(f"Eval loss: {avg_eval_loss:.4f}")
                    
                    if write:
                        write.add_scalar("loss/eval", avg_eval_loss, global_step)
                        
            if max_steps and global_step >= max_steps:
                break
            
        if write:
            write.close()
        print("training complete")