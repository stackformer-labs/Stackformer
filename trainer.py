import torch
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

class Trainer:
    def __init__(self,
                model,
                train_dataset,
                eval_dataset,
                train_batch_size,
                eval_batch_size,
                output_dir,
                num_epoch,
                lr: float,
                scheduler_type=None, 
                optimizer_type="adamw",
                weight_decay=0.01, 
                warmup_steps=0,
                grad_accumulation_step=1, 
                max_eval_step=None,
                max_steps=None,
                Save_step=None,
                Save_epoch=None,
                max_epoch=None,
                model_to_resume=None,
                resume_training=False,
                seed=42,
                device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.num_epoch = num_epoch
        self.max_steps = max_steps
        self.max_epoch = max_epoch
        self.max_eval_step = max_eval_step
        self.lr = lr
        self.scheduler_type = scheduler_type
        self.output_dir = output_dir
        self.model_to_resume = model_to_resume
        self.resume_training = resume_training
        self.Save_step = Save_step
        self.Save_epoch = Save_epoch
        self.grad_accumulation_step = grad_accumulation_step
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.device = device
    
    # --- random seed ---
    def set_seed(self, seed):
        torch.manual_seed(seed)
        if self.device == 'cuda' and torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
        
    # --- scheduler ---
    def get_scheduler(self, scheduler_type, total_training_steps, optimizer):
        if scheduler_type is None:
            return None
        elif scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=total_training_steps
            )
        elif scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_training_steps
            )
        elif scheduler_type == "cosine_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=4 
            )
        elif scheduler_type == "cosineannealing":
            return CosineAnnealingLR(optimizer, T_max=total_training_steps)
        elif scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(optimizer, T_0=total_training_steps//4, T_mult=2)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    # --- optimizer ---
    def get_optimizer(self, optimizer_type, model, lr, weight_decay):
        if optimizer_type.lower() == "adamw":
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_type}")

    # --- validate model ---
    def eval_model(self, model, eval_loader, max_val_steps):
        eval_loss = 0
        model.eval()
        max_val_steps = min(max_val_steps or len(eval_loader), len(eval_loader))
        with torch.no_grad():
            pbar = tqdm(eval_loader, total=max_val_steps, desc="Evaluating", leave=False)
            for step, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model(inputs)  # shape: [B, T, V]
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    targets.view(-1), ignore_index=-100)
                pbar.set_postfix(loss=loss.item())
                eval_loss += loss.item()
                if step + 1 >= max_val_steps:
                    break
        model.train()
        avg_eval_loss = eval_loss / max_val_steps
        return avg_eval_loss
    
    # --- train dataloader ---
    def get_train_loader(self, train_dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=True if self.device == 'cuda' else False
        )
        return train_loader

    # --- validation dataloader ---
    def get_eval_loader(self, eval_dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=batch_size,
            shuffle=False,
            generator=generator,
            pin_memory=True if self.device == 'cuda' else False
        )
        return eval_loader
    
    # --- save model ---
    def save_model(self, model, optimizer, scheduler, epoch, num_epoch, loss, global_step, output_dir, name):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'current_epoch': epoch,
            'num_epoch': num_epoch,
            'loss': loss,
            'global_step': global_step,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        os.makedirs(output_dir, exist_ok=True)
        path = f'{output_dir}/checkpoint_{name}.pt'
        torch.save(checkpoint, path)
        print(f'Saved training state to {path}')
        
    def load_checkpoint(self, path, model, optimizer, scheduler):
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['current_epoch']
        global_step = checkpoint['global_step']
        loss = checkpoint['loss']
        num_epoch = checkpoint['num_epoch']
        # RNG
        torch.set_rng_state(checkpoint['rng_state']['torch'])
        if torch.cuda.is_available() and checkpoint['rng_state']['cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
        return {
            'current_epoch': current_epoch,
            'num_epoch': num_epoch,
            'global_step': global_step,
            'loss': loss
        }
    
    # --- train ---
    def train(self):
        # --- seed --- 
        self.set_seed(self.seed)
        # --- dataloader ---
        train_loader = self.get_train_loader(self.train_dataset, self.train_batch_size, self.seed)
        eval_loader = self.get_eval_loader(self.eval_dataset, self.eval_batch_size, self.seed)
        
        # --- Calculate the total step ---
        steps_per_epoch = len(train_loader) // self.grad_accumulation_step
        total_training_steps = self.max_steps if self.max_steps is not None else steps_per_epoch * self.num_epoch
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        model = self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, model, self.lr, self.weight_decay)
        criterion = torch.nn.functional.cross_entropy
        scheduler = self.get_scheduler(self.scheduler_type, total_training_steps, optimizer)
        
        global_step = 0
        start_epoch = 0
        num_epoch = self.num_epoch
        accumulated_steps = 0

        if self.resume_training and self.model_to_resume:
            ckpt_data = self.load_checkpoint(self.model_to_resume, model, optimizer, scheduler)
            start_epoch = ckpt_data['current_epoch']
            global_step = ckpt_data['global_step']
            print(f"Resuming training from epoch {start_epoch}, step {global_step}")
            
        for epoch in range(start_epoch, num_epoch):
            model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                # --- load the inputs and targets ---
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # --- get prediction from model ---
                output = model(inputs)
                
                # --- calculate loss ---
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                loss = loss / self.grad_accumulation_step
                loss.backward()
                
                pbar.set_postfix(loss=loss.item() * self.grad_accumulation_step)
                epoch_loss += loss.item() * self.grad_accumulation_step
                accumulated_steps += 1
                
                # --- gradient accumulation ---
                if accumulated_steps % self.grad_accumulation_step == 0:
                    # Gradient clipping for stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_steps = 0
                
                # Check max steps
                if self.max_steps is not None and global_step >= self.max_steps:
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir, 
                        name=f'final_step_epoch_{epoch+1}_step_{global_step}'
                    )
                    return
                
                # Save at specific steps
                if (self.Save_step is not None and 
                    global_step > 0 and 
                    global_step % self.Save_step == 0):
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir, 
                        name=f'epoch_{epoch+1}_step_{global_step}'
                    )

            # Handle remaining accumulated gradients at epoch end
            if accumulated_steps > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} finished - Training Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
            print(f"Eval loss: {avg_eval_loss:.4f}")
            
            # Check max epoch
            if self.max_epoch is not None and (epoch+1) == self.max_epoch:
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=avg_epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, 
                    name=f'final_model_epoch_{epoch+1}_step_{global_step}'
                )
                return
            
            # Save at specific epochs
            if (self.Save_epoch is not None and 
                (epoch + 1) % self.Save_epoch == 0):
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=avg_epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, 
                    name=f'epoch_{epoch+1}_step_{global_step}'
                )