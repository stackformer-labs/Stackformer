import os
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self,
                model,
                train_dataset,
                eval_dataset,
                train_batch_size,
                eval_batch_size,
                vocab_size,
                output_dir,
                num_epoch,
                lr: float,
                scheduler_type, 
                Save_step,
                Save_epoch,
                grad_accumulation_step, 
                optimizer_type,
                weight_decay, 
                warmup_steps,
                max_steps,
                max_eval_step,
                max_epoch = None,
                model_to_resume=None,
                resume_training=False,
                seed=42,
                device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.vocab_size = vocab_size
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
    def set_seed(self,seed):
        torch.manual_seed(seed)
        if self.device == 'cuda': 
            torch.cuda.manual_seed_all(seed)
        
    # --- scheduler ---
    def get_scheduler(self, scheduler_type, total_training_steps, optimizer):
        if scheduler_type == "linear":
            return LinearLR(optimizer,total_iters = total_training_steps)
        elif scheduler_type == "cosineannealing":
            return CosineAnnealingLR(optimizer, T_max=total_training_steps)
        
    # --- optimizer ---
    def get_optimizer(self, optimizer_type, model, lr, weight_decay):
        if optimizer_type.lower() == "adamw":
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay,momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_type}")

    # --- validate model ---
    def eval_model(self, model, eval_loader, max_val_steps):
        eval_loss = 0
        model.eval()
        with torch.no_grad():
            pbar = tqdm(eval_loader, total=max_val_steps, desc="Evaluating", leave=False)
            for step,eval_batch in enumerate(pbar):
                inputs,targets = eval_batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model(inputs)
                loss = torch.nn.functional.cross_entropy(output,targets)
                pbar.set_postfix(loss=loss.item())
                eval_loss += loss.item()
                if step+1 >= max_val_steps:
                    break
        model.train()
        avg_eval_loss = eval_loss / min(len(eval_loader), max_val_steps)
        return avg_eval_loss
    
    # --- train dataloader ---
    def get_train_loader(self, train_dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True,
        # num_workers=2,
        # worker_init_fn=worker_init_fn,
        generator=generator
        )
        return train_loader

    # --- validation dataloader ---
    def get_eval_loader(self, eval_dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
        shuffle=False,
        #num_workers=2,
        # worker_init_fn=worker_init_fn,
        generator=generator
        )
        return eval_loader
    # --- save model ---
    def save_model(self, model, optimizer, scheduler,
                epoch, num_epoch, loss, global_step, output_dir, name):
        checkpoint = { 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'current_epoch': epoch,
            'num_epoch': num_epoch,
            'loss': loss,
            'global_step': global_step,
            'rng_state':{
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            }
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
            torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'][0])
        return {
            'current_epoch': current_epoch,
            'num_epoch': num_epoch,
            'global_step': global_step,
            'loss': loss}
    
    # --- train ---
    def train(self):
        # --- seed --- 
        self.set_seed(self.seed)
        # --- dataloader ---
        train_loader = self.get_train_loader(self.train_dataset, self.train_batch_size, self.seed)
        eval_loader = self.get_eval_loader(self.eval_dataset, self.eval_batch_size, self.seed)
        
        # --- Calculate the total step ---
        total_training_steps = self.max_steps if self.max_steps is not None else len(train_loader) // self.grad_accumulation_step * self.num_epoch
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        model = self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, model, self.lr, self.weight_decay)
        criterion = torch.nn.functional.cross_entropy
        scheduler = self.get_scheduler(self.scheduler_type, total_training_steps, optimizer)
        
        global_step = 0
        start_epoch = 0
        num_epoch = self.num_epoch
        loss = torch.tensor(0.0, device=self.device)
        ckpt_global_step = 0

        if self.resume_training:
            ckpt_data = self.load_checkpoint(self.model_to_resume, model, optimizer, scheduler)
            start_epoch = ckpt_data['current_epoch']
            num_epoch = ckpt_data['num_epoch']
            ckpt_global_step = ckpt_data['global_step']
            loss = ckpt_data['loss']
            
        for epoch in range(start_epoch,num_epoch):
            model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epoch}",leave=False)
            for batch_idx, batch in enumerate(pbar):
                if self.resume_training and global_step < ckpt_global_step:
                    global_step += 1
                    continue
                global_step += 1
                # --- load the inputs and targets ---
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # --- get_prediction from model ---
                logits = model(inputs)  # [B, T, V]
                # --- calculate loss ---
                loss = criterion(logits.view(-1, logits.shape[-1]), targets.view(-1))
                loss = loss / self.grad_accumulation_step
                loss.backward()
                pbar.set_postfix(loss=loss.item() * self.grad_accumulation_step)
                epoch_loss += loss.item() * self.grad_accumulation_step
                # --- gradient accumulation ---
                if (batch_idx + 1) % self.grad_accumulation_step == 0 or global_step >= self.max_steps:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                
                if self.max_steps is not None and global_step >= self.max_steps:
                    self.save_model(model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, name= f'epoch_{epoch+1}_step_{global_step}')
                    return
                if self.Save_step and (batch_idx + 1) % self.Save_step == 0:
                    self.save_model(model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, name = f'epoch_{epoch+1}_step_{global_step}' )

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} finished Loss: {avg_epoch_loss:.4f}")
            avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
            print(f"Eval loss: {avg_eval_loss:.4f}")
            
            if self.max_epoch is not None and (epoch+1) == self.max_epoch:
                    self.save_model(model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, name=f'epoch_{epoch+1}_step_{global_step}')
                    return
            if self.Save_epoch and (epoch + 1) % self.Save_epoch == 0:
                self.save_model(model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir, name=f'epoch_{epoch+1}_step_{global_step}')
