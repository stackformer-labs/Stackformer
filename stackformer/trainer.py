import torch
import os
from torch.utils.data import DataLoader,Subset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm

class Trainer:
    """
    A comprehensive trainer class for training transformer models with advanced features.
    
    This trainer provides a complete training pipeline with support for:
    - Multiple optimizers (AdamW, SGD)
    - Various learning rate schedulers
    - Gradient accumulation for large models
    - Automatic checkpointing and resuming
    - Built-in evaluation during training
    - Flexible saving strategies
    """
    
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
                scheduler_type=None, 
                optimizer_type="adamw",
                eval_per_epoch = 1,
                eval_per_step = None,
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
        """
        Initialize the Trainer with all necessary parameters.
        
        Args:
            model: PyTorch model to train (e.g., transformer model)
            train_dataset: PyTorch Dataset containing training data
            eval_dataset: PyTorch Dataset containing evaluation data
            train_batch_size (int): Number of samples per training batch
            eval_batch_size (int): Number of samples per evaluation batch
            vocab_size (int): Size of the vocabulary (for loss calculation)
            output_dir (str): Directory to save checkpoints and models
            num_epoch (int): Number of training epochs
            lr (float): Learning rate for optimizer
            scheduler_type (str, optional): Type of LR scheduler. Options:
                - None: No scheduler
                - "linear": Linear decay with warmup
                - "cosine": Cosine annealing with warmup
                - "cosine_restarts": Cosine with hard restarts
                - "cosineannealing": Standard cosine annealing
                - "cosine_warm_restarts": Cosine annealing with warm restarts
            optimizer_type (str): Type of optimizer ("adamw" or "sgd")
            eval_per_epoch (int): Evaluate every N epochs (default: 1)
            eval_per_step (int, optional): Evaluate every N steps
            weight_decay (float): Weight decay for regularization (default: 0.01)
            warmup_steps (int): Number of warmup steps for scheduler (default: 0)
            grad_accumulation_step (int): Gradient accumulation steps (default: 1)
            max_eval_step (int, optional): Maximum evaluation steps per eval
            max_steps (int, optional): Maximum training steps (overrides epochs)
            Save_step (int, optional): Save checkpoint every N steps
            Save_epoch (int, optional): Save checkpoint every N epochs
            max_epoch (int, optional): Maximum epochs (early stopping)
            model_to_resume (str, optional): Path to checkpoint to resume from
            resume_training (bool): Whether to resume training (default: False)
            seed (int): Random seed for reproducibility (default: 42)
            device (str): Device to train on ("cpu", "cuda", "mps")
        """
        self.model = model
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.eval_dataset = eval_dataset
        self.eval_batch_size = eval_batch_size
        self.vocab_size = vocab_size
        self.num_epoch = num_epoch
        self.max_steps = max_steps
        self.max_epoch = max_epoch
        self.eval_per_epoch = eval_per_epoch
        self.eval_per_step = eval_per_step
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
    
    def set_seed(self, seed):
        """
        Set random seed for reproducible training.
        """
        torch.manual_seed(seed)
        if self.device == 'cuda' and torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
        
    def get_scheduler(self, scheduler_type, total_training_steps, optimizer):
        """
        Create and return a learning rate scheduler.
        """
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
                num_cycles=4  # Number of restarts
            )
        elif scheduler_type == "cosineannealing":
            return CosineAnnealingLR(optimizer, T_max=total_training_steps)
        elif scheduler_type == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(optimizer, T_0=total_training_steps//4, T_mult=2)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    def get_optimizer(self, optimizer_type, model, lr, weight_decay):
        """
        Create and return an optimizer.
        """
        if optimizer_type.lower() == "adamw":
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_type}")

    def eval_model(self, model, eval_loader, max_val_steps):
        """
        Evaluate the model on validation data with a clean progress bar.
        """
        eval_loss = 0
        model.eval()
        max_val_steps = min(max_val_steps or len(eval_loader), len(eval_loader))
        with torch.no_grad():
            pbar = tqdm(eval_loader, total=max_val_steps, desc="Evaluating", leave=False)
            for step, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                output = model(inputs)  # [B, T, V]
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)),
                    targets.view(-1),
                    ignore_index=-100
                )
                eval_loss += loss.item()
                if step + 1 >= max_val_steps:
                    break
            pbar.close()
        model.train()
        avg_eval_loss = eval_loss / max_val_steps
        return avg_eval_loss

    
    def get_train_loader(self, train_dataset, batch_size, seed):
        """
        Create a DataLoader for training data.
        """
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

    def get_eval_loader(self, eval_dataset, batch_size, seed):
        """
        Create a DataLoader for evaluation data.
        """
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
    
    def save_model(self, model, optimizer, scheduler, epoch, num_epoch, loss, global_step, 
                accumulated_steps, batch_idx_to_resume, output_dir, name):
        """
        Save a complete training checkpoint.
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'current_epoch': epoch,
            'num_epoch': num_epoch,
            'loss': loss,
            'accumulated_steps': accumulated_steps,
            'global_step': global_step,
            'batch_idx_to_resume': batch_idx_to_resume,
        }
        os.makedirs(output_dir, exist_ok=True)
        path = f'{output_dir}/checkpoint_{name}.pt'
        torch.save(checkpoint, path)
        print(f'Saved training state to {path}')
        
    def load_checkpoint(self, path, model, optimizer, scheduler):
        """
        Load a training checkpoint and restore training state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_epoch = checkpoint['current_epoch']
        global_step = checkpoint['global_step']
        loss = checkpoint['loss']
        num_epoch = checkpoint['num_epoch']
        accumulated_steps = checkpoint['accumulated_steps']
        batch_idx_to_resume = checkpoint['batch_idx_to_resume']
        return {
            'current_epoch': current_epoch,
            'num_epoch': num_epoch,
            'accumulated_steps' : accumulated_steps,
            'batch_idx_to_resume': batch_idx_to_resume,
            'global_step': global_step,
            'loss': loss
        }
    
    def train(self):
        """
        Main training loop that handles the complete training process.
        """
        # Set random seed for reproducibility
        self.set_seed(self.seed)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        model = self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, model, self.lr, self.weight_decay)
        criterion = torch.nn.functional.cross_entropy
        # scheduler will be built later, after total_training_steps is known

        # Default training state
        global_step = 0
        start_epoch = 1
        num_epoch = self.num_epoch
        batch_idx_to_resume = 0
        accumulated_steps = 0

        # Handle checkpoint resume
        if self.resume_training and self.model_to_resume:
            # temporarily build a scheduler with dummy steps (will be replaced later)
            scheduler_tmp = self.get_scheduler(self.scheduler_type, 1, optimizer)
            ckpt_data = self.load_checkpoint(self.model_to_resume, model, optimizer, scheduler_tmp)

            start_epoch = ckpt_data['current_epoch']
            global_step = ckpt_data['global_step']
            num_epoch = ckpt_data['num_epoch']
            batch_idx_to_resume = ckpt_data['batch_idx_to_resume']
            accumulated_steps = ckpt_data['accumulated_steps']

            if batch_idx_to_resume == 0:
                start_epoch += 1
                
            print(f"♻️ Resuming training from epoch {start_epoch}, step {global_step}, batch {batch_idx_to_resume}")

        # Create loaders
        train_loader_full = self.get_train_loader(self.train_dataset, self.train_batch_size, self.seed)
        eval_loader = self.get_eval_loader(self.eval_dataset, self.eval_batch_size, self.seed)

        # Keep the original number of batches for reporting & pbar
        original_num_batches = len(train_loader_full)

        if self.resume_training and batch_idx_to_resume > 0:
            total_samples = len(train_loader_full.dataset)
            start_sample = batch_idx_to_resume * train_loader_full.batch_size
            subset_indices = range(start_sample, total_samples)
            resumed_dataset = Subset(train_loader_full.dataset, subset_indices)

            effective_train_loader = DataLoader(
                resumed_dataset,
                batch_size=train_loader_full.batch_size,
                shuffle=False,
                num_workers=train_loader_full.num_workers
            )
        else:
            effective_train_loader = train_loader_full

        # Training steps and scheduler (based on original full size)
        steps_per_epoch = original_num_batches // self.grad_accumulation_step
        total_training_steps = self.max_steps if self.max_steps is not None else steps_per_epoch * num_epoch

        scheduler = self.get_scheduler(self.scheduler_type, total_training_steps, optimizer)

        # Print training information
        print(f"🧠 Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🏋️ Number of train samples: {len(self.train_dataset):,}")
        print(f"🧪 Number of eval samples: {len(self.eval_dataset):,}")
        print(f"🏃 Train steps per epoch (batches): {original_num_batches:,}")   # <-- fixed
        print(f"🧭 Eval steps per epoch (batches): {len(eval_loader):,}")
        print("\n","---"*15,'\n')

        # Main training loop
        for epoch in range(start_epoch, num_epoch + 1):
            model.train()
            epoch_loss = 0
            current_loss = 0

            # pbar uses full dataset count, starts at resume point
            pbar = tqdm(effective_train_loader, total=original_num_batches, desc=f"Epoch {epoch}/{num_epoch}",initial=batch_idx_to_resume)
            
            # Now continue training normally, pbar already advanced to the resume point
            for batch_idx, batch in enumerate(pbar, start=batch_idx_to_resume):
                if epoch == start_epoch and self.resume_training and batch_idx == batch_idx_to_resume:
                    batch_idx_to_resume = 0   # reset after resuming
                
                # Load batch data
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                output = model(inputs)
                
                # Calculate loss
                loss = criterion(
                    output.view(-1, self.vocab_size), 
                    targets.view(-1), 
                    ignore_index=-100
                )
                loss = loss / self.grad_accumulation_step
                loss.backward()
                
                # Store current loss for display
                current_loss = loss.item() * self.grad_accumulation_step
                epoch_loss += current_loss
                accumulated_steps += 1
                
                # Gradient accumulation and optimization step
                if accumulated_steps % self.grad_accumulation_step == 0:
                    # Gradient clipping for stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_steps = 0
                
                # Update progress bar with current loss anr lr
                pbar.set_postfix(
                    loss=f"{current_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}" if scheduler is not None else "n/a"
                )
                
                is_last_step = (self.max_steps is not None and global_step >= self.max_steps)
                
                # Step-based evaluation
                if (self.eval_per_step is not None and global_step+1 % self.eval_per_step == 0) or is_last_step:
                    avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
                    print(f"🎯 Eval loss: {avg_eval_loss:.4f}")

                # Check for max steps termination
                if is_last_step:
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir,
                        batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps, 
                        name=f'final_step_epoch_{epoch}_step_{global_step}'
                    )
                    return
                
                # Step-based checkpoint saving
                if (self.Save_step is not None and 
                    global_step > 0 and 
                    global_step % self.Save_step == 0):
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir,
                        batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
                        name=f'epoch_{epoch}_step_{global_step}'
                    )
            
            # Close the progress bar to ensure clean output
            pbar.close()
            
            # Handle remaining accumulated gradients at epoch end
            if accumulated_steps > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            is_last_epoch = (self.max_epoch is not None and (epoch) == self.max_epoch)
            
            # Epoch-based evaluation
            if (self.eval_per_epoch is not None and (epoch) % self.eval_per_epoch == 0) or is_last_epoch:
                avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
                print(f"❄️ Eval loss: {avg_eval_loss:.4f}")
            
            # Check for max epoch termination
            if is_last_epoch:
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir,
                    batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
                    name=f'final_model_epoch_{epoch}_step_{global_step}'
                )
                return
            
            # Print final epoch loss
            avg_epoch_loss = epoch_loss / original_num_batches
            print(f"🔥 Epoch loss: {avg_epoch_loss:.4f}")
            
            # Epoch-based checkpoint saving
            if (self.Save_epoch is not None and 
                (epoch + 1) % self.Save_epoch == 0):
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir,
                    batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
                    name=f'epoch_{epoch}_step_{global_step}'
                )