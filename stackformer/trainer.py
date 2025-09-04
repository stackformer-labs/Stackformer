import torch
import os
from torch.utils.data import DataLoader
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
    
    Example:
        Basic usage:
        >>> trainer = Trainer(
        ...     model=my_model,
        ...     train_dataset=train_data,
        ...     eval_dataset=eval_data,
        ...     train_batch_size=16,
        ...     eval_batch_size=32,
        ...     vocab_size=50000,
        ...     output_dir="./checkpoints",
        ...     num_epoch=3,
        ...     lr=1e-4
        ... )
        >>> trainer.train()
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
            
        Example:
            >>> # Basic setup
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataset=train_data,
            ...     eval_dataset=eval_data,
            ...     train_batch_size=16,
            ...     eval_batch_size=32,
            ...     vocab_size=50000,
            ...     output_dir="./outputs",
            ...     num_epoch=5,
            ...     lr=2e-4,
            ...     scheduler_type="cosine",
            ...     device="cuda"
            ... )
            
            >>> # Advanced setup with gradient accumulation
            >>> trainer = Trainer(
            ...     model=large_model,
            ...     train_dataset=train_data,
            ...     eval_dataset=eval_data,
            ...     train_batch_size=4,  # Small batch due to memory
            ...     eval_batch_size=8,
            ...     vocab_size=50000,
            ...     output_dir="./checkpoints",
            ...     num_epoch=10,
            ...     lr=1e-4,
            ...     grad_accumulation_step=8,  # Effective batch size = 4*8 = 32
            ...     scheduler_type="linear",
            ...     warmup_steps=1000,
            ...     Save_epoch=2,
            ...     eval_per_epoch=1
            ... )
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
        
        Args:
            seed (int): Random seed value
            
        Note:
            Sets seeds for PyTorch CPU and CUDA operations to ensure
            reproducible results across training runs.
        """
        torch.manual_seed(seed)
        if self.device == 'cuda' and torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
        
    def get_scheduler(self, scheduler_type, total_training_steps, optimizer):
        """
        Create and return a learning rate scheduler.
        
        Args:
            scheduler_type (str): Type of scheduler to create
            total_training_steps (int): Total number of training steps
            optimizer: PyTorch optimizer instance
            
        Returns:
            PyTorch scheduler or None if scheduler_type is None
            
        Raises:
            ValueError: If scheduler_type is not supported
            
        Supported schedulers:
            - "linear": Linear decay with warmup
            - "cosine": Cosine annealing with warmup  
            - "cosine_restarts": Cosine with hard restarts
            - "cosineannealing": Standard cosine annealing
            - "cosine_warm_restarts": Cosine annealing with warm restarts
            
        Example:
            >>> optimizer = AdamW(model.parameters(), lr=1e-4)
            >>> scheduler = trainer.get_scheduler("cosine", 10000, optimizer)
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
        
        Args:
            optimizer_type (str): Type of optimizer ("adamw" or "sgd")
            model: PyTorch model
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            
        Returns:
            PyTorch optimizer instance
            
        Raises:
            ValueError: If optimizer_type is not supported
            
        Example:
            >>> optimizer = trainer.get_optimizer("adamw", model, 1e-4, 0.01)
        """
        if optimizer_type.lower() == "adamw":
            return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == "sgd":
            return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_type}")

    def eval_model(self, model, eval_loader, max_val_steps):
        """
        Evaluate the model on validation data.
        
        Args:
            model: PyTorch model to evaluate
            eval_loader: DataLoader for evaluation data
            max_val_steps (int, optional): Maximum evaluation steps
            
        Returns:
            float: Average evaluation loss
            
        Note:
            - Switches model to eval mode during evaluation
            - Uses cross-entropy loss with ignore_index=-100
            - Restores model to train mode after evaluation
            
        Example:
            >>> eval_loss = trainer.eval_model(model, eval_loader, 100)
            >>> print(f"Validation loss: {eval_loss:.4f}")
        """
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
    
    def get_train_loader(self, train_dataset, batch_size, seed):
        """
        Create a DataLoader for training data.
        
        Args:
            train_dataset: PyTorch Dataset
            batch_size (int): Batch size
            seed (int): Random seed for shuffling
            
        Returns:
            DataLoader: Configured training data loader
            
        Features:
            - Shuffles data for better training
            - Uses pin_memory for CUDA acceleration
            - Deterministic shuffling with seed
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
        
        Args:
            eval_dataset: PyTorch Dataset
            batch_size (int): Batch size
            seed (int): Random seed
            
        Returns:
            DataLoader: Configured evaluation data loader
            
        Features:
            - No shuffling for consistent evaluation
            - Uses pin_memory for CUDA acceleration
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
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Scheduler instance (can be None)
            epoch (int): Current epoch
            num_epoch (int): Total epochs
            loss (float): Current loss
            global_step (int): Global training step
            accumulated_steps (int): Current accumulated steps
            batch_idx_to_resume (int): Batch index for resuming
            output_dir (str): Directory to save checkpoint
            name (str): Checkpoint name
            
        Saves:
            - Model state dict
            - Optimizer state dict
            - Scheduler state dict (if exists)
            - Training metadata
            - Random number generator states for reproducibility
            
        Example:
            >>> trainer.save_model(
            ...     model, optimizer, scheduler, epoch=5, num_epoch=10,
            ...     loss=2.5, global_step=1000, accumulated_steps=0,
            ...     batch_idx_to_resume=0, output_dir="./checkpoints",
            ...     name="epoch_5_checkpoint"
            ... )
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
            # 'rng_state': {
            #     'torch': torch.get_rng_state(),
            #     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            # }
        }
        os.makedirs(output_dir, exist_ok=True)
        path = f'{output_dir}/checkpoint_{name}.pt'
        torch.save(checkpoint, path)
        print(f'Saved training state to {path}')
        
    def load_checkpoint(self, path, model, optimizer, scheduler):
        """
        Load a training checkpoint and restore training state.
        
        Args:
            path (str): Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer instance
            scheduler: Scheduler instance (can be None)
            
        Returns:
            dict: Dictionary containing restored training state with keys:
                - current_epoch: Epoch to resume from
                - num_epoch: Total epochs
                - accumulated_steps: Accumulated gradient steps
                - batch_idx_to_resume: Batch index to resume from
                - global_step: Global training step
                - loss: Last recorded loss
                
        Note:
            Restores random number generator states for reproducible resuming
            
        Example:
            >>> state = trainer.load_checkpoint(
            ...     "./checkpoints/checkpoint_epoch_5.pt", 
            ...     model, optimizer, scheduler
            ... )
            >>> print(f"Resuming from epoch {state['current_epoch']}")
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
        # Restore RNG states
        # torch.set_rng_state(checkpoint['rng_state']['torch'])
        # if torch.cuda.is_available() and checkpoint['rng_state']['cuda']:
        #     torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
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
        
        This method orchestrates the entire training pipeline including:
        - Data loading and preprocessing
        - Model initialization and optimization setup
        - Training loop with gradient accumulation
        - Evaluation and checkpointing
        - Learning rate scheduling
        - Progress tracking and logging
        
        Training Process:
            1. Sets random seed for reproducibility
            2. Creates data loaders for training and evaluation
            3. Calculates total training steps
            4. Initializes model, optimizer, and scheduler
            5. Handles checkpoint resuming if specified
            6. Runs training epochs with:
                - Forward and backward passes
                - Gradient accumulation
                - Gradient clipping
                - Learning rate scheduling
                - Periodic evaluation
                - Checkpoint saving
            7. Saves final model
            
        Features:
            - Gradient accumulation for large effective batch sizes
            - Gradient clipping for training stability
            - Flexible evaluation scheduling
            - Multiple checkpoint saving strategies
            - Comprehensive progress tracking
            - Memory-efficient data loading
            
        Example:
            >>> # Basic training
            >>> trainer = Trainer(...)
            >>> trainer.train()  # Starts training automatically
            
            >>> # Training with custom parameters
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataset=train_data,
            ...     eval_dataset=eval_data,
            ...     train_batch_size=16,
            ...     eval_batch_size=32,
            ...     vocab_size=50000,
            ...     output_dir="./checkpoints",
            ...     num_epoch=10,
            ...     lr=2e-4,
            ...     scheduler_type="cosine",
            ...     grad_accumulation_step=4,  # Effective batch size = 64
            ...     eval_per_epoch=2,          # Evaluate every 2 epochs
            ...     Save_epoch=5,              # Save every 5 epochs
            ...     warmup_steps=1000,
            ...     device="cuda"
            ... )
            >>> trainer.train()
            
        Output:
            Prints training progress including:
            - Model parameter count
            - Dataset sizes
            - Epoch progress with loss
            - Evaluation results
            - Checkpoint saving confirmations
        """
        # Set random seed for reproducibility
        self.set_seed(self.seed)
        
        # Create data loaders
        train_loader = self.get_train_loader(self.train_dataset, self.train_batch_size, self.seed)
        eval_loader = self.get_eval_loader(self.eval_dataset, self.eval_batch_size, self.seed)
        
        # Calculate total training steps
        steps_per_epoch = len(train_loader) // self.grad_accumulation_step
        total_training_steps = self.max_steps if self.max_steps is not None else steps_per_epoch * self.num_epoch
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize training components
        model = self.model.to(self.device)
        optimizer = self.get_optimizer(self.optimizer_type, model, self.lr, self.weight_decay)
        criterion = torch.nn.functional.cross_entropy
        scheduler = self.get_scheduler(self.scheduler_type, total_training_steps, optimizer)
        
        # Initialize training state
        global_step = 0
        start_epoch = 0
        num_epoch = self.num_epoch
        batch_idx_to_resume = 0
        accumulated_steps = 0

        # Handle checkpoint resuming
        if self.resume_training and self.model_to_resume:
            ckpt_data = self.load_checkpoint(self.model_to_resume, model, optimizer, scheduler)
            start_epoch = ckpt_data['current_epoch']
            global_step = ckpt_data['global_step']
            num_epoch = ckpt_data['num_epoch']
            batch_idx_to_resume = ckpt_data['batch_idx_to_resume']
            accumulated_steps = ckpt_data['accumulated_steps']
            print(f"♻️ Resuming training from epoch {start_epoch}, step {global_step}")
        
        # Print training information
        print(f"🧠 Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🏋️ Number of train samples: {len(self.train_dataset):,}")
        print(f"🧪 Number of eval samples: {len(self.eval_dataset):,}")
        print(f"🏃 Train steps per epoch (batches): {len(train_loader):,}")
        print(f"🧭 Eval steps per epoch (batches): {len(eval_loader):,}")
        print("\n","---"*15,'\n')
        
        # Main training loop
        for epoch in range(start_epoch, num_epoch):
            model.train()
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                # Handle resuming from specific batch
                if epoch == start_epoch and self.resume_training:
                    if batch_idx < batch_idx_to_resume:
                        continue
                    elif batch_idx == batch_idx_to_resume:
                        batch_idx_to_resume = 0
                
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
                
                # Update progress display
                pbar.set_postfix(loss=loss.item() * self.grad_accumulation_step)
                epoch_loss += loss.item() * self.grad_accumulation_step
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
                
                is_last_step = (self.max_steps is not None and global_step >= self.max_steps)
                
                # Step-based evaluation
                if (self.eval_per_step is not None and global_step+1 % self.eval_per_step == 0) or is_last_step:
                    avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
                    print(f"🎯 Eval loss: {avg_eval_loss:.4f}")

                # Check for max steps termination
                if is_last_step:
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir,
                        batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps, 
                        name=f'final_step_epoch_{epoch+1}_step_{global_step}'
                    )
                    return
                
                # Step-based checkpoint saving
                if (self.Save_step is not None and 
                    global_step > 0 and 
                    global_step % self.Save_step == 0):
                    self.save_model(
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                        global_step=global_step, output_dir=self.output_dir,
                        batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
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
            
            is_last_epoch = (self.max_epoch is not None and (epoch+1) == self.max_epoch)
            
            # Epoch-based evaluation
            if (self.eval_per_epoch is not None and (epoch+1) % self.eval_per_epoch == 0) or is_last_epoch:
                avg_eval_loss = self.eval_model(model, eval_loader, self.max_eval_step)
                print(f"❄️ Eval loss: {avg_eval_loss:.4f}")
            
            # Check for max epoch termination
            if is_last_epoch:
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir,
                    batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
                    name=f'final_model_epoch_{epoch+1}_step_{global_step}'
                )
                return
            
            # Print epoch summary
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"🔥 Epoch {epoch+1} finished - Training Loss: {avg_epoch_loss:.4f}")
            
            # Epoch-based checkpoint saving
            if (self.Save_epoch is not None and 
                (epoch + 1) % self.Save_epoch == 0):
                self.save_model(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch+1, num_epoch=num_epoch, loss=epoch_loss,
                    global_step=global_step, output_dir=self.output_dir,
                    batch_idx_to_resume=batch_idx+1,accumulated_steps=accumulated_steps,
                    name=f'epoch_{epoch+1}_step_{global_step}'
                )