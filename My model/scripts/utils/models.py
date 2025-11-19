"""
models.py - PRODUCTION VERSION with CMGAN Audio Processing & PESQ Evaluation
==============================================================================
Complete integration with CMGAN-style audio processing and PESQ-based model selection.

FEATURES:
1. CMGAN-style audio processing (RMS norm, power compression, n_fft=320, hop=160)
2. PESQ evaluation during training (every 5 epochs)
3. Dual model selection: best.pt (loss) and best_pesq.pt (PESQ)
4. Clean, professional logging (standard research code style)
5. Progress tracking with minimal console output
"""

import os
import shutil
import timeit
import numpy as np
import soundfile as sf
import torch
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

# Import PESQ library (optional)
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("? Warning: PESQ library not installed. Install with: pip install pesq")

from configs import (
    exp_conf, train_conf, test_conf,
    TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR,
    VALID_CLEAN_DIR, VALID_NOISY_DIR,
    TEST_CLEAN_DIR, TEST_NOISY_DIR,
    validate_data_dirs,
    get_test_snr_dirs
)
from utils.utils import getLogger, numParams, countFrames, lossMask, wavNormalize
from utils.pipeline_modules import NetFeeder, Resynthesizer
from utils.data_utils import create_dataloaders, create_test_dataloader_only
from utils.networks import Net
from utils.criteria import LossFunction


class CheckPoint(object):
    """Checkpoint management with scheduler state support"""

    def __init__(self, ckpt_info=None, net_state_dict=None,
                 optim_state_dict=None, scheduler_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict
        self.scheduler_state_dict = scheduler_state_dict

    def save(self, filename, is_best, best_model=None):
        """Save checkpoint to file"""
        torch.save(self, filename)
        if is_best and best_model:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        """Load checkpoint from file"""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'No checkpoint found at {filename}')
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict
        self.scheduler_state_dict = getattr(ckpt, 'scheduler_state_dict', None)


def lossLog(log_file, ckpt, logging_period=None):
    """Write loss to CSV file"""
    ckpt_info = ckpt.ckpt_info

    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch,iter,tr_loss,cv_loss\n')

    with open(log_file, 'a') as f:
        f.write('{},{},{:.4f},{:.4f}\n'.format(
            ckpt_info['cur_epoch'] + 1,
            ckpt_info['cur_iter'] + 1,
            ckpt_info['tr_loss'],
            ckpt_info['cv_loss']
        ))


def pesqLog(log_file, epoch, pesq_score):
    """Write PESQ score to CSV file"""
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch,pesq\n')

    with open(log_file, 'a') as f:
        f.write('{},{:.4f}\n'.format(epoch, pesq_score))


class Model(object):
    """
    Main model class with CMGAN-style audio processing and PESQ evaluation
    """

    def __init__(self):
        """Initialize model with CMGAN-style configuration"""
        # Audio parameters
        self.in_norm = exp_conf['in_norm']
        self.sample_rate = exp_conf['sample_rate']

        # CMGAN-style STFT parameters
        self.n_fft = exp_conf['n_fft']
        self.hop_length = exp_conf['hop_length']
        self.power = exp_conf['power_compression']

        # For backwards compatibility
        self.win_size = self.n_fft
        self.hop_size = self.hop_length

    def train(self):
        """Training procedure with CMGAN-style audio processing and PESQ evaluation"""

        # ============================================================
        # STEP 1: VALIDATE DATA DIRECTORIES
        # ============================================================
        print("\n" + "="*70)
        print("INITIALIZING TRAINING")
        print("="*70)

        try:
            validate_data_dirs(mode='train')
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? ERROR: {e}")
            print("Please check your configs.py paths!")
            return

        # ============================================================
        # STEP 2: LOAD CONFIGURATION
        # ============================================================

        # Training config
        self.ckpt_dir = train_conf['ckpt_dir']
        self.resume_model = train_conf['resume_model']
        self.time_log = train_conf['time_log']
        self.lr = train_conf['lr']
        self.plateau_factor = train_conf['plateau_factor']
        self.plateau_patience = train_conf['plateau_patience']
        self.plateau_threshold = train_conf['plateau_threshold']
        self.plateau_min_lr = train_conf['plateau_min_lr']
        self.clip_norm = train_conf['clip_norm']
        self.max_n_epochs = train_conf['max_n_epochs']
        self.early_stop_patience = train_conf['early_stop_patience']
        self.batch_size = train_conf['batch_size']
        self.num_workers = train_conf['num_workers']
        self.loss_log = train_conf['loss_log']
        self.unit = train_conf['unit']
        self.segment_size = train_conf['segment_size']
        self.segment_shift = train_conf['segment_shift']
        self.max_length_seconds = train_conf['max_length_seconds']

        # PESQ config
        self.pesq_eval_frequency = train_conf.get('pesq_eval_frequency', 5)
        self.pesq_num_samples = train_conf.get('pesq_num_samples', 100)
        self.save_best_pesq = train_conf.get('save_best_pesq', True)
        self.pesq_log = train_conf.get('pesq_log', 'pesq.csv')

        # Device setup
        self.gpu_ids = tuple(map(int, train_conf['gpu_ids'].split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ============================================================
        # STEP 3: SETUP LOGGER
        # ============================================================

        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)
        logger.info('='*70)
        logger.info('TRAINING - CMGAN-STYLE AUDIO PROCESSING + PESQ EVALUATION')
        logger.info('='*70)
        logger.info(f'Sample rate: {self.sample_rate} Hz')
        logger.info(f'STFT: n_fft={self.n_fft} ({self.n_fft/self.sample_rate*1000:.1f}ms), hop={self.hop_length} ({self.hop_length/self.sample_rate*1000:.2f}ms)')
        logger.info(f'Power compression: {self.power}')
        logger.info(f'Normalization: RMS (CMGAN-style)')
        logger.info(f'Processing mode: {self.unit}')
        if self.unit == 'seg':
            logger.info(f'Segment: size={self.segment_size}s, shift={self.segment_shift}s')
        logger.info(f'Batch size: {self.batch_size}')
        logger.info(f'Initial LR: {self.lr}, Min LR: {self.plateau_min_lr}')
        logger.info(f'Max epochs: {self.max_n_epochs}')
        logger.info(f'Device: {self.device}')
        if PESQ_AVAILABLE:
            logger.info(f'PESQ: Enabled (every {self.pesq_eval_frequency} epochs, {self.pesq_num_samples} samples)')
        else:
            logger.info('PESQ: Disabled (library not installed)')
        logger.info('='*70 + '\n')

        print(f"? Configuration loaded")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  PESQ: {'Enabled' if PESQ_AVAILABLE else 'Disabled'}")

        # ============================================================
        # STEP 4: CREATE DATALOADERS
        # ============================================================

        print("\n" + "-"*70)
        print("Loading datasets...")

        cache_dir = os.path.join(self.ckpt_dir, 'cache')

        try:
            train_loader, valid_loader, _ = create_dataloaders(
                train_clean_dir=TRAIN_CLEAN_DIR,
                train_noisy_dir=TRAIN_NOISY_DIR,
                valid_clean_dir=VALID_CLEAN_DIR,
                valid_noisy_dir=VALID_NOISY_DIR,
                test_clean_dir=TEST_CLEAN_DIR,
                test_noisy_dir=TEST_NOISY_DIR,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sample_rate=self.sample_rate,
                unit=self.unit,
                segment_size=self.segment_size,
                segment_shift=self.segment_shift,
                max_length_seconds=self.max_length_seconds,
                pin_memory=True,
                drop_last=True,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f'Failed to create dataloaders: {e}')
            print(f"? ERROR: {e}")
            raise

        self.logging_period = len(train_loader)

        logger.info(f'Train: {len(train_loader.dataset)} samples, {self.logging_period} iterations/epoch')
        logger.info(f'Valid: {len(valid_loader.dataset)} samples\n')

        print(f"? Datasets loaded")
        print(f"  Train: {len(train_loader.dataset)} samples ({self.logging_period} iters/epoch)")
        print(f"  Valid: {len(valid_loader.dataset)} samples")

        # ============================================================
        # STEP 5: INITIALIZE MODEL
        # ============================================================

        print("\n" + "-"*70)
        print("Initializing model...")

        net = Net()
        net = net.to(self.device)
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)
            logger.info(f'Using DataParallel with {len(self.gpu_ids)} GPUs')

        param_count = numParams(net)
        logger.info(f'Parameters: {param_count:,d} ({param_count*32/8/(2**20):.2f} MB)\n')

        print(f"? Model initialized ({param_count:,d} parameters)")

        # Pipeline modules
        feeder = NetFeeder(self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)
        resynthesizer = Resynthesizer(self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)
        criterion = LossFunction(device=self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)

        optimizer = Adam(net.parameters(), lr=self.lr, amsgrad=False)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.plateau_factor,
            patience=self.plateau_patience, threshold=self.plateau_threshold,
            threshold_mode='rel', cooldown=2, min_lr=self.plateau_min_lr, verbose=False
        )

        # ============================================================
        # STEP 6: CHECKPOINT INITIALIZATION
        # ============================================================

        ckpt_info = {
            'cur_epoch': 0,
            'cur_iter': 0,
            'tr_loss': None,
            'cv_loss': None,
            'best_loss': float('inf'),
            'best_pesq': -1.0,
            'global_step': 0,
            'min_lr_epoch_count': 0
        }
        global_step = 0
        min_lr_epoch_count = 0

        # Resume if needed
        if self.resume_model:
            print("\n" + "-"*70)
            print(f"Resuming from checkpoint: {self.resume_model}")

            logger.info('='*70)
            logger.info('RESUMING FROM CHECKPOINT')
            logger.info('='*70)
            logger.info(f'Loading: {self.resume_model}')

            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)

            # Load network
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.' + key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)

            # Load optimizer
            optim_state = ckpt.optim_state_dict
            for state in optim_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            optimizer.load_state_dict(optim_state)

            # Load info
            ckpt_info = ckpt.ckpt_info
            global_step = ckpt_info.get('global_step', 0)
            min_lr_epoch_count = ckpt_info.get('min_lr_epoch_count', 0)

            logger.info(f'Resumed from epoch {ckpt_info["cur_epoch"] + 1}')
            logger.info(f'Best loss: {ckpt_info["best_loss"]:.4f}')
            if 'best_pesq' in ckpt_info:
                logger.info(f'Best PESQ: {ckpt_info["best_pesq"]:.4f}')
            logger.info('='*70 + '\n')

            print(f"? Resumed from epoch {ckpt_info['cur_epoch'] + 1}")

        # ============================================================
        # STEP 7: TRAINING LOOP
        # ============================================================

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70 + "\n")

        logger.info('Starting training loop...\n')

        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            epoch_num = ckpt_info['cur_epoch'] + 1
            accu_tr_loss = 0.
            accu_n_frames = 0
            net.train()

            epoch_start_time = timeit.default_timer()

            # ==================== TRAINING PHASE ====================
            for n_iter, batch in enumerate(train_loader):
                global_step += 1

                # Get batch
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)

                # Forward pass
                feat, lbl, norm_factors = feeder(mix, sph)
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)

                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(feat, global_step=global_step)

                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)

                # Backward pass
                loss.backward()
                if self.clip_norm > 0.0:
                    clip_grad_norm_(net.parameters(), self.clip_norm)
                optimizer.step()

                # Accumulate loss
                running_loss = loss.data.item()
                accu_tr_loss += running_loss * n_frames_sum
                accu_n_frames += n_frames_sum

                # Console logging (every 50 iterations)
                if (n_iter + 1) % 50 == 0 or (n_iter + 1) == self.logging_period:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_tr_loss = accu_tr_loss / accu_n_frames
                    progress_pct = 100 * (n_iter + 1) / self.logging_period

                    print(f"Epoch [{epoch_num:3d}/{self.max_n_epochs}] "
                          f"[{n_iter+1:4d}/{self.logging_period}] ({progress_pct:5.1f}%) | "
                          f"Loss: {running_loss:.4f} (avg: {avg_tr_loss:.4f}) | "
                          f"LR: {current_lr:.6f}")

            # ==================== VALIDATION PHASE ====================

            avg_tr_loss = accu_tr_loss / accu_n_frames

            print("\n" + "-"*70)
            print(f"Validating epoch {epoch_num}...")

            avg_cv_loss = self.validate(net, valid_loader, criterion, feeder, global_step)
            net.train()

            # ==================== PESQ EVALUATION (OPTIONAL) ====================

            pesq_score = None
            if PESQ_AVAILABLE and epoch_num % self.pesq_eval_frequency == 0:
                print("-"*70)
                print(f"Evaluating PESQ (epoch {epoch_num})...")

                pesq_score = self.evaluate_pesq(
                    net, valid_loader, feeder, resynthesizer, global_step
                )

                if pesq_score is not None:
                    logger.info(f'PESQ Score: {pesq_score:.4f}')
                    pesqLog(os.path.join(self.ckpt_dir, self.pesq_log), epoch_num, pesq_score)
                    print(f"? PESQ: {pesq_score:.4f}")

            # ==================== CHECKPOINT MANAGEMENT ====================

            # Update checkpoint info
            ckpt_info['cur_iter'] = n_iter
            ckpt_info['global_step'] = global_step
            ckpt_info['tr_loss'] = avg_tr_loss
            ckpt_info['cv_loss'] = avg_cv_loss

            # Check for best loss
            is_best_loss = avg_cv_loss < ckpt_info['best_loss']
            if is_best_loss:
                ckpt_info['best_loss'] = avg_cv_loss
                min_lr_epoch_count = 0

            # Check for best PESQ
            is_best_pesq = False
            if pesq_score is not None:
                if pesq_score > ckpt_info['best_pesq']:
                    ckpt_info['best_pesq'] = pesq_score
                    is_best_pesq = True

            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_cv_loss)
            new_lr = optimizer.param_groups[0]['lr']

            lr_reduced = abs(old_lr - new_lr) > 1e-8

            # Early stopping logic
            if new_lr <= self.plateau_min_lr:
                if not is_best_loss:
                    min_lr_epoch_count += 1

                    if min_lr_epoch_count >= self.early_stop_patience:
                        logger.info('='*70)
                        logger.info('EARLY STOPPING')
                        logger.info(f'No improvement for {self.early_stop_patience} epochs at min LR')
                        logger.info(f'Best loss: {ckpt_info["best_loss"]:.4f}')
                        if PESQ_AVAILABLE:
                            logger.info(f'Best PESQ: {ckpt_info["best_pesq"]:.4f}')
                        logger.info('='*70)

                        self._save_checkpoint(ckpt_info, net, optimizer, scheduler,
                                            is_best_loss=False, is_best_pesq=False)
                        print(f"\n? Training stopped by early stopping")
                        print(f"  Best loss: {ckpt_info['best_loss']:.4f}")
                        if PESQ_AVAILABLE:
                            print(f"  Best PESQ: {ckpt_info['best_pesq']:.4f}")
                        return
            else:
                min_lr_epoch_count = 0

            ckpt_info['min_lr_epoch_count'] = min_lr_epoch_count

            # ==================== LOGGING & SAVING ====================

            epoch_time = timeit.default_timer() - epoch_start_time

            # Console summary
            print("-"*70)
            print(f"Epoch [{epoch_num:3d}/{self.max_n_epochs}] Summary:")
            print(f"  Train Loss:  {avg_tr_loss:.4f}")
            print(f"  Valid Loss:  {avg_cv_loss:.4f} {'? BEST' if is_best_loss else ''}")
            if pesq_score is not None:
                print(f"  PESQ:        {pesq_score:.4f} {'? BEST' if is_best_pesq else ''}")
            print(f"  LR:          {new_lr:.6f} {'? REDUCED' if lr_reduced else ''}")
            print(f"  Time:        {epoch_time:.1f}s")
            print("-"*70 + "\n")

            # Logger summary
            logger.info('='*70)
            logger.info(f'Epoch {epoch_num}/{self.max_n_epochs} Summary')
            logger.info('='*70)
            logger.info(f'Train Loss: {avg_tr_loss:.4f}')
            logger.info(f'Valid Loss: {avg_cv_loss:.4f} {"(BEST)" if is_best_loss else ""}')
            if pesq_score is not None:
                logger.info(f'PESQ: {pesq_score:.4f} {"(BEST)" if is_best_pesq else ""}')
            logger.info(f'LR: {new_lr:.6f} {"(REDUCED)" if lr_reduced else ""}')
            logger.info(f'Best Loss: {ckpt_info["best_loss"]:.4f}')
            if PESQ_AVAILABLE and ckpt_info['best_pesq'] > 0:
                logger.info(f'Best PESQ: {ckpt_info["best_pesq"]:.4f}')
            logger.info(f'Time: {epoch_time:.1f}s')
            logger.info('='*70 + '\n')

            # Save checkpoint
            self._save_checkpoint(ckpt_info, net, optimizer, scheduler,
                                is_best_loss=is_best_loss, is_best_pesq=is_best_pesq)

            # Write loss log
            ckpt = CheckPoint(ckpt_info, None, None, None)
            lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt)

            # Next epoch
            ckpt_info['cur_epoch'] += 1

        # ==================== TRAINING COMPLETED ====================

        print("="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total epochs: {ckpt_info['cur_epoch']}")
        print(f"Best loss: {ckpt_info['best_loss']:.4f}")
        if PESQ_AVAILABLE and ckpt_info['best_pesq'] > 0:
            print(f"Best PESQ: {ckpt_info['best_pesq']:.4f}")
        print("="*70 + "\n")

        logger.info('='*70)
        logger.info('TRAINING COMPLETED')
        logger.info(f'Total epochs: {ckpt_info["cur_epoch"]}')
        logger.info(f'Best loss: {ckpt_info["best_loss"]:.4f}')
        if PESQ_AVAILABLE and ckpt_info['best_pesq'] > 0:
            logger.info(f'Best PESQ: {ckpt_info["best_pesq"]:.4f}')
        logger.info('='*70)

        return

    def _save_checkpoint(self, ckpt_info, net, optimizer, scheduler,
                        is_best_loss=False, is_best_pesq=False):
        """Save checkpoint with dual model selection"""
        model_path = os.path.join(self.ckpt_dir, 'models')
        os.makedirs(model_path, exist_ok=True)

        if len(self.gpu_ids) > 1:
            ckpt = CheckPoint(ckpt_info, net.module.state_dict(),
                            optimizer.state_dict(), scheduler.state_dict())
        else:
            ckpt = CheckPoint(ckpt_info, net.state_dict(),
                            optimizer.state_dict(), scheduler.state_dict())

        # Save latest
        ckpt.save(os.path.join(model_path, 'latest.pt'), False, None)

        # Save best loss model
        if is_best_loss:
            shutil.copyfile(
                os.path.join(model_path, 'latest.pt'),
                os.path.join(model_path, 'best.pt')
            )

        # Save best PESQ model
        if is_best_pesq and self.save_best_pesq:
            shutil.copyfile(
                os.path.join(model_path, 'latest.pt'),
                os.path.join(model_path, 'best_pesq.pt')
            )

    def validate(self, net, cv_loader, criterion, feeder, global_step):
        """Validation procedure"""
        accu_cv_loss = 0.
        accu_n_frames = 0

        model = net.module if isinstance(net, DataParallel) else net
        model.eval()

        with torch.no_grad():
            for batch in cv_loader:
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                feat, lbl, norm_factors = feeder(mix, sph)
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)

                est = model(feat, global_step=global_step)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)

                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)

                accu_cv_loss += loss.data.item() * n_frames_sum
                accu_n_frames += n_frames_sum

        avg_cv_loss = accu_cv_loss / accu_n_frames
        return avg_cv_loss

    def evaluate_pesq(self, net, valid_loader, feeder, resynthesizer, global_step):
        """
        Evaluate PESQ on validation set

        Args:
            net: Network model
            valid_loader: Validation dataloader
            feeder: NetFeeder instance
            resynthesizer: Resynthesizer instance
            global_step: Current global step

        Returns:
            avg_pesq: Average PESQ score, or None if error
        """
        if not PESQ_AVAILABLE:
            return None

        model = net.module if isinstance(net, DataParallel) else net
        model.eval()

        pesq_scores = []
        samples_processed = 0

        with torch.no_grad():
            for batch in valid_loader:
                if samples_processed >= self.pesq_num_samples:
                    break

                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples']  # Keep on CPU for indexing

                # Forward pass
                feat, lbl, norm_factors = feeder(mix, sph)
                est = model(feat, global_step=global_step)

                # Resynthesize
                sph_est = resynthesizer(est, mix, norm_factors)

                # Compute PESQ for each sample in batch
                for i in range(mix.shape[0]):
                    if samples_processed >= self.pesq_num_samples:
                        break

                    # FIXED: Convert tensor to Python int before slicing
                    if isinstance(n_samples, torch.Tensor):
                        n_samples_i = n_samples[i].item()
                    else:
                        n_samples_i = n_samples[i]

                    # Get audio samples with correct length
                    ref_audio = sph[i, :n_samples_i].cpu().numpy()
                    est_audio = sph_est[i, :n_samples_i].cpu().numpy()

                    # Skip if audio is empty or too short
                    if len(ref_audio) < 100 or len(est_audio) < 100:
                        continue

                    # Normalize to [-1, 1]
                    ref_max = np.abs(ref_audio).max()
                    est_max = np.abs(est_audio).max()
                    
                    if ref_max > 1e-8:
                        ref_audio = ref_audio / ref_max
                    if est_max > 1e-8:
                        est_audio = est_audio / est_max

                    try:
                        # Compute PESQ (mode='wb' for wideband 16kHz)
                        score = pesq(self.sample_rate, ref_audio, est_audio, 'wb')
                        pesq_scores.append(score)
                        samples_processed += 1
                    except Exception as e:
                        # Skip this sample if PESQ computation fails
                        continue

        if len(pesq_scores) == 0:
            return None

        avg_pesq = np.mean(pesq_scores)
        return avg_pesq

    def test(self):
        """Testing procedure with auto-detection of SNR levels"""

        # ============================================================
        # STEP 1: DETECT SNR STRUCTURE
        # ============================================================

        print("\n" + "="*70)
        print("INITIALIZING TESTING")
        print("="*70)

        try:
            snr_dirs = get_test_snr_dirs()
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? ERROR: {e}")
            return

        print(f"\n? Detected {len(snr_dirs)} SNR levels:")
        for snr_val, snr_dir in snr_dirs:
            num_files = len([f for f in os.listdir(snr_dir) if f.endswith('.wav')])
            print(f"   SNR {snr_val:+3d} dB: {num_files} files")

        # ============================================================
        # STEP 2: LOAD CONFIGURATION
        # ============================================================

        self.model_file = test_conf['model_file']
        self.ckpt_dir = train_conf['ckpt_dir']
        self.est_path = train_conf['est_path']
        self.write_ideal = test_conf['write_ideal']

        self.gpu_ids = tuple(map(int, train_conf['gpu_ids'].split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.est_path, exist_ok=True)

        # ============================================================
        # STEP 3: SETUP LOGGER
        # ============================================================

        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)
        logger.info('='*70)
        logger.info('TESTING - CMGAN-STYLE AUDIO PROCESSING')
        logger.info('='*70)
        logger.info(f'Model: {self.model_file}')
        logger.info(f'SNR levels: {[snr for snr, _ in snr_dirs]}')
        logger.info(f'Output: {self.est_path}')
        logger.info(f'Device: {self.device}')
        logger.info('='*70 + '\n')

        # ============================================================
        # STEP 4: LOAD MODEL
        # ============================================================

        print("\n" + "-"*70)
        print("Loading model...")

        net = Net()
        net = net.to(self.device)

        param_count = numParams(net)
        logger.info(f'Parameters: {param_count:,d} ({param_count*32/8/(2**20):.2f} MB)\n')

        # Create utilities
        feeder = NetFeeder(self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)
        resynthesizer = Resynthesizer(self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)
        criterion = LossFunction(device=self.device, n_fft=self.n_fft, hop_length=self.hop_size, power=self.power)

        # Load checkpoint
        if not os.path.isfile(self.model_file):
            logger.error(f'Model not found: {self.model_file}')
            print(f"? ERROR: Model not found: {self.model_file}")
            return

        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)

        logger.info(f'Loaded: {self.model_file}')
        logger.info(f'Epoch: {ckpt.ckpt_info["cur_epoch"] + 1}, Loss: {ckpt.ckpt_info["best_loss"]:.4f}\n')

        print(f"? Model loaded (epoch {ckpt.ckpt_info['cur_epoch'] + 1})")

        net.eval()

        cache_dir = os.path.join(self.ckpt_dir, 'cache')

        # ============================================================
        # STEP 5: PROCESS ALL SNR LEVELS
        # ============================================================

        print("\n" + "="*70)
        print("PROCESSING TEST DATA")
        print("="*70 + "\n")

        overall_stats = {'snr_results': {}, 'total_processed': 0, 'total_errors': 0}

        for snr_idx, (snr_value, snr_noisy_dir) in enumerate(snr_dirs):
            print(f"SNR {snr_value:+3d} dB ({snr_idx + 1}/{len(snr_dirs)})...")

            logger.info('='*70)
            logger.info(f'SNR {snr_value:+3d} dB')
            logger.info('='*70)

            # Create output directory
            snr_output_dir = os.path.join(self.est_path, f'snr_{snr_value:+03d}')
            os.makedirs(snr_output_dir, exist_ok=True)

            # Create dataloader
            try:
                test_loader = create_test_dataloader_only(
                    test_clean_dir=TEST_CLEAN_DIR,
                    test_noisy_dir=snr_noisy_dir,
                    batch_size=test_conf['batch_size'],
                    num_workers=test_conf['num_workers'],
                    sample_rate=self.sample_rate,
                    unit='utt',
                    segment_size=6.0,
                    segment_shift=6.0,
                    max_length_seconds=6.0,
                    pin_memory=True,
                    cache_dir=cache_dir
                )
            except Exception as e:
                logger.error(f'Failed to create dataloader: {e}')
                print(f"  ? ERROR: {e}")
                continue

            logger.info(f'Samples: {len(test_loader)}')

            # Process
            accu_loss = 0.
            accu_n_frames = 0
            processed_count = 0
            error_count = 0

            with torch.no_grad():
                for k, batch in enumerate(test_loader):
                    try:
                        mix = batch['mix'].to(self.device)
                        sph = batch['sph'].to(self.device)
                        n_samples = batch['n_samples']
                        n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                        feat, lbl, norm_factors = feeder(mix, sph)
                        loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)

                        est = net(feat, global_step=None)
                        loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples, norm_factors)

                        if isinstance(n_frames, torch.Tensor):
                            n_frames_sum = n_frames.sum().item()
                        else:
                            n_frames_sum = sum(n_frames)

                        accu_loss += loss.data.item() * n_frames_sum
                        accu_n_frames += n_frames_sum

                        # Resynthesize
                        sph_idl = resynthesizer(lbl, mix, norm_factors)
                        sph_est = resynthesizer(est, mix, norm_factors)

                        # Convert to numpy
                        mix_np = mix[0].cpu().numpy()
                        sph_np = sph[0].cpu().numpy()
                        sph_est_np = sph_est[0].cpu().numpy()
                        sph_idl_np = sph_idl[0].cpu().numpy()

                        # Normalize
                        mix_np, sph_np, sph_est_np, sph_idl_np = wavNormalize(
                            mix_np, sph_np, sph_est_np, sph_idl_np
                        )

                        filename_base = os.path.splitext(batch['filenames'][0])[0]

                        # Save
                        sf.write(os.path.join(snr_output_dir, f'{filename_base}_mix.wav'),
                                mix_np, self.sample_rate)
                        sf.write(os.path.join(snr_output_dir, f'{filename_base}_sph.wav'),
                                sph_np, self.sample_rate)
                        sf.write(os.path.join(snr_output_dir, f'{filename_base}_sph_est.wav'),
                                sph_est_np, self.sample_rate)

                        if self.write_ideal:
                            sf.write(os.path.join(snr_output_dir, f'{filename_base}_sph_idl.wav'),
                                    sph_idl_np, self.sample_rate)

                        processed_count += 1

                        # Progress (every 50 samples)
                        if (k + 1) % 50 == 0:
                            print(f"  Progress: {k + 1}/{len(test_loader)} ({100*(k+1)/len(test_loader):.1f}%)")

                    except Exception as e:
                        error_count += 1
                        logger.error(f'Error processing {batch["filenames"][0]}: {e}')
                        continue

            # Report
            if accu_n_frames > 0:
                avg_loss = accu_loss / accu_n_frames

                overall_stats['snr_results'][snr_value] = {
                    'processed': processed_count,
                    'errors': error_count,
                    'avg_loss': avg_loss
                }
                overall_stats['total_processed'] += processed_count
                overall_stats['total_errors'] += error_count

                logger.info(f'Processed: {processed_count}, Errors: {error_count}, Loss: {avg_loss:.4f}')
                print(f"  ? Processed {processed_count} samples (Loss: {avg_loss:.4f})\n")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================

        print("="*70)
        print("TESTING COMPLETED")
        print("="*70)
        print(f"Total samples: {overall_stats['total_processed']}")
        print(f"Total errors: {overall_stats['total_errors']}")
        print(f"\nResults: {self.est_path}/")
        for snr_value in sorted(overall_stats['snr_results'].keys()):
            result = overall_stats['snr_results'][snr_value]
            print(f"  ? snr_{snr_value:+03d}/ - {result['processed']} samples (Loss: {result['avg_loss']:.4f})")
        print("="*70 + "\n")

        logger.info('='*70)
        logger.info('TESTING COMPLETED')
        logger.info(f'Total samples: {overall_stats["total_processed"]}')
        logger.info(f'Total errors: {overall_stats["total_errors"]}')
        logger.info('='*70)

        return


if __name__ == '__main__':
    import sys

    model = Model()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            model.train()
        elif sys.argv[1] == 'test':
            model.test()
        else:
            print(f'? Unknown command: {sys.argv[1]}')
            print('Usage: python models.py [train|test]')
    else:
        print('Usage: python models.py [train|test]')
