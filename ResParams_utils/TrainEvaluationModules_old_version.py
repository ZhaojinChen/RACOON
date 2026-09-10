import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ResParams_utils.data_utils_image_cropping_logic import plot_mri_orthoview_comparison,load_image_func
import psutil
import pynvml
import time
import torch.nn.functional as F
import pandas as pd

def prepare_batch(batch, device=None, non_blocking=False, dtype=None, renorm_scale=True,scale_factor=1.0):
    '''
    Here's the info we save in the data_utils (this is for futher analysis)
        return {
           #'image_raw': image_raw, 
           'image': image,
           'subject_id': subject_id,
           'age': age,
           'sex':sex,
           'lin_rate':lin_rate,
           'Affine_param':Affine_param   
        }'''
    x = batch['image'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    y = batch['Affine_param'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    if renorm_scale:
        y[:,3:6] = (1-y[:,3:6])*scale_factor
    #x_raw =batch['image_raw'].to(device=device, dtype=dtype, non_blocking=non_blocking)
    subject_id = batch['subject_id']
    age = batch['age']
    sex = batch['sex']
    lin_rate = batch['lin_rate']
    # we didn't assume any co-vars here
    return x, y, subject_id,age,sex,lin_rate,{}

def save_checkpoint(epoch,model,optimizer,train_loss, val_loss,is_best,output_dir,best_model_dir):
    state ={
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'train_loss':train_loss,
        'val_loss':val_loss,
    }

    # define the file name pattern
    FILENAME_PATTERN = (
        f"Epoch_{epoch}_"
        f"TrainLoss_{train_loss:.4f}_"
        f"ValLoss_{val_loss:.4f}_.pt.tar"
    )

    checkpoint_path = os.path.join(output_dir,'checkpoints',FILENAME_PATTERN)

    torch.save(state,checkpoint_path)

    if is_best:
        best_path = os.path.join(best_model_dir, FILENAME_PATTERN)
        torch.save(state, best_path)
        print(f"New best model saved to {best_path}")

def save_png_from_inputs(inputs, inputs_raw, labels,base_save_path,outline_path='/data/dadmah/chezha/mni_icbm152_t1_tal_nlin_sym_09c_outline.mnc'):
    #...save image in the path
    #load outline_data
    outline_image= load_image_func(outline_path)
    for idx, (input_data, input_raw_data, label_data) in enumerate(zip(inputs, inputs_raw, labels)):
        input_data_cpu = input_data.cpu().detach().squeeze()
        input_raw_data_cpu = input_raw_data.cpu().detach()
        label_val = label_data.cpu().item() if isinstance(label_data, torch.Tensor) else label_data
        
        try:
            plot_mri_orthoview_comparison(
                input_raw_data_cpu,   
                input_data_cpu,   
                outline_image,       
                label_val,          
                base_save_path         
            )
        except Exception as e:
            print(f"Error plotting sample {idx}: {e}")
            continue

GB = 1024 * 1024 * 1024

def get_gpu_memory_summary():
    """
    Retrieves the VRAM usage for all available GPUs using pynvml.
    Returns:
        list of dict: [{'index': 0, 'used_gb': 1.5, 'total_gb': 12.0, 'percent': 12.5}, ...]
    """
    gpu_memories = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_gb = info.total / GB
            used_gb = info.used / GB
            percent = (info.used / info.total) * 100 if info.total > 0 else 0.0
            
            gpu_memories.append({
                'index': i, 
                'used_gb': used_gb, 
                'total_gb': total_gb, 
                'percent': percent
            })
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        return [] 
    return gpu_memories


def log_combined_memory_status(message="Memory Check"):
    """
    Logs both System RAM and GPU VRAM usage in the format: used/total (percentage).
    """

    mem = psutil.virtual_memory()
    system_total_gb = mem.total / GB
    system_used_gb = mem.used / GB
    system_percent = mem.percent
    print(f"  -> System RAM: {system_used_gb:.2f} GB used / {system_total_gb:.2f} GB total ({system_percent:.1f}%)")
    
    gpu_memories = get_gpu_memory_summary()
    
    if gpu_memories:
        for gpu in gpu_memories:
            print(f"  -> GPU {gpu['index']} VRAM: {gpu['used_gb']:.2f} GB used / {gpu['total_gb']:.2f} GB total ({gpu['percent']:.1f}%)")
    else:
        print("  -> GPU VRAM: Monitoring N/A (pynvml error or CUDA not available)")

def get_grad_norm(model, prefix_tuple):
    """Calculates the L2 norm of the gradient for layers matching the given prefixes."""
    total_norm = 0.0
    found_layer = False
    
    # Iterate through named parameters of the model
    for name, param in model.named_parameters():
        # Check if the parameter name starts with one of the prefixes and has a gradient
        if param.grad is not None and name.startswith(prefix_tuple):
            # Calculate the L2 norm for this parameter's gradient
            norm = param.grad.data.norm(2)
            total_norm += norm.item() ** 2
            found_layer = True
    
    if found_layer:
        # Return the final L2 norm (square root of the sum of squared norms)
        return total_norm ** 0.5
    else:
        # Return NaN or a sentinel value if no matching layer was found
        return float('nan')
    
# training and validation epochs
def train_epoch(model,dataloader,criterion,optimizer,device,max_iterations=None, log_ram_every_n_batches=50,grad_head_prefix=("fnn_rmse.4",),scale_factor=1.0):
    model.train()
    total_loss = 0
    total_samples = 0
    
    train_bar = tqdm(dataloader, desc="Training", leave=False)
    #for batch in dataloader:
    for batch_idx, batch in enumerate(train_bar):
        inputs, Affine_params, _,_,_,_, _ = prepare_batch(batch,device=device,scale_factor=scale_factor)
        #Check min=inputs.min() max=inputs.max() inputs.mean() inputs.std().
        #Check whether the dataloader modified the image too much
        #if batch_idx < 10:
        #    save_png_from_inputs(inputs, inputs_raw, RMSE,base_save_path=f'/data/dadmah/chezha/Results/training_results/CNN/pretrain2/train_example_{batch_idx}.png')
        if (batch_idx + 1) % log_ram_every_n_batches == 0:
            log_combined_memory_status(f"Batch {batch_idx + 1}")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss=criterion(outputs,Affine_params)

        loss.backward()
        grad_norm = get_grad_norm(model,grad_head_prefix)
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']

        bs = Affine_params.size(0)
        total_loss+=loss.item() * bs
        total_samples += bs


        train_bar.set_postfix(
            loss=f"{loss.item():.4f}", 
            grad_norm=f"{grad_norm:.6f}", # Display the calculated gradient norm
            lr=f"{current_lr:.6f}"
        )

        if max_iterations is not None and (batch_idx + 1) >= max_iterations:
            break

    avg_loss = total_loss / total_samples if total_samples else 0.0

    return avg_loss, current_lr

def validation_epoch(model,dataloader,criterion,device,max_iterations=None,scale_factor=1.0):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
            inputs, Affine_params, _,_,_,_, _ = prepare_batch(batch,device=device,scale_factor=scale_factor)
            #Check inputs.min() inputs.max() inputs.mean() inputs.std()
            #if batch_idx < 10:
            #    save_png_from_inputs(inputs, inputs_raw, RMSE,base_save_path=f'/data/dadmah/chezha/Results/training_results/CNN/pretrain2/validation_example_{batch_idx}.png')
            outputs = model(inputs)
            loss = criterion(outputs, Affine_params)
            
            # Accumulate metrics
            bs = Affine_params.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break
    avg_loss = total_loss / total_samples if total_samples else 0.0

    return avg_loss

'''
def evaluate_dataset(model,dataloader,device,max_iterations=None):
    model.eval()
    all_preds = []
    all_true = []
    all_params = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluation", leave=False)):
            inputs, Affine_params, subject_id,age,sex,lin_rate, _ = prepare_batch(batch,device=device)
            outputs = model(inputs)
            predicted = outputs.data
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(RMSE.cpu().numpy())
            all_params.extend(params.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                #max_iteration is for test plot code
                break
    return np.array(all_preds),np.array(all_true),np.array(all_params),np.array(all_labels)  
'''

def evaluate_dataset(model, dataloader, device, max_iterations=None,scale_factor=1.0):
    model.eval()
    
    all_subject_ids = []
    all_ages = []
    all_sexes = []
    all_lin_rates = []
    all_true_affine = []
    all_pred_affine = []

    with torch.no_grad(): # This disables gradient tracking safely
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            inputs, affine_true, subject_id, age, sex, lin_rate, _ = prepare_batch(batch, device=device,scale_factor=scale_factor)
            
            # Inference
            outputs = model(inputs) 
            pred_np = outputs.detach().cpu().numpy()
            true_np = affine_true.detach().cpu().numpy()
            
            all_subject_ids.extend(subject_id)
            all_ages.extend(age.cpu().numpy() if torch.is_tensor(age) else age)
            all_sexes.extend(sex.cpu().numpy() if torch.is_tensor(sex) else sex)
            all_lin_rates.extend(lin_rate.cpu().numpy() if torch.is_tensor(lin_rate) else lin_rate)
            
            all_true_affine.append(true_np)
            all_pred_affine.append(pred_np)

            if max_iterations is not None and (batch_idx + 1) >= max_iterations:
                break

    return {
        "subject_id": all_subject_ids,
        "age": np.array(all_ages),
        "sex": np.array(all_sexes),
        "lin_rate": np.array(all_lin_rates),
        "true_affine": np.vstack(all_true_affine),
        "pred_affine": np.vstack(all_pred_affine)
    }

def save_results_to_csv(eval_results, output_path):
    """Formats the evaluation dictionary into a CSV file."""
    rows = []
    num_samples = len(eval_results["subject_id"])
    
    for i in range(num_samples):
        row = {
            'subject_id': eval_results["subject_id"][i],
            'age': eval_results["age"][i],
            'sex': eval_results["sex"][i],
            'lin_rate': eval_results["lin_rate"][i]
        }
        
        # Expand Predicted Affine Params (column per parameter)
        pred_params = eval_results["pred_affine"][i]
        for j, val in enumerate(pred_params):
            row[f'pred_param_{j}'] = val
            
        # Expand True Affine Params (column per parameter)
        true_params = eval_results["true_affine"][i]
        for j, val in enumerate(true_params):
            row[f'true_affine_{j}'] = val
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved results to: {output_path}")
    return df