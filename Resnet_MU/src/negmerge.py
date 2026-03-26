import torch
import copy
from torchvision import models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

original_state_dict = torch.load("resnet18_cifar10_original.pt", map_location=device)

# Calculate Task Vectors
def perform_negmerge(pool_path="./model_pool", num_variants=27, scaling_lambda=1.0):
    task_vectors = []
    
    print(f"Loading {num_variants} variants and calculating task vectors...")
    for i in range(1, num_variants + 1):
        variant_state_dict = torch.load(f"{pool_path}/variant_{i}.pt", map_location=device)
        
        # Task Vector = Fine-tuned Weights - Original Weights
        task_vector = {key: variant_state_dict[key] - original_state_dict[key] 
                       for key in original_state_dict.keys()}
        task_vectors.append(task_vector)

    # Compute Sign Consensus
    merged_task_vector = {}
    keys = original_state_dict.keys()
    
    print("Applying Sign-Consensus Mask...")
    for key in keys:
        # Stack all task vectors for this parameter: [27, ...param_shape]
        stacked = torch.stack([tv[key] for tv in task_vectors])
        
        # Get signs: +1, -1, or 0
        signs = torch.sign(stacked)
        
        # Consensus: Check if all 27 signs are identical to the first one
        # This acts as a filter for 'sign unanimity'
        is_consistent = (signs == signs[0]).all(dim=0)
        
        # Final merged value: Average of all vectors * the binary consistency mask
        avg_vector = stacked.mean(dim=0)
        merged_task_vector[key] = avg_vector * is_consistent.float()

    # Negation: theta_unlearn = theta_ori - (lambda * merged_task_vector)
    unlearned_state_dict = copy.deepcopy(original_state_dict)
    for key in keys:
        unlearned_state_dict[key] -= scaling_lambda * merged_task_vector[key]
        
    return unlearned_state_dict

# Execute and Save
unlearned_weights = perform_negmerge(scaling_lambda=1.0)
torch.save(unlearned_weights, "resnet18_cifar10_negmerge_final.pt")

print("NegMerge complete! Final unlearned model saved.")