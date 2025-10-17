import torch.nn.functional as F

def generate_gradcam(model, input_tensor, target_class):
    """
    Generate Grad-CAM heatmap for a given class
    
    Args:
        model: The model
        input_tensor: Preprocessed image tensor [1, 3, 224, 224]
        target_class: Index of class to visualize
        
    Returns:
        cam: Numpy array [224, 224] with values 0-1
    """
    features = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal features
        features = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
    
    target_layer = model.features[-1]
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # forward pass
    model.zero_grad()
    output = model(input_tensor)
    
    # and backward 
    target = output[0, target_class]
    target.backward()
    
    # Compute CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    # Resize to input size
    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Cleanup the garbage
    handle_forward.remove()
    handle_backward.remove()
    
    return cam.squeeze().cpu().detach().numpy()