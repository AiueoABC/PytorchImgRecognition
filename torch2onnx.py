import torch.onnx
import time
import onnxruntime as ort
"""
Ref: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""


def convert2onnx(pth_model_path):
    model = torch.load(model_path, output_path)
    print(model)
    model.cpu()
    model.eval()

    # batch_size = 1  # In here, only one image is consumed
    # x = torch.randn(batch_size, 3, 600, 600, requires_grad=True)  # Generate dummy input image
    x = torch.randn(1, 3, 600, 600, requires_grad=True)  # Generate dummy input image

    # Export the model
    torch.onnx.export(model,                                       # model being run
                      x,                                           # model input (or a tuple for multiple inputs)
                      f'{output_path}/onnx_model.onnx',            # where to save the model (can be a file or file-like object)
                      export_params=True,                          # store the trained parameter weights inside the model file
                      opset_version=10,                            # the ONNX version to export the model to
                      do_constant_folding=True,                    # whether to execute constant folding for optimization
                      input_names=['input'],                       # the model's input names
                      output_names=['output'],                     # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})


def onnx_inference(onnx_model_path, pil_image):
    # Make onnxruntime session
    session = ort.InferenceSession(onnx_model_path)
    # Get inputs/outputs info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    output_name = outputs[0].name
    # prepare transform
    transform = transforms.Compose([
        transforms.Resize(input_shape.shape[2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    
    # pil_image => image_tensor => image_data (what onnx wants)
    image_tensor = transform_norm(pil_image)
    image_tensor = image_tensor.unsqueeze(0)
    image_data = image_tensor.to('cpu').detach().numpy().copy()
    # inference
    s = time.time()
    outputs_index = session.run([output_name], {input_name: image_data})
    print(f'InferenceTime: {time.time() - s} sec(s)')
    output = outputs_index[0]
    m = torch.nn.Softmax(dim=0)
    processed_output = m(torch.tensor(np.array(output[0]))).detach().numpy().copy()
    score_value = max(processed_output)
    index = np.argmax(processed_output)
    return index, score_value
