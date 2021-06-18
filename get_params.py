import torch


def get_params(model_path):
    params = torch.load(model_path)['model']
    num_params = 0
    for wts in params.values():
        num_params += torch.numel(wts)
    
    return round(num_params/(10**6),2)

if __name__=='__main__':
    model_paths = [
    '/home/tanmayg/Data/gpv-1-output/gpv_single_head/ckpts/model_22.pth',
    '/home/tanmayg/Data/gpv-1-output/gpv_multi_head/ckpts/model_22.pth'
    ]

    for model_path in model_paths:
        print(model_path)
        print(get_params(model_path))
