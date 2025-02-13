import os
import urllib.request

def download_lpips_weights():
    model_urls = {
        'vgg': 'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth?raw=true',
    }
    model_dir = 'C:\\Users\\andrew\\Anaconda3\\envs\\yolov8\\lib\\site-packages\\lpips\\weights\\v0.1'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for model_name, url in model_urls.items():
        model_path = os.path.join(model_dir, f'{model_name}.pth')
        if not os.path.exists(model_path):
            print(f'Downloading {model_name} weights...')
            urllib.request.urlretrieve(url, model_path)
            print(f'{model_name} weights downloaded and saved to {model_path}')
        else:
            print(f'{model_name} weights already exist at {model_path}')

download_lpips_weights()