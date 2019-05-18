from sklearn.externals import joblib
from pathlib import Path
from src.torch_model import Net
from src.data_processing import *

def load_model(model_name):

    net = Net()
    filename = f'{Path(__file__).parent.parent}/models/{model_name}.sav'
    net.load_state_dict(torch.load(filename))
    net.eval()

    return net


net = load_model('first_')
print(net.eval())

char_list = generate_char_wip()


with torch.no_grad():
    for img in char_list:
        plot_image(img)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(torch.float32)
        output = net(img)
        _, predicted = torch.max(output.data, 1)
        print(predicted)
