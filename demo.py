import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
import cv2
from models.moran import MORAN

model_path = './demo.pth'
img_path = './demo/0.png'
alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

################# 1-load model #######################################
cuda_flag = False
if torch.cuda.is_available():
    cuda_flag = True
    MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=cuda_flag)
    MORAN = MORAN.cuda()
else:
    MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=cuda_flag)

print('loading pretrained model from %s' % model_path)
if cuda_flag:
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location='cpu')
MORAN_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") # remove `module.`
    MORAN_state_dict_rename[name] = v
MORAN.load_state_dict(MORAN_state_dict_rename)

for p in MORAN.parameters():
    p.requires_grad = False
MORAN.eval()
################# 2-load data #######################################
converter = utils.strLabelConverterForAttention(alphabet, ':')#得到所有待识别字符的类别编号
transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image) #读取灰度图像并将其转换成100*32(w,h), image:1x32x100

if cuda_flag:
    image = image.cuda()
image = image.view(1, *image.size()) # 1x1x32x100
image = Variable(image)
text = torch.LongTensor(1 * 5)
length = torch.IntTensor(1)
text = Variable(text)
length = Variable(length)

max_iter = 20
t, l = converter.encode('0'*max_iter) # 初始化文本内容和文本长度t=20*'0', l=20
utils.loadData(text, t) #将初始化的值赋值到text和l上
utils.loadData(length, l)

################# 3-模型输出 #######################################
output = MORAN(image, length, text, text, test=True, debug=True) #这里初始双向的结果

preds, preds_reverse = output[0] #双向结果
demo = output[1] #test debug阶段输出矫正的文本

_, preds = preds.max(1)
_, preds_reverse = preds_reverse.max(1)

sim_preds = converter.decode(preds.data, length.data) #将预测的文本概率转换成文本, jewelers$e$e$e$
sim_preds = sim_preds.strip().split('$')[0] #jewelers
sim_preds_reverse = converter.decode(preds_reverse.data, length.data) #srelewej$$$$$seej$$$
sim_preds_reverse = sim_preds_reverse.strip().split('$')[0]#srelewej

print('\nResult:\n' + 'Left to Right: ' + sim_preds + '\nRight to Left: ' + sim_preds_reverse + '\n\n')

cv2.imshow("demo", demo)
cv2.waitKey()
