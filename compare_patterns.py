import cv2 as cv
import torch
from torchvision import models, transforms
from numpy import dot
from numpy.linalg import norm
import math
import numpy as np
import io

from candlesticks import get_upload_details

resnet = models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # ResNet expected size, this basically fixed my issues, i was resizing outside this function before and did not realise it was needed
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(img):
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(input_tensor)
        features = features.squeeze().numpy()
    return features

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

pattern_names = [["hammer", "bullish_marubozu", "bullish_engulfing", "tweezer_bottom", "tws", "mds"],["shooting_star", "bearish_marubozu", "bearish_engulfing", "tweezer_top", "tbc", "eds"]]
pattern_candlesticks = [4,1,2,2,3,3] # number of candlesticks in each pattern
pattern_emb = np.load("patterns_emb.npy")

def find_best_pattern(img):
    candlesticks_cnt, max_height = get_upload_details(img)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    (h, w, _) = img_rgb.shape

    best_score = -1
    best_coord = (0,0)
    best_pattern = ""

    # sliding window
    for i in range(len(pattern_names)):
        for j in range(len(pattern_names[i])):
            x_window = math.floor(w * pattern_candlesticks[j]/candlesticks_cnt) - 1 # candlestick width times n number of candlesticks in pattern
            y_window = max_height
            x_stride = math.floor(w/candlesticks_cnt) # overlap by 1 candlestick width
            y_stride = math.floor(h/8) # arbitrary

            heatmap = np.zeros(( (h-y_window)//y_stride+1, (w-x_window)//x_stride+1 ))

            current_pattern_name = pattern_names[i][j]
            current_pattern_emb = pattern_emb[i][j]

            for yi, y in enumerate(range(0, h - y_window, y_stride)):
                for xi, x in enumerate(range(0, w - x_window, x_stride)):
                    patch = img_rgb[y:y+y_window, x:x+x_window]
                    if patch.shape[0] < 7 or patch.shape[1] < 7:
                        continue  # skip tiny patches at edges
                    patch_emb = get_embedding(patch)
                    score = cosine_similarity(patch_emb, current_pattern_emb)
                    if score > best_score:
                        best_score = score
                        best_coord = (x, y)
                        best_pattern = current_pattern_name
                    heatmap[yi, xi] = score

    x, y = best_coord
    annotated = img.copy()
    # draw rectangle of highest sim
    cv.rectangle(annotated, (x, y), (x + x_window, y + y_window), (0, 255, 0), 2)

    _, buffer = cv.imencode(".jpg", annotated)
    img_bytes = io.BytesIO(buffer)

    return best_pattern, best_coord, float(best_score), img_bytes