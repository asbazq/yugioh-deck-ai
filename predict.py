import argparse
import time

import cv2
import torch

from data.preprocess.torch import detector_preprocessing
from utils.image_utils import make_square_shape
from models.torch import Detector
from db import ChromaDBConnection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", dest="image")
    parser.add_argument("-m", "--model-path", dest="model_path")
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()
    chroma_db = ChromaDBConnection()
    origin_img = cv2.imread(args.image)[:, :, ::-1]

    student_inputs, _, _ = make_square_shape(origin_img, 640)
    student_inputs = detector_preprocessing(student_inputs.copy())
    student_inputs = student_inputs[None, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_model = Detector()
    pre_model_dict = torch.load(args.model_path, map_location="cpu")
    student_model.load_state_dict(pre_model_dict)
    student_model.to(device)
    student_model.eval()

    with torch.no_grad():
        # if not torch.cuda.is_available():
        #     student_inputs = student_inputs.cuda()
        student_inputs = student_inputs.to(device)
        pred_det, pred_embed = student_model(student_inputs)
    input_shape = student_inputs.shape[2:4]
    det_results = student_model.postprocess(
        pred_det, pred_embed, input_shape, origin_img[None, :]
    )


    for i, det in enumerate(det_results):
        card_names = []
        card_ids = []
        for embed in det.embeds:
            res = chroma_db.search_by_embed(embed.tolist(), 5)[0]
            names = list(map(lambda x: x["name"], res))
            ids = list(map(lambda x: x["id"], res))
            card_names.append(names)
            card_ids.append(ids)
        det.names = card_names
        det.ids = card_ids
        det.save(f"res_{i}.png")
        print(card_names)
    print(time.time() - start)



if __name__ == "__main__":
    main()
