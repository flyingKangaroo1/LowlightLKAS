# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
#

from . import nms_impl

def nms(boxes, scores, overlap, top_k):
    print(f"Boxes device: {boxes.device}, Scores device: {scores.device}", overlap, top_k)
    return nms_impl.nms_forward(boxes, scores, overlap, top_k)
    # return nms_forward(boxes, scores, overlap, top_k)
