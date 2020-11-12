"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

import json
from collections import OrderedDict
from itertools import combinations

import cv2
import os
from libs.draw import Draw
from libs.geodist import social_distance, get_crop
from libs.geometric import get_polygon, get_point, get_line
from libs.person_trackers import PersonTrackers, TrackableObject
from libs.validate import validate
from openvino.inference_engine import IENetwork, IECore


class SocialDistance(object):
    def __init__(self):
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file_path) as f:
            cfg = json.load(f)
            validate(cfg)
        self.running = True
        self.videosource = cfg.get("video")
        self.model_modelfile = cfg.get("pedestrian_model_weights")
        self.model_configfile = cfg.get("pedestrian_model_description")
        self.model_modelfile_reid = cfg.get("reidentification_model_weights")
        self.model_configfile_reid = cfg.get("reidentification_model_description")
        self.coords = cfg.get("coords")
        # OPENVINO VARS
        self.ov_input_blob = None
        self.out_blob = None
        self.net = None
        self.ov_n = None
        self.ov_c = None
        self.ov_h = None
        self.ov_w = None
        self.ov_input_blob_reid = None
        self.out_blob_reid = None
        self.net_reid = None
        self.ov_n_reid = None
        self.ov_c_reid = None
        self.ov_h_reid = None
        self.ov_w_reid = None
        # PROCESSOR VARS
        self.confidence_threshold = .85
        self.iterations = 4  # ~ 5 feets
        self.trackers = []
        self.max_disappeared = 90
        self.polygon = None
        self.trackers = PersonTrackers(OrderedDict())
        self.min_w = 99999
        self.max_w = 1

    def load_openvino(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile, weights=self.model_modelfile)
            self.ov_input_blob = next(iter(net.inputs))
            self.out_blob = next(iter(net.outputs))
            self.net = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n, self.ov_c, self.ov_h, self.ov_w = net.inputs[self.ov_input_blob].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino error:{e}")
        self.load_openvino_reid()

    def load_openvino_reid(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile_reid, weights=self.model_modelfile_reid)
            self.ov_input_blob_reid = next(iter(net.inputs))
            self.out_blob_reid = next(iter(net.outputs))
            self.net_reid = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid = net.inputs[self.ov_input_blob_reid].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino reidentification error:{e}")

    def config_env(self, frame):
        h, w = frame.shape[:2]
        self.trackers.clear()

        polylist = []

        for pair in self.coords:
            polylist.append([int(pair[0] * w / 100), int(pair[1] * h / 100)])

        self.polygon = get_polygon(polylist)

    def get_frame(self):
        h = w = None
        try:
            cap = cv2.VideoCapture(self.videosource)
        except Exception as e:
            raise Exception(f"Video source error: {e}")

        while self.running:
            has_frame, frame = cap.read()
            if has_frame:
                if frame.shape[1] > 2000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .3), int(frame.shape[0] * .3)))

                elif frame.shape[1] > 1000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .8), int(frame.shape[0] * .8)))

                if w is None or h is None:
                    h, w = frame.shape[:2]
                    print(frame.shape)
                    self.config_env(frame)

                yield frame
            else:
                self.running = False
        return None

    def process_frame(self, frame):
        _frame = frame.copy()
        trackers = []

        frame = cv2.resize(frame, (self.ov_w, self.ov_h))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((self.ov_n, self.ov_c, self.ov_h, self.ov_w))

        self.net.start_async(request_id=0, inputs={self.ov_input_blob: frame})

        if self.net.requests[0].wait(-1) == 0:
            res = self.net.requests[0].outputs[self.out_blob]

            frame = _frame
            h, w = frame.shape[:2]
            out = res[0][0]
            for i, detection in enumerate(out):

                confidence = detection[2]
                if confidence > self.confidence_threshold and int(detection[1]) == 1:  # 1 => CLASS Person

                    xmin = int(detection[3] * w)
                    ymin = int(detection[4] * h)
                    xmax = int(detection[5] * w)
                    ymax = int(detection[6] * h)

                    if get_line([[xmin, ymax], [xmax, ymax]]).length < self.min_w:
                        self.min_w = get_line([[xmin, ymax], [xmax, ymax]]).length
                    elif get_line([[xmin, ymax], [xmax, ymax]]).length > self.max_w:
                        self.max_w = get_line([[xmin, ymax], [xmax, ymax]]).length

                    cX = int((xmin + xmax) / 2.0)
                    cY = int(ymax)
                    point = get_point([cX, cY])
                    if not self.polygon.contains(point):
                        continue

                    trackers.append(
                        TrackableObject((xmin, ymin, xmax, ymax), None, (cX, cY))
                    )
                    Draw.rectangle(frame, (xmin, ymin, xmax, ymax), "green", 2)

        for tracker in trackers:
            person = frame[tracker.bbox[1]:tracker.bbox[3], tracker.bbox[0]:tracker.bbox[2]]

            try:
                person = cv2.resize(person, (self.ov_w_reid, self.ov_h_reid))
            except cv2.error as e:
                print(f"CV2 RESIZE ERROR: {e}")
                continue

            person = person.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            person = person.reshape((self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid))

            self.net_reid.start_async(request_id=0, inputs={self.ov_input_blob: person})

            if self.net_reid.requests[0].wait(-1) == 0:
                res = self.net_reid.requests[0].outputs[self.out_blob_reid]
                tracker.reid = res

        self.trackers.similarity(trackers)
        if len(self.trackers.trackers) > 0:
            track_tuples = list(combinations(self.trackers.trackers.keys(), 2))
            for trackup in track_tuples:
                l1 = self.trackers.trackers[trackup[0]].bbox
                l2 = self.trackers.trackers[trackup[1]].bbox

                if l1[3] < l2[3]:
                    a = (l1[0], l1[3])
                    b = (l1[2], l1[3])
                    c = (l2[0], l2[3])
                    d = (l2[2], l2[3])
                else:
                    c = (l1[0], l1[3])
                    d = (l1[2], l1[3])
                    a = (l2[0], l2[3])
                    b = (l2[2], l2[3])

                h, w = frame.shape[:2]
                result = social_distance((h, w), a, b, c, d, self.iterations, self.min_w, self.max_w)
                if result["alert"]:
                    xmin, ymin, xmax, ymax = get_crop(l1, l2)
                    Draw.rectangle(frame, l1, "yellow", 2)
                    Draw.rectangle(frame, l2, "yellow", 2)
                    Draw.rectangle(frame, (xmin, ymin, xmax, ymax), "red", 3)
        return frame

    def render(self, frame):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()

    def run(self):
        self.load_openvino()
        for frame in self.get_frame():
            frame = self.process_frame(frame)
            self.render(frame)


if __name__ == '__main__':
    try:
        sd = SocialDistance()
        sd.run()
    except Exception as exception:
        print(exception)
