from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    '''
    Hungarian algorithm or Munkres Assignment Algorithm
    '''
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_1, bb_2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]

    ## Example
    >>> bb_1 = np.array([[20, 30, 70, 80], [50, 60, 100, 120]])
    >>> bb_2 = np.array([[40, 50, 80, 90], [70, 80, 110, 130], [10, 20, 50, 60]])
    >>> iou_matrix = iou_batch(bb_1, bb_2)

    Here the resulting iou_matrix is a 2D array of shape (2, 3),
    where each element represents the IoU between a bounding box from bb_1 and a bounding box from bb_2.
    For example, iou_matrix[0, 1] represents the IoU between the first bounding box from bb_1 and the second bounding box from bb_2.

    The resulting iou_matrix for the given example would be:
    >>> array([[0.27272727, 0.16216216, 0.], [0., 0.16216216, 0.10526316]])

    Here's how the IoU values are computed for each pair of bounding boxes:

    - IoU between the first bounding box from bb_1 and the first bounding box from bb_2 is 0.27272727.
    - IoU between the first bounding box from bb_1 and the second bounding box from bb_2 is 0.16216216.
    - IoU between the first bounding box from bb_1 and the third bounding box from bb_2 is 0.0.
    - IoU between the second bounding box from bb_1 and the first bounding box from bb_2 is 0.0.
    - IoU between the second bounding box from bb_1 and the second bounding box from bb_2 is 0.16216216.
    - IoU between the second bounding box from bb_1 and the third bounding box from bb_2 is 0.10526316.
    """

    bb_1 = np.expand_dims(bb_1, 1)
    bb_2 = np.expand_dims(bb_2, 0)

    xx1 = np.maximum(bb_1[..., 0], bb_2[..., 0])
    yy1 = np.maximum(bb_1[..., 1], bb_2[..., 1])
    xx2 = np.minimum(bb_1[..., 2], bb_2[..., 2])
    yy2 = np.minimum(bb_1[..., 3], bb_2[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_1[..., 2] - bb_1[..., 0]) * (bb_1[..., 3] - bb_1[..., 1])
        + (bb_2[..., 2] - bb_2[..., 0]) * (bb_2[..., 3] - bb_2[..., 1])
        - wh
    )
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w//2.
    y = bbox[1] + h//2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    result = np.array([x[0]-w//2, x[1]-h//2, x[0]+w//2, x[1]+h//2])

    if (score == None):
        return result.reshape((1, 4))
    else:
        return result.reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hit_streak = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    The function has two outputs:
    matches: It contains the indices of the matched pairs between detections and trackers. Each row represents a matched pair of indices (detection index, tracker index). These indices indicate which detections are associated with which trackers based on some criteria (e.g., IoU threshold).
    unmatched_detections: It is an array that contains the indices of the detections that do not have any associated tracker.
    """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1: # after applying the threshold, if we get only one tracked bbox for each detection or nothing then that is clean mathing.
            matched_indices = np.stack(np.where(a), axis=1)
        else: # if there are conflict for matching, find best matches which maximizes the IOU.
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))


    unmatched_detections = []
    for d in range(len(detections)):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections)


class Sort(object):
    def __init__(self, max_age=8, min_hits=3, iou_threshold=0.3):
        """
        ### Params:
        - max_age:  It defines the maximum number of consecutive frames for which a tracker can remain unmatched (i.e., not associated with any detection) before it is considered as a lost track and removed.
        - min_hits: It determines the minimum number of consecutive detections required for a track to be considered valid and returned as an output.
        - iou_threshold: It is a threshold value used for matching detections to existing tracks.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # get predictions from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            if np.any(np.isnan(pos)):
                self.trackers.pop(t)
                continue

            trk[:] = [*pos, 0]

        matched, unmatched_dets = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        for i, trk in enumerate(self.trackers):
            d = trk.get_state()[0]
            
            if (trk.time_since_update == 0) and (trk.hit_streak >= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))

            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            return np.concatenate(ret)

        return np.empty((0, 5))


if __name__ == '__main__':
    dets = np.array([
        [0, 12, 10, 15, 0.8],
        [10, 10, 100, 100, 0.98]
    ])
    algo = Sort(max_age=50)
    algo.update(dets)
