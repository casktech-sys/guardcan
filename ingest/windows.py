"""Time-window features for event-level CAN intrusion detection.

The per-frame detector (features.py) scores every frame on its own. Here the bus is cut into tumbling windows
and, per window, the frames of every CAN ID are counted:

  * counts[w, k]  frames of key k in window w. Keys are the IDs seen in normal training traffic; every other ID
                  shares one last "unknown ID" key.

Why 100 ms (WINDOW_S). CAN messages are mostly periodic, with periods of 10-100 ms on typical vehicles. On the
HCRL captures 17 of 27 IDs repeat every 10 ms, 4 every 20 ms, one every 50 ms, three every 100 ms and two every
1 s. A window must hold at least one period of an ID for its count to mean anything, so it has to be at least as
long as the longest common period: 100 ms. At 100 ms a 10 ms ID sends about 10 frames per window, and phase jitter
changes that by about one frame. Shorter windows hold one or two frames of the 20-100 ms IDs, so a single late
frame swings their count by 50-100 %. Longer windows add latency: a decision can come no earlier than the end of a
window. The 1 s IDs get 0 or 1 frame per window. That is still enough to see one flooded, but not to see one go
silent.

Windows tile each segment (one recording, or one part of it, e.g. its test period) from the segment's first
frame. The last window of a segment can be shorter; it can only hold fewer frames, never more.
"""
import numpy as np

WINDOW_S = 0.100


class WindowIndex:
    """Frames -> windows. seg[i] is the segment of frame i (any integer code); t must be sorted within a segment.

    Attributes, per window: seg, start, end (seconds); per frame: win (the global window index of the frame).
    Window indices are contiguous within a segment and in time order, so window w-1 precedes w in the same segment
    unless w is the segment's first window (first[w]).
    """

    def __init__(self, t, seg, width=WINDOW_S):
        t = np.asarray(t, dtype=np.float64)
        seg = np.asarray(seg)
        self.width = width
        codes, inv = np.unique(seg, return_inverse=True)
        t0 = np.full(len(codes), np.inf); t1 = np.full(len(codes), -np.inf)
        np.minimum.at(t0, inv, t); np.maximum.at(t1, inv, t)
        n = np.floor((t1 - t0) / width).astype(np.int64) + 1          # windows per segment, last one may be short
        offset = np.r_[0, np.cumsum(n)[:-1]]
        self.win = offset[inv] + np.floor((t - t0[inv]) / width).astype(np.int64)
        self.n_windows = int(n.sum())
        self.seg = np.repeat(codes, n)
        local = np.arange(self.n_windows) - np.repeat(offset, n)
        self.start = np.repeat(t0, n) + local * width
        self.end = np.minimum(self.start + width, np.repeat(t1, n))
        self.first = local == 0

    def count(self, key, n_keys, mask=None):
        """counts[w, k] = frames of key k in window w (int32), optionally only frames where mask is True."""
        w, k = (self.win, key) if mask is None else (self.win[mask], np.asarray(key)[mask])
        flat = np.bincount(w * n_keys + k, minlength=self.n_windows * n_keys)
        return flat.reshape(self.n_windows, n_keys).astype(np.int32)

    def any(self, flag):
        """Per window: does any frame of the window have flag set."""
        return np.bincount(self.win, weights=np.asarray(flag, dtype=np.float64), minlength=self.n_windows) > 0


def id_keys(can_id, known_ids):
    """CAN ID -> key: its index in the sorted known_ids, or len(known_ids) (the shared unknown-ID key)."""
    known_ids = np.asarray(known_ids)
    can_id = np.asarray(can_id)
    pos = np.searchsorted(known_ids, can_id)
    hit = (pos < len(known_ids)) & (known_ids[np.minimum(pos, len(known_ids) - 1)] == can_id)
    return np.where(hit, pos, len(known_ids)).astype(np.int64)
