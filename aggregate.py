"""Conservative offline evidence aggregation, without joining unrelated track IDs."""
from collections import defaultdict
import numpy as np


def aggregate(records, fps, threshold=0.87, min_hits=3, max_gap_seconds=0.2):
    tracks = defaultdict(list)
    previous = {}
    segments = defaultdict(int)
    for record in records:
        for person in record['people']:
            tid = person['track_id']
            box = np.array(person['bbox'])
            if tid in previous:
                last_frame, old = previous[tid]
                distance = np.linalg.norm((box[:2]+box[2:]-old[:2]-old[2:])/2)
                diagonal = max(20, np.linalg.norm(old[2:]-old[:2]))
                if record['frame']-last_frame > max(1, round(fps*0.6)) or distance > diagonal:
                    segments[tid] += 1
            previous[tid] = record['frame'], box
            person['segment_id'] = f'{tid}.{segments[tid]}'
            person['staff'] = False
            person['interpolated'] = False
            tracks[person['segment_id']].append((record['frame'], person))
    summaries = []
    frame_lookup = {r['frame']:r for r in records}
    for segment, items in tracks.items():
        hits = [(f,p) for f,p in items if p.get('tag_score', 0) >= threshold]
        #require a cluster of evidence rather than isolated matches over a long track
        left = 0
        clustered_hits = 0
        for right, (frame, _) in enumerate(hits):
            while hits[left][0] < frame - round(2 * fps):
                left += 1
            clustered_hits = max(clustered_hits, right - left + 1)
        staff = clustered_hits >= min_hits
        peak = max(p.get('tag_score', 0) for _,p in items)
        summaries.append({'segment_id':segment, 'track_id':items[0][1]['track_id'],
                          'first_frame':items[0][0], 'last_frame':items[-1][0],
                          'observations':len(items), 'tag_hits':len(hits), 'max_hits_in_two_seconds':clustered_hits, 'peak_tag_score':peak, 'staff':staff})
        for _, person in items:
            person['staff'] = staff
            person['track_tag_score'] = peak
        if not staff:
            continue
        for (left, a), (right, b) in zip(items, items[1:]):
            gap = right-left-1
            if not 0 < gap <= round(fps*max_gap_seconds):
                continue
            for index in range(left+1,right):
                alpha=(index-left)/(right-left)
                box=((1-alpha)*np.array(a['bbox'])+alpha*np.array(b['bbox'])).tolist()
                frame_lookup[index]['people'].append({'track_id':a['track_id'],'segment_id':segment,
                    'bbox':box,'person_score':None,'tag_score':0.0,'tag_bbox':None,
                    'staff':True,'interpolated':True,'track_tag_score':peak})
    return summaries
