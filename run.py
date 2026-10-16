"""Detect people, track identities and aggregate name-tag evidence in a video."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from pathlib import Path
import cv2
import pandas as pd
from aggregate import aggregate
from evaluate import evaluate
from visualize import annotate, figures, render_video

ROOT=Path(__file__).resolve().parent


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=ROOT/'sample.mp4')
    p.add_argument('--out',type=Path,default=ROOT/'outputs'/'sample')
    p.add_argument('--model',type=Path,default=ROOT/'models'/'yolo11s.pt')
    p.add_argument('--device',default='auto',help='auto, cpu, mps or cuda:0')
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--rotations',type=int,nargs='+',default=[0,90,180,270],choices=[0,90,180,270])
    p.add_argument('--motion-fallback',action=argparse.BooleanOptionalAction,default=True,
                   help='Supplement missing detections with motion proposals; requires a fixed camera')
    p.add_argument('--tag-every',type=int,default=2,help='Search for tags every N frames; people are tracked every frame')
    p.add_argument('--tag-threshold',type=float,default=.85)
    p.add_argument('--min-hits',type=int,default=2)
    p.add_argument('--max-gap',type=float,default=.2,help='Maximum same-track interpolation gap in seconds')
    p.add_argument('--templates',type=Path,default=ROOT/'assets')
    p.add_argument('--labels',type=Path,help='Optional reviewed frame,staff_present CSV')
    p.add_argument('--reuse-tracks',action='store_true',help='Reuse cache only if video, model, code and settings match')
    p.add_argument('--reuse-tags',action='store_true',help='Reuse validated tag scores when changing aggregation settings')
    p.add_argument('--people-only',action='store_true',help='Run person detection/tracking and save diagnostic video')
    p.add_argument('--preview',action='store_true',help='Show causal tag decisions during processing; Q closes preview')
    p.add_argument('--play',action='store_true',help='Play the final offline result while saving it')
    p.add_argument('--max-frames',type=int,help='Limit processing for a quick smoke test')
    return p


def collect_tracks(args, meta):
    from detect import PersonDetector, MotionProposals
    from track import PersonTracker
    source_hash=file_hash(args.video)
    code_hash=hashlib.sha256(''.join(file_hash(ROOT/p) for p in ['detect.py','track.py']).encode()).hexdigest()
    key={'video_sha256':source_hash,'model':str(args.model.resolve()),'imgsz':args.imgsz,
         'rotations':args.rotations,'motion_fallback':args.motion_fallback,'max_frames':args.max_frames,
         'code_sha256':code_hash}
    cache=args.out/'tracks-cache.json'
    if args.reuse_tracks and cache.exists() and args.model.exists():
        payload=json.loads(cache.read_text())
        key['model_sha256']=file_hash(args.model)
        if payload['key']!=key:raise ValueError('Track cache does not match this input/configuration. Omit --reuse-tracks to rebuild it.')
        return payload['records'],payload['inference_seconds'],True
    detector=PersonDetector(args.model,args.device,args.imgsz,args.rotations)
    tracker=PersonTracker(meta['fps'])
    motion=MotionProposals(meta['fps']) if args.motion_fallback else None
    cap=cv2.VideoCapture(str(args.video));records=[];start=time.perf_counter()
    try:
        while args.max_frames is None or len(records)<args.max_frames:
            ok,frame=cap.read()
            if not ok:break
            detections=detector(frame)
            if motion is not None:detections=motion.supplement(frame,detections)
            tracked=tracker.update(detections,frame)
            people=[{'bbox':[float(v) for v in row[:4]],'track_id':int(row[4]),'person_score':float(row[5]),
                     'tag_score':0.0,'tag_bbox':None} for row in tracked]
            index=len(records)
            records.append({'frame':index,'timestamp':index/meta['fps'],'people':people})
            if index%50==0:print(f'Tracking {index}/{meta["reported_frames"]}: {len(people)} tracks',flush=True)
    finally:cap.release()
    if not records:raise ValueError('No frames decoded')
    expected=min(meta['reported_frames'],args.max_frames or meta['reported_frames'])
    if expected>0 and len(records)!=expected:raise ValueError(f'Decoded {len(records)} frames; expected {expected}. Check video integrity.')
    elapsed=time.perf_counter()-start
    key['model_sha256']=file_hash(args.model)
    cache.write_text(json.dumps({'key':key,'inference_seconds':elapsed,'records':records}))
    return records,elapsed,False


def tag_records(args, records):
    from tag import TagDetector
    detector=TagDetector(args.templates)
    cap=cv2.VideoCapture(str(args.video));hits=defaultdict(deque);confirmed=set();show=args.preview;start=time.perf_counter()
    fps=cap.get(cv2.CAP_PROP_FPS)
    previous={}
    pool=ThreadPoolExecutor(max_workers=4)
    try:
        for record in records:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Unexpected video end during tag matching')
            matches = list(pool.map(lambda person: detector(frame, person['bbox']), record['people'])) if record['frame']%args.tag_every==0 else []
            for person_index, person in enumerate(record['people']):
                tid=person['track_id']; index=record['frame']
                if tid in previous:
                    last,old=previous[tid]
                    box=person['bbox']
                    distance=((box[0]+box[2]-old[0]-old[2])**2+(box[1]+box[3]-old[1]-old[3])**2)**.5/2
                    diagonal=max(20,((old[2]-old[0])**2+(old[3]-old[1])**2)**.5)
                    if index-last>round(fps*.6) or distance>diagonal:
                        hits[tid].clear();confirmed.discard(tid)
                previous[tid]=(index,person['bbox'])
                while hits[tid] and hits[tid][0]<index-round(2*fps):hits[tid].popleft()
                if matches:
                    score,box=matches[person_index]
                    person['tag_score']=score
                    person['tag_bbox']=[int(v) for v in box] if box is not None else None
                    if score>=args.tag_threshold:hits[tid].append(index)
                if len(hits[tid])>=args.min_hits:confirmed.add(tid)
                person['staff']=tid in confirmed
            if show:
                cv2.imshow('FootFallCam - processing preview (Q closes preview)',annotate(frame,record,args.tag_threshold,live=True))
                if cv2.waitKey(1)&0xff==ord('q'):show=False;cv2.destroyAllWindows()
            if record['frame']%100==0:print(f'Tag matching {record["frame"]}/{len(records)}',flush=True)
    finally:
        cap.release()
        pool.shutdown()
        if show:cv2.destroyAllWindows()
    return time.perf_counter()-start


def save_tables(records, out):
    frames=[];people=[]
    for record in records:
        staff=[p for p in record['people'] if p['staff']]
        primary=max(staff,key=lambda p:p['track_tag_score']) if staff else None
        b=primary['bbox'] if primary else None
        frames.append({'frame':record['frame'],'timestamp':record['timestamp'],
            'staff_present':int(bool(staff)),'staff_count':len(staff),
            'x':(b[0]+b[2])/2 if b else None,'y':(b[1]+b[3])/2 if b else None,
            'primary_track_id':primary['track_id'] if primary else None})
        for p in record['people']:
            b=p['bbox'];tb=p.get('tag_bbox') or [None]*4
            people.append({'frame':record['frame'],'timestamp':record['timestamp'],'track_id':p['track_id'],
                'segment_id':p['segment_id'],'staff':int(p['staff']), 'x':(b[0]+b[2])/2,'y':(b[1]+b[3])/2,
                'x1':b[0],'y1':b[1],'x2':b[2],'y2':b[3],'person_score':p['person_score'],
                'tag_score':p['tag_score'],'track_tag_score':p['track_tag_score'],'interpolated':int(p['interpolated']),
                'tag_x1':tb[0],'tag_y1':tb[1],'tag_x2':tb[2],'tag_y2':tb[3]})
    results=pd.DataFrame(frames);results.to_csv(out/'results.csv',index=False)
    columns=['frame','timestamp','track_id','segment_id','staff','x','y','x1','y1','x2','y2','person_score',
             'tag_score','track_tag_score','interpolated','tag_x1','tag_y1','tag_x2','tag_y2']
    pd.DataFrame(people,columns=columns).to_csv(out/'people.csv',index=False)
    return results


def main():
    cv2.setNumThreads(1)
    p=parser();args=p.parse_args()
    if not args.video.is_file():p.error(f'Video does not exist: {args.video}')
    if args.tag_every<1 or args.min_hits<1 or not 0<args.tag_threshold<=1 or args.max_gap<0:
        p.error('Invalid tag interval, threshold, hit count or gap')
    if args.max_frames is not None and args.max_frames<1:p.error('--max-frames must be positive')
    if args.imgsz<32:p.error('--imgsz must be at least 32')
    if args.labels and not args.labels.is_file():p.error('Label file does not exist')
    args.out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.video))
    meta={'fps':cap.get(cv2.CAP_PROP_FPS),'reported_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    if meta['fps']<=0 or meta['width']<=0:p.error('Cannot read video dimensions or frame rate')
    start=time.perf_counter()
    records,track_seconds,reused=collect_tracks(args,meta)
    tag_seconds=0
    if not args.people_only:
        tag_key={'tracks_sha256':file_hash(args.out/'tracks-cache.json'),
                 'tag_code_sha256':file_hash(ROOT/'tag.py'),'tag_every':args.tag_every,
                 'templates':{p.name:file_hash(p) for p in sorted(args.templates.glob('tag-*.png'))}}
        tag_cache=args.out/'tags-cache.json'
        if args.reuse_tags and tag_cache.exists():
            cached=json.loads(tag_cache.read_text())
            if cached['key']!=tag_key:raise ValueError('Tag cache does not match. Omit --reuse-tags to recompute.')
            records=cached['records'];tag_seconds=cached['tag_seconds']
        else:
            tag_seconds=tag_records(args,records)
            tag_cache.write_text(json.dumps({'key':tag_key,'tag_seconds':tag_seconds,'records':records}))
    summaries=aggregate(records,meta['fps'],args.tag_threshold,args.min_hits,args.max_gap)
    (args.out/'track-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    results=save_tables(records,args.out)
    metrics=evaluate(results,args.labels)
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Rendering annotated video and plots...',flush=True)
    background=render_video(args.video,records,args.out,args.tag_threshold,args.play)
    figures(results,records,background,args.out)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest={'video':str(args.video.resolve()),'video_sha256':file_hash(args.video),**meta,
        'processed_frames':len(records),'config':config,'tracking_seconds':track_seconds,
        'tag_seconds':tag_seconds,'inference_fps':len(records)/(track_seconds+tag_seconds),
        'wall_seconds':time.perf_counter()-start,'reused_tracks':reused,
        'staff_frames':int(results.staff_present.sum()),'staff_segments':sum(t['staff'] for t in summaries),
        'coordinate_definition':'Person bounding-box centre, original-image pixels; origin top-left; x right, y down.',
        'timing_assumption':'Constant frame rate; timestamps = zero-based frame / reported FPS.',
        'tag_score_definition':'Template correlation, not a calibrated probability.',
        'mode':'Offline aggregation can use later tag evidence to label earlier observations.',
        'templates':{p.name:file_hash(p) for p in sorted(args.templates.glob('tag-*.png'))}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['processed_frames','staff_frames','staff_segments','inference_fps','wall_seconds']},indent=2))
    print(f'Outputs: {args.out.resolve()}')


if __name__=='__main__':main()
