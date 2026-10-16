"""Video annotation and static result figures."""
from collections import defaultdict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def annotate(frame, record, threshold, live=False):
    image = frame.copy()
    count=sum(p.get('staff',False) for p in record['people'])
    for person in record['people']:
        staff = person.get('staff',False)
        color = (70,220,70) if staff else (150,150,150)
        box=np.round(person['bbox']).astype(int)
        cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2 if staff else 1)
        label=f"{'STAFF' if staff else 'person'} #{person['track_id']}"
        if person.get('interpolated'):label+=' (estimated)'
        cv2.putText(image,label,(int(box[0]),max(40,int(box[1])-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        if person.get('tag_bbox') and person.get('tag_score',0)>=threshold:
            tag=np.array(person['tag_bbox'],dtype=int)
            cv2.rectangle(image,tuple(tag[:2]),tuple(tag[2:]),(0,220,255),2)
            cv2.putText(image,f"tag {person['tag_score']:.2f}",(int(tag[0]),max(40,int(tag[1])-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(image.shape[1],30),(25,25,25),-1)
    mode='CAUSAL PREVIEW' if live else 'OFFLINE RESULT'
    cv2.putText(image,f"{mode} | {record['timestamp']:.2f}s | frame {record['frame']} | staff: {count}",
                (10,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return image


def render_video(video, records, out, threshold, show=False):
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'avc1'),fps,(width,height))
    if not writer.isOpened():
        writer=cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        cap.release(); raise RuntimeError('No usable MP4 encoder available')
    first=None
    try:
        for record in records:
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f"Video ended before frame {record['frame']}")
            if first is None:first=frame.copy()
            image=annotate(frame,record,threshold)
            writer.write(image)
            if show:
                cv2.imshow('FootFallCam - offline result (Q closes preview)',image)
                if cv2.waitKey(max(1,round(1000/fps))) & 0xff == ord('q'):
                    show=False;cv2.destroyAllWindows()
    finally:
        cap.release();writer.release()
        if show:cv2.destroyAllWindows()
    return first


def figures(results, records, background, out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,2.4),layout='constrained')
    ax.step(results.timestamp,results.staff_present,where='post',color='#16864a')
    ax.fill_between(results.timestamp,results.staff_present,step='post',alpha=.18,color='#16864a')
    ax.set(xlabel='Time (seconds)',yticks=[0,1],yticklabels=['Absent','Present'],ylim=(-.1,1.2),title='Predicted staff presence')
    fig.savefig(out/'timeline.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7.5),layout='constrained')
    ax.imshow(cv2.cvtColor(background,cv2.COLOR_BGR2RGB),alpha=.6)
    paths=defaultdict(list)
    for record in records:
        for p in record['people']:
            if p.get('staff'):
                b=p['bbox'];paths[p['segment_id']].append((record['frame'],(b[0]+b[2])/2,(b[1]+b[3])/2))
    for key,path in paths.items():
        data=np.array(path); split=np.flatnonzero(np.diff(data[:,0])>1)+1
        for n,part in enumerate(np.split(data,split)):
            ax.plot(part[:,1],part[:,2],linewidth=1.5,label=f'Track {key}' if n==0 else None)
    if paths:ax.legend(loc='upper left',fontsize=7)
    ax.set(title='Staff paths: person-box centres in image pixels',xlabel='x (pixels)',ylabel='y (pixels)')
    fig.savefig(out/'trajectory.png',dpi=170);plt.close(fig)
