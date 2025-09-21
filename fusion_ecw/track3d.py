import numpy as np

def pixel_to_cam(u,v,z,K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return np.array([X,Y,Z], dtype=np.float32)

class Track3DEstimator:
    def __init__(self):
        self.state = {}  # id -> dict(x,y,z,vx,vy,vz,t_last)

    def update(self, dets, Dfused, K, tstamp):
        out=[]
        for d in dets:
            tid = d["id"]; x1,y1,x2,y2 = map(int, d["bbox"])
            cx = (x1+x2)//2; cy = (y1+y2)//2
            # robust depth: median inside inner 40% patch
            w = int((x2-x1)*0.4); h=int((y2-y1)*0.4)
            xi1=max(cx-w//2,0); yi1=max(cy-h//2,0)
            xi2=min(cx+w//2, Dfused.shape[1]-1); yi2=min(cy+h//2, Dfused.shape[0]-1)
            z = np.median(Dfused[yi1:yi2+1, xi1:xi2+1])
            if not np.isfinite(z) or z<=0: continue
            X,Y,Z = pixel_to_cam(cx,cy,z,K)

            if tid in self.state:
                st = self.state[tid]
                dt = max(1e-3, tstamp - st["t"])
                vx, vy, vz = (X-st["x"])/dt, (Y-st["y"])/dt, (Z-st["z"])/dt
            else:
                vx=vy=vz=0.0

            self.state[tid] = {"x":X,"y":Y,"z":Z,"vx":vx,"vy":vy,"vz":vz,"t":tstamp}
            out.append({"id":tid,"x":X,"y":Y,"z":Z,"vx":vx,"vy":vy,"vz":vz})
        return out
