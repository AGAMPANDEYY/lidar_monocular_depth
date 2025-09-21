import argparse, os, cv2, numpy as np

def draw_points(vis, pts, radius=4):
    for i,(x,y) in enumerate(pts):
        cv2.circle(vis, (int(x),int(y)), radius, (0,0,255), -1)
        cv2.putText(vis, str(i+1), (int(x)+6,int(y)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

def auto_detect(img_path, cols, rows, out_csv, out_dbg):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (cols, rows))
    if not ret:
        print("[auto] Chessboard NOT detected. Try manual mode or adjust cols/rows.")
        return False

    # refine
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)  # (N,1,2)
    pts = corners.reshape(-1,2)  # row-major TL->BR

    # save
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    np.savetxt(out_csv, pts, fmt="%.4f", delimiter=",")
    print(f"[auto] Saved {pts.shape[0]} points to {out_csv}")

    # debug overlay
    vis = img.copy()
    cv2.drawChessboardCorners(vis, (cols,rows), corners, True)
    draw_points(vis, pts)
    cv2.imwrite(out_dbg, vis)
    print(f"[auto] Wrote debug: {out_dbg}")
    return True

def manual_pick(img_path, out_csv, out_dbg, num_points):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    pts = []
    win = "Click points (u=undo, c=clear, Enter=save, Esc=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))

    cv2.setMouseCallback(win, on_click)

    while True:
        vis = img.copy()
        draw_points(vis, pts)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('u') and pts:
            pts.pop()
        elif key == ord('c'):
            pts = []
        elif key in (13, 10):  # Enter
            if num_points and len(pts) != num_points:
                print(f"[manual] Need {num_points} points, you have {len(pts)}.")
                continue
            break
        elif key == 27:  # Esc
            cv2.destroyWindow(win)
            print("[manual] Cancelled.")
            return False

    cv2.destroyWindow(win)
    pts_np = np.array(pts, dtype=np.float32)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    np.savetxt(out_csv, pts_np, fmt="%.2f", delimiter=",")
    print(f"[manual] Saved {pts_np.shape[0]} points to {out_csv}")

    vis = img.copy()
    draw_points(vis, pts_np)
    cv2.imwrite(out_dbg, vis)
    print(f"[manual] Wrote debug: {out_dbg}")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path to checkerboard image (front view)")
    ap.add_argument("--cols", type=int, help="inner corners across (for auto mode)")
    ap.add_argument("--rows", type=int, help="inner corners down (for auto mode)")
    ap.add_argument("--out_csv", default="calibration/image_points2d.csv")
    ap.add_argument("--manual", action="store_true", help="manual clicking mode")
    ap.add_argument("--num", type=int, default=None, help="expected number of clicks in manual mode")
    args = ap.parse_args()

    dbg = os.path.splitext(args.out_csv)[0] + "_debug.png"

    if args.manual:
        ok = manual_pick(args.img, args.out_csv, dbg, args.num)
        if not ok:
            exit(1)
    else:
        if not args.cols or not args.rows:
            raise SystemExit("For auto mode, pass --cols and --rows (inner corners).")
        ok = auto_detect(args.img, args.cols, args.rows, args.out_csv, dbg)
        if not ok:
            print("Tip: run in manual mode:\n"
                  f"python calibration/create_image_points_csv.py --img {args.img} --manual --num 20 --out_csv {args.out_csv}")
            exit(1)
