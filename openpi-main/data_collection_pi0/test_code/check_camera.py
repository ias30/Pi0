#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h5_cam_healthcheck.py
对 H5 录制文件中的三路相机做“体检”：
- 帧时间戳间隔 Δt 统计（中位数 / 95分位 / 零间隔比例）
- 近似 FPS（1/median(Δt)）
- 唯一时间戳数量
- 抽样重复帧比例（基于像素哈希）
- 连续重复帧最长长度（抽样窗口内）

用法:
  python h5_cam_healthcheck.py /path/to/episode.h5
可选:
  --sample 600       # 每路最多抽样帧数（默认600）
  --seed 42
  --plot             # 画Δt直方图和时间戳曲线
  --csv out.csv      # 输出汇总到CSV
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


CAM_GROUPS = [
    ("camera_high",        "High Camera"),
    ("camera_left_wrist",  "Left Wrist Camera"),
    ("camera_right_wrist", "Right Wrist Camera"),
]


def img_digest(img: np.ndarray) -> str:
    """
    低分辨率+SHA1：快速估算图像内容是否重复。
    下采样可以加快速度且对重复检测足够鲁棒。
    """
    if img.ndim == 3:
        small = img[::8, ::8, :].tobytes()
    else:
        small = img[::8, ::8].tobytes()
    return hashlib.sha1(small).hexdigest()


def sample_indices(n: int, k: int, seed: int = 0):
    """从 [0, n) 中均匀+随机混合抽样，不放回。"""
    if n <= 0:
        return []
    k = max(1, min(k, n))
    # 先等距，再微扰，保证覆盖头尾且均匀分布
    lin = np.linspace(0, n - 1, num=k, dtype=int)
    rng = random.Random(seed)
    # 小幅度扰动
    jitt = [min(n - 1, max(0, i + rng.randint(-1, 1))) for i in lin]
    return sorted(set(jitt))


def analyze_one_cam(f: h5py.File, cam_id: str, sample_cap: int = 600, seed: int = 0, do_plot: bool = False):
    """
    返回字典统计信息；如果相机组缺失，则返回 None。
    也可选择绘图（Δt 直方图、时间戳轨迹）。
    """
    grp = f.get(f"observations/cameras/{cam_id}")
    if grp is None:
        return None

    color = grp.get("color")
    ts_ds = grp.get("local_timestamps")
    if color is None or ts_ds is None:
        return None

    # 时间戳
    ts = np.asarray(ts_ds[:], dtype=np.float64)
    n = len(ts)
    if n == 0:
        return {
            "frames": 0,
            "unique_ts": 0,
            "fps_med": 0.0,
            "dt_med": np.nan,
            "dt_p95": np.nan,
            "dt_zero_ratio": 0.0,
            "dup_ratio": np.nan,
            "max_consec_dups": 0,
        }

    # Δt 统计
    dt = np.diff(ts)
    pos = dt[dt > 0]
    fps_med = (1.0 / np.median(pos)) if pos.size else 0.0
    dt_med = np.median(pos) if pos.size else np.nan
    dt_p95 = np.quantile(pos, 0.95) if pos.size else np.nan
    dt_zero_ratio = float((dt <= 0).mean()) if dt.size else 0.0

    # 抽样做重复帧估算
    k = min(sample_cap, n)
    idxs = sample_indices(n, k, seed=seed)
    digests = []
    max_consec = 1
    curr_consec = 1
    prev_hash = None

    for i in idxs:
        img = color[i]  # HDF5 懒加载
        h = img_digest(img)
        digests.append(h)
        if prev_hash is not None:
            if h == prev_hash:
                curr_consec += 1
                max_consec = max(max_consec, curr_consec)
            else:
                curr_consec = 1
        prev_hash = h

    uniq = len(set(digests))
    dup_ratio = 1.0 - (uniq / len(digests)) if digests else np.nan

    # 可选绘图
    if do_plot and HAS_PLT:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        # Δt 直方图
        if dt.size:
            axes[0].hist(dt[dt >= 0], bins=40)
            axes[0].set_title(f"{cam_id} Δt histogram (s)")
            axes[0].set_xlabel("Δt (s)")
            axes[0].set_ylabel("count")

        # 时间戳轨迹
        axes[1].plot(ts, ".", markersize=2)
        axes[1].set_title(f"{cam_id} timestamps")
        axes[1].set_xlabel("frame index")
        axes[1].set_ylabel("t (s)")
        plt.tight_layout()

    return {
        "frames": n,
        "unique_ts": int(np.unique(ts).size),
        "fps_med": float(fps_med),
        "dt_med": float(dt_med) if not np.isnan(dt_med) else np.nan,
        "dt_p95": float(dt_p95) if not np.isnan(dt_p95) else np.nan,
        "dt_zero_ratio": float(dt_zero_ratio),
        "dup_ratio": float(dup_ratio) if not np.isnan(dup_ratio) else np.nan,
        "max_consec_dups": int(max_consec),
    }


def main():
    ap = argparse.ArgumentParser(description="H5 三路相机体检：时间戳与重复帧统计")
    ap.add_argument("h5_file", type=str, help="HDF5 文件路径")
    ap.add_argument("--sample", type=int, default=600, help="每路最多抽样帧数用于重复帧估算（默认600）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（抽样）")
    ap.add_argument("--plot", action="store_true", help="绘制 Δt 直方图 与 时间戳轨迹（需要 matplotlib）")
    ap.add_argument("--csv", type=str, default="", help="将统计结果写入 CSV 文件")
    args = ap.parse_args()

    p = Path(args.h5_file)
    if not p.exists():
        print(f"❌ 文件不存在: {p}")
        return

    try:
        f = h5py.File(str(p), "r")
    except Exception as e:
        print(f"❌ 无法打开 H5: {e}")
        return

    # 若存在 global_timestamps，顺便打印长度，便于对比
    gt = f.get("observations/global_timestamps")
    if gt is not None:
        gts = np.asarray(gt[:], dtype=np.float64)
        print(f"Global timeline: frames={len(gts)}  span={gts[-1]-gts[0]:.3f}s" if len(gts) else "Global timeline: empty")
    else:
        print("⚠️ 无 observations/global_timestamps（不影响本脚本相机端统计）")

    print("\n=== Camera Health Check ===")
    rows = []
    for cam_id, label in CAM_GROUPS:
        stats = analyze_one_cam(
            f, cam_id,
            sample_cap=args.sample,
            seed=args.seed,
            do_plot=args.plot
        )
        if stats is None:
            print(f"{cam_id:20s}  (missing)")
            continue

        print(f"{cam_id:20s}  frames={stats['frames']:5d}  unique_ts={stats['unique_ts']:5d}  "
              f"fps~{stats['fps_med']:.2f}  dt_med={stats['dt_med']:.4f}s  p95={stats['dt_p95']:.4f}s  "
              f"Δt<=0比例={stats['dt_zero_ratio']:.2%}  抽样重复≈{stats['dup_ratio']:.2%}  "
              f"最长连续重复={stats['max_consec_dups']}")

        rows.append({
            "camera": cam_id,
            **stats
        })

    f.close()

    if args.plot and HAS_PLT:
        plt.show()

    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=[
                "camera", "frames", "unique_ts", "fps_med", "dt_med",
                "dt_p95", "dt_zero_ratio", "dup_ratio", "max_consec_dups"
            ])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n✅ CSV 已写出: {outp}")


if __name__ == "__main__":
    main()
