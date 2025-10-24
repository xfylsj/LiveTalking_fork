



# test_semaphore_leak.py
import multiprocessing as mp
import sys

def main():
    # 用 spawn 更稳定复现（Linux/Mac 均可）
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # 已设置过则忽略

    # 创建一堆同步原语（底层都是 SemLock/semaphore）
    num = 15
    leaks = []
    leaks += [mp.Event() for _ in range(num)]
    leaks += [mp.BoundedSemaphore(1) for _ in range(num)]
    leaks += [mp.Lock() for _ in range(num)]
    leaks += [mp.Queue() for _ in range(num)]  # Queue 也会建立 semaphores

    # 不做任何清理，直接退出
    print(f"Created {len(leaks)} objects; exiting without cleanup to trigger leak warning.")
    sys.exit(0)

if __name__ == "__main__":
    main()