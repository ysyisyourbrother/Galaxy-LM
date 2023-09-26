import subprocess
import threading

class Tegrastats:
    def __init__(self, interval, log_file):
        self.interval = interval
        self.log_file = log_file
        self.cond = threading.Condition()

    def start_tegrastats(self):
        # 需要sudo权限才可以获取power consumption
        # 可以编辑/etc/sudoers获得不用密码的sudo权限
        tegrastats_cmd = f"sudo tegrastats --interval {self.interval}"
        cmd = f"{{ echo $(date -u) & {tegrastats_cmd}; }} > {self.log_file}"
        self.process = subprocess.Popen(cmd, shell=True)

    
    def stop_tegrastats(self):
        # 关闭也需要sudo权限
        stop_cmd = f"sudo killall -9 tegrastats"
        self.process = subprocess.Popen(stop_cmd, shell=True)

    def run(self):
        self.start_tegrastats()

        self.cond.acquire()
        self.cond.wait()
        self.cond.release()

        self.stop_tegrastats()

