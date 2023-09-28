from email import parser
import threading

from .tegrastats import Tegrastats
from .parse import Parse

class TegraMonitor():
    def __init__(self, interval, tegra_monitor_log_file="output_log.txt"):
        # Logging interval in milliseconds for tegrastats
        self.interval = interval
        self.tegra_monitor_log_file = tegra_monitor_log_file
    
        self.tegrastats = Tegrastats(interval, tegra_monitor_log_file)
        self.parser = Parse(interval, tegra_monitor_log_file)

    def start_tegra_monitor(self):
        monitor_thread = threading.Thread(target=self.tegrastats.run)
        monitor_thread.start()
    
    def end_tegra_monitor(self):
        self.tegrastats.cond.acquire()
        self.tegrastats.cond.notify()
        self.tegrastats.cond.release()

    def get_avg_power_consumption(self):
        """ Get power consumption from tegra_monitor_log_file
        POM_5V_IN measures the overall 5V input to the board, it represents the total power consumption. 
        You don’t need to add POM_5V_GPU and POM_5V_CPU to it. 
        The GPU/CPU rails are provided for extra reference.
        """
        counter = 0
        total_time = 0
        total_power_consumption = 0
        with open(self.tegra_monitor_log_file, 'r') as log:
            data = log.readlines()
            # 遍历所有的data line
            for line in data[1:]:
                counter += 1
                total_time = total_time + self.interval
                lookup_table = self.parser.parse_data(line)

                for key in lookup_table.keys():
                    # Jetson Nano
                    if "Current POM" in key and "IN" in key:
                        total_power_consumption += lookup_table[key]
                    # Jetson TX2 & NX
                    elif "Current VDD" in key and "IN" in key:
                        total_power_consumption += lookup_table[key]

        return total_power_consumption/counter, total_time
