import threading

"""
Implementation of a thread-safe counter with many producers and many consumers.
"""
class Counter:
    def __init__(self, initial_count):
        self.count = initial_count
        self.cond = threading.Condition()

    def decrement(self):
        self.cond.acquire()
        self.count -= 1
        self.cond.notify_all()
        self.cond.release()

    def wait(self):
        self.cond.acquire()
        while self.count > 0:
            self.cond.wait()
        self.cond.release()