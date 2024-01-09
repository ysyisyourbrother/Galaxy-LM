# TegrastatsMonitor

## 依赖：
需要支持一下package：
1. Linux中的`killall`指令
2. Jetson 设备需要支持`tegrastat`指令的运行
3. sudo权限

## 使用流程
1. 从init文件中import TegraMonitor，并在自己的项目中初始化这个类实例monitor
    ```python
    # 在自己的项目中import这个包
    from monitor import TegraMonitor
    # 实例化类；interval:多久采样一次
    monitor = TegraMonitor(interval=500)
    ```
2. 在要执行的代码前启动monitor
    ```python
    # 如果使用GPU，最好进行一次synchronize
    torch.cuda.synchronize()
    monitor.start_tegrastats()
    ```
3. 执行完成推理后结束monitor
    ```python
    # 如果使用GPU，最好进行一次synchronize
    torch.cuda.synchronize()
    monitor.stop_tegrastats()
    ```
4. 获取能耗
   ```python
   monitor.get_avg_power_consumption()
   ```
