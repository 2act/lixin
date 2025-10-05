from datetime import datetime


class _ColorLogger:
    _COLORS = {
        "debug": "\033[36m",  # 青色
        "info": "\033[32m",  # 绿色
        "warning": "\033[33m",  # 黄色
        "error": "\033[31m",  # 红色
        "critical": "\033[1;31m"  # 粗体红色
    }
    _RESET = "\033[0m"

    _enable_color = True  # 是否启用彩色输出，matlab命令行窗口不能启用

    def __init__(self):
        self._start_time = None

    def disable_color(self):
        self._enable_color = False

    def _print(self, *args, level="info", sep=" ", end="\n"):
        """打印带颜色和时间戳的日志，支持可变参数"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = self._COLORS.get(level, "")
        msg = sep.join(str(arg) for arg in args)
        if self._enable_color:
            print(f"{color}[{now}] {msg}{self._RESET}", end=end)
        else:
            print(f"[{now}] {msg}", end=end)

    def debug(self, *args):
        self._print(*args, level="debug")

    def info(self, *args):
        self._print(*args, level="info")

    def warning(self, *args):
        self._print(*args, level="warning")

    def error(self, *args):
        self._print(*args, level="error")

    def critical(self, *args):
        self._print(*args, level="critical")

    def start_timer(self):
        """启动计时器"""
        self._start_time = datetime.now()
        self.debug("计时器启动")
        return self._start_time

    def elapsed(self, t0=None, label="耗时"):
        """打印自启动以来的耗时"""
        if self._start_time is None and t0 is None:
            self.warning("未启动计时器，请先调用 log.start_timer()")
            return
        if t0:  # 传递了时间则用传递的时间，支持多个计时
            elapsed_time = datetime.now() - t0
        else:
            elapsed_time = datetime.now() - self._start_time
        self.debug(f"{label}: {elapsed_time.total_seconds():.4f} 秒")
        return elapsed_time.total_seconds()


# 单例导出
log = _ColorLogger()
