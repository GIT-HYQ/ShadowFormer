import os
import datetime
import sys

class Logger:
    def __init__(self, log_dir, resume=False):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # 文件名格式：20260220_1800.log
        suffix = "resume" if resume else "start"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        self.log_path = os.path.join(self.log_dir, f"train_{timestamp}_{suffix}.log")
        
        # 初始写入
        self.info(f"Logger initialized at {self.log_path}")

    def _format_msg(self, level, *args, **kwargs):
        """格式化消息，支持可变参数"""
        # 合并所有位置参数，模仿 print()
        msg = " ".join(map(str, args))
        # 处理关键字参数中的特定前缀（可选）
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{level}] {msg}"

    def info(self, *args):
        """通用信息记录"""
        msg = self._format_msg("INFO", *args)
        self._write(msg)

    def warn(self, *args):
        """警告信息记录"""
        msg = self._format_msg("WARN", *args)
        self._write(msg)

    def error(self, *args):
        """错误信息记录"""
        msg = self._format_msg("ERROR", *args)
        self._write(msg)

    def _write(self, msg):
        """写入文件并同步输出到控制台"""
        print(msg)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush() # 核心：实时刷入磁盘，防止崩溃丢失

    def log_params(self, epoch, iter, losses: dict, params: dict = None):
        """
        专为 ANM 训练优化的格式化方法
        losses: {'L1': 0.1, 'Noise': 0.05}
        params: {'alpha': (min, max), 'beta': (min, max)}
        """
        loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in losses.items()])
        msg = f"Epoch [{epoch}] Iter [{iter}] >> {loss_str}"
        
        if params:
            param_str = " | ".join([f"{k}: [{v[0]:.2e}, {v[1]:.2e}]" for k, v in params.items()])
            msg += f" || {param_str}"
            
        self.info(msg)