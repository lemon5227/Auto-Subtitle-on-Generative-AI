# 基于生成式AI的视频自动生成系统

使用了OpenAI的Whisper大模型，所以英文识别效果最好，如果需要其他语言可以选择其他特化的大模型进行接入，未来会加入选择大模型功能

# install 安装
因为使用了Whisper 所以需要 Python 3.9.9 and PyTorch 1.10.1  可以使用miniconda配置环境，下面我会贴出我的conda info 以供参考
```
conda version : 23.11.0
    conda-build version : not installed
         python version : 3.11.5.final.0
                 solver : libmamba (default)
       virtual packages : __archspec=1=skylake
                          __conda=23.11.0=0
                          __cuda=12.5=0
                          __glibc=2.35=0
                          __linux=5.15.153.1=0
                          __unix=0=0
       base environment : /home/lemon/miniconda3  (writable)
      conda av data dir : /home/lemon/miniconda3/etc/conda
  conda av metadata url : None
           channel URLs : https://conda.anaconda.org/conda-forge/linux-64
                          https://conda.anaconda.org/conda-forge/noarch
                          https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /home/lemon/miniconda3/pkgs
                          /home/lemon/.conda/pkgs
       envs directories : /home/lemon/miniconda3/envs
                          /home/lemon/.conda/envs
               platform : linux-64
             user-agent : conda/23.11.0 requests/2.31.0 CPython/3.11.5 Linux/5.15.153.1-microsoft-standard-WSL2 ubuntu/22.04.4 glibc/2.35 solver/libmamba conda-libmamba-solver/23.12.0 libmambapy/1.5.3
                UID:GID : 1000:1000
             netrc file : None
           offline mode : False
```
配置好环境以后可以使用下面的命令运行
```
python web03.py
```
