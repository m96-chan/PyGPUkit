@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d D:\Projects\m96-chan\PyGPUkit
nvcc -arch=sm_86 dump_fragments.cu -o dump_fragments.exe
if exist dump_fragments.exe (
    echo Compilation succeeded
    dump_fragments.exe
) else (
    echo Compilation failed
)
