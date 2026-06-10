@echo off 
setlocal 
set PY=D:\annaconda\envs\SN\python.exe 
set BATCH=100 
set START=0 
if not exist %%PY%% (echo Python not found: %%PY%% & goto :end) 
:loop 
echo Running %%PY%% json_converter.py --start %%START%% --limit %%BATCH%% 
%%PY%% json_converter.py --start %%START%% --limit %%BATCH%% 
if errorlevel 1 goto :end 
goto :loop 
set /a START=%%START%%+%%BATCH%% 
:end 
echo Stopped at start=%%START%% 
endlocal 
