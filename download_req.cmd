@echo off

set installer_url=https://repo.anaconda.com/archive/Anaconda3-2021.05-Windows-x86_64.exe
set installer_file=Anaconda3-2021.05-Windows-x86_64.exe
set install_path=C:\Anaconda3

echo Downloading Anaconda installer...
curl -o "%installer_file%" "%installer_url%"

echo Installing Anaconda...
start /wait "" "%installer_file%" /InstallationType=JustMe /AddToPath=1 /RegisterPython=0 /S /D="%install_path%"

echo Cleaning up installer...
del "%installer_file%"

echo Creating a new environment...
call "%install_path%\Scripts\activate.bat"
conda create --name ShellEnv python=3.10
conda activate ShellEnv

echo Installing dependencies...
pip install -r requirements.txt