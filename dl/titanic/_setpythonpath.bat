REM set DOCK="C:\Program Files\Docker\Docker\Resources\bin"
REM set CANACONDA3="C:\ProgramData\Anaconda3";"C:\ProgramData\Anaconda3\Library\mingw-w64\bin";"C:\ProgramData\Anaconda3\Library\usr\bin";"C:\ProgramData\Anaconda3\Library\bin";"C:\ProgramData\Anaconda3\Scripts"
REM set JAVA="C:\ProgramData\Oracle\Java\javapath"
set DANACONDA3=D:\nb_tmp\program_files\Anaconda3
set ANDESTECH=D:\nb_tmp\program_files\Andestech\AndeSight201MCU\toolchains\nds32le-elf-mculib-v3m\bin
set MAKETOOLS=D:\nb_tmp\program_files\Andestech\Tools
REM set TOOLSDIR=".\Tools";".\Tools\NMake"
REM set PATH=%ANDESTECH%;%TOOLSDIR%;%DOCK%;%CANACONDA3%;%JAVA%;%DANACONDA3%;%PATH%
set PATH=%MAKETOOLS%;%ANDESTECH%;%DANACONDA3%;%PATH%