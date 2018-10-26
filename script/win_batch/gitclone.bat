REM "C:\\Program Files\\Git\\bin\\sh.exe" --login -i -c "git clone --progress  git@gitx.com:A_project/specialA.git 2>&1 | tee -a log.txt"
SET LOG_FILE=%1

"C:\\Program Files\\Git\\bin\\sh.exe" --login -i -c "git clone --progress  git@gitx.com:A_project/specialA.git 2>&1 | tee -a %LOG_FILE% "
exit