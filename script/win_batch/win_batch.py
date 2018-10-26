import time
from subprocess import Popen, PIPE

cmdInterface=1
#command Interface 1: Local command(windows) 2: Command via SSH 3: Local Command (Linux)
#test loop number
testNumber=0


activeCommand = {
    'gitclone'       : "start gitclone.bat",
    'gitpull'        : "start xxxxx.bat ",}

def get_command(strCommand):
    print('origin:',  strCommand)
    print('get:', activeCommand.get(strCommand))
    if activeCommand.get(strCommand) == None:
        return 0
    else:
        return (''.join(activeCommand.get(strCommand)))

def testHeader(outstream):
    out_buffer = ''
    if outstream==0:
        print(decode('+-------------------------------------%'))
        print(decode('|'),'                              Test No  '+str(testNumber)+'                                ')
        print(decode('|'),str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        print(decode('p-------------------------------------q'))
    elif outstream==1:
        out_buffer += "=======================================\n"
        out_buffer += "|       Test No " + str(testNumber)+"\n"
        out_buffer += "=======================================\n"
        out_buffer += str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))+'\n'
        return out_buffer


def save_file_log(fileString, data_buffer, openType):
    file = open(fileString, openType)
    file.write(data_buffer)
    file.close()


def OEM_winCommand_w_log_test(commandString):
    time.sleep(1)

    commandResultString=''
    if cmdInterface == 1:
        commandResultString = commandString + '_win'
    elif cmdInterface == 3:
        commandResultString=commandString+'_linux'
    
    commandResultString+='.txt'
    save_file_log(commandResultString, testHeader(1), "a+")
    testItem = 'Execute Command:'+ commandString +' by Win/Linux Test'
    save_file_log(commandResultString, winCommandExecute(get_command(commandString)+ ' '+commandResultString), "a+")


def winCommandExecute(command):
    p = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    stdout_data, stderr_data = p.communicate()
    if cmdInterface == 1:
        print(stdout_data.decode('big5'))
    elif cmdInterface == 3:
        print(stdout_data.decode())
    #save_file_log("this_is_test", stdout_data.decode('big5'), "a+")
    if stderr_data == '':
        print('Execute: ', command, ' Fail')
        print(stderr_data)
        if cmdInterface == 1:
            return stderr_data.decode('big5')
        elif cmdInterface == 3:
            return stdout_data.decode()
    else:
        print('Execute: ', command, ' Success')
        if cmdInterface == 1:
            return stdout_data.decode('big5')
        elif cmdInterface == 3:
            return stdout_data.decode()

def main():
    OEM_winCommand_w_log_test("gitclone")


if __name__ == '__main__':
    main()


