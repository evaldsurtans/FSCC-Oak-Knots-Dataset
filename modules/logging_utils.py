import json
import os
import sys
import threading
import traceback
from datetime import datetime
import time
from modules.file_utils import FileUtils
import os
import psutil
import numpy as np

from modules.json_utils import json_dumps

rootLogger = None
current_logger_path = None
current_logger_suffix = 'auto'

# https://misc.flogisoft.com/bash/tip_colors_and_formatting
MAPPING = {
    'DEBUG'   : 92, # green
    'INFO'    : 36, # cyan
    'WARNING' : 33, # yellow
    'ERROR'   : 31, # red
    'CRITICAL': 41, # white on red bg
}

PREFIX = '\033['
SUFFIX = '\033[0m'

def log_process_id(process_name, event, args):
    try:
        if process_name is not None and args.is_proc_tracker:
            pid = os.getpid()
            tid = threading.get_ident()
            process = psutil.Process(pid)
            mem_usage = process.memory_info().rss // (1024 * 1024)

            # needed for stats
            cpu_usages = []
            for _ in range(10):
                time.sleep(0.01)
                cpu_usages.append(process.cpu_percent(interval=0.01) / psutil.cpu_count())
            cpu_usage = np.mean(cpu_usages)

            FileUtils.createDir('./logs')
            filename = os.path.abspath('./logs/' + datetime.now().strftime('%y-%m-%d_pid') + '.csv')
            mode = 'w'
            if os.path.exists(filename):
                mode = 'a'
            with open(filename, mode) as fp:
                FileUtils.lock_file(fp)
                if mode == 'w':
                    fp.write(','.join(['pid', 'tid', 'process_name', 'event', 'mem_usage', 'cpu_usage']) + '\n')
                fp.write(','.join([str(it) for it in [pid, tid, process_name, event, mem_usage, cpu_usage]]) + '\n')
                FileUtils.unlock_file(fp)
    except Exception as e:
        print(str(e))

IS_SENDING_ERROR_TO_DEVELOPERS = False

class LoggingUtils:
    def __init__(self):
        LoggingUtils.create()

    @staticmethod
    def current_file_name():
        global current_logger_suffix
        filename = os.path.abspath('./logs/' + datetime.utcnow().strftime(f'%y-%m-%d_{current_logger_suffix}') + '.log')
        return filename

    @staticmethod
    def remove_handlers():
        global rootLogger, current_logger_path, current_logger_suffix
        current_logger_path = None
        current_logger_suffix = 'auto'

    @staticmethod
    def create(suffix=None):
        global rootLogger, current_logger_path, current_logger_suffix
        try:
            if suffix is not None:
                current_logger_suffix = suffix
            filename = LoggingUtils.current_file_name()
            if current_logger_path == filename and os.path.exists(filename):
                return

            current_logger_path = filename
            FileUtils.createDir('./logs')

        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def write_log(message, color_code):
        global current_logger_path
        try:
            LoggingUtils.create() # make sure new file every day is being created
            str_datetime = datetime.utcnow().strftime("%Y.%m.%d %H:%M:%S")
            print(f'{str_datetime} [{os.getpid()}] [{threading.current_thread().name}] {PREFIX}{MAPPING[color_code]}m[{color_code}]{SUFFIX} {message}')
            with open(current_logger_path, 'a' if os.path.exists(current_logger_path) else 'w') as fp:
                fp.write(f'{str_datetime} [{os.getpid()}] [{threading.current_thread().name}] [{color_code}] {message}\n')
        except:
            pass

    @staticmethod
    def info(message):
        try:
            LoggingUtils.write_log(message, 'INFO')
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def debug_dict(dict_value):
        try:
            if not isinstance(dict_value, dict):
                dict_value = dict_value.__dict__
            LoggingUtils.write_log(json_dumps(dict_value, indent=4), 'DEBUG')
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def debug(message):
        try:
            LoggingUtils.write_log(message, 'DEBUG')
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def error(message, is_send_to_developers=True):
        try:
            message = message.replace("\\n", "\n")
            LoggingUtils.write_log(message, 'ERROR')
            if is_send_to_developers:
                LoggingUtils.send_error_to_developers('message', message)
            time.sleep(1)
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def warn(message):
        try:
            LoggingUtils.write_log(message, 'WARNING')
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    @staticmethod
    def exception(e, extra_message=''):
        LoggingUtils.error(str(e), is_send_to_developers=False)
        exc_type, exc_value, exc_tb = sys.exc_info()
        desc = '\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)).replace('\\n', '\n')
        desc = desc.replace("\\n", "\n")
        LoggingUtils.error(desc + extra_message, is_send_to_developers=False)
        if len(extra_message):
            extra_message = ' \n' + extra_message
        LoggingUtils.send_error_to_developers(str(e), desc + extra_message)
        time.sleep(1)


    @staticmethod
    def send_error_to_developers(msg, desc='', is_force=False):
        try:
            if 'asya_ver' in os.environ and os.environ['asya_ver']:
                if os.environ['asya_ver'] == 'production':
                    global IS_SENDING_ERROR_TO_DEVELOPERS
                    if not IS_SENDING_ERROR_TO_DEVELOPERS:
                        IS_SENDING_ERROR_TO_DEVELOPERS = True

                        excludes = [
                            'email_not_registered',
                            'conversation_id is missing',
                            'hangup detected',
                            'email_already_registered',
                            'connect to iTunes',
                            'setJSExceptionHandler',
                            'Billing is unavailable',
                            'timeout',
                            'Connection reset by peer',
                            'WebSocket.error',
                            'Connection timed out',
                            'email_not_confirmed',
                            'fetchWithTimeout',
                            'already_registered_waiting_for_confirmation',
                            'RNBackgroundFetch',
                            'connection appears to be offline',
                            'timed out',
                            '_saveImage',
                            'Broken pipe',
                            'in the simulator',
                            'simulator',
                            'auth_token_invalid',
                            'Recipient syntax error',
                            'Connection refused',
                            '.php',
                            'AuthorizationError error 1001',
                            'INVALID_STATE_ERR',
                            'E_USER_CANCELLED',
                            'The operation couldn’t be completed',
                            'Nothing to pop',
                            'Nothing to dismiss',
                            'Nothing to',
                            'Nothing to popeen expired for too long'

                        ]
                        desc = str(desc)
                        is_excluded = False
                        for exclude in excludes:
                            if exclude in desc or exclude in msg:
                                is_excluded = True
                                break

                        if not is_excluded or is_force:
                            from modules_core.manager_push_notifications import ManagerPushNotifications, PushNotificationTypes
                            from modules_core.manager_user_authentification import ManagerUserAuthentification

                            ManagerUserAuthentification.send_mail_notification(
                                'evalds@asya.ai',
                                f'server time:{int(time.time())} ' + msg,
                                desc
                            )
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))
        IS_SENDING_ERROR_TO_DEVELOPERS = False
