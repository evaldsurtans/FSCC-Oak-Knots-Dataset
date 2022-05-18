import os
import time
import logging
import sys
import traceback
import traceback, sys

from modules.file_utils import FileUtils

class CsvUtils(object):

    # results for group of tests
    @staticmethod
    def create(args):
        try:
            if args.report and len(args.report) > 0:
                filename = os.path.join('reports', args.report) + '.csv'
                if not os.path.exists(filename):
                    headers = args.params_report
                    if not args.params_grid is None:
                        headers += args.params_grid
                    with open(filename, 'w') as outfile:
                        FileUtils.lock_file(outfile)
                        outfile.write(','.join(headers) + '\n')
                        outfile.flush()
                        os.fsync(outfile)
                        FileUtils.unlock_file(outfile)
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    # results for group of tests
    @staticmethod
    def add_results(args, state):
        try:
            if args.report and len(args.report) > 0:
                filename = os.path.join('reports', args.report) + '.csv'

                if not os.path.exists(filename):
                    if not os.path.exists('./reports'):
                        os.mkdir('./reports')
                    with open(filename, 'w') as outfile:
                        FileUtils.lock_file(outfile)
                        outfile.write(','.join(args.params_report) + '\n')
                        outfile.flush()
                        os.fsync(outfile)
                        FileUtils.unlock_file(outfile)

                lines_all = []
                with open(filename, 'r+') as outfile:
                    FileUtils.lock_file(outfile)
                    raw_lines = outfile.readlines()
                    if len(raw_lines) > 0:
                        header_line = raw_lines[0].strip()
                        headers = header_line.split(',')
                    else:
                        headers = args.params_report
                        lines_all.append(headers)

                    for line in raw_lines:
                        line = line.strip()
                        if len(line) > 0 and ',' in line:
                            parts = line.split(',')
                            lines_all.append(parts)

                    line_new = []
                    for key in headers:
                        #! gather from state
                        if key in state:
                            line_new.append(str(state[key]))
                        # ! gather also from args
                        elif key in vars(args):
                            line_new.append(str(getattr(args, key)))
                        # ! if not found empty
                        else:
                            line_new.append('')

                    # look for existing line to override
                    part_idx_id = headers.index('id')
                    is_exist = False
                    try:
                        for idx_line in range(1, len(lines_all)):
                            parts = lines_all[idx_line]
                            part_id = parts[part_idx_id]
                            if str(args.id) == part_id.strip():
                                lines_all[idx_line] = line_new
                                is_exist = True
                                break
                    except Exception as e:
                        print(str(e))
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        print(traceback.format_exception(exc_type, exc_value, exc_tb))

                    if not is_exist:
                        lines_all.append(line_new)

                    outfile.truncate(0)
                    outfile.seek(0)
                    outfile.flush()
                    rows = [','.join(it) for it in lines_all]
                    outfile.write('\n'.join(rows))
                    outfile.flush()
                    os.fsync(outfile)
                    FileUtils.unlock_file(outfile)
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    # results for each test instance/task
    @staticmethod
    def create_local(args):
        try:
            if args.name and len(args.name) > 0:
                filename = os.path.join('runs', args.name, args.name) + '.csv'
                if not os.path.exists(filename):
                    headers = args.params_report_local
                    with open(filename, 'w') as outfile:
                        FileUtils.lock_file(outfile)
                        outfile.write(','.join(headers) + '\n')
                        outfile.flush()
                        os.fsync(outfile)
                        FileUtils.unlock_file(outfile)
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))

    # results for each test instance/task
    @staticmethod
    def add_results_local(args, state):
        try:
            if args.report and len(args.report) > 0:
                filename = os.path.join('runs', args.name, args.name) + '.csv'
                if os.path.exists(filename):
                    with open(filename, 'r') as outfile:
                        header_line = outfile.readline().strip()
                        headers = header_line.split(',')

                    values = []
                    for key in headers:
                        if key in state:
                            values.append(str(state[key]))
                        elif key in vars(args):
                            values.append(str(getattr(args, key)))
                        else:
                            values.append('')

                    with open(filename, 'a') as outfile:
                        outfile.write(','.join(values) + '\n')
                else:
                    print(f'missing: {filename}')
        except Exception as e:
            print(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(traceback.format_exception(exc_type, exc_value, exc_tb))