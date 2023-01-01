from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
import json, numpy as np
import os
# from main import czhou_classification as czhou
# from main import biLSTM_classification as lstm
from main import roberta_classification as robert

import requests
from lxml import etree


tables_path = "main/static/table"
model_path = "model"
chart_type = ["Scatterplot", "Bar Chart", "Line Chart", "Pie Chart"]
task_to_chart = {   # Characterize Distribution (7) 增加 pie (3)
    0: [0, 1, 2, 3], 1: [3, 1], 2: [1], 3: [3], 4: [1, 2, 0], 5: [1, 2, 0],
    6: [0, 2, 3, 1], 7: [0, 1, 2, 3], 8: [1, 0, 3], 9: [3, 1, 0], 10: [2, 0, 1]
}

categories = ['Others', 'Retrieve Value', 'Filter', 'Compute Derived Value', 'Find Extremum', 'Sort',
              'Determine Range', 'Characterize Distribution', 'Find Anomalies', 'Cluster', 'Correlate']

'''
Retrieve Value
Filter
Compute Derived Value
Find Extremum
Sort
Determine Range
Characterize Distribution
Find Anomalies
Cluster
Correlate
'''

with open("main/static/table/table.json", "r", encoding='utf-8') as fp:
    tables_info = json.load(fp)
tables_name = list(tables_info.keys())

freenli_tables_name = []
log_tables_name = []
for ti in tables_name:
    if not ti.startswith("log_"):
        freenli_tables_name.append(ti)
    else:
        log_tables_name.append(ti)

current_log_table = log_tables_name[0]


# Create your views here.
def index(request):
    if request.method == "GET":
        '''
        for parent, dirnames, filename in os.walk(tables_path):
            tables_name = filename
        '''
        table_name = freenli_tables_name[0]
        table_path = "/".join([tables_path, table_name])
        if table_name.endswith(".csv"):
            table = pd.read_csv(table_path)
        elif table_name.endswith(".json"):
            table = pd.read_json(table_path)
        else:   # elif file_name.endswith(".xlsx"):
            table = pd.read_excel(table_path)

        columns = list(tables_info[table_name].keys())

        return render(request, "index.html", {"tables_name": freenli_tables_name, "table_columns": columns, "current_table": "",
                                              "table_values": json.loads(table[columns].to_json(orient='values'))})
    '''
    if request.method == "POST":
        try:
            file_obj = request.FILES.get("table")
            file_name = str(time.time()) + file_obj.name

            file = open(file_name, "wb")
            for i in file_obj.chunks():
                file.write(i)
            file.close()

            if file_name.endswith(".csv"):
                table = pd.read_csv(file_name)
            # elif file_name.endswith(".xlsx"):
            else:
                table = pd.read_excel(file_name)

            return JsonResponse({"result": True, "name": file_name, "table_columns": list(table.columns),
                                 "table_values": json.loads(table.iloc[:5].to_json(orient='values'))})  # to_json将DataFrame转为json
        except Exception as e:
            print(repr(e))
            return JsonResponse({"result": False})
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)
        # return render(request, "index.html", {"result": "AIforVisPOST", "data": file_name})
    '''
    return None


def get_table(request):
    if request.method == "GET":
        try:
            table_name = request.GET.get("table_name")
            table_path = "/".join([tables_path, table_name])
            if table_name.endswith(".csv"):
                table = pd.read_csv(table_path)
            elif table_name.endswith(".json"):
                table = pd.read_json(table_path)
            else:  # elif file_name.endswith(".xlsx"):
                table = pd.read_excel(table_path)

            columns = list(tables_info[table_name].keys())
            if table_name.startswith("log_"):
                global current_log_table
                current_log_table = table_name

            return JsonResponse({"result": True, "table_columns": columns,
                                 "table_values": json.loads(table[columns].to_json(orient='values'))})  # to_json将DataFrame转为json
        except Exception as e:
            return JsonResponse({"result": False, "error": repr(e)})
    return None


def nlp_analyse(request):
    if request.method == "GET":
        try:
            nlp_text = request.GET.get("nlp_text")
            table_name = request.GET.get("table_name")
            columns_name = json.loads(request.GET.get("columns_name"))

            columns_type = []

            # task = int(request.GET.get("task"))
            # task = int(json.dumps(task_int64, cls=NpEncoder))  # task_int64 是numpy.int64类型的，无法序列化
            task = robert.predict([nlp_text])[0]
            charts_id = []
            task_name = []

            for ti in task:
                charts_id.append(task_to_chart[ti])
                task_name.append(categories[ti])

            table_path = "/".join([tables_path, table_name])
            if table_name.endswith(".csv"):
                table = pd.read_csv(table_path)
            elif table_name.endswith(".json"):
                table = pd.read_json(table_path)
            else:  # elif file_name.endswith(".xlsx"):
                table = pd.read_excel(table_path)

            for cni in columns_name:
                columns_type.append(tables_info[table_name][cni])

            '''
            data = []
            for column, type in tables_info[table_name].items():
                if column.lower() in sentence:
                    columns_involved.append(type)
                    data.append(list(table[column]))
            '''

            return JsonResponse({"result": True, "charts": charts_id, "columns_type": columns_type, "task_name": task_name,
                                 "task": task, "data": json.loads(table[columns_name].to_json(orient='values'))})
        except Exception as e:
            return JsonResponse({"result": False, "error": repr(e)})
    return None


def get_sentences_tasks(request):
    try:
        str_sentences = request.GET.get("sentences") if request.method == "GET" else request.POST.get("sentences")
        sentences = json.loads(str_sentences)
        task_result = robert.predict(sentences)

        ip, location = get_ip_info(request)
        with open("./main/static/table/log_api.csv", "a", encoding='utf-8') as fp:
            fp.write("\n%s,%s,%s,%s" % (pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ip, "", location))

        return JsonResponse({"result": True, "tasks": json.dumps(task_result)})  # , cls=NpEncoder
    except Exception as e:
        return JsonResponse({"result": False, "error": repr(e)})


def classification(request):
    return render(request, "sentences_classification.html")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def access_log(request):
    if request.method == "GET":
        log = pd.read_csv("./main/static/table/" + current_log_table)
        return render(request, "index.html", {"tables_name": log_tables_name, "table_columns": ['Time', 'Ip', 'Id', 'Location'],
                                              "table_values": json.loads(log.to_json(orient='values')), "current_table": current_log_table})
    elif request.method == "POST":
        try:
            record = ["\n%s" % pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            type = request.POST.get("type")
            table_name = "log_%s.csv" % type
            if not tables_info.get(table_name):
                print(table_name, type)
                return JsonResponse({"result": False, "error": "The log_file does not exist!"})
            for key, value in dict(request.POST).items():
                if key == "type":
                    continue
                # recode += ",%s" % value[0]
                # recode += ",%s" % request.POST.get(key)  # 这种方式会有错误 UnicodeEncodeError: 'ascii' codec can't encode character u'\xa1' in position 0: ordinal not in range(128)
                record.append(str(request.POST.get(key)))
                # print(key, value)
            with open("./main/static/table/" + table_name, "a", encoding='utf-8') as fp:
                fp.write(",".join(record))
        except Exception as e:
            return JsonResponse({"result": False, "error": repr(e)})
        return JsonResponse({"result": True})



def get_ip_info(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')  # 判断是否使用代理
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]  # 使用代理获取真实的ip
    else:
        ip = request.META.get('REMOTE_ADDR')  # 未使用代理获取IP

    url = 'https://www.hao7188.com/ip/%s.html' % ip
    try:
        result = requests.get(url).text
        parse_html = etree.HTML(result)
        location = parse_html.xpath('//body/div/table[1]/tbody/tr[2]/td/span/text()')[0]
    except Exception as e:
        location = ""

    return [ip, location]


def create_log(request):
    try:
        if request.method == "GET":
            type = request.GET.get("type")
            property = json.loads(request.GET.get("property", '["Time", "Ip", "Id", "Location"]'))
            property_type = json.loads(request.GET.get("property_type", '[0, 0, 1, 0]'))
            force = int(request.GET.get("force", '0'))

            table_name = "log_%s.csv" % type
            if tables_info.get(table_name) and not force:
                return JsonResponse({"result": False, "error": "The log_file is already existed!"})

            with open("./main/static/table/" + table_name, "w", encoding='utf-8') as fp:
                fp.write(','.join(property))

            tables_info[table_name] = {}
            for pi in range(len(property)):
                tables_info[table_name][property[pi]] = property_type[pi]
            log_tables_name.append(table_name)

            return JsonResponse({"result": True})
    except Exception as e:
        return JsonResponse({"result": False, "error": repr(e)})
    finally:
        with open("main/static/table/table.json", "w", encoding='utf-8') as fp:
            json.dump(tables_info, fp, indent=2)


def delete_log(request):
    try:
        if request.method == "GET":
            global current_log_table
            type = request.GET.get("type")

            table_name = "log_%s.csv" % type
            table_path = "./main/static/table/" + table_name
            if os.path.exists(table_path):
                os.remove(table_path)
                tables_info.pop(table_name, 'no such key')
                log_tables_name.remove(table_name)
                if current_log_table == table_name:
                    current_log_table = log_tables_name[0]
            return JsonResponse({"result": True})
    except Exception as e:
        return JsonResponse({"result": False, "error": repr(e)})
    finally:
        with open("main/static/table/table.json", "w", encoding='utf-8') as fp:
            json.dump(tables_info, fp, indent=2)
